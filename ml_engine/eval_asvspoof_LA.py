"""
eval_asvspoof_LA.py - Smoke test on ASVspoof 2021 LA eval split.

Filters to eval split only (col 7 == 'eval').
Samples 500 bonafide + 500 spoof (seed=42).
Runs wav2vec2 embedding + LinearHead inference at threshold=0.70.
"""
import os, sys, hashlib, random, time, csv, warnings
os.environ['WANDB_MODE'] = 'disabled'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ─────────────────────────────────────────────────────────────────────
FLAC_DIR   = Path(r"C:\vdfk_engine_dev\data\public_dataset\asvspoof_2021_archive\ASVspoof2021_LA_eval\ASVspoof2021_LA_eval\flac")
LABEL_FILE = Path(r"C:\vdfk_engine_dev\data\public_dataset\asvspoof_2021_archive\LA-keys-full\keys\LA\CM\trial_metadata.txt")
CACHE_DIR  = Path(r"C:\vdfk_engine_dev\data\asvspoof2021_LA\embeddings")
W2V2_DIR   = Path(r"C:\vdfk_engine_dev\ml_engine\models\wav2vec2-large-960h")
MODEL_PATH = Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned\best_model.pth")
RESULTS    = Path(r"C:\vdfk_engine_dev\ml_engine\results\asvspoof2021_LA_smoke_results.csv")

SAMPLE_RATE  = 16000
TARGET_LEN_S = 10.0
TARGET_LEN   = int(SAMPLE_RATE * TARGET_LEN_S)
THRESHOLD    = 0.70
SEED         = 42
N_PER_CLASS  = 500


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


class LinearHead(nn.Module):
    def __init__(self, in_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


def _eer(y_true, y_scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


# ── Load and sample label file ────────────────────────────────────────────────
def load_sample():
    bonafide, spoof = [], []
    with open(LABEL_FILE, "r") as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) < 8:
                continue
            file_id = cols[1]
            label   = cols[5]   # bonafide / spoof
            split   = cols[7]   # eval / progress
            if split != "eval":
                continue
            path = FLAC_DIR / f"{file_id}.flac"
            if label == "bonafide":
                bonafide.append((path, 0))
            elif label == "spoof":
                spoof.append((path, 1))

    rng = random.Random(SEED)
    rng.shuffle(bonafide)
    rng.shuffle(spoof)
    selected = bonafide[:N_PER_CLASS] + spoof[:N_PER_CLASS]
    rng.shuffle(selected)

    print(f"Label file: {len(bonafide)} bonafide + {len(spoof)} spoof in eval split")
    print(f"Sampled   : {N_PER_CLASS} bonafide + {N_PER_CLASS} spoof = {len(selected)} files")
    return selected


# ── Precompute embeddings ─────────────────────────────────────────────────────
def precompute_embeddings(files, device):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    import miniaudio

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    already = sum(
        1 for p, _ in files
        if (CACHE_DIR / f"{stable_hash(str(p.resolve()) + f'|{TARGET_LEN_S}')}.npy").exists()
    )
    if already == len(files):
        print(f"\n[Step 1] All {len(files)} embeddings cached — skipping backbone.\n")
        return

    to_compute = len(files) - already
    print(f"\n[Step 1] {already} cached, {to_compute} to compute. Loading backbone ...")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(W2V2_DIR), local_files_only=True)
    backbone  = Wav2Vec2Model.from_pretrained(str(W2V2_DIR), local_files_only=True)
    backbone  = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    print(f"[Step 1] Backbone on {device}. Extracting {to_compute} embeddings ...\n")

    computed = skipped = errors = 0
    t0 = time.time()

    for i, (path, _) in enumerate(files, 1):
        key = stable_hash(str(path.resolve()) + f"|{TARGET_LEN_S}")
        npy = CACHE_DIR / f"{key}.npy"

        if npy.exists():
            skipped += 1
            continue
        try:
            decoded = miniaudio.flac_read_file_f32(str(path))
            wav = np.array(decoded.samples, dtype=np.float32)
            # Mix down to mono if multi-channel
            if decoded.nchannels > 1:
                wav = wav.reshape(-1, decoded.nchannels).mean(axis=1)
            # Resample if needed
            if decoded.sample_rate != SAMPLE_RATE:
                import librosa as _lr
                wav = _lr.resample(wav, orig_sr=decoded.sample_rate, target_sr=SAMPLE_RATE)
            if len(wav) > TARGET_LEN:
                wav = wav[:TARGET_LEN]
            elif len(wav) < TARGET_LEN:
                wav = np.pad(wav, (0, TARGET_LEN - len(wav)), mode="constant")

            inputs = processor(wav, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=False)
            with torch.no_grad():
                out    = backbone(inputs.input_values.to(device))
                pooled = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            np.save(npy, pooled)
            computed += 1
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")
            errors += 1

    elapsed = time.time() - t0
    print(f"[Step 1] Done: {computed} computed, {skipped} skipped, "
          f"{errors} errors — {elapsed:.1f} s  ({computed/max(elapsed,1):.1f} f/s)\n")

    del backbone
    torch.cuda.empty_cache()


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(files, device):
    print(f"[Step 2] Loading model from {MODEL_PATH}")
    model = LinearHead(in_dim=1024).to(device).eval()
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    )
    print(f"[Step 2] Inference on {len(files)} files at threshold={THRESHOLD}\n")

    rows = []
    all_scores, all_labels = [], []
    tp = tn = fp = fn = missing = 0
    t0 = time.time()

    for path, label in files:
        key = stable_hash(str(path.resolve()) + f"|{TARGET_LEN_S}")
        npy = CACHE_DIR / f"{key}.npy"

        if not npy.exists():
            missing += 1
            continue

        emb = torch.from_numpy(np.load(npy)).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(emb)).item()

        pred_fake   = score >= THRESHOLD
        actual_fake = label == 1

        if   pred_fake and     actual_fake: tp += 1
        elif not pred_fake and not actual_fake: tn += 1
        elif pred_fake and not actual_fake: fp += 1
        else:                               fn += 1

        all_scores.append(score)
        all_labels.append(label)
        rows.append({
            "file":       path.name,
            "true_label": "spoof" if actual_fake else "bonafide",
            "verdict":    "FAKE"  if pred_fake   else "REAL",
            "correct":    int(pred_fake == actual_fake),
            "ml_score":   f"{score:.4f}",
        })

    elapsed = time.time() - t0
    return rows, np.array(all_scores), np.array(all_labels), tp, tn, fp, fn, missing, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device   : {device}")
    print(f"Model    : {MODEL_PATH}")
    print(f"Threshold: {THRESHOLD}\n")

    if device.type == 'cpu':
        raise RuntimeError("GPU not available — aborting.")

    t_total = time.time()

    files = load_sample()

    # Missing file check
    missing_paths = [p for p, _ in files if not p.exists()]
    if missing_paths:
        print(f"WARNING: {len(missing_paths)} files not found on disk, e.g. {missing_paths[0].name}")

    precompute_embeddings(files, device)

    rows, scores, labels, tp, tn, fp, fn, missing, infer_elapsed = run_inference(files, device)

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total    if total     else 0.0
    fpr_val  = fp / (fp + tn)       if (fp + tn) else 0.0
    fnr_val  = fn / (fn + tp)       if (fn + tp) else 0.0
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        auroc = float('nan')
    eer = _eer(labels, scores)

    elapsed_total = time.time() - t_total

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file","true_label","verdict","correct","ml_score"])
        writer.writeheader()
        writer.writerows(rows)
    with open(RESULTS, "a", newline="") as f:
        f.write(f"\nSUMMARY,accuracy={accuracy:.4f},EER={eer:.4f},"
                f"FPR={fpr_val:.4f},FNR={fnr_val:.4f},"
                f"TP={tp},TN={tn},FP={fp},FN={fn},AUROC={auroc:.4f}\n")

    W = 65
    print(f"\n{'='*W}")
    print(f"  ASVspoof 2021 LA — Smoke Test (500 bonafide + 500 spoof)")
    print(f"  Model    : {MODEL_PATH.name}")
    print(f"  Threshold: {THRESHOLD}  |  Missing embeddings: {missing}")
    print(f"  {'Metric':<10}  {'Value':>10}  {'Target':>10}")
    print(f"  {'-'*38}")
    print(f"  {'Accuracy':<10}  {accuracy*100:>9.2f}%  {'> 70%':>10}  {'PASS' if accuracy > 0.70 else 'FAIL'}")
    print(f"  {'EER':<10}  {eer*100:>9.2f}%  {'< 30%':>10}  {'PASS' if eer < 0.30 else 'FAIL'}")
    print(f"  {'FPR':<10}  {fpr_val*100:>9.2f}%  {'< 15%':>10}  {'PASS' if fpr_val < 0.15 else 'FAIL'}")
    print(f"  {'FNR':<10}  {fnr_val*100:>9.2f}%  {'—':>10}")
    print(f"  {'AUROC':<10}  {auroc:>10.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*W}")
    print(f"\n  Results saved -> {RESULTS}")
    print(f"  Total time: {elapsed_total:.1f} s  (inference: {infer_elapsed:.1f} s)")
    print(f"{'='*W}")
