"""
infer_only.py - Inference-only evaluation using pre-cached embeddings.

Skips embedding precomputation entirely.
Reads threshold from threshold.txt.
"""
import os, sys, hashlib, time, csv, warnings
os.environ['WANDB_MODE'] = 'disabled'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

TEST_REAL  = Path(r"C:\vdfk_engine_dev\data\test\real")
TEST_FAKE  = Path(r"C:\vdfk_engine_dev\data\test\fake")
CACHE_DIR  = Path(r"C:\vdfk_engine_dev\data\test\embeddings")
MODEL_PATH = Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned\best_model.pth")
THR_PATH   = Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned\threshold.txt")
RESULTS    = Path(r"C:\vdfk_engine_dev\ml_engine\results\test_finetuned_t070_results.csv")

SAMPLE_RATE  = 16000
TARGET_LEN_S = 10.0
REPORT_EVERY = 1000

BASELINE_ACC = 0.6676
BASELINE_EER = 0.3333
TARGET_ACC   = 0.70
TARGET_EER   = 0.30
TARGET_FPR   = 0.15


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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    threshold = float(THR_PATH.read_text().strip()) if THR_PATH.exists() else 0.70
    print(f"Threshold: {threshold}")

    files = (
        [(p, 0) for p in sorted(TEST_REAL.glob("*.wav"))] +
        [(p, 1) for p in sorted(TEST_FAKE.glob("*.wav"))]
    )
    n_real = sum(1 for _, l in files if l == 0)
    n_fake = sum(1 for _, l in files if l == 1)
    print(f"Test set: {n_real} real + {n_fake} fake = {len(files)} files")
    print(f"Embeddings: {CACHE_DIR}\n")

    print(f"Loading model from {MODEL_PATH}")
    model = LinearHead(in_dim=1024).to(device).eval()
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    )
    print(f"Running inference at threshold={threshold}\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_scores, all_labels = [], []
    tp = tn = fp = fn = missing = 0
    t0 = time.time()

    for i, (path, label) in enumerate(files, 1):
        key = stable_hash(str(path.resolve()) + f"|{TARGET_LEN_S}")
        npy = CACHE_DIR / f"{key}.npy"

        if not npy.exists():
            missing += 1
            continue

        emb = torch.from_numpy(np.load(npy)).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(emb)).item()

        pred_fake   = score >= threshold
        actual_fake = label == 1

        if   pred_fake and     actual_fake: tp += 1
        elif not pred_fake and not actual_fake: tn += 1
        elif pred_fake and not actual_fake: fp += 1
        else:                               fn += 1

        all_scores.append(score)
        all_labels.append(label)

        rows.append({
            "file":       path.name,
            "true_label": "spoof" if actual_fake else "bona-fide",
            "verdict":    "FAKE"  if pred_fake   else "REAL",
            "correct":    int(pred_fake == actual_fake),
            "ml_score":   f"{score:.4f}",
        })

        if i % REPORT_EVERY == 0:
            total_so_far = tp + tn + fp + fn
            acc_so_far   = (tp + tn) / total_so_far if total_so_far else 0
            elapsed      = time.time() - t0
            rate         = i / elapsed
            eta_s        = (len(files) - i) / max(rate, 1e-6)
            print(f"  [{i:>6}/{len(files)}] acc={acc_so_far:.3f} "
                  f"tp={tp} tn={tn} fp={fp} fn={fn} | "
                  f"{rate:.0f} f/s | ETA {eta_s/60:.1f} min")

    elapsed = time.time() - t0

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total    if total     else 0.0
    fpr_val  = fp / (fp + tn)       if (fp + tn) else 0.0
    fnr_val  = fn / (fn + tp)       if (fn + tp) else 0.0
    try:
        auroc = roc_auc_score(np.array(all_labels), np.array(all_scores))
    except Exception:
        auroc = float('nan')
    eer = _eer(np.array(all_labels), np.array(all_scores))

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
    print(f"  TEST SET EVALUATION — Finetuned Model (threshold=0.70)")
    print(f"  Files: {total}  |  Threshold: {threshold}  |  Missing: {missing}")
    print(f"  {'Metric':<10}  {'Finetuned':>11}  {'Original':>11}  {'Target':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Accuracy':<10}  {accuracy*100:>10.2f}%  {BASELINE_ACC*100:>10.2f}%  {TARGET_ACC*100:>9.1f}%"
          f"  {'PASS' if accuracy > TARGET_ACC else 'FAIL'}")
    print(f"  {'EER':<10}  {eer*100:>10.2f}%  {BASELINE_EER*100:>10.2f}%  {TARGET_EER*100:>9.1f}%"
          f"  {'PASS' if eer < TARGET_EER else 'FAIL'}")
    print(f"  {'FPR':<10}  {fpr_val*100:>10.2f}%  {'?':>11}  {TARGET_FPR*100:>9.1f}%"
          f"  {'PASS' if fpr_val < TARGET_FPR else 'FAIL'}")
    print(f"  {'FNR':<10}  {fnr_val*100:>10.2f}%  {'?':>11}  {'—':>10}")
    print(f"  {'AUROC':<10}  {auroc:>11.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*W}")
    print(f"\n  Delta vs original baseline:")
    print(f"    Accuracy : {(accuracy - BASELINE_ACC)*100:+.2f}%")
    print(f"    EER      : {(eer - BASELINE_EER)*100:+.2f}%")
    print(f"\n  Results saved -> {RESULTS}")
    print(f"  Inference time: {elapsed:.1f} s")
    print(f"{'='*W}")
