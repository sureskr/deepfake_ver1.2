"""
finetune_v2.py - Fine-tune V2 on combined calibration set.

Sources:
  - Original in-the-wild: 4,726 files (embeddings already cached)
  - ASVspoof 2021 LA:     1,000 files (FLAC, embeddings to compute)

Starts from: models/finetuned/best_model.pth  (V1 checkpoint)
Saves to:    models/v1_finetuned_v2/best_model.pth
"""
import os, sys, hashlib, random, time, json, warnings
os.environ['WANDB_MODE'] = 'disabled'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_curve

# ── Paths ─────────────────────────────────────────────────────────────────────
# Original calibration (wav, cached embeddings)
ORIG_REAL_DIR  = Path(r"C:\vdfk_engine_dev\data\calibration\real")
ORIG_FAKE_DIR  = Path(r"C:\vdfk_engine_dev\data\calibration\fake")
ORIG_CACHE_DIR = Path(r"C:\vdfk_engine_dev\data\calibration\embeddings")

# ASVspoof LA calibration (flac, needs embedding compute)
LA_REAL_DIR    = Path(r"C:\vdfk_engine_dev\data\calibration\asvspoof2021_LA\real")
LA_FAKE_DIR    = Path(r"C:\vdfk_engine_dev\data\calibration\asvspoof2021_LA\fake")
LA_CACHE_DIR   = Path(r"C:\vdfk_engine_dev\data\calibration\asvspoof2021_LA\embeddings")

W2V2_DIR       = Path(r"C:\vdfk_engine_dev\ml_engine\models\wav2vec2-large-960h")
CHECKPOINT     = Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned\best_model.pth")
OUT_DIR        = Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_finetuned_v2")
LOG_DIR        = Path(r"C:\vdfk_engine_dev\ml_engine\logs")

SAMPLE_RATE  = 16000
TARGET_LEN_S = 10.0
TARGET_LEN   = int(SAMPLE_RATE * TARGET_LEN_S)
SEED         = 42
LR           = 1e-4
EPOCHS       = 10
PATIENCE     = 3
BATCH_SIZE   = 32
VAL_FRAC     = 0.20
WEIGHT_FAKE  = 1.7


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


# ── Step 1: Precompute LA embeddings (FLAC via miniaudio) ─────────────────────
def precompute_la_embeddings(device):
    import miniaudio
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    files = (
        [(p, 0) for p in sorted(LA_REAL_DIR.glob("*.flac"))] +
        [(p, 1) for p in sorted(LA_FAKE_DIR.glob("*.flac"))]
    )
    LA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    already = sum(
        1 for p, _ in files
        if (LA_CACHE_DIR / f"{stable_hash(str(p.resolve()) + f'|{TARGET_LEN_S}')}.npy").exists()
    )
    if already == len(files):
        print(f"[Step 1] All {len(files)} LA embeddings cached — skipping.\n")
        return

    to_compute = len(files) - already
    print(f"[Step 1] LA: {already} cached, {to_compute} to compute. Loading backbone ...")

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
        npy = LA_CACHE_DIR / f"{key}.npy"

        if npy.exists():
            skipped += 1
            continue
        try:
            decoded = miniaudio.flac_read_file_f32(str(path))
            wav = np.array(decoded.samples, dtype=np.float32)
            if decoded.nchannels > 1:
                wav = wav.reshape(-1, decoded.nchannels).mean(axis=1)
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


# ── Step 2: Load all embeddings from both caches ──────────────────────────────
def load_all_embeddings():
    sources = [
        # (real_dir, fake_dir, cache_dir, glob_pattern, label)
        (ORIG_REAL_DIR, ORIG_FAKE_DIR, ORIG_CACHE_DIR, "*.wav"),
        (LA_REAL_DIR,   LA_FAKE_DIR,   LA_CACHE_DIR,   "*.flac"),
    ]

    X, y = [], []
    stats = []
    for real_dir, fake_dir, cache_dir, pattern in sources:
        files = (
            [(p, 0) for p in sorted(real_dir.glob(pattern))] +
            [(p, 1) for p in sorted(fake_dir.glob(pattern))]
        )
        loaded = missing = 0
        for path, label in files:
            key = stable_hash(str(path.resolve()) + f"|{TARGET_LEN_S}")
            npy = cache_dir / f"{key}.npy"
            if not npy.exists():
                missing += 1
                continue
            X.append(np.load(npy))
            y.append(label)
            loaded += 1
        stats.append((cache_dir.parent.name, loaded, missing))

    for name, loaded, missing in stats:
        flag = f"  WARNING: {missing} missing" if missing else ""
        print(f"  {name}: {loaded} embeddings loaded{flag}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Step 3: Threshold sweep on validation set ─────────────────────────────────
def threshold_sweep(model, X_va, y_va, device):
    dl = DataLoader(TensorDataset(torch.from_numpy(X_va)), batch_size=256)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            preds.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    preds = np.array(preds)

    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    best_f1, best_thr = 0.0, 0.5

    print(f"\n  {'Thr':>5}  {'F1':>7}  {'Acc':>7}  {'FPR':>7}  {'FNR':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*62}")

    for thr in thresholds:
        pred_bin = (preds >= thr).astype(int)
        f1  = f1_score(y_va, pred_bin, zero_division=0)
        acc = accuracy_score(y_va, pred_bin)
        tp  = int(((pred_bin==1)&(y_va==1)).sum())
        fp  = int(((pred_bin==1)&(y_va==0)).sum())
        tn  = int(((pred_bin==0)&(y_va==0)).sum())
        fn  = int(((pred_bin==0)&(y_va==1)).sum())
        fpr = fp/(fp+tn) if (fp+tn) else 0.0
        fnr = fn/(fn+tp) if (fn+tp) else 0.0
        tag = " <-- best" if f1 > best_f1 else ""
        print(f"  {thr:>5.2f}  {f1:>7.4f}  {acc:>7.4f}  {fpr:>7.4f}  {fnr:>7.4f}  "
              f"{tp:>5}  {fp:>5}  {fn:>5}{tag}")
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    thr_file = OUT_DIR / "threshold.txt"
    thr_file.write_text(f"{best_thr:.4f}\n")
    print(f"\n  Best threshold: {best_thr:.4f}  (F1={best_f1:.4f})")
    print(f"  Saved -> {thr_file}")
    return best_thr, best_f1


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        raise RuntimeError("GPU not available.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    # ── Step 1: Precompute LA embeddings ──────────────────────────────────────
    precompute_la_embeddings(device)

    # ── Step 2: Load combined embeddings ─────────────────────────────────────
    print("Loading embeddings from both calibration sources ...")
    X, y = load_all_embeddings()
    n_real = int((y==0).sum()); n_fake = int((y==1).sum())
    print(f"Combined: {len(X)} total  ({n_fake} fake / {n_real} real)  shape={X.shape}\n")

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    rng      = np.random.default_rng(SEED)
    real_idx = np.where(y == 0)[0]; rng.shuffle(real_idx)
    fake_idx = np.where(y == 1)[0]; rng.shuffle(fake_idx)
    n_val_real = max(1, int(len(real_idx) * VAL_FRAC))
    n_val_fake = max(1, int(len(fake_idx) * VAL_FRAC))
    val_idx   = np.concatenate([real_idx[:n_val_real], fake_idx[:n_val_fake]])
    train_idx = np.concatenate([real_idx[n_val_real:], fake_idx[n_val_fake:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx],   y[val_idx]
    print(f"Train: {len(X_tr)}  ({int(y_tr.sum())} fake)  "
          f"Val: {len(X_va)}  ({int(y_va.sum())} fake)")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                          batch_size=BATCH_SIZE, shuffle=False)

    # ── Load V1 checkpoint ────────────────────────────────────────────────────
    model = LinearHead(in_dim=X.shape[1]).to(device)
    model.load_state_dict(
        torch.load(str(CHECKPOINT), map_location='cpu', weights_only=False)
    )
    print(f"\nLoaded checkpoint: {CHECKPOINT}")

    pos_weight = torch.tensor([WEIGHT_FAKE]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Loss: BCEWithLogitsLoss(pos_weight={WEIGHT_FAKE})")
    print(f"Settings: lr={LR}, epochs={EPOCHS}, patience={PATIENCE}, batch={BATCH_SIZE}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_eer     = float('inf')
    patience_cnt = 0
    history      = []

    print(f"  {'Epoch':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
          f"{'Va Loss':>8}  {'Va Acc':>7}  {'Va EER':>7}  {'Status'}")
    print(f"  {'-'*70}")

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses, tr_preds, tr_labels = [], [], []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            tr_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            tr_labels.extend(yb.cpu().numpy())

        tr_loss = float(np.mean(tr_losses))
        tr_acc  = accuracy_score(tr_labels, (np.array(tr_preds) > 0.5).astype(int))

        model.eval()
        va_losses, va_preds, va_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                va_losses.append(criterion(logits, yb).item())
                va_preds.extend(torch.sigmoid(logits).cpu().numpy())
                va_labels.extend(yb.cpu().numpy())

        va_preds  = np.array(va_preds)
        va_labels = np.array(va_labels)
        va_loss   = float(np.mean(va_losses))
        va_acc    = accuracy_score(va_labels, (va_preds > 0.5).astype(int))
        va_eer    = _eer(va_labels, va_preds)

        improved = va_eer < best_eer
        if improved:
            best_eer = va_eer
            patience_cnt = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pth")
            status = "BEST *"
        else:
            patience_cnt += 1
            status = f"no improvement {patience_cnt}/{PATIENCE}"

        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            va_loss=va_loss, va_acc=va_acc, va_eer=va_eer))

        print(f"  {epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.4f}  {va_eer:>7.4f}  {status}")

        if patience_cnt >= PATIENCE:
            print(f"\n  Early stopping — no improvement for {PATIENCE} epochs.")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min  |  Best val EER: {best_eer:.4f}")

    # ── Save history ──────────────────────────────────────────────────────────
    hist_file = LOG_DIR / "finetune_v2_history.json"
    with open(hist_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved -> {hist_file}")

    # ── Reload best and sweep threshold ───────────────────────────────────────
    model.load_state_dict(
        torch.load(str(OUT_DIR / "best_model.pth"), map_location=device, weights_only=False)
    )
    print(f"\n{'='*65}")
    print(f"  Threshold sweep on {len(X_va)} val files (best model reloaded)")
    print(f"{'='*65}")
    best_thr, best_f1 = threshold_sweep(model, X_va, y_va, device)

    print(f"\n{'='*65}")
    print(f"  V2 fine-tune complete in {elapsed/60:.1f} min")
    print(f"  Epochs trained  : {len(history)}")
    print(f"  Best val EER    : {best_eer:.4f}")
    print(f"  Best threshold  : {best_thr:.4f}  (F1={best_f1:.4f})")
    print(f"  Model saved     : {OUT_DIR}/best_model.pth")
    print(f"  Threshold saved : {OUT_DIR}/threshold.txt")
    print(f"{'='*65}")
