"""
finetune_calibration.py - Three-stage fine-tuning pipeline.

Step 1: Precompute wav2vec2-large embeddings for all calibration files (GPU).
Step 2: Fine-tune LinearHead on cached embeddings with 80/20 val split.
Step 3: Threshold sweep (0.05-0.95) to maximise F1 on val split.

Outputs:
  models/finetuned/best_model.pth
  models/finetuned/threshold.txt
  logs/finetune_history.json
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
REAL_DIR   = Path(r"C:\vdfk_engine_dev\data\calibration\real")
FAKE_DIR   = Path(r"C:\vdfk_engine_dev\data\calibration\fake")
CACHE_DIR  = Path(r"C:\vdfk_engine_dev\data\calibration\embeddings")
W2V2_DIR   = Path(r"C:\vdfk_engine_dev\ml_engine\models\wav2vec2-large-960h")
BASE_MODEL = Path(r"C:\vdfk_engine_dev\ml_engine\models\base_model.pth")
OUT_DIR    = Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned")
LOG_DIR    = Path(r"C:\vdfk_engine_dev\ml_engine\logs")

SAMPLE_RATE  = 16000
TARGET_LEN_S = 10.0
TARGET_LEN   = int(SAMPLE_RATE * TARGET_LEN_S)
SEED         = 42
LR           = 1e-4
EPOCHS       = 5
PATIENCE     = 2
BATCH_SIZE   = 32
VAL_FRAC     = 0.20


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Model ─────────────────────────────────────────────────────────────────────
class LinearHead(nn.Module):
    def __init__(self, in_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Precompute embeddings
# ─────────────────────────────────────────────────────────────────────────────
def step1_precompute(device: torch.device):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    import librosa

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = (
        [(p, 0) for p in sorted(REAL_DIR.glob("*.wav"))] +
        [(p, 1) for p in sorted(FAKE_DIR.glob("*.wav"))]
    )
    n_real = sum(1 for _, l in files if l == 0)
    n_fake = sum(1 for _, l in files if l == 1)
    print(f"[Step 1] {len(files)} files  ({n_real} real, {n_fake} fake)")
    print(f"[Step 1] Cache: {CACHE_DIR}")
    print(f"[Step 1] Device: {device}")

    # Check how many already cached
    already = sum(
        1 for p, _ in files
        if (CACHE_DIR / f"{stable_hash(str(p.resolve()) + f'|{TARGET_LEN_S}')}.npy").exists()
    )
    if already == len(files):
        print(f"[Step 1] All {len(files)} embeddings already cached — skipping backbone load.\n")
        return files

    print(f"[Step 1] {already} cached, {len(files)-already} to compute. Loading backbone ...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(W2V2_DIR), local_files_only=True)
    backbone  = Wav2Vec2Model.from_pretrained(str(W2V2_DIR), local_files_only=True)
    backbone  = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"[Step 1] Backbone loaded. Starting embedding extraction ...\n")

    computed = skipped = errors = 0
    t0 = time.time()

    for i, (path, label) in enumerate(files, 1):
        key      = stable_hash(str(path.resolve()) + f"|{TARGET_LEN_S}")
        npy_path = CACHE_DIR / f"{key}.npy"

        if npy_path.exists():
            skipped += 1
        else:
            try:
                wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
                if len(wav) > TARGET_LEN:
                    wav = wav[:TARGET_LEN]
                elif len(wav) < TARGET_LEN:
                    wav = np.pad(wav, (0, TARGET_LEN - len(wav)), mode="constant")

                inputs = processor(wav, sampling_rate=SAMPLE_RATE,
                                   return_tensors="pt", padding=False)
                with torch.no_grad():
                    out    = backbone(inputs.input_values.to(device))
                    pooled = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                np.save(npy_path, pooled)
                computed += 1
            except Exception as e:
                print(f"  ERROR {path.name}: {e}")
                errors += 1

        if i % 500 == 0:
            elapsed = time.time() - t0
            rate    = (computed + skipped) / elapsed
            eta_s   = (len(files) - i) / max(rate, 1e-6)
            print(f"  [{i:>5}/{len(files)}] computed={computed} skipped={skipped} "
                  f"errors={errors} | {rate:.1f} f/s | ETA {eta_s/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\n[Step 1] Done: {computed} computed, {skipped} skipped, "
          f"{errors} errors — {elapsed/60:.1f} min\n")

    # Release backbone to free VRAM before training
    del backbone
    torch.cuda.empty_cache()
    return files


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Fine-tune head
# ─────────────────────────────────────────────────────────────────────────────
def _load_embeddings(files):
    X, y, missing = [], [], 0
    for path, label in files:
        key      = stable_hash(str(path.resolve()) + f"|{TARGET_LEN_S}")
        npy_path = CACHE_DIR / f"{key}.npy"
        if not npy_path.exists():
            missing += 1
            continue
        X.append(np.load(npy_path))
        y.append(label)
    if missing:
        print(f"  WARNING: {missing} embeddings missing from cache — skipped.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _eer(y_true, y_scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


def step2_finetune(files, device: torch.device):
    print("[Step 2] Loading embeddings into memory ...")
    X, y = _load_embeddings(files)
    print(f"[Step 2] {len(X)} embeddings loaded  shape={X.shape}  "
          f"({int(y.sum())} fake / {int((y==0).sum())} real)")

    # Stratified 80/20 split
    rng      = np.random.default_rng(SEED)
    real_idx = np.where(y == 0)[0]; rng.shuffle(real_idx)
    fake_idx = np.where(y == 1)[0]; rng.shuffle(fake_idx)

    n_val_real = max(1, int(len(real_idx) * VAL_FRAC))
    n_val_fake = max(1, int(len(fake_idx) * VAL_FRAC))

    val_idx   = np.concatenate([real_idx[:n_val_real],   fake_idx[:n_val_fake]])
    train_idx = np.concatenate([real_idx[n_val_real:],   fake_idx[n_val_fake:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx],   y[val_idx]
    print(f"[Step 2] Train: {len(X_tr)}  ({int(y_tr.sum())} fake)  "
          f"Val: {len(X_va)}  ({int(y_va.sum())} fake)\n")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                          batch_size=BATCH_SIZE, shuffle=False)

    # Load base model
    model = LinearHead(in_dim=X.shape[1]).to(device)
    model.load_state_dict(
        torch.load(str(BASE_MODEL), map_location='cpu', weights_only=False)
    )
    print(f"[Step 2] Base model loaded from {BASE_MODEL}")
    print(f"[Step 2] Training: lr={LR}, epochs={EPOCHS}, patience={PATIENCE}, batch={BATCH_SIZE}\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_eer = float('inf')
    patience_cnt = 0
    history = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  {'Epoch':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
          f"{'Va Loss':>8}  {'Va Acc':>7}  {'Va EER':>7}")
    print(f"  {'-'*52}")

    for epoch in range(1, EPOCHS + 1):
        # Train
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

        # Validate
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

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                   va_loss=va_loss, va_acc=va_acc, va_eer=va_eer)
        history.append(row)

        improved = va_eer < best_eer
        marker   = " => BEST" if improved else f" (no improvement {patience_cnt+1}/{PATIENCE})"
        print(f"  {epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.4f}  {va_eer:>7.4f}{marker}")

        if improved:
            best_eer = va_eer
            patience_cnt = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    print(f"\n[Step 2] Best val EER: {best_eer:.4f}")
    print(f"[Step 2] Best model saved -> {OUT_DIR / 'best_model.pth'}\n")

    with open(LOG_DIR / "finetune_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history, (X_va, y_va)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────
def step3_threshold_sweep(val_data, device: torch.device):
    X_va, y_va = val_data
    print(f"[Step 3] Threshold sweep on {len(X_va)} val files ...")

    model = LinearHead(in_dim=X_va.shape[1]).to(device).eval()
    model.load_state_dict(
        torch.load(str(OUT_DIR / "best_model.pth"), map_location=device, weights_only=False)
    )

    dl = DataLoader(TensorDataset(torch.from_numpy(X_va)), batch_size=128)
    preds = []
    with torch.no_grad():
        for (xb,) in dl:
            preds.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    preds = np.array(preds)

    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    best_f1, best_thr = 0.0, 0.5

    print(f"\n  {'Thr':>5}  {'F1':>7}  {'Acc':>7}  {'FPR':>7}  {'FNR':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*60}")

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
    print(f"\n[Step 3] Best threshold: {best_thr:.4f}  F1={best_f1:.4f}")
    print(f"[Step 3] Saved -> {thr_file}")
    return best_thr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        raise RuntimeError("GPU not available — check CUDA installation.")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    t_total = time.time()

    files             = step1_precompute(device)
    history, val_data = step2_finetune(files, device)
    best_thr          = step3_threshold_sweep(val_data, device)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} min")
    print(f"  Best model  -> {OUT_DIR}/best_model.pth")
    print(f"  Threshold   -> {best_thr:.4f}")
    print(f"  History     -> {LOG_DIR}/finetune_history.json")
    print(f"{'='*60}")
