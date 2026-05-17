"""
finetune_continue.py - Continue fine-tuning from saved checkpoint.

Loads best_model.pth from previous run (epoch 5).
Trains up to 20 more epochs with:
  - patience=5
  - lr=1e-4
  - BCEWithLogitsLoss(pos_weight=1.7)  fake weighted higher
  - Same 80/20 val split (seed=42)
  - Embeddings already cached — no backbone needed
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
REAL_DIR    = Path(r"C:\vdfk_engine_dev\data\calibration\real")
FAKE_DIR    = Path(r"C:\vdfk_engine_dev\data\calibration\fake")
CACHE_DIR   = Path(r"C:\vdfk_engine_dev\data\calibration\embeddings")
CHECKPOINT  = Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned\best_model.pth")
OUT_DIR     = Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned")
LOG_DIR     = Path(r"C:\vdfk_engine_dev\ml_engine\logs")

TARGET_LEN_S = 10.0
SEED         = 42
LR           = 1e-4
EPOCHS       = 20
PATIENCE     = 5
BATCH_SIZE   = 32
VAL_FRAC     = 0.20
WEIGHT_FAKE  = 1.7   # pos_weight for BCEWithLogitsLoss


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


# ── Load embeddings (already cached) ─────────────────────────────────────────
def load_embeddings():
    files = (
        [(p, 0) for p in sorted(REAL_DIR.glob("*.wav"))] +
        [(p, 1) for p in sorted(FAKE_DIR.glob("*.wav"))]
    )
    X, y, missing = [], [], 0
    for path, label in files:
        key = stable_hash(str(path.resolve()) + f"|{TARGET_LEN_S}")
        npy = CACHE_DIR / f"{key}.npy"
        if not npy.exists():
            missing += 1
            continue
        X.append(np.load(npy))
        y.append(label)
    if missing:
        print(f"WARNING: {missing} embeddings not in cache — check Step 1 ran cleanly.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Threshold sweep ───────────────────────────────────────────────────────────
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

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    # ── Load embeddings ───────────────────────────────────────────────────────
    print("\nLoading cached embeddings ...")
    X, y = load_embeddings()
    print(f"Loaded {len(X)} embeddings  shape={X.shape}  "
          f"({int(y.sum())} fake / {int((y==0).sum())} real)")

    # ── Same 80/20 split as original run (seed=42) ────────────────────────────
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

    # ── Load checkpoint ───────────────────────────────────────────────────────
    model = LinearHead(in_dim=X.shape[1]).to(device)
    model.load_state_dict(
        torch.load(str(CHECKPOINT), map_location='cpu', weights_only=False)
    )
    print(f"\nLoaded checkpoint: {CHECKPOINT}")

    # ── Loss with class weight ────────────────────────────────────────────────
    pos_weight = torch.tensor([WEIGHT_FAKE]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Loss: BCEWithLogitsLoss(pos_weight={WEIGHT_FAKE})  "
          f"[fake weighted {WEIGHT_FAKE}x vs real]")
    print(f"Settings: lr={LR}, epochs={EPOCHS}, patience={PATIENCE}, batch={BATCH_SIZE}\n")

    # ── Training ──────────────────────────────────────────────────────────────
    best_eer     = float('inf')
    patience_cnt = 0
    history      = []

    print(f"  {'Epoch':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
          f"{'Va Loss':>8}  {'Va Acc':>7}  {'Va EER':>7}  {'Status'}")
    print(f"  {'-'*70}")

    t0 = time.time()
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

    # ── Append history ────────────────────────────────────────────────────────
    hist_file = LOG_DIR / "finetune_history.json"
    existing  = []
    if hist_file.exists():
        existing = json.loads(hist_file.read_text())
    # Offset epoch numbers to continue from previous run
    prev_epochs = len(existing)
    for row in history:
        row['epoch'] += prev_epochs
    with open(hist_file, 'w') as f:
        json.dump(existing + history, f, indent=2)

    # ── Reload best and run threshold sweep ───────────────────────────────────
    model.load_state_dict(
        torch.load(str(OUT_DIR / "best_model.pth"), map_location=device, weights_only=False)
    )
    print(f"\n{'='*65}")
    print(f"  Threshold sweep on {len(X_va)} val files (best model reloaded)")
    print(f"{'='*65}")
    best_thr, best_f1 = threshold_sweep(model, X_va, y_va, device)

    print(f"\n{'='*65}")
    print(f"  Continue run complete in {elapsed/60:.1f} min")
    print(f"  Epochs trained  : {len(history)}  (total: {prev_epochs + len(history)})")
    print(f"  Best val EER    : {best_eer:.4f}")
    print(f"  Best threshold  : {best_thr:.4f}  (F1={best_f1:.4f})")
    print(f"  Model saved     : {OUT_DIR}/best_model.pth")
    print(f"  Threshold saved : {OUT_DIR}/threshold.txt")
    print(f"{'='*65}")
