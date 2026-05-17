"""
train_v1_2_4_proper.py — v1.2.3-style training on v1_2_4 calibration (frozen Wav2Vec2 + linear head).

Same recipe as production v1.2.3 (frozen backbone, 1024→1 head, no aug, no scheduler, no unfreeze),
but train on prepared v1_2_4 calibration (~17k files) with updated pos_weight.

Default: print config summary and exit (no training). Run training with:
  py -3.11 train_v1_2_4_proper.py --train
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from functools import partial
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ── Paths ──────────────────────────────────────────────────────────────────
W2V2_DIR = Path(r"C:\vdfk_engine_dev\ml_engine\models\wav2vec2-large-960h")
V13_HEAD_CKPT = Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_finetuned\best_model.pth")
V13_HEAD_FALLBACKS = [
    Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_finetuned_v2\best_model.pth"),
    Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned\best_model.pth"),
]

TRAIN_REAL = Path(r"C:\vdfk_engine_dev\data\v1_2_4\calibration\real")
TRAIN_FAKE = Path(r"C:\vdfk_engine_dev\data\v1_2_4\calibration\fake")
VAL_REAL = Path(r"C:\vdfk_engine_dev\data\v1_2_4\validation\real")
VAL_FAKE = Path(r"C:\vdfk_engine_dev\data\v1_2_4\validation\fake")

OUT_DIR = Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_2_4_proper")
OUT_MODEL = OUT_DIR / "best_model.pth"
LOG_DIR = Path(r"C:\vdfk_engine_dev\ml_engine\logs")
LOG_JSONL = LOG_DIR / "train_v1_2_4_proper.jsonl"

SAMPLE_RATE = 16000
TARGET_LEN_S = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_LEN_S)
SEED = 42

LR = 1e-4
BATCH_SIZE = 32
PATIENCE = 3
MAX_EPOCHS = 50
# N_real / N_fake on calibration = 6685 / 10361
POS_WEIGHT = 6685.0 / 10361.0

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def eer_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


def resolve_v13_head_checkpoint() -> Path | None:
    if V13_HEAD_CKPT.is_file():
        return V13_HEAD_CKPT
    for p in V13_HEAD_FALLBACKS:
        if p.is_file():
            return p
    return None
def load_head_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            inner = ckpt.get(k)
            if isinstance(inner, dict) and inner and any(torch.is_tensor(v) for v in inner.values()):
                return inner
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError(f"Cannot find state dict in {ckpt_path}")


def strip_head_prefix(state: dict) -> dict:
    if not state or not any(str(k).startswith("head.") for k in state):
        return state
    return {str(k)[len("head.") :]: v for k, v in state.items() if str(k).startswith("head.")}


class LinearHeadLarge(nn.Module):
    """Single linear layer 1024→1 (same as v1.2.3 / inference.LinearHeadLarge)."""

    def __init__(self, in_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class FrozenW2V2Linear(nn.Module):
    def __init__(self, backbone: Wav2Vec2Model, head: LinearHeadLarge):
        super().__init__()
        self.wav2vec2 = backbone
        self.head = head

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.wav2vec2(input_values)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.head(pooled)


class AudioFolderDataset(Dataset):
    """Lists (path, label); label 0=real, 1=fake."""

    def __init__(self, real_dir: Path, fake_dir: Path) -> None:
        self.samples: list[tuple[Path, int]] = []
        for label, dir_path in ((0, real_dir), (1, fake_dir)):
            for p in sorted(dir_path.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
                    continue
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def load_audio_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".flac":
        import miniaudio

        dec = miniaudio.flac_read_file_f32(str(path))
        wav = np.asarray(dec.samples, dtype=np.float32)
        if dec.nchannels > 1:
            wav = wav.reshape(-1, dec.nchannels).mean(axis=1)
        if dec.sample_rate != sample_rate:
            import librosa as lr

            wav = lr.resample(wav, orig_sr=dec.sample_rate, target_sr=sample_rate).astype(
                np.float32
            )
        return wav
    import librosa

    wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return wav.astype(np.float32, copy=False)


def collate_audio_no_aug(
    batch, processor: Wav2Vec2FeatureExtractor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed front crop / pad to TARGET_SAMPLES — no random crop, no noise or gain (v1.2.3-style)."""
    paths, labels = zip(*batch)
    waves = []
    for path in paths:
        wav = load_audio_mono(path)
        if len(wav) > TARGET_SAMPLES:
            wav = wav[:TARGET_SAMPLES]
        elif len(wav) < TARGET_SAMPLES:
            wav = np.pad(wav, (0, TARGET_SAMPLES - len(wav)), mode="constant")
        waves.append(wav.astype(np.float32))

    inputs = processor(
        list(waves), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
    )
    return inputs.input_values, torch.tensor(labels, dtype=torch.float32)


def init_head(head: LinearHeadLarge) -> None:
    nn.init.xavier_uniform_(head.classifier.weight)
    if head.classifier.bias is not None:
        nn.init.zeros_(head.classifier.bias)


def load_v13_head_weights(head: LinearHeadLarge, ckpt_path: Path) -> None:
    sd = strip_head_prefix(load_head_state_dict(ckpt_path))
    sub = {k: v for k, v in sd.items() if str(k).startswith("classifier.")}
    if len(sub) == 2:
        head.load_state_dict(sub, strict=True)
        return
    if {"classifier.weight", "classifier.bias"} <= set(sd.keys()):
        head.load_state_dict(
            {k: v for k, v in sd.items() if k in ("classifier.weight", "classifier.bias")},
            strict=True,
        )
        return
    raise RuntimeError(
        f"No classifier.* tensors in {ckpt_path} (sample keys: {list(sd.keys())[:12]})"
    )


def freeze_full_backbone(m: Wav2Vec2Model) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Adam | None,
) -> dict:
    train = optimizer is not None
    if train:
        model.wav2vec2.eval()
        model.head.train()
    else:
        model.eval()
    losses: list[float] = []
    scores: list[float] = []
    labs: list[float] = []

    if train:
        for xb, yb in tqdm(loader, desc="train", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labs.extend(yb.cpu().numpy().tolist())
    else:
        with torch.no_grad():
            for xb, yb in tqdm(loader, desc="val", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                losses.append(criterion(logits, yb).item())
                scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labs.extend(yb.cpu().numpy().tolist())

    y = np.array(labs)
    s = np.array(scores)
    pred = (s >= 0.5).astype(int)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "eer": float(eer_binary(y, s)),
    }


def print_config_summary(
    n_train: int,
    n_train_real: int,
    n_train_fake: int,
    n_val: int,
    n_val_real: int,
    n_val_fake: int,
    device: torch.device,
) -> None:
    print()
    print("=" * 72)
    print("train_v1_2_4_proper - config summary (v1.2.3 recipe, v1_2_4 data)")
    print("=" * 72)
    print(f"  Wav2Vec2:        fully FROZEN (local {W2V2_DIR})")
    print(f"  Head:            single Linear 1024->1 (LinearHeadLarge)")
    print(f"  Head init:       load classifier from {V13_HEAD_CKPT}")
    print(f"  Optimizer:       Adam(lr={LR}, train head only)")
    print(f"  Scheduler:       none")
    print(f"  Loss:            BCEWithLogitsLoss(pos_weight={POS_WEIGHT:.6f}  # 6685/10361)")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Early stopping:  patience={PATIENCE} on val EER lower is better (max epochs {MAX_EPOCHS})")
    print(f"  Augmentation:    none")
    print(f"  Seed:            {SEED}")
    print(f"  Device:          {device}")
    print("-" * 72)
    print(f"  Train (calibration): {n_train} files  (real={n_train_real}, fake={n_train_fake})")
    print(f"  Val (validation):    {n_val} files    (real={n_val_real}, fake={n_val_fake})")
    print(f"  Output:            {OUT_MODEL}")
    print("=" * 72)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="v1.2.3-style train on v1_2_4 calibration.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training; omit to only print the config summary (default).",
    )
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = AudioFolderDataset(TRAIN_REAL, TRAIN_FAKE)
    val_ds = AudioFolderDataset(VAL_REAL, VAL_FAKE)
    n_train_real = sum(1 for _, y in train_ds.samples if y == 0)
    n_train_fake = sum(1 for _, y in train_ds.samples if y == 1)
    n_val_real = sum(1 for _, y in val_ds.samples if y == 0)
    n_val_fake = sum(1 for _, y in val_ds.samples if y == 1)

    print_config_summary(
        len(train_ds),
        n_train_real,
        n_train_fake,
        len(val_ds),
        n_val_real,
        n_val_fake,
        device,
    )

    if not args.train:
        print("Not training (default). Pass --train to start the run.\n")
        return

    head_ckpt = resolve_v13_head_checkpoint()
    if head_ckpt is None:
        raise SystemExit(
            f"Missing v1.2.3 head checkpoint. Tried {V13_HEAD_CKPT} "
            f"and fallbacks: {V13_HEAD_FALLBACKS}"
        )
    if head_ckpt != V13_HEAD_CKPT:
        print(f"[WARN] Primary missing; loading head init from {head_ckpt}\n")

    if not W2V2_DIR.is_dir():
        raise SystemExit(f"Missing local Wav2Vec2: {W2V2_DIR}")
    if len(train_ds) == 0:
        raise SystemExit("Training dataset is empty.")
    if len(val_ds) == 0:
        raise SystemExit("Validation dataset is empty.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(W2V2_DIR), local_files_only=True)
    backbone = Wav2Vec2Model.from_pretrained(str(W2V2_DIR), local_files_only=True)
    freeze_full_backbone(backbone)
    head = LinearHeadLarge(in_dim=backbone.config.hidden_size)
    try:
        load_v13_head_weights(head, head_ckpt)
        print(f"Loaded linear head weights from {head_ckpt}")
    except RuntimeError as e:
        print(f"[WARN] Could not load v1.2.3 head: {e}; training from fresh head init.")
        init_head(head)

    model = FrozenW2V2Linear(backbone, head).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(POS_WEIGHT, device=device, dtype=torch.float32)
    )
    optimizer = Adam((p for p in head.parameters() if p.requires_grad), lr=LR)

    collate_fn = partial(collate_audio_no_aug, processor=processor)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    best_eer = float("inf")
    patience_left = PATIENCE
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = run_epoch(model, train_loader, device, criterion, optimizer)
        va = run_epoch(model, val_loader, device, criterion, optimizer=None)

        row = {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "elapsed_sec": time.perf_counter() - t0,
        }
        with open(LOG_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | train loss {tr['loss']:.4f} f1 {tr['f1']:.4f} | "
            f"val loss {va['loss']:.4f} f1 {va['f1']:.4f} eer {va['eer']:.4f}"
        )

        if va["eer"] < best_eer - 1e-8:
            best_eer = va["eer"]
            best_epoch = epoch
            patience_left = PATIENCE
            payload = {
                "format": "v1_2_4_proper",
                "model_name": "facebook/wav2vec2-large-960h-local",
                "embed_dim": int(backbone.config.hidden_size),
                "wav2vec2_path": str(W2V2_DIR.resolve()),
                "epoch": epoch,
                "val_metrics": va,
                "train_metrics": tr,
                "head_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "train_config": {
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "patience": PATIENCE,
                    "max_epochs": MAX_EPOCHS,
                    "pos_weight": POS_WEIGHT,
                    "optimizer": "Adam",
                    "scheduler": None,
                    "augmentation": False,
                    "backbone_frozen": True,
                    "head": "Linear1024to1",
                    "seed": SEED,
                    "early_stopping_metric": "val_eer",
                },
            }
            torch.save(payload, str(OUT_MODEL))
            print(f"  >>> Saved best to {OUT_MODEL} (val EER {best_eer:.4f})")
        else:
            patience_left -= 1
            print(f"  (no val EER improvement) patience {PATIENCE - patience_left}/{PATIENCE}")
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}; best epoch {best_epoch} val EER {best_eer:.4f}")
                break

    print(f"Done. best val EER={best_eer:.4f} @ epoch {best_epoch}. Checkpoint: {OUT_MODEL}")


if __name__ == "__main__":
    main()
