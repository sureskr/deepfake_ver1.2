"""
train_v1_2_4.py — End-to-end fine-tune Wav2Vec2 (last 2 encoder layers) + classifier head.

- No precomputed embeddings: full forward through backbone each step.
- Device: CUDA (required).

Run (overnight / background):
  cd C:\\vdfk_engine_dev\\ml_engine
  python train_v1_2_4.py

Smoke test (few train batches only, skips validation / checkpointing):
  python train_v1_2_4.py --max-batches 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
from functools import partial
import time
import warnings
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ── Paths ───────────────────────────────────────────────────────────────────
W2V2_DIR = Path(r"C:\vdfk_engine_dev\ml_engine\models\wav2vec2-large-960h")
BASE_HEAD = Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_finetuned\best_model.pth")
BASE_HEAD_FALLBACKS = [
    Path(r"C:\vdfk_engine_dev\ml_engine\models\finetuned\best_model.pth"),
    Path(r"C:\vdfk_engine_dev\ml_engine\models\base_model.pth"),
]
OUT_MODEL = Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_2_4\best_model.pth")
LOG_DIR = Path(r"C:\vdfk_engine_dev\ml_engine\logs")

TRAIN_REAL = Path(r"C:\vdfk_engine_dev\data\v1_2_4\calibration\real")
TRAIN_FAKE = Path(r"C:\vdfk_engine_dev\data\v1_2_4\calibration\fake")
VAL_REAL = Path(r"C:\vdfk_engine_dev\data\v1_2_4\validation\real")
VAL_FAKE = Path(r"C:\vdfk_engine_dev\data\v1_2_4\validation\fake")
TEST_REAL = Path(r"C:\vdfk_engine_dev\data\v1_2_4\test\real")
TEST_FAKE = Path(r"C:\vdfk_engine_dev\data\v1_2_4\test\fake")

SAMPLE_RATE = 16000
TARGET_LEN_S = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_LEN_S)
SEED = 42

LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
EPOCHS = 15
PATIENCE = 5
BATCH_SIZE = 8
# pos_weight = N_real / N_fake = 6685/10361 = 0.645
# Fake is majority class (61%) so we down-weight it
# Higher threshold (0.70) handles FP/FN tradeoff at inference
POS_WEIGHT = 0.645
DROPOUT = 0.3
DROPOUT_HEAD_MID = 0.1
HEAD_HIDDEN = 256
WEIGHT_DECAY = 0.01

# Balanced ASVspoof-heavy held-out check each epoch (after VAD), fixed seed
SECONDARY_VAL_PER_CLASS = 375
SCHEDULER_ETA_MIN = 1e-6

# User: "last 2 layers (23, 24)" — HF encoder uses 0..23 for 24 layers → unfreeze 22, 23
UNFREEZE_LAYER_IDX = (22, 23)

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

MIN_VAD_SECONDS = 0.5
MIN_VAD_RMS = 0.001


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def eer_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


def audio_passes_vad(wav: np.ndarray) -> bool:
    min_samples = int(MIN_VAD_SECONDS * SAMPLE_RATE)
    if len(wav) < min_samples:
        return False
    rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float64)))))
    return rms >= MIN_VAD_RMS


def init_head_linear_modules(head: nn.Module) -> None:
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _add_gaussian_noise_snr(wav: np.ndarray, snr_db: float) -> np.ndarray:
    sig = wav.astype(np.float32, copy=False)
    p = float(np.mean(np.square(sig)) + 1e-12)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = p / snr_lin
    noise = np.random.randn(len(sig)).astype(np.float32) * np.sqrt(noise_var)
    return np.clip(sig + noise, -1.0, 1.0)


def resolve_head_checkpoint() -> Path | None:
    if BASE_HEAD.is_file():
        return BASE_HEAD
    for p in BASE_HEAD_FALLBACKS:
        if p.is_file():
            print(f"[WARN] Using fallback head checkpoint: {p}")
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


class ClassifierHead(nn.Module):
    """MLP: dropout → Linear(d→256) → ReLU → dropout → Linear(256→1)."""

    def __init__(self, in_dim: int = 1024, dropout: float = DROPOUT, hidden: int = HEAD_HIDDEN):
        super().__init__()
        self.dropout_p_in = dropout
        self.dropout_p_mid = DROPOUT_HEAD_MID
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout_p_in, self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_p_mid, self.training)
        return self.fc2(x).squeeze(-1)


class Wav2Vec2Deepfake(nn.Module):
    def __init__(self, backbone: Wav2Vec2Model, head: ClassifierHead):
        super().__init__()
        self.wav2vec2 = backbone
        self.head = head

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.wav2vec2(input_values)
        h = out.last_hidden_state
        pooled = h.mean(dim=1)
        return self.head(pooled)


def freeze_backbone_except_layers(backbone: Wav2Vec2Model, layer_indices: tuple[int, ...]) -> None:
    for p in backbone.parameters():
        p.requires_grad = False
    for idx in layer_indices:
        if idx < 0 or idx >= len(backbone.encoder.layers):
            raise ValueError(f"Bad layer index {idx}; encoder has {len(backbone.encoder.layers)} layers")
        for p in backbone.encoder.layers[idx].parameters():
            p.requires_grad = True


class AudioFolderDataset(Dataset):
    def __init__(self, real_dir: Path, fake_dir: Path, *, vad_filter: bool = False) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.vad_skipped = 0
        for label, dir_path in ((0, real_dir), (1, fake_dir)):
            for p in sorted(dir_path.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
                    continue
                if vad_filter:
                    try:
                        w = load_audio_mono(p)
                    except Exception:
                        self.vad_skipped += 1
                        continue
                    if not audio_passes_vad(w):
                        self.vad_skipped += 1
                        continue
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def load_audio_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load mono float32 waveform; FLAC via miniaudio (Windows-safe), others via librosa."""
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


def collate_audio(
    batch, processor: Wav2Vec2FeatureExtractor, train: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    paths, labels = zip(*batch)
    waves = []
    for path in paths:
        wav = load_audio_mono(path)
        if len(wav) > TARGET_SAMPLES:
            if train:
                start = random.randint(0, len(wav) - TARGET_SAMPLES)
                wav = wav[start : start + TARGET_SAMPLES]
            else:
                wav = wav[:TARGET_SAMPLES]
        elif len(wav) < TARGET_SAMPLES:
            wav = np.pad(wav, (0, TARGET_SAMPLES - len(wav)), mode="constant")
        wav = wav.astype(np.float32)
        if train:
            if random.random() < 0.5:
                snr_db = random.uniform(20.0, 40.0)
                wav = _add_gaussian_noise_snr(wav, snr_db)
            if random.random() < 0.5:
                wav = np.clip(wav * random.uniform(0.8, 1.2), -1.0, 1.0)
        waves.append(wav)

    inputs = processor(
        list(waves), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
    )
    return inputs.input_values, torch.tensor(labels, dtype=torch.float32)


def main(max_batches: int | None = None) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (torch.cuda.is_available() is False).")

    if max_batches is not None and max_batches < 1:
        raise SystemExit("--max-batches must be >= 1 when provided.")

    device = torch.device("cuda")
    set_seed()

    if max_batches is not None:
        print(
            f"[smoke] --max-batches={max_batches}: "
            "will run at most that many training batches, then stop (no val / save)."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    log_jsonl = LOG_DIR / "train_v1_2_4.jsonl"

    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(W2V2_DIR), local_files_only=True)
    backbone = Wav2Vec2Model.from_pretrained(str(W2V2_DIR), local_files_only=True)
    head = ClassifierHead(in_dim=backbone.config.hidden_size, dropout=DROPOUT)

    head_ckpt = resolve_head_checkpoint()
    if head_ckpt is not None:
        sd = load_head_state_dict(head_ckpt)
        try:
            head.load_state_dict(sd, strict=True)
            print(f"Loaded classifier head from {head_ckpt}")
        except RuntimeError:
            print(
                f"[WARN] Head checkpoint incompatible with MLP head ({head_ckpt}); "
                "training head from scratch."
            )
            init_head_linear_modules(head)
    else:
        print(
            "[WARN] No head checkpoint found at v1_finetuned/finetuned/base paths — "
            "starting classifier from default init."
        )
        init_head_linear_modules(head)

    freeze_backbone_except_layers(backbone, UNFREEZE_LAYER_IDX)
    model = Wav2Vec2Deepfake(backbone, head).to(device)

    backbone_params = [
        p for n, p in model.named_parameters() if p.requires_grad and n.startswith("wav2vec2")
    ]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")]
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=SCHEDULER_ETA_MIN
    )

    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(POS_WEIGHT, device=device, dtype=torch.float32)
    )

    train_ds = AudioFolderDataset(TRAIN_REAL, TRAIN_FAKE, vad_filter=True)
    val_ds = AudioFolderDataset(VAL_REAL, VAL_FAKE, vad_filter=True)
    print(
        f"[VAD] Skipped train={train_ds.vad_skipped} | val={val_ds.vad_skipped} "
        f"(too short, load error, or RMS < {MIN_VAD_RMS})"
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    test_pool = AudioFolderDataset(TEST_REAL, TEST_FAKE, vad_filter=True)
    print(f"[VAD] Test split pool: skipped {test_pool.vad_skipped} files")
    idx_real = [i for i, (_, y) in enumerate(test_pool.samples) if y == 0]
    idx_fake = [i for i, (_, y) in enumerate(test_pool.samples) if y == 1]
    rng_sub = random.Random(SEED)
    rng_sub.shuffle(idx_real)
    rng_sub.shuffle(idx_fake)
    n_real_sub = min(SECONDARY_VAL_PER_CLASS, len(idx_real))
    n_fake_sub = min(SECONDARY_VAL_PER_CLASS, len(idx_fake))
    secondary_indices = idx_real[:n_real_sub] + idx_fake[:n_fake_sub]
    test_secondary_ds = Subset(test_pool, secondary_indices)
    print(
        f"[secondary val] {len(test_secondary_ds)} files "
        f"({n_real_sub} real + {n_fake_sub} fake, cap {SECONDARY_VAL_PER_CLASS}/class)"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=partial(collate_audio, processor=processor, train=True),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=partial(collate_audio, processor=processor, train=False),
    )
    test_secondary_loader = DataLoader(
        test_secondary_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=partial(collate_audio, processor=processor, train=False),
    )

    best_mean_eer = float("inf")
    patience_cnt = 0
    best_epoch = 0

    def log_epoch(row: dict) -> None:
        sec_e = row["secondary_eer"]
        sec_s = f"{sec_e:.4f}" if sec_e is not None else "n/a"
        print(
            f"Epoch {row['epoch']}/{EPOCHS} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} eer {row['val_eer']:.4f} | "
            f"sec-eer {sec_s} mean-eer {row['mean_eer']:.4f} | "
            f"lr_backbone {row['lr_backbone']:.2e} lr_head {row['lr_head']:.2e}"
        )
        with open(log_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    global_step = 0
    t0 = time.perf_counter()

    torch.backends.cudnn.benchmark = True

    smoke_batch_times: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses, tr_pred, tr_lab = [], [], []
        pbar = tqdm(train_loader, desc=f"train ep{epoch}", leave=False)
        smoke_stop = False
        for xb, yb in pbar:
            t_batch = time.perf_counter()
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optimizer.step()
            if max_batches is not None:
                smoke_batch_times.append(time.perf_counter() - t_batch)
            tr_losses.append(loss.item())
            tr_pred.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            tr_lab.extend(yb.cpu().numpy().tolist())
            global_step += 1
            pbar.set_postfix(loss=float(np.mean(tr_losses[-50:])))
            if max_batches is not None and global_step >= max_batches:
                smoke_stop = True
                break

        if smoke_stop:
            times = smoke_batch_times
            mean_t = float(np.mean(times)) if times else 0.0
            print(
                f"[smoke] Ran {len(times)} batches on {device}. "
                f"Time/batch: first={times[0]:.3f}s, "
                f"mean={mean_t:.3f}s, "
                f"last={times[-1]:.3f}s"
            )
            print(f"[smoke] Done (stopped early). Full training: omit --max-batches.")
            return

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        tr_acc = accuracy_score(np.array(tr_lab), (np.array(tr_pred) >= 0.5).astype(int))

        model.eval()
        va_losses, va_pred, va_lab = [], [], []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"val ep{epoch}", leave=False):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                va_losses.append(crit(logits, yb).item())
                va_pred.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                va_lab.extend(yb.cpu().numpy().tolist())

        va_loss = float(np.mean(va_losses)) if va_losses else 0.0
        va_pred_a = np.array(va_pred)
        va_lab_a = np.array(va_lab)
        va_acc = accuracy_score(va_lab_a, (va_pred_a >= 0.5).astype(int))
        va_eer = eer_binary(va_lab_a, va_pred_a)

        sec_losses, sec_pred, sec_lab = [], [], []
        with torch.no_grad():
            for xb, yb in tqdm(
                test_secondary_loader, desc=f"test-secondary ep{epoch}", leave=False
            ):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                sec_losses.append(crit(logits, yb).item())
                sec_pred.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                sec_lab.extend(yb.cpu().numpy().tolist())
        sec_loss = float(np.mean(sec_losses)) if sec_losses else 0.0
        sec_pred_a = np.array(sec_pred)
        sec_lab_a = np.array(sec_lab)
        sec_acc = (
            accuracy_score(sec_lab_a, (sec_pred_a >= 0.5).astype(int))
            if len(sec_lab_a)
            else 0.0
        )
        sec_eer = eer_binary(sec_lab_a, sec_pred_a) if len(sec_lab_a) else float("nan")
        mean_eer = (float(va_eer) + float(sec_eer)) / 2.0 if not np.isnan(sec_eer) else float(va_eer)

        lrs = [g["lr"] for g in optimizer.param_groups]
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": float(tr_acc),
            "val_loss": va_loss,
            "val_acc": float(va_acc),
            "val_eer": float(va_eer),
            "secondary_loss": sec_loss,
            "secondary_acc": float(sec_acc),
            "secondary_eer": float(sec_eer) if not np.isnan(sec_eer) else None,
            "mean_eer": float(mean_eer),
            "lr_backbone": float(lrs[0]),
            "lr_head": float(lrs[1]),
            "elapsed_sec": time.perf_counter() - t0,
        }
        log_epoch(row)
        scheduler.step()

        if mean_eer < best_mean_eer - 1e-8:
            best_mean_eer = mean_eer
            best_epoch = epoch
            patience_cnt = 0
            payload = {
                "format": "v1_2_4_end_to_end",
                "model_name": "facebook/wav2vec2-large-960h-local",
                "embed_dim": backbone.config.hidden_size,
                "wav2vec2_path": str(W2V2_DIR.resolve()),
                "unfrozen_encoder_layer_indices_0_based": list(UNFREEZE_LAYER_IDX),
                "epoch": epoch,
                "val_eer": float(va_eer),
                "secondary_eer": float(sec_eer) if not np.isnan(sec_eer) else None,
                "mean_eer": float(best_mean_eer),
                "val_acc": float(va_acc),
                "train_config": {
                    "lr_backbone": LR_BACKBONE,
                    "lr_head": LR_HEAD,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "pos_weight": POS_WEIGHT,
                    "dropout": DROPOUT,
                    "dropout_head_mid": DROPOUT_HEAD_MID,
                    "head_hidden": HEAD_HIDDEN,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                    "patience": PATIENCE,
                    "scheduler": "CosineAnnealingLR",
                    "scheduler_T_max": EPOCHS,
                    "scheduler_eta_min": SCHEDULER_ETA_MIN,
                    "secondary_val_per_class": SECONDARY_VAL_PER_CLASS,
                },
                "backbone_state_dict": {k: v.cpu() for k, v in backbone.state_dict().items()},
                "head_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
            }
            torch.save(payload, str(OUT_MODEL))
            sec_e = float(sec_eer) if not np.isnan(sec_eer) else float("nan")
            sec_str = f"{sec_e:.4f}" if not np.isnan(sec_e) else "n/a"
            print(
                f"  >>> Saved new best to {OUT_MODEL} "
                f"(mean EER {best_mean_eer:.4f}; val {va_eer:.4f}; sec {sec_str})"
            )
        else:
            patience_cnt += 1
            print(f"  (no improve on mean EER) patience {patience_cnt}/{PATIENCE}")
            if patience_cnt >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch}; best was epoch {best_epoch} "
                    f"mean EER {best_mean_eer:.4f}"
                )
                break

    print(
        f"Done. Best mean EER={best_mean_eer:.4f} @ epoch {best_epoch}. Checkpoint: {OUT_MODEL}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 v1.2.4 (CUDA required).")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        metavar="N",
        help="Smoke test: stop after N training batches (skips validation and checkpoint save).",
    )
    args = parser.parse_args()
    main(max_batches=args.max_batches)
