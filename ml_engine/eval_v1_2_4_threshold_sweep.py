"""
Evaluate v1_2_4 checkpoint: optional threshold sweep (F1) or fixed threshold (e.g. 0.70).
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import train_v1_2_4 as t

CKPT = Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_2_4\best_model.pth")
W2V2 = t.W2V2_DIR
BATCH = 16
NUM_WORKERS = 4
THRESH_LO, THRESH_HI, THRESH_STEP = 0.3, 0.9, 0.01


def eer_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


def fpr_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thresh: float) -> float:
    neg = y_true == 0
    if not np.any(neg):
        return float("nan")
    pred_fake = y_score >= thresh
    fp = np.sum(pred_fake & neg)
    tn = np.sum((~pred_fake) & neg)
    return float(fp / (fp + tn))


def fnr_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thresh: float) -> float:
    """FNR on fake / positives: missed fake rate among true fakes."""
    pos = y_true == 1
    if not np.any(pos):
        return float("nan")
    pred_fake = y_score >= thresh
    fn = np.sum((~pred_fake) & pos)
    tp = np.sum(pred_fake & pos)
    return float(fn / (fn + tp))


def load_model(device: torch.device) -> tuple[torch.nn.Module, Wav2Vec2FeatureExtractor]:
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(W2V2), local_files_only=True)
    backbone = Wav2Vec2Model.from_pretrained(str(W2V2), local_files_only=True)
    head = t.ClassifierHead(in_dim=backbone.config.hidden_size, dropout=t.DROPOUT)
    head.load_state_dict(ckpt["head_state_dict"], strict=True)
    backbone.load_state_dict(ckpt["backbone_state_dict"], strict=True)
    t.freeze_backbone_except_layers(backbone, t.UNFREEZE_LAYER_IDX)
    model = t.Wav2Vec2Deepfake(backbone, head).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    real_dir: Path,
    fake_dir: Path,
    device: torch.device,
    name: str,
    vad: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ds = t.AudioFolderDataset(real_dir, fake_dir, vad_filter=vad)
    print(f"[{name}] samples={len(ds)} | vad_skipped={ds.vad_skipped}")
    if len(ds) == 0:
        return np.array([]), np.array([])
    loader = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=partial(t.collate_audio, processor=processor, train=False),
    )
    scores: list[float] = []
    labels: list[int] = []
    for xb, yb in tqdm(loader, desc=name, leave=False):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels.extend(yb.numpy().astype(int).tolist())
    return np.asarray(labels), np.asarray(scores, dtype=np.float64)


def sweep_and_report(y_true: np.ndarray, y_score: np.ndarray, title: str) -> None:
    if len(y_true) == 0:
        print(f"{title}: (empty)")
        return
    eer = eer_binary(y_true, y_score)
    thresholds = np.arange(THRESH_LO, THRESH_HI + 1e-9, THRESH_STEP)
    best_f1, best_t = -1.0, THRESH_LO
    for t_ in thresholds:
        pred = (y_score >= t_).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t_)
    pred = (y_score >= best_t).astype(int)
    acc = accuracy_score(y_true, pred)
    fpr = fpr_at_threshold(y_true, y_score, best_t)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    print(f"\n=== {title} ===")
    print(f"N={len(y_true)} | EER (score-based)={eer*100:.2f}%")
    print(f"Best F1 threshold in [{THRESH_LO}, {THRESH_HI}] step {THRESH_STEP}: {best_t:.2f}")
    print(f"  F1={best_f1*100:.2f}% | Acc={acc*100:.2f}% | Precision={prec*100:.2f}% | Recall={rec*100:.2f}%")
    print(f"  FPR (on real / negatives @ threshold)={fpr*100:.2f}%")


def report_fixed_threshold(
    y_true: np.ndarray, y_score: np.ndarray, title: str, thresh: float
) -> None:
    if len(y_true) == 0:
        print(f"{title}: (empty)")
        return
    eer = eer_binary(y_true, y_score)
    pred = (y_score >= thresh).astype(int)
    acc = accuracy_score(y_true, pred)
    fpr = fpr_at_threshold(y_true, y_score, thresh)
    fnr = fnr_at_threshold(y_true, y_score, thresh)
    print(f"\n=== {title} ===")
    print(f"N={len(y_true)} | fixed threshold={thresh:.2f}")
    print(f"  Accuracy={acc*100:.2f}%")
    print(f"  EER (score-based)={eer*100:.2f}%")
    print(f"  FPR (real / negatives)={fpr*100:.2f}%")
    print(f"  FNR (fake / positives, missed-fake rate)={fnr*100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v1.2.4 checkpoint")
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        metavar="T",
        help="If set, skip F1 sweep and report metrics at this score threshold only.",
    )
    parser.add_argument(
        "--only",
        choices=("both", "v14", "ritw"),
        default="both",
        help="Which corpus to run (default: both).",
    )
    args = parser.parse_args()

    if not CKPT.is_file():
        sys.exit(f"Missing checkpoint: {CKPT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | checkpoint: {CKPT}")
    model, processor = load_model(device)

    v4_real = Path(r"C:\vdfk_engine_dev\data\v1_2_4\test\real")
    v4_fake = Path(r"C:\vdfk_engine_dev\data\v1_2_4\test\fake")
    ritw_real = Path(r"C:\vdfk_engine_dev\data\test\real")
    ritw_fake = Path(r"C:\vdfk_engine_dev\data\test\fake")

    if args.fixed_threshold is not None:
        t_fixed = float(args.fixed_threshold)
        if args.only in ("both", "v14"):
            y_t, y_s = predict_dataset(
                model, processor, v4_real, v4_fake, device, "v1_2_4 test", vad=False
            )
            report_fixed_threshold(y_t, y_s, "v1.2.4 test (LA + ASV5 held-out copy)", t_fixed)
        if args.only in ("both", "ritw"):
            y_t2, y_s2 = predict_dataset(
                model, processor, ritw_real, ritw_fake, device, "release_in_the_wild test", vad=False
            )
            report_fixed_threshold(
                y_t2, y_s2, "release_in_the_wild locked test", t_fixed
            )
        if args.only == "ritw":
            print("\n--- v1.2.3 reference (user-provided, same threshold=0.70 assumed) ---")
            print("Accuracy: 73.72% | EER: 28.28% | FPR: 12.37% | FNR: (not provided)")
        return

    if args.only in ("both", "v14"):
        y_t, y_s = predict_dataset(model, processor, v4_real, v4_fake, device, "v1_2_4 test", vad=False)
        sweep_and_report(y_t, y_s, "v1.2.4 test (LA + ASV5 held-out copy)")
    if args.only in ("both", "ritw"):
        y_t2, y_s2 = predict_dataset(
            model, processor, ritw_real, ritw_fake, device, "release_in_the_wild test", vad=False
        )
        sweep_and_report(y_t2, y_s2, "release_in_the_wild locked test")

    print("\n--- v1.2.3 reference (user-provided) ---")
    print("Accuracy: 73.72% | EER: 28.28% | FPR: 12.37%")


if __name__ == "__main__":
    main()
