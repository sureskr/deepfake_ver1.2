"""
Evaluate v1_2_4_proper (frozen Wav2Vec2 + linear head) on RitW locked test at fixed threshold.
Writes CSV with metrics and v1.2.3 reference row for comparison.

  py -3.11 eval_v1_2_4_proper_ritw.py --threshold 0.55 --out-csv results\\foo.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import train_v1_2_4_proper as p

DEFAULT_CKPT = Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_2_4_proper\best_model.pth")
DEFAULT_RITW_REAL = Path(r"C:\vdfk_engine_dev\data\test\real")
DEFAULT_RITW_FAKE = Path(r"C:\vdfk_engine_dev\data\test\fake")
DEFAULT_OUT_CSV = Path(r"C:\vdfk_engine_dev\ml_engine\results\test_v1_2_4_proper_results.csv")

BATCH = 16
NUM_WORKERS = 4


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
    pos = y_true == 1
    if not np.any(pos):
        return float("nan")
    pred_fake = y_score >= thresh
    fn = np.sum((~pred_fake) & pos)
    tp = np.sum(pred_fake & pos)
    return float(fn / (fn + tp))


def load_model(
    ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, Wav2Vec2FeatureExtractor]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(p.W2V2_DIR), local_files_only=True)
    backbone = Wav2Vec2Model.from_pretrained(str(p.W2V2_DIR), local_files_only=True)
    p.freeze_full_backbone(backbone)
    head = p.LinearHeadLarge(in_dim=backbone.config.hidden_size)
    head_sd = ckpt.get("head_state_dict")
    if not isinstance(head_sd, dict):
        sys.exit("Checkpoint missing head_state_dict")
    head.load_state_dict(head_sd, strict=True)
    model = p.FrozenW2V2Linear(backbone, head).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    real_dir: Path,
    fake_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    ds = p.AudioFolderDataset(real_dir, fake_dir)
    print(f"RITW samples N={len(ds)}")
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
        collate_fn=partial(p.collate_audio_no_aug, processor=processor),
    )
    scores: list[float] = []
    labels: list[int] = []
    for xb, yb in tqdm(loader, desc="ritw_proper"):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels.extend(yb.numpy().astype(int).tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="RitW eval for v1_2_4_proper linear head.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_RITW_REAL)
    parser.add_argument("--fake-dir", type=Path, default=DEFAULT_RITW_FAKE)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="cuda requires torch.cuda.is_available()",
    )
    args = parser.parse_args()

    ckpt_path: Path = args.checkpoint.expanduser().resolve()
    out_csv: Path = args.out_csv.expanduser().resolve()
    real_dir: Path = args.real_dir.expanduser().resolve()
    fake_dir: Path = args.fake_dir.expanduser().resolve()
    thresh = float(args.threshold)

    if not ckpt_path.is_file():
        sys.exit(f"Missing checkpoint: {ckpt_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    print(f"Device: {device} | checkpoint: {ckpt_path} | threshold: {thresh}")

    model, processor = load_model(ckpt_path, device)
    print("Loaded v1_2_4_proper (frozen backbone + linear head from checkpoint).")

    y_true, y_score = predict_all(model, processor, device, real_dir, fake_dir)
    if len(y_true) == 0:
        sys.exit("Empty dataset.")

    eer = eer_binary(y_true, y_score)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")

    pred = (y_score >= thresh).astype(int)
    acc = float(accuracy_score(y_true, pred))
    fpr = fpr_at_threshold(y_true, y_score, thresh)
    fnr = fnr_at_threshold(y_true, y_score, thresh)

    print(f"\n=== v1_2_4_proper @ RitW, threshold={thresh:g} ===")
    print(f"N={len(y_true)}")
    print(f"Accuracy={acc*100:.2f}%")
    print(f"EER (score-based)={eer*100:.2f}%")
    print(f"FPR (reals)={fpr*100:.2f}%")
    print(f"FNR (fakes)={fnr*100:.2f}%")
    print(f"AUROC={auroc:.4f}" if not np.isnan(auroc) else "AUROC=n/a")

    print("\n=== v1.2.3 reference (your table @ 0.70 RitW; not re-tuned to 0.55) ===")
    print("Accuracy=73.72% | EER=28.28% | FPR=12.37% | FNR=49.78% | AUROC=(n/a)")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "threshold",
                "n_samples",
                "accuracy",
                "eer",
                "fpr",
                "fnr",
                "auroc",
            ]
        )
        w.writerow(
            [
                "v1_2_4_proper",
                str(thresh),
                len(y_true),
                f"{acc*100:.4f}",
                f"{eer*100:.4f}",
                f"{fpr*100:.4f}",
                f"{fnr*100:.4f}",
                f"{auroc:.6f}" if not np.isnan(auroc) else "",
            ]
        )
        w.writerow(
            [
                "v1_2_3_reference",
                "0.70",
                len(y_true),
                "73.7200",
                "28.2800",
                "12.3700",
                "49.7800",
                "",
            ]
        )

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
