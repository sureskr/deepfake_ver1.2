"""
Threshold sweep for v1_2_4_proper on v1_2_4 held-out test (data/v1_2_4/test).

Sweep 0.30-0.90 step 0.05. Pick threshold with FPR closest to 12% (v1.2.3-style),
tie-break by higher accuracy.
"""
from __future__ import annotations

import csv
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import train_v1_2_4_proper as p

CKPT = Path(r"C:\vdfk_engine_dev\ml_engine\models\v1_2_4_proper\best_model.pth")
TEST_REAL = Path(r"C:\vdfk_engine_dev\data\v1_2_4\test\real")
TEST_FAKE = Path(r"C:\vdfk_engine_dev\data\v1_2_4\test\fake")
OUT_CSV = Path(r"C:\vdfk_engine_dev\ml_engine\results\v1_2_4_proper_v14test_threshold_sweep.csv")

FPR_TARGET_PCT = 12.0
THRESH_LO, THRESH_HI, THRESH_STEP = 0.30, 0.90, 0.05
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


def load_model(device: torch.device) -> tuple[torch.nn.Module, Wav2Vec2FeatureExtractor]:
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
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
) -> tuple[np.ndarray, np.ndarray]:
    ds = p.AudioFolderDataset(TEST_REAL, TEST_FAKE)
    print(f"v1_2_4 test samples N={len(ds)}")
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
    for xb, yb in tqdm(loader, desc="v14_test_proper"):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels.extend(yb.numpy().astype(int).tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def main() -> None:
    if not CKPT.is_file():
        sys.exit(f"Missing checkpoint: {CKPT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | checkpoint: {CKPT}")

    model, processor = load_model(device)
    y_true, y_score = predict_all(model, processor, device)
    if len(y_true) == 0:
        sys.exit("Empty dataset.")

    eer = eer_binary(y_true, y_score)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")

    print(f"\nScore-based EER (threshold-independent)={eer*100:.4f}%")
    print(f"AUROC={auroc:.6f}" if not np.isnan(auroc) else "AUROC=n/a")

    thresholds = np.arange(THRESH_LO, THRESH_HI + 1e-9, THRESH_STEP)
    rows: list[dict] = []
    for t in thresholds:
        tt = float(t)
        pred = (y_score >= tt).astype(int)
        fpr = fpr_at_threshold(y_true, y_score, tt)
        fnr = fnr_at_threshold(y_true, y_score, tt)
        acc = float(accuracy_score(y_true, pred))
        rows.append(
            {
                "threshold": tt,
                "accuracy_pct": acc * 100.0,
                "fpr_pct": fpr * 100.0,
                "fnr_pct": fnr * 100.0,
                "precision_pct": float(precision_score(y_true, pred, zero_division=0)) * 100.0,
                "recall_pct": float(recall_score(y_true, pred, zero_division=0)) * 100.0,
                "fpr_dist_to_12": abs(fpr * 100.0 - FPR_TARGET_PCT),
            }
        )

    best = min(
        rows,
        key=lambda r: (r["fpr_dist_to_12"], -r["accuracy_pct"]),
    )

    print("\n=== Full sweep (v1_2_4_proper @ v1_2_4 test) ===")
    hdr = (
        f"{'thr':>5}  {'Acc%':>7}  {'FPR%':>7}  {'FNR%':>7}  {'Prec%':>7}  {'Rec%':>7}  "
        f"{'|FPR-12%|':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = "  <- closest FPR to 12% (then best Acc)" if r is best else ""
        print(
            f"{r['threshold']:5.2f}  {r['accuracy_pct']:7.2f}  {r['fpr_pct']:7.2f}  "
            f"{r['fnr_pct']:7.2f}  {r['precision_pct']:7.2f}  {r['recall_pct']:7.2f}  "
            f"{r['fpr_dist_to_12']:9.3f}{mark}"
        )

    print(
        f"\nChosen: threshold={best['threshold']:.2f}  FPR={best['fpr_pct']:.2f}%  "
        f"accuracy={best['accuracy_pct']:.2f}%  (target FPR ~{FPR_TARGET_PCT}%)"
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "accuracy_pct",
                "fpr_pct",
                "fnr_pct",
                "precision_pct",
                "recall_pct",
                "fpr_dist_to_12pct",
                "eer_score_pct",
                "auroc",
                "n_samples",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "threshold": f"{r['threshold']:.2f}",
                    "accuracy_pct": f"{r['accuracy_pct']:.4f}",
                    "fpr_pct": f"{r['fpr_pct']:.4f}",
                    "fnr_pct": f"{r['fnr_pct']:.4f}",
                    "precision_pct": f"{r['precision_pct']:.4f}",
                    "recall_pct": f"{r['recall_pct']:.4f}",
                    "fpr_dist_to_12pct": f"{r['fpr_dist_to_12']:.4f}",
                    "eer_score_pct": f"{eer*100:.4f}",
                    "auroc": f"{auroc:.6f}" if not np.isnan(auroc) else "",
                    "n_samples": len(y_true),
                }
            )
        w.writerow(
            {
                "threshold": f"BEST_FPR12_then_ACC={best['threshold']:.2f}",
                "accuracy_pct": f"{best['accuracy_pct']:.4f}",
                "fpr_pct": f"{best['fpr_pct']:.4f}",
                "fnr_pct": f"{best['fnr_pct']:.4f}",
                "precision_pct": "",
                "recall_pct": "",
                "fpr_dist_to_12pct": f"{best['fpr_dist_to_12']:.4f}",
                "eer_score_pct": f"{eer*100:.4f}",
                "auroc": f"{auroc:.6f}" if not np.isnan(auroc) else "",
                "n_samples": len(y_true),
            }
        )

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
