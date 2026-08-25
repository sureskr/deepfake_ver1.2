"""
train_xgboost_v1_2_6.py — XGBoost on Wav2Vec2 statistical functionals.

Alternative classifier skeleton. Keep the frozen Wav2Vec2-large-960h feature
extractor, but replace the linear head with gradient-boosted trees over RICH
temporal functionals (mean/std/min/max/percentiles per dim) rather than a single
mean-pooled vector. Rationale:
  - Trees learn non-linear boundaries; fakes from diverse TTS engines occupy
    different regions of embedding space a single hyperplane cannot separate.
  - Functionals preserve temporal structure (e.g. unnatural smoothness) that
    mean-pooling throws away — exactly the rich tabular input trees thrive on.

Trains on the same data/v1_2_5 splits as v1.2.5 and reports EER on the SAME
held-out modern-TTS test set, broken down by source, for an apples-to-apples
comparison with the neural head.

Feature extraction is cached to disk, so hyperparameter re-runs are fast.

Usage:
  python train_xgboost_v1_2_6.py --config ml_engine/config/training_v1_2_5.json --train
  python train_xgboost_v1_2_6.py --config ml_engine/config/training_v1_2_5.json  # dry run: counts only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent / "ml_engine"))
from training_config import load_training_config  # noqa: E402
from train_v1_2_4_proper import load_audio_mono  # noqa: E402
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
# Functionals computed over the time axis of the Wav2Vec2 hidden states.
FUNCTIONAL_NAMES = ["mean", "std", "min", "max", "p10", "p50", "p90"]


def eer(y: np.ndarray, s: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[i])


def list_split(real_dir: Path, fake_dir: Path) -> list[tuple[Path, int, str]]:
    rows: list[tuple[Path, int, str]] = []
    for label, d in ((0, real_dir), (1, fake_dir)):
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_EXT:
                source = p.name.split("_", 1)[0]  # asvspoof / mlaad
                rows.append((p, label, source))
    return rows


def functionals(hidden: np.ndarray) -> np.ndarray:
    """(T, D) hidden states -> (len(FUNCTIONAL_NAMES) * D,) feature vector."""
    feats = [
        hidden.mean(axis=0),
        hidden.std(axis=0),
        hidden.min(axis=0),
        hidden.max(axis=0),
        np.percentile(hidden, 10, axis=0),
        np.percentile(hidden, 50, axis=0),
        np.percentile(hidden, 90, axis=0),
    ]
    return np.concatenate(feats).astype(np.float32)


@torch.no_grad()
def extract_features(
    rows, processor, backbone, target_samples, device, bs=32
) -> np.ndarray:
    X: list[np.ndarray] = []
    backbone.eval()
    t0 = time.time()
    for i in range(0, len(rows), bs):
        waves = []
        for p, _, _ in rows[i : i + bs]:
            w = load_audio_mono(p)
            if len(w) > target_samples:
                w = w[:target_samples]
            elif len(w) < target_samples:
                w = np.pad(w, (0, target_samples - len(w)), mode="constant")
            waves.append(w.astype(np.float32))
        inp = processor(list(waves), sampling_rate=16000, return_tensors="pt", padding=False)
        hidden = backbone(inp.input_values.to(device)).last_hidden_state  # (B, T, D)
        h = hidden.cpu().numpy()
        for j in range(h.shape[0]):
            X.append(functionals(h[j]))
        if (i // bs) % 10 == 0:
            done = min(i + bs, len(rows))
            print(f"    features {done}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(X)


def cached_features(name, rows, cache_dir: Path, extractor_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.npz"
    paths = [str(p) for p, _, _ in rows]
    if cache.is_file():
        d = np.load(cache, allow_pickle=True)
        if list(d["paths"]) == paths:
            print(f"  [{name}] using cached features {d['X'].shape}")
            return d["X"], d["y"], d["src"]
        print(f"  [{name}] cache stale (file list changed) — recomputing")
    print(f"  [{name}] extracting features for {len(rows)} files...")
    X = extractor_fn(rows)
    y = np.array([lbl for _, lbl, _ in rows])
    src = np.array([s for _, _, s in rows])
    np.savez_compressed(cache, X=X, y=y, src=src, paths=np.array(paths))
    return X, y, src


def per_source_report(y, s, src, threshold=0.55) -> dict:
    r = {
        "n": int(len(y)),
        "eer_overall": eer(y, s),
        "acc@0.50": float(((s >= 0.50).astype(int) == y).mean()),
        "acc@0.55": float(((s >= threshold).astype(int) == y).mean()),
    }
    for source in sorted(set(src[y == 1])):
        mask = (y == 0) | ((y == 1) & (src == source))
        r[f"eer[{source}]"] = eer(y[mask], s[mask])
        fake_mask = (y == 1) & (src == source)
        r[f"recall@0.55[{source}]"] = float((s[fake_mask] >= threshold).mean())
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="ml_engine/config/training_v1_2_5.json")
    ap.add_argument("--train", action="store_true", help="Run training; omit to just print counts.")
    ap.add_argument("--feature-cache", default="data/features_v1_2_6")
    ap.add_argument("--out", default="ml_engine/models/xgboost_v1_2_6")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    args = ap.parse_args()

    cfg = load_training_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = list_split(cfg.train_real_dir, cfg.train_fake_dir)
    val_rows = list_split(cfg.val_real_dir, cfg.val_fake_dir)
    test_rows = list_split(cfg.held_out_test_real_dir, cfg.held_out_test_fake_dir)

    def counts(rows):
        y = np.array([l for _, l, _ in rows])
        return len(rows), int((y == 0).sum()), int((y == 1).sum())

    print("=" * 70)
    print("train_xgboost_v1_2_6 — XGBoost on Wav2Vec2 functionals")
    print("=" * 70)
    print(f"  Feature extractor: frozen Wav2Vec2-large ({cfg.wav2vec2_dir})")
    print(f"  Functionals:       {FUNCTIONAL_NAMES}  -> {len(FUNCTIONAL_NAMES)}x1024 dims")
    print(f"  Classifier:        XGBoost (n_est={args.n_estimators}, lr={args.learning_rate}, "
          f"depth={args.max_depth})")
    print(f"  Device (features): {device}")
    for nm, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        n, nr, nf = counts(rows)
        print(f"  {nm:5s}: {n:5d} files  (real={nr}, fake={nf})")
    print("=" * 70)

    if not args.train:
        print("\nNot training (default). Pass --train to extract features + fit XGBoost.\n")
        return

    import xgboost as xgb

    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(cfg.wav2vec2_dir), local_files_only=True)
    backbone = Wav2Vec2Model.from_pretrained(str(cfg.wav2vec2_dir), local_files_only=True).to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    def extractor(rows):
        return extract_features(rows, processor, backbone, cfg.target_samples, device)

    cache_dir = Path(args.feature_cache)
    Xtr, ytr, _ = cached_features("train", train_rows, cache_dir, extractor)
    Xva, yva, _ = cached_features("val", val_rows, cache_dir, extractor)
    Xte, yte, ste = cached_features("test", test_rows, cache_dir, extractor)
    print(f"\nFeature matrices: train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.3,  # high-dim features: sample columns aggressively
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=args.early_stopping_rounds,
        tree_method="hist",
        n_jobs=-1,
    )
    print("\nFitting XGBoost (early stopping on val AUC)...")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best_it = clf.best_iteration
    print(f"  best_iteration={best_it}  best_val_auc={clf.best_score:.4f}")

    s_val = clf.predict_proba(Xva)[:, 1]
    s_test = clf.predict_proba(Xte)[:, 1]
    val_eer = eer(yva, s_val)
    print(f"  val EER: {val_eer:.4f}")

    report = per_source_report(yte, s_test, ste)
    print("\n=== XGBoost v1.2.6 — held-out modern-TTS test set ===")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(out_dir / "model.json"))
    importances = clf.feature_importances_
    top = np.argsort(importances)[::-1][:20]
    payload = {
        "format": "xgboost_v1_2_6",
        "feature_extractor": "wav2vec2-large-960h frozen",
        "functionals": FUNCTIONAL_NAMES,
        "n_features": int(Xtr.shape[1]),
        "best_iteration": int(best_it),
        "best_val_auc": float(clf.best_score),
        "val_eer": float(val_eer),
        "test_report": report,
        "top20_feature_idx": top.tolist(),
        "params": {
            "n_estimators": args.n_estimators, "learning_rate": args.learning_rate,
            "max_depth": args.max_depth, "colsample_bytree": 0.3, "subsample": 0.8,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved model -> {out_dir/'model.json'}  and results -> {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
