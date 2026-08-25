"""
eval_external_datasets.py — score any set of v1.2.x checkpoints against any test_data/ subset.

All four shipped models share the same frozen Wav2Vec2-large-960h front end, so audio is
read once, pushed through the backbone once, and BOTH feature views are derived from the
same hidden states:

  * neural heads (v1.2.4 / v1.2.5 / v1.2.6) — mean-pooled 1024-d vector,
  * XGBoost v1.2.6 — 7 functionals (mean, std, min, max, p10, p50, p90) over the time
    axis -> 7,168 dims, exactly as built by train_xgboost_v1_2_6.py.

The "mean" functional IS the mean-pool, so every model sees identical audio and identical
features. Audio handling matches both training and the original v1.2.4 eval path: mono
16 kHz, front crop / zero-pad to 10 s, Wav2Vec2FeatureExtractor normalisation.

Metrics per (dataset, model): EER + threshold, AUROC, both with bootstrap CIs; accuracy /
precision / recall / F1 / confusion / FPR / FNR at the production threshold 0.55 AND at the
model's own EER threshold; per-class score distributions; ROC/DET curve points; per-subgroup
breakdowns (TTS engine, codec, attack ID, speaker). Across models: paired bootstrap CIs on
EER differences and McNemar tests on per-file correctness.

Usage:
  python eval_external_datasets.py                          # all models, all datasets
  python eval_external_datasets.py --datasets in_the_wild --models v1_2_4 xgb_v1_2_6
  python eval_external_datasets.py --decode librosa --datasets deepfake-audio  # decode check
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# --- XGBoost worker mode ------------------------------------------------------------
# torch and xgboost link different OpenMP runtimes on macOS; loading both in one process
# segfaults or deadlocks. The trees are therefore scored in a short-lived subprocess that
# imports xgboost and never touches torch. This dispatch has to run before `import torch`.
if len(sys.argv) > 1 and sys.argv[1] == "--xgb-worker":
    import numpy as _np
    import xgboost as _xgb

    _model_path, _feat_path, _out_path = sys.argv[2:5]
    _booster = _xgb.Booster()
    _booster.load_model(_model_path)
    _X = _np.load(_feat_path)["X"]
    _scores = _booster.predict(_xgb.DMatrix(_X))
    _np.save(_out_path, _np.asarray(_scores, dtype=_np.float64))
    sys.exit(0)

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

BASE = Path(__file__).resolve().parent
TEST_DATA = BASE / "test_data"
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

SAMPLE_RATE = 16000
TARGET_SECONDS = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SECONDS)
PRODUCTION_THRESHOLD = 0.55
FUNCTIONAL_NAMES = ["mean", "std", "min", "max", "p10", "p50", "p90"]
SEED = 42

BACKBONE_ID = "facebook/wav2vec2-large-960h"

NEURAL_MODELS = {
    "v1_2_4": "ml_engine/models/v1_2_4_proper/best_model.pth",
    "v1_2_5": "ml_engine/models/v1_2_5_proper/best_model.pth",
    "v1_2_6": "ml_engine/models/v1_2_6_proper/best_model.pth",
}
XGB_MODELS = {"xgb_v1_2_6": "ml_engine/models/xgboost_v1_2_6/model.json"}
ALL_MODELS = list(NEURAL_MODELS) + list(XGB_MODELS)

DATASETS = ["deepfake-audio", "asvspoof5", "mlaad", "in_the_wild", "codecfake"]

# v1.2.4 numbers from eval reports 01-05 (ML engine, 200 files each). The harness is
# considered validated if it reproduces these.
V1_2_4_BASELINES = {
    "deepfake-audio": {"accuracy": 0.290, "eer": 0.850},
    "asvspoof5": {"accuracy": 0.755, "eer": 0.250},
    "mlaad": {"accuracy": 0.445, "eer": 0.620},
    "in_the_wild": {"accuracy": 0.720, "eer": 0.260},
    "codecfake": {"accuracy": 0.545, "eer": 0.410},
}
# Reports 01/03 quote accuracy at 0.55; reports 02/04/05 quote inference.py's accuracy at
# 0.50. Both are computed, and the check passes if either matches.
BASELINE_TOLERANCE = 0.02


# --------------------------------------------------------------------------------------
# audio + features
# --------------------------------------------------------------------------------------
def load_audio_native(path: Path) -> np.ndarray:
    """Loader used by v1.2.5/v1.2.6/XGBoost training (miniaudio for FLAC, librosa else)."""
    if path.suffix.lower() == ".flac":
        import miniaudio

        dec = miniaudio.flac_read_file_f32(str(path))
        wav = np.asarray(dec.samples, dtype=np.float32)
        if dec.nchannels > 1:
            wav = wav.reshape(-1, dec.nchannels).mean(axis=1)
        if dec.sample_rate != SAMPLE_RATE:
            import librosa as lr

            wav = lr.resample(wav, orig_sr=dec.sample_rate, target_sr=SAMPLE_RATE)
        return wav.astype(np.float32, copy=False)
    import librosa

    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32, copy=False)


def load_audio_librosa(path: Path) -> np.ndarray:
    """Loader used by the original deployment_v1_2_4/inference.py eval path."""
    import librosa

    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32, copy=False)


DECODERS = {"native": load_audio_native, "librosa": load_audio_librosa}


def crop_or_pad(wav: np.ndarray) -> np.ndarray:
    if len(wav) > TARGET_SAMPLES:
        return wav[:TARGET_SAMPLES]
    if len(wav) < TARGET_SAMPLES:
        return np.pad(wav, (0, TARGET_SAMPLES - len(wav)), mode="constant")
    return wav


def functionals(hidden: np.ndarray) -> np.ndarray:
    """(T, D) hidden states -> (7*D,) — identical to train_xgboost_v1_2_6.functionals."""
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


def list_dataset(dataset: str) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    for label, sub in ((0, "real"), (1, "fake")):
        d = TEST_DATA / dataset / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXT:
                rows.append((p, label))
    return rows


@torch.no_grad()
def extract_features(rows, processor, backbone, decode: str, batch_size: int = 8) -> np.ndarray:
    decoder = DECODERS[decode]
    out: list[np.ndarray] = []
    t0 = time.time()
    for i in range(0, len(rows), batch_size):
        waves = [crop_or_pad(decoder(p)).astype(np.float32) for p, _ in rows[i : i + batch_size]]
        inp = processor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False)
        hidden = backbone(inp.input_values).last_hidden_state.cpu().numpy()
        for j in range(hidden.shape[0]):
            out.append(functionals(hidden[j]))
        done = min(i + batch_size, len(rows))
        print(f"    features {done}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    return np.stack(out)


def cached_features(dataset: str, rows, cache_dir: Path, decode: str, extractor) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if decode == "native" else f"_{decode}"
    cache = cache_dir / f"{dataset}{suffix}.npz"
    paths = [str(p) for p, _ in rows]
    if cache.is_file():
        d = np.load(cache, allow_pickle=True)
        if list(d["paths"]) == paths:
            print(f"  [{dataset}] cached features {d['X'].shape}")
            return d["X"]
        print(f"  [{dataset}] cache stale — recomputing")
    print(f"  [{dataset}] extracting features for {len(rows)} files (decode={decode})...")
    X = extractor(rows)
    np.savez_compressed(cache, X=X, paths=np.array(paths))
    return X


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------
@dataclass
class LinearHead:
    name: str
    weight: np.ndarray  # (1024,)
    bias: float
    checkpoint_meta: dict

    def score(self, X: np.ndarray) -> np.ndarray:
        pooled = X[:, :1024]  # the "mean" functional is the mean-pooled vector
        logits = pooled @ self.weight + self.bias
        return 1.0 / (1.0 + np.exp(-logits))


def load_linear_head(name: str, path: Path) -> LinearHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = ck.get("head_state_dict") or {}
    if "classifier.weight" not in head:
        raise KeyError(f"{path}: expected a linear head (classifier.weight), got {list(head)}")
    if ck.get("backbone_state_dict"):
        raise RuntimeError(
            f"{path} carries fine-tuned backbone weights; the shared-forward-pass "
            "assumption of this harness does not hold for it."
        )
    w = head["classifier.weight"].detach().numpy().astype(np.float64).reshape(-1)
    b = float(head["classifier.bias"].detach().numpy().reshape(-1)[0])
    meta = {
        "format": ck.get("format"),
        "epoch": ck.get("epoch"),
        "val_metrics": {
            k: float(v) for k, v in (ck.get("val_metrics") or {}).items() if isinstance(v, (int, float))
        },
        "train_config": {
            k: v for k, v in (ck.get("train_config") or {}).items() if isinstance(v, (str, int, float, bool))
        },
    }
    return LinearHead(name, w, b, meta)


@dataclass
class XGBModel:
    name: str
    model_path: Path
    checkpoint_meta: dict

    def score(self, X: np.ndarray) -> np.ndarray:
        """Score in a subprocess — see the OpenMP note at the top of this file."""
        with tempfile.TemporaryDirectory() as tmp:
            feat_path = Path(tmp) / "features.npz"
            out_path = Path(tmp) / "scores.npy"
            np.savez(feat_path, X=X.astype(np.float32))
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--xgb-worker",
                 str(self.model_path), str(feat_path), str(out_path)],
                check=True,
            )
            return np.load(out_path)


def load_xgb(name: str, path: Path) -> XGBModel:
    if not path.is_file():
        raise SystemExit(f"XGBoost model not found: {path}")
    meta_path = path.parent / "results.json"
    meta = {}
    if meta_path.is_file():
        raw = json.loads(meta_path.read_text())
        meta = {k: raw[k] for k in ("best_iteration", "best_val_auc", "val_eer", "n_features", "params") if k in raw}
    meta["num_features_in_booster"] = int(
        subprocess.check_output(
            [sys.executable, "-c",
             "import sys,xgboost as x;b=x.Booster();b.load_model(sys.argv[1]);print(b.num_features())",
             str(path)],
            text=True,
        ).strip()
    )
    return XGBModel(name, path, meta)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def eer_and_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(fpr[i]), float(thr[i])


def threshold_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "fnr": fn / (fn + tp) if fn + tp else float("nan"),
    }


def score_distribution(s: np.ndarray) -> dict:
    return {
        "n": int(len(s)),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "deciles": [float(v) for v in np.percentile(s, np.arange(10, 100, 10))],
    }


def bootstrap_indices(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Class-stratified resampling: keeps the 100/100 real/fake design of each test set."""
    real_idx = np.flatnonzero(y == 0)
    fake_idx = np.flatnonzero(y == 1)
    out = np.empty((n_boot, len(y)), dtype=np.int64)
    for b in range(n_boot):
        out[b] = np.concatenate(
            [rng.choice(real_idx, len(real_idx), replace=True),
             rng.choice(fake_idx, len(fake_idx), replace=True)]
        )
    return out


def ci(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "n_boot": int(v.size)}


def roc_points(y: np.ndarray, s: np.ndarray) -> dict:
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.where(np.isinf(thr), np.nan, thr)
    return {
        "fpr": [round(float(v), 6) for v in fpr],
        "tpr": [round(float(v), 6) for v in tpr],
        "fnr": [round(float(1 - v), 6) for v in tpr],
        "thresholds": [None if np.isnan(v) else round(float(v), 6) for v in thr],
    }


def subgroup_metrics(y: np.ndarray, s: np.ndarray, groups: np.ndarray, eer_thr: float) -> dict:
    """Per (subgroup, class) metrics: fake subgroups are scored against ALL real files,
    real subgroups report false-flag rates. Some datasets (In-the-Wild) use the same
    subgroup label — a speaker — for both real and fake audio, so a group that carries both
    classes is reported as two rows, "<group> (fake)" and "<group> (real)"."""
    real_mask = y == 0
    out = {}
    for g in sorted(set(groups.tolist())):
        g_mask = groups == g
        classes = sorted(set(y[g_mask].tolist()))
        for cls in classes:
            mask = g_mask & (y == cls)
            key = str(g) if len(classes) == 1 else f"{g} ({'fake' if cls else 'real'})"
            entry = {
                "n": int(mask.sum()),
                "class": "fake" if cls else "real",
                "score_mean": float(s[mask].mean()),
                "score_median": float(np.median(s[mask])),
                "score_std": float(np.std(s[mask], ddof=1)) if mask.sum() > 1 else 0.0,
            }
            if cls == 1:
                sub = real_mask | mask
                e, t = eer_and_threshold(y[sub], s[sub])
                entry.update({
                    "eer_vs_all_real": e,
                    "eer_threshold": t,
                    "recall@0.55": float((s[mask] >= PRODUCTION_THRESHOLD).mean()),
                    "recall@eer_threshold": float((s[mask] >= eer_thr).mean()),
                    "detected@0.55": int((s[mask] >= PRODUCTION_THRESHOLD).sum()),
                })
            else:
                entry.update({
                    "fpr@0.55": float((s[mask] >= PRODUCTION_THRESHOLD).mean()),
                    "fpr@eer_threshold": float((s[mask] >= eer_thr).mean()),
                    "flagged@0.55": int((s[mask] >= PRODUCTION_THRESHOLD).sum()),
                })
            out[key] = entry
    return out


def evaluate(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray) -> dict:
    e, t = eer_and_threshold(y, s)
    auroc = float(roc_auc_score(y, s))
    boot_eer, boot_auc = [], []
    for idx in boot_idx:
        yb, sb = y[idx], s[idx]
        if len(set(yb.tolist())) < 2:
            continue
        boot_eer.append(eer_and_threshold(yb, sb)[0])
        boot_auc.append(roc_auc_score(yb, sb))
    return {
        "n": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "eer": e,
        "eer_ci95": ci(np.array(boot_eer)),
        "eer_threshold": t,
        "auroc": auroc,
        "auroc_ci95": ci(np.array(boot_auc)),
        "at_production_threshold": threshold_metrics(y, s, PRODUCTION_THRESHOLD),
        "at_threshold_0.50": threshold_metrics(y, s, 0.50),
        "at_eer_threshold": threshold_metrics(y, s, t),
        "score_distribution": {
            "real": score_distribution(s[y == 0]),
            "fake": score_distribution(s[y == 1]),
            "separation_mean_fake_minus_real": float(s[y == 1].mean() - s[y == 0].mean()),
        },
        "roc_curve": roc_points(y, s),
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar on per-file correctness (b = A right/B wrong, c = A wrong/B right)."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    p = float(stats.binomtest(b, n, 0.5).pvalue) if n else 1.0
    return {"b_a_correct_b_wrong": b, "c_a_wrong_b_correct": c, "n_discordant": n, "p_value": p}


def compare_models(y: np.ndarray, scores: dict[str, np.ndarray], boot_idx: np.ndarray) -> dict:
    names = list(scores)
    eers = {m: eer_and_threshold(y, scores[m]) for m in names}
    out = {}
    for i, a in enumerate(names):
        for b_name in names[i + 1 :]:
            sa, sb = scores[a], scores[b_name]
            diffs = []
            for idx in boot_idx:
                yb = y[idx]
                if len(set(yb.tolist())) < 2:
                    continue
                diffs.append(
                    eer_and_threshold(yb, sa[idx])[0] - eer_and_threshold(yb, sb[idx])[0]
                )
            diffs = np.array(diffs)
            delta = eers[a][0] - eers[b_name][0]
            interval = ci(diffs)
            correct_055_a = ((sa >= PRODUCTION_THRESHOLD).astype(int) == y)
            correct_055_b = ((sb >= PRODUCTION_THRESHOLD).astype(int) == y)
            correct_eer_a = ((sa >= eers[a][1]).astype(int) == y)
            correct_eer_b = ((sb >= eers[b_name][1]).astype(int) == y)
            out[f"{a}_vs_{b_name}"] = {
                "eer_a": eers[a][0],
                "eer_b": eers[b_name][0],
                "eer_delta_a_minus_b": delta,
                "eer_delta_ci95": interval,
                "eer_delta_significant": bool(
                    not np.isnan(interval["lo"]) and (interval["lo"] > 0 or interval["hi"] < 0)
                ),
                "bootstrap_p_two_sided": float(
                    min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
                ) if diffs.size else float("nan"),
                "mcnemar@0.55": mcnemar(correct_055_a, correct_055_b),
                "mcnemar@own_eer_thresholds": mcnemar(correct_eer_a, correct_eer_b),
            }
    return out


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE, text=True).strip()
    except Exception:
        return "unknown"


def library_versions() -> dict:
    import librosa, sklearn, transformers

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "librosa": librosa.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": stats.__name__ and __import__("scipy").__version__,
    }
    try:  # queried in the worker process, which is the one that actually loads it
        versions["xgboost"] = subprocess.check_output(
            [sys.executable, "-c", "import xgboost; print(xgboost.__version__)"], text=True
        ).strip()
    except Exception:
        pass
    return versions


def mlaad_views(labels: dict, rows) -> dict[str, np.ndarray]:
    """Boolean masks splitting MLAAD into held-out vs training (contaminated) engines."""
    contamination = np.array(
        [labels[f"{'fake' if lbl else 'real'}/{p.name}"].get("contamination", "unknown")
         for p, lbl in rows]
    )
    y = np.array([lbl for _, lbl in rows])
    return {
        "mlaad_held_out_engines": (y == 0) | (contamination == "held_out"),
        "mlaad_seen_engines": (y == 0) | (contamination == "seen_in_training"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--decode", choices=list(DECODERS), default="native",
                    help="FLAC decode path: 'native' (miniaudio, as in v1.2.5/v1.2.6 training) "
                         "or 'librosa' (the original v1.2.4 eval path).")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--feature-cache", default="data/features_external")
    ap.add_argument("--labels", default="paper_data/subgroup_labels.json")
    ap.add_argument("--out", default="results_external_v1_2_6.json")
    ap.add_argument("--csv", default="paper_data/external_scores.csv")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_grad_enabled(False)

    labels_path = BASE / args.labels
    all_labels = json.loads(labels_path.read_text()) if labels_path.is_file() else {}

    print("Loading models...")
    models: dict[str, object] = {}
    for name in args.models:
        if name in NEURAL_MODELS:
            models[name] = load_linear_head(name, BASE / NEURAL_MODELS[name])
        elif name in XGB_MODELS:
            models[name] = load_xgb(name, BASE / XGB_MODELS[name])
        else:
            raise SystemExit(f"unknown model: {name}")
        print(f"  {name}: ok")

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    print(f"Loading backbone {BACKBONE_ID} (from local HF cache)...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(BACKBONE_ID)
    backbone = Wav2Vec2Model.from_pretrained(BACKBONE_ID)
    backbone.eval()

    rng = np.random.default_rng(SEED)
    results = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_sha(),
            "seed": SEED,
            "decode_path": args.decode,
            "n_bootstrap": args.bootstrap,
            "production_threshold": PRODUCTION_THRESHOLD,
            "backbone": BACKBONE_ID,
            "audio": {"sample_rate": SAMPLE_RATE, "target_seconds": TARGET_SECONDS,
                      "crop": "front crop / zero pad"},
            "functionals": FUNCTIONAL_NAMES,
            "models": {
                name: {
                    "path": (NEURAL_MODELS | XGB_MODELS)[name],
                    "kind": "linear_head_1024" if name in NEURAL_MODELS else "xgboost_functionals_7168",
                    "checkpoint_meta": m.checkpoint_meta,
                }
                for name, m in models.items()
            },
            "library_versions": library_versions(),
        },
        "datasets": {},
        "per_file_scores_csv": args.csv,
    }

    csv_rows = []
    for dataset in args.datasets:
        rows = list_dataset(dataset)
        if not rows:
            print(f"[{dataset}] no audio found — skipping")
            continue
        print(f"\n=== {dataset}: {len(rows)} files ===")
        y = np.array([lbl for _, lbl in rows])

        X = cached_features(
            dataset, rows, BASE / args.feature_cache, args.decode,
            lambda r: extract_features(r, processor, backbone, args.decode, args.batch_size),
        )
        scores = {name: np.asarray(m.score(X), dtype=np.float64) for name, m in models.items()}

        ds_labels = all_labels.get(dataset, {})
        keys = [f"{'fake' if lbl else 'real'}/{p.name}" for p, lbl in rows]
        groups = np.array([ds_labels.get(k, {}).get("subgroup", "unknown") for k in keys])

        boot_idx = bootstrap_indices(y, args.bootstrap, rng)
        entry = {"n_files": len(rows), "models": {}, "model_comparisons": compare_models(y, scores, boot_idx)}
        for name, s in scores.items():
            m = evaluate(y, s, boot_idx)
            m["per_subgroup"] = subgroup_metrics(y, s, groups, m["eer_threshold"])
            entry["models"][name] = m

        # MLAAD: held-out vs training engines must never be blended for v1.2.5/v1.2.6.
        if dataset == "mlaad" and ds_labels:
            entry["contamination_views"] = {}
            for view, mask in mlaad_views(ds_labels, rows).items():
                yv = y[mask]
                boot_v = bootstrap_indices(yv, args.bootstrap, rng)
                view_scores = {n: s[mask] for n, s in scores.items()}
                v_entry = {
                    "n_files": int(mask.sum()),
                    "models": {},
                    "model_comparisons": compare_models(yv, view_scores, boot_v),
                }
                for name, s in view_scores.items():
                    vm = evaluate(yv, s, boot_v)
                    vm["per_subgroup"] = subgroup_metrics(yv, s, groups[mask], vm["eer_threshold"])
                    v_entry["models"][name] = vm
                entry["contamination_views"][view] = v_entry

        # harness validation against the published v1.2.4 baselines
        if "v1_2_4" in scores and dataset in V1_2_4_BASELINES:
            base = V1_2_4_BASELINES[dataset]
            got = entry["models"]["v1_2_4"]
            acc055 = got["at_production_threshold"]["accuracy"]
            acc050 = got["at_threshold_0.50"]["accuracy"]
            acc_delta = min(abs(acc055 - base["accuracy"]), abs(acc050 - base["accuracy"]))
            eer_delta = abs(got["eer"] - base["eer"])
            entry["harness_validation"] = {
                "reported_accuracy": base["accuracy"],
                "reproduced_accuracy@0.55": acc055,
                "reproduced_accuracy@0.50": acc050,
                "accuracy_abs_delta": acc_delta,
                "reported_eer": base["eer"],
                "reproduced_eer": got["eer"],
                "eer_abs_delta": eer_delta,
                "tolerance": BASELINE_TOLERANCE,
                "passed": bool(acc_delta <= BASELINE_TOLERANCE and eer_delta <= BASELINE_TOLERANCE),
            }
            v = entry["harness_validation"]
            print(f"  harness check v1.2.4: acc {base['accuracy']:.3f} -> "
                  f"{acc055:.3f}/{acc050:.3f} (@0.55/@0.50), EER {base['eer']:.3f} -> "
                  f"{got['eer']:.3f}  {'PASS' if v['passed'] else 'FAIL'}")

        for i, (p, lbl) in enumerate(rows):
            meta = ds_labels.get(keys[i], {})
            row = {
                "dataset": dataset,
                "filename": f"{'fake' if lbl else 'real'}/{p.name}",
                "label": lbl,
                "label_name": "fake" if lbl else "real",
                "subgroup": meta.get("subgroup", "unknown"),
                "contamination": meta.get("contamination", ""),
                "orig_filename": meta.get("orig_filename") or "",
                "decode_path": args.decode,
            }
            for name in scores:
                row[f"score_{name}"] = round(float(scores[name][i]), 6)
            csv_rows.append(row)

        for name in scores:
            m = entry["models"][name]
            print(f"  {name:11s} EER {m['eer']:.3f} "
                  f"[{m['eer_ci95']['lo']:.3f},{m['eer_ci95']['hi']:.3f}]  "
                  f"AUROC {m['auroc']:.3f}  acc@0.55 {m['at_production_threshold']['accuracy']:.3f}  "
                  f"acc@EER {m['at_eer_threshold']['accuracy']:.3f}")

        results["datasets"][dataset] = entry

    out_path = BASE / args.out
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    if csv_rows:
        csv_path = BASE / args.csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Wrote {csv_path}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
