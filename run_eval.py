"""
Evaluate physics engine + combined (physics AND ML) on the test_data/ directory.
Reuses ML scores from results.json (already computed by deployment_v1_2_4/inference.py).
"""
import os
import sys
import json
import numpy as np
import soundfile as sf

# Add deepfake-engine to path so we can import the physics engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepfake-engine"))
from engine import detect_from_array

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
ML_RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
ML_THRESHOLD = 0.55  # v1.2.4 production threshold

# Load ML results (scores are in order: real_000..real_099, fake_000..fake_099)
with open(ML_RESULTS) as f:
    ml_data = json.load(f)
ml_scores = ml_data["predictions"]
ml_labels = ml_data["targets"]

# Build file list in the same order the ML inference used (real/ then fake/)
files = []
real_dir = os.path.join(TEST_DIR, "real")
fake_dir = os.path.join(TEST_DIR, "fake")
for p in sorted(os.listdir(real_dir)):
    files.append((os.path.join(real_dir, p), 0))
for p in sorted(os.listdir(fake_dir)):
    files.append((os.path.join(fake_dir, p), 1))

assert len(files) == len(ml_scores), f"File count mismatch: {len(files)} vs {len(ml_scores)}"

# Run physics engine on each file
print(f"Running physics engine on {len(files)} files...\n")

physics_verdicts = []
physics_checks_failed = []
combined_verdicts = []

for i, (path, label) in enumerate(files):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    result = detect_from_array(audio, sr, audio_path=path)

    physics_verdicts.append(result.verdict)
    physics_checks_failed.append(result.checks_failed)

    # Combined: AND logic — FAKE only if both physics AND ML say FAKE
    ml_verdict = "FAKE" if ml_scores[i] >= ML_THRESHOLD else "REAL"
    if result.verdict == "FAKE" and ml_verdict == "FAKE":
        combined_verdicts.append("FAKE")
    else:
        combined_verdicts.append("REAL")

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(files)}] processed...")

labels = np.array([l for _, l in files])

# --- Physics engine metrics ---
phys_preds = np.array([1 if v == "FAKE" else 0 for v in physics_verdicts])
phys_tp = int(((phys_preds == 1) & (labels == 1)).sum())
phys_tn = int(((phys_preds == 0) & (labels == 0)).sum())
phys_fp = int(((phys_preds == 1) & (labels == 0)).sum())
phys_fn = int(((phys_preds == 0) & (labels == 1)).sum())
phys_acc = (phys_tp + phys_tn) / len(labels)

# --- Combined engine metrics ---
comb_preds = np.array([1 if v == "FAKE" else 0 for v in combined_verdicts])
comb_tp = int(((comb_preds == 1) & (labels == 1)).sum())
comb_tn = int(((comb_preds == 0) & (labels == 0)).sum())
comb_fp = int(((comb_preds == 1) & (labels == 0)).sum())
comb_fn = int(((comb_preds == 0) & (labels == 1)).sum())
comb_acc = (comb_tp + comb_tn) / len(labels)

# --- ML engine metrics (from results.json, at 0.55 threshold) ---
ml_preds = np.array([1 if s >= ML_THRESHOLD else 0 for s in ml_scores])
ml_tp = int(((ml_preds == 1) & (labels == 1)).sum())
ml_tn = int(((ml_preds == 0) & (labels == 0)).sum())
ml_fp = int(((ml_preds == 1) & (labels == 0)).sum())
ml_fn = int(((ml_preds == 0) & (labels == 1)).sum())
ml_acc = (ml_tp + ml_tn) / len(labels)

print(f"\n{'='*60}")
print(f"  RESULTS — {len(labels)} files (100 real + 100 fake)")
print(f"  Dataset: garystafford/deepfake-audio-detection (HuggingFace)")
print(f"{'='*60}")

print(f"\n  ML Engine (v1.2.4, threshold={ML_THRESHOLD}):")
print(f"    Accuracy : {ml_acc*100:.1f}%")
print(f"    TP={ml_tp}  TN={ml_tn}  FP={ml_fp}  FN={ml_fn}")
print(f"    FPR={ml_fp/(ml_fp+ml_tn)*100:.1f}%  FNR={ml_fn/(ml_fn+ml_tp)*100:.1f}%")

print(f"\n  Physics Engine (>=2 checks fail = FAKE):")
print(f"    Accuracy : {phys_acc*100:.1f}%")
print(f"    TP={phys_tp}  TN={phys_tn}  FP={phys_fp}  FN={phys_fn}")
print(f"    FPR={phys_fp/(phys_fp+phys_tn)*100:.1f}%  FNR={phys_fn/(phys_fn+phys_tp)*100:.1f}%")

print(f"\n  Combined Engine (FAKE only if BOTH agree):")
print(f"    Accuracy : {comb_acc*100:.1f}%")
print(f"    TP={comb_tp}  TN={comb_tn}  FP={comb_fp}  FN={comb_fn}")
print(f"    FPR={comb_fp/(comb_fp+comb_tn)*100:.1f}%  FNR={comb_fn/(comb_fn+comb_tp)*100:.1f}%")

print(f"\n{'='*60}\n")

# Save all results
all_results = {
    "dataset": "garystafford/deepfake-audio-detection",
    "n_files": len(labels),
    "ml_threshold": ML_THRESHOLD,
    "ml_engine": {"accuracy": ml_acc, "tp": ml_tp, "tn": ml_tn, "fp": ml_fp, "fn": ml_fn},
    "physics_engine": {"accuracy": phys_acc, "tp": phys_tp, "tn": phys_tn, "fp": phys_fp, "fn": phys_fn},
    "combined_engine": {"accuracy": comb_acc, "tp": comb_tp, "tn": comb_tn, "fp": comb_fp, "fn": comb_fn},
    "per_file": [
        {
            "file": os.path.basename(files[i][0]),
            "label": "fake" if labels[i] == 1 else "real",
            "ml_score": round(ml_scores[i], 4),
            "ml_verdict": "FAKE" if ml_scores[i] >= ML_THRESHOLD else "REAL",
            "physics_verdict": physics_verdicts[i],
            "physics_checks_failed": physics_checks_failed[i],
            "combined_verdict": combined_verdicts[i],
        }
        for i in range(len(files))
    ],
}

out_path = os.path.join(os.path.dirname(__file__), "results_all_engines.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Detailed results saved to: {out_path}")
