"""
eval_combined_v2.py - Confidence-based two-stage deepfake detector.

Stage 1 - ML runs on every file:
  ml_score >= 0.75  -> FAKE  (high confidence, skip physics)
  ml_score <= 0.25  -> REAL  (high confidence, skip physics)
  0.25 < score < 0.75 -> uncertain -> Stage 2

Stage 2 - Physics runs only on uncertain files:
  checks_failed >= 2 -> FAKE  (physics confirmed)
  checks_failed < 2  -> REAL  (physics cleared)

Baselines (release_in_the_wild, 400 files):
  Physics alone : 67.50% acc, 32.00% EER, 11.00% FPR, 54.00% FNR
  ML alone      : 66.76% acc, 33.33% EER
  Combined AND  : 59.75% acc, 38.25% EER,  3.50% FPR, 77.00% FNR
  Target        : >70% acc, <10% FPR, <50% FNR
"""
import sys, os, csv, random, argparse, subprocess
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import detect

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ML_HIGH  = 0.75
DEFAULT_ML_LOW   = 0.25
DEFAULT_PHYS_MIN = 2

# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class V2Result:
    verdict:       str    # FAKE | REAL
    confidence:    str    # HIGH | MEDIUM
    reason:        str    # ml_highconf_fake | ml_highconf_real |
                          # physics_confirmed | physics_cleared
    ml_score:      float
    checks_failed: int    # -1 when physics was skipped

# ── ML subprocess ─────────────────────────────────────────────────────────────
def start_ml_worker() -> subprocess.Popen:
    worker = os.path.join(_HERE, "ml_worker.py")
    proc = subprocess.Popen(
        [sys.executable, worker],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,   # inherit parent stderr so GPU messages are visible
        text=True,
        bufsize=1,
    )
    line = proc.stdout.readline().strip()
    if line != "READY":
        raise RuntimeError(f"ml_worker unexpected start: {line!r}")
    line = proc.stdout.readline().strip()
    if line != "LOADED":
        raise RuntimeError(f"ml_worker load failed: {line!r}")
    return proc


def _ml_score(proc: subprocess.Popen, path: str) -> float:
    proc.stdin.write(path + "\n")
    proc.stdin.flush()
    resp = proc.stdout.readline().strip()
    if resp.startswith("ERROR"):
        raise RuntimeError(f"ml_worker: {resp}")
    return float(resp)


def stop_ml_worker(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.close()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# ── Core two-stage logic ──────────────────────────────────────────────────────
def predict(proc: subprocess.Popen, path: str,
            ml_high: float, ml_low: float, phys_min: int) -> V2Result:
    score = _ml_score(proc, path)

    # Stage 1: high-confidence ML
    if score >= ml_high:
        return V2Result("FAKE", "HIGH", "ml_highconf_fake", score, -1)
    if score <= ml_low:
        return V2Result("REAL", "HIGH", "ml_highconf_real", score, -1)

    # Stage 2: uncertain — run physics
    phys = detect(path)
    cf   = phys.checks_failed
    if cf >= phys_min:
        return V2Result("FAKE", "MEDIUM", "physics_confirmed", score, cf)
    return V2Result("REAL", "MEDIUM", "physics_cleared", score, cf)


# ── EER ───────────────────────────────────────────────────────────────────────
def _eer(y_true, y_scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Confidence-based two-stage deepfake detector v2")
    parser.add_argument("--dataset",  type=str, required=True,
                        help="Path to release_in_the_wild directory")
    parser.add_argument("--max",      type=int, default=200,
                        help="Max files per class (default 200 -> 400 total)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--ml_high",  type=float, default=DEFAULT_ML_HIGH,
                        help="ML score >= this -> FAKE immediately (default 0.75)")
    parser.add_argument("--ml_low",   type=float, default=DEFAULT_ML_LOW,
                        help="ML score <= this -> REAL immediately (default 0.25)")
    parser.add_argument("--phys_min", type=int,   default=DEFAULT_PHYS_MIN,
                        help="Physics checks_failed >= this -> FAKE (default 2)")
    args = parser.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    meta_path = os.path.join(args.dataset, "meta.csv")
    if not os.path.exists(meta_path):
        sys.exit(f"meta.csv not found: {meta_path}")

    real_entries, fake_entries = [], []
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            path = os.path.join(args.dataset, row["file"])
            if row["label"] == "bona-fide":
                real_entries.append((path, "bona-fide"))
            elif row["label"] == "spoof":
                fake_entries.append((path, "spoof"))

    rng = random.Random(args.seed)
    rng.shuffle(real_entries)
    rng.shuffle(fake_entries)
    selected = real_entries[:args.max] + fake_entries[:args.max]
    rng.shuffle(selected)

    n_real = min(args.max, len(real_entries))
    n_fake = min(args.max, len(fake_entries))
    print(f"Dataset : {args.dataset}")
    print(f"meta.csv: {len(real_entries)} bona-fide, {len(fake_entries)} spoof")
    print(f"Evaluating: {n_real} real + {n_fake} fake = {len(selected)} files")
    print(f"Thresholds: ML high={args.ml_high}, ML low={args.ml_low}, "
          f"physics_min_fails={args.phys_min}\n")

    # ── Start ML worker ───────────────────────────────────────────────────────
    print("Pre-warming ML model ...", flush=True)
    proc = start_ml_worker()
    print("ML model ready.\n", flush=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    rows         = []
    tp = tn = fp = fn = 0
    ml_scores_all = []
    labels_all    = []
    reason_counts = {}
    physics_run   = 0

    for i, (path, true_label) in enumerate(selected, 1):
        if not os.path.exists(path):
            print(f"  SKIP {os.path.basename(path)} (not found)")
            continue

        try:
            result = predict(proc, path, args.ml_high, args.ml_low, args.phys_min)
        except Exception as e:
            print(f"  ERROR {os.path.basename(path)}: {e}")
            continue

        predicted_fake = result.verdict == "FAKE"
        actually_fake  = true_label == "spoof"

        if   predicted_fake and     actually_fake: tp += 1
        elif not predicted_fake and not actually_fake: tn += 1
        elif predicted_fake and not actually_fake: fp += 1
        else:                                      fn += 1

        ml_scores_all.append(result.ml_score)
        labels_all.append(1 if actually_fake else 0)
        reason_counts[result.reason] = reason_counts.get(result.reason, 0) + 1
        if result.checks_failed >= 0:
            physics_run += 1

        rows.append({
            "file":          os.path.basename(path),
            "true":          true_label,
            "verdict":       result.verdict,
            "correct":       int(predicted_fake == actually_fake),
            "confidence":    result.confidence,
            "reason":        result.reason,
            "ml_score":      f"{result.ml_score:.4f}",
            "checks_failed": result.checks_failed,
        })

        if i % 25 == 0:
            print(f"  [{i:>3}/{len(selected)}] "
                  f"tp={tp} tn={tn} fp={fp} fn={fn} | "
                  f"phys_run={physics_run} ...")

    stop_ml_worker(proc)

    # ── Metrics ───────────────────────────────────────────────────────────────
    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total    if total     else 0.0
    fpr_val  = fp / (fp + tn)       if (fp + tn) else 0.0
    fnr_val  = fn / (fn + tp)       if (fn + tp) else 0.0

    arr_scores = np.array(ml_scores_all)
    arr_labels = np.array(labels_all)
    try:
        auroc = roc_auc_score(arr_labels, arr_scores)
    except Exception:
        auroc = float("nan")
    try:
        eer = _eer(arr_labels, arr_scores)
    except Exception:
        eer = float("nan")

    # ── Report ────────────────────────────────────────────────────────────────
    W = 68
    print(f"\n{'='*W}")
    print(f"  release_in_the_wild -- Confidence-Based Combined v2")
    print(f"  Files evaluated : {total}  |  ML high={args.ml_high}  "
          f"ML low={args.ml_low}  phys_min={args.phys_min}")
    print(f"  {'Metric':<10}  {'v2':>9}  {'AND':>9}  {'Physics':>9}  {'ML-only':>9}")
    print(f"  {'-'*52}")
    print(f"  {'Accuracy':<10}  {accuracy*100:>8.2f}%  {'59.75%':>9}  {'67.50%':>9}  {'66.76%':>9}")
    print(f"  {'EER':<10}  {eer*100:>8.2f}%  {'38.25%':>9}  {'32.00%':>9}  {'33.33%':>9}")
    print(f"  {'FPR':<10}  {fpr_val*100:>8.2f}%  {'3.50%':>9}  {'11.00%':>9}  {'?':>9}")
    print(f"  {'FNR':<10}  {fnr_val*100:>8.2f}%  {'77.00%':>9}  {'54.00%':>9}  {'?':>9}")
    print(f"  AUROC      : {auroc:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*W}")

    print(f"\n  Decision path breakdown ({total} files):")
    hf = reason_counts.get('ml_highconf_fake', 0)
    hr = reason_counts.get('ml_highconf_real', 0)
    pc = reason_counts.get('physics_confirmed', 0)
    pcl= reason_counts.get('physics_cleared',  0)
    print(f"    ml_highconf_fake   (ML >= {args.ml_high}) -> FAKE  : {hf:>4}  ({100*hf/total:.1f}%)")
    print(f"    ml_highconf_real   (ML <= {args.ml_low}) -> REAL  : {hr:>4}  ({100*hr/total:.1f}%)")
    print(f"    physics_confirmed  (uncertain, fails >= {args.phys_min}) -> FAKE : {pc:>4}  ({100*pc/total:.1f}%)")
    print(f"    physics_cleared    (uncertain, fails <  {args.phys_min}) -> REAL : {pcl:>4}  ({100*pcl/total:.1f}%)")
    print(f"    Physics ran on {physics_run}/{total} files ({100*physics_run/total:.1f}% of total)")

    acc_pass = accuracy > 0.70
    fpr_pass = fpr_val  < 0.10
    fnr_pass = fnr_val  < 0.50
    print(f"\n  Targets:")
    print(f"    Accuracy > 70%  : {'PASS' if acc_pass else 'FAIL'}  ({accuracy*100:.2f}%)")
    print(f"    FPR < 10%       : {'PASS' if fpr_pass else 'FAIL'}  ({fpr_val*100:.2f}%)")
    print(f"    FNR < 50%       : {'PASS' if fnr_pass else 'FAIL'}  ({fnr_val*100:.2f}%)")

    out_csv = "results_combined_v2.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(out_csv, "a", newline="") as f:
            f.write(f"\nSUMMARY,accuracy={accuracy:.4f},EER={eer:.4f},"
                    f"FPR={fpr_val:.4f},FNR={fnr_val:.4f},"
                    f"TP={tp},TN={tn},FP={fp},FN={fn}\n")
        print(f"\nResults saved -> {out_csv}")


if __name__ == "__main__":
    main()
