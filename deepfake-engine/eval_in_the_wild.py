"""
Evaluate engine.py on the release_in_the_wild dataset.
Uses meta.csv (columns: file, speaker, label) where label is 'bona-fide' or 'spoof'.
"""
import sys, os, csv, random, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from engine import detect, _compute_eer

DATASET = r"D:\Backup_data_from_GPU_gaming_computer\_dataset_for_deepfake_training\release_in_the_wild"
OUT_CSV = "results_in_the_wild.csv"


def main(max_per_class: int = 100, seed: int = 42, dataset_dir: str = ""):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = dataset_dir if dataset_dir else DATASET
    meta_path = os.path.join(dataset, "meta.csv")
    if not os.path.exists(meta_path):
        sys.exit(f"meta.csv not found: {meta_path}")

    real_entries = []
    fake_entries = []
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            path = os.path.join(dataset, row["file"])
            if row["label"] == "bona-fide":
                real_entries.append((path, "bona-fide"))
            elif row["label"] == "spoof":
                fake_entries.append((path, "spoof"))

    rng = random.Random(seed)
    rng.shuffle(real_entries)
    rng.shuffle(fake_entries)

    selected = real_entries[:max_per_class] + fake_entries[:max_per_class]
    rng.shuffle(selected)

    n_real = min(max_per_class, len(real_entries))
    n_fake = min(max_per_class, len(fake_entries))
    print(f"Dataset: {dataset}")
    print(f"meta.csv: {len(real_entries)} bona-fide, {len(fake_entries)} spoof")
    print(f"Evaluating: {n_real} real + {n_fake} fake = {len(selected)} files\n")

    rows = []
    tp = tn = fp = fn = 0
    scores_real = []   # checks_failed for real files
    scores_fake = []   # checks_failed for fake files

    for i, (path, true_label) in enumerate(selected, 1):
        if not os.path.exists(path):
            print(f"  SKIP {os.path.basename(path)} (not found)")
            continue

        try:
            result = detect(path)
        except Exception as e:
            print(f"  ERROR {os.path.basename(path)}: {e}")
            continue

        predicted_fake = result.verdict == "FAKE"
        actually_fake  = true_label == "spoof"

        if   predicted_fake and     actually_fake: tp += 1
        elif not predicted_fake and not actually_fake: tn += 1
        elif predicted_fake and not actually_fake: fp += 1
        else:                                      fn += 1

        (scores_fake if actually_fake else scores_real).append(result.checks_failed)

        rows.append({
            "file":          os.path.basename(path),
            "true":          true_label,
            "predicted":     result.verdict.lower(),
            "correct":       int(predicted_fake == actually_fake),
            "checks_failed": result.checks_failed,
            "jitter_pct":    result.jitter_shimmer.detail.get("jitter_pct", ""),
            "shimmer_db":    result.jitter_shimmer.detail.get("shimmer_db", ""),
            "mean_hnr_db":   result.subglottal.detail.get("mean_hnr_db", ""),
            "cv_jitter_pct": result.temporal_consistency.detail.get("cv_jitter_pct", ""),
            "cv_hnr_pct":    result.temporal_consistency.detail.get("cv_hnr_pct", ""),
            "phase_dev":     result.phase_deviation.detail.get("phase_deviation", ""),
            "flux_cv_pct":   result.spectral_flux.detail.get("flux_cv_pct", ""),
            "sba_score":     result.subband_asymmetry.detail.get("sba_score", ""),
        })

        if i % 50 == 0:
            print(f"  [{i}/{len(selected)}] processed …")

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total    if total     else 0.0
    fpr      = fp / (fp + tn)       if (fp + tn) else 0.0
    fnr      = fn / (fn + tp)       if (fn + tp) else 0.0
    eer      = _compute_eer(scores_real, scores_fake)

    print(f"\n{'='*60}")
    print(f"  release_in_the_wild  —  {total} files")
    print(f"  Accuracy  : {accuracy*100:.2f} %")
    print(f"  EER       : {eer*100:.2f} %")
    print(f"  FPR       : {fpr*100:.2f} %  (real flagged as fake)")
    print(f"  FNR       : {fnr*100:.2f} %  (fake flagged as real)")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*60}")

    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(OUT_CSV, "a", newline="") as f:
            f.write(f"\nSUMMARY,accuracy={accuracy:.4f},EER={eer:.4f},"
                    f"FPR={fpr:.4f},FNR={fnr:.4f},"
                    f"TP={tp},TN={tn},FP={fp},FN={fn}\n")
        print(f"\nResults saved -> {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=100,
                        help="Max files per class (default 100 -> 200 total)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="",
                        help="Path to release_in_the_wild directory (overrides built-in default)")
    args = parser.parse_args()
    main(max_per_class=args.max, seed=args.seed, dataset_dir=args.dataset)
