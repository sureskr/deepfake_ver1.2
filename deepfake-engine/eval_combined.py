"""
Evaluate CombinedDetector on ASVspoof, release_in_the_wild, or CVoiceFake datasets.
Compares physics-only baseline vs combined (physics + VeriFauX ML).

ASVspoof dev baselines:
  Physics alone : 75.5% accuracy, 17.5% EER
  Target        : 80%+ accuracy

release_in_the_wild baselines:
  Physics alone : 67.5% accuracy, 32.0% EER
  ML alone      : 75.1% accuracy, 24.5% EER
  Target        : beat both, FPR < 15%

CVoiceFake English baselines:
  Physics alone : 52.5% accuracy, 47.5% EER
  Target        : beat physics alone
"""
import sys, os, csv, random, argparse, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from combined_engine import CombinedDetector
from engine import _compute_eer

_WILD_DATASET = "/Volumes/Hiru kutty/Backup_data_from_GPU_gaming_computer/_dataset_for_deepfake_training/release_in_the_wild"
OUT_CSV = "results_combined.csv"


def _load_asvspoof(dataset_dir: str, protocol_path: str):
    """Parse ASVspoof protocol: speaker file_id - - bonafide|spoof"""
    real_entries, fake_entries = [], []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id, label = parts[1], parts[4]
            path = os.path.join(dataset_dir, file_id + ".flac")
            if label == "bonafide":
                real_entries.append((path, "bona-fide"))
            elif label == "spoof":
                fake_entries.append((path, "spoof"))
    return real_entries, fake_entries


def _load_wild(dataset_dir: str):
    """Parse release_in_the_wild meta.csv"""
    real_entries, fake_entries = [], []
    meta_path = os.path.join(dataset_dir, "meta.csv")
    if not os.path.exists(meta_path):
        sys.exit(f"meta.csv not found: {meta_path}")
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            path = os.path.join(dataset_dir, row["file"])
            if row["label"] == "bona-fide":
                real_entries.append((path, "bona-fide"))
            elif row["label"] == "spoof":
                fake_entries.append((path, "spoof"))
    return real_entries, fake_entries


_CVOICEFAKE_FAKE_DIRS = [
    "griffin_lim_generated",
    "vctk_multi_band_melgan.v2_generated",
    "vctk_parallel_wavegan.v1_generated",
    "vctk_style_melgan.v1_generated",
    "world_generated",
]


def _load_cvoicefake(dataset_dir: str):
    """
    CVoiceFake folder layout:
      Bonafide/           -> real MP3s
      <vocoder>_generated/ -> fake MP3s (one dir per vocoder)
    Returns (real_entries, fake_entries) as (path, label) tuples.
    """
    real_dir = os.path.join(dataset_dir, "Bonafide")
    real_entries = [(p, "bona-fide") for p in glob.glob(os.path.join(real_dir, "*.mp3"))]

    fake_entries = []
    for vdir in _CVOICEFAKE_FAKE_DIRS:
        vpath = os.path.join(dataset_dir, vdir)
        if os.path.isdir(vpath):
            fake_entries.extend(
                [(p, "spoof") for p in glob.glob(os.path.join(vpath, "*.mp3"))]
            )

    return real_entries, fake_entries


def main(max_per_class: int = 100, seed: int = 42, ml_threshold: float = 0.5,
         dataset_dir: str = "", protocol_path: str = ""):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    is_cvoicefake = False
    if dataset_dir and protocol_path:
        real_entries, fake_entries = _load_asvspoof(dataset_dir, protocol_path)
        dataset_label = dataset_dir
    elif dataset_dir and os.path.isdir(os.path.join(dataset_dir, "Bonafide")):
        real_entries, fake_entries = _load_cvoicefake(dataset_dir)
        dataset_label = dataset_dir
        is_cvoicefake = True
    else:
        if not dataset_dir:
            dataset_dir = _WILD_DATASET
        real_entries, fake_entries = _load_wild(dataset_dir)
        dataset_label = dataset_dir

    rng = random.Random(seed)
    rng.shuffle(real_entries)
    rng.shuffle(fake_entries)

    selected = real_entries[:max_per_class] + fake_entries[:max_per_class]
    rng.shuffle(selected)

    n_real = min(max_per_class, len(real_entries))
    n_fake = min(max_per_class, len(fake_entries))
    print(f"Dataset: {dataset_label}")
    if is_cvoicefake:
        print(f"Structure: Bonafide/ + {len(_CVOICEFAKE_FAKE_DIRS)} vocoder dirs")
        print(f"Total available: {len(real_entries)} bona-fide, {len(fake_entries)} spoof")
    else:
        print(f"meta.csv: {len(real_entries)} bona-fide, {len(fake_entries)} spoof")
    print(f"Evaluating: {n_real} real + {n_fake} fake = {len(selected)} files")
    print(f"ML threshold: {ml_threshold}\n")

    del real_entries, fake_entries
    import gc; gc.collect()

    # librosa needed for MP3 decoding (CVoiceFake)
    if is_cvoicefake:
        try:
            import librosa as _librosa
        except ImportError:
            sys.exit("librosa required for MP3: pip install librosa")

    detector = CombinedDetector(ml_threshold=ml_threshold)
    print("Pre-warming ML model …", flush=True)
    detector._ensure_ml_loaded()
    print("ML model ready.\n", flush=True)

    rows = []
    tp = tn = fp = fn = 0
    scores_real = []
    scores_fake = []

    source_counts = {}

    for i, (path, true_label) in enumerate(selected, 1):
        if not os.path.exists(path):
            print(f"  SKIP {os.path.basename(path)} (not found)")
            continue

        try:
            if is_cvoicefake:
                audio, sr = _librosa.load(path, sr=None, mono=True)
                audio = audio.astype(np.float32)
                result = detector.predict_from_array(audio, sr, audio_path=path)
            else:
                result = detector.predict(path)
        except Exception as e:
            print(f"  ERROR {os.path.basename(path)}: {e}")
            continue

        predicted_fake = result.verdict == "FAKE"
        actually_fake  = true_label == "spoof"

        if   predicted_fake and     actually_fake: tp += 1
        elif not predicted_fake and not actually_fake: tn += 1
        elif predicted_fake and not actually_fake: fp += 1
        else:                                      fn += 1

        # Use ml_score for EER if available, else physics checks_failed (negated)
        eer_score = result.ml_score if result.ml_score is not None else (result.physics_checks_failed / 7.0)
        (scores_fake if actually_fake else scores_real).append(eer_score)

        source_counts[result.source] = source_counts.get(result.source, 0) + 1

        rows.append({
            "file":                 os.path.basename(path),
            "true":                 true_label,
            "predicted":            result.verdict.lower(),
            "correct":              int(predicted_fake == actually_fake),
            "verdict":              result.verdict,
            "confidence":           result.confidence,
            "source":               result.source,
            "physics_verdict":      result.physics_verdict,
            "physics_checks_failed": result.physics_checks_failed,
            "ml_score":             "" if result.ml_score is None else f"{result.ml_score:.4f}",
            "ml_verdict":           result.ml_verdict or "",
        })

        if i % 25 == 0:
            print(f"  [{i}/{len(selected)}] tp={tp} tn={tn} fp={fp} fn={fn} …")

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total    if total     else 0.0
    fpr      = fp / (fp + tn)       if (fp + tn) else 0.0
    fnr      = fn / (fn + tp)       if (fn + tp) else 0.0

    # EER on ml_score / normalised physics score
    from sklearn.metrics import roc_auc_score
    all_scores = scores_real + scores_fake
    all_labels = [0] * len(scores_real) + [1] * len(scores_fake)
    try:
        auroc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auroc = float("nan")

    eer = _compute_eer(scores_real, scores_fake)

    # Baseline labels depend on dataset
    if is_cvoicefake:
        ds_label    = "CVoiceFake English"
        phys_acc    = "52.50"; phys_eer = "47.50"; phys_fpr = "45.00%"; phys_fnr = "50.00%"
        acc_target  = 0.525;   eer_target = 0.475
        acc_tgt_str = "52.50"; eer_tgt_str = "47.50"
        out_csv     = "results_combined_cvoicefake.csv"
    elif dataset_dir and protocol_path:
        ds_label    = "ASVspoof2019 LA dev"
        phys_acc    = "75.50"; phys_eer = "17.50"; phys_fpr = "?"; phys_fnr = "?"
        acc_target  = 0.80;    eer_target = 0.175
        acc_tgt_str = "80.00"; eer_tgt_str = "17.50"
        out_csv     = OUT_CSV
    else:
        ds_label    = "release_in_the_wild"
        phys_acc    = "67.50"; phys_eer = "32.00"; phys_fpr = "11.00%"; phys_fnr = "54.00%"
        acc_target  = 0.7510;  eer_target = 0.245
        acc_tgt_str = "75.10"; eer_tgt_str = "24.50"
        out_csv     = OUT_CSV

    print(f"\n{'='*65}")
    print(f"  {ds_label} — Combined (physics + VeriFauX ML)")
    print(f"  Files evaluated : {total}")
    print(f"  ML threshold    : {ml_threshold}")
    print(f"  {'Metric':<12}  {'Combined':>10}  {'Physics':>10}")
    print(f"  {'-'*38}")
    print(f"  {'Accuracy':<12}  {accuracy*100:>9.2f}%  {phys_acc:>9}%")
    print(f"  {'EER':<12}  {eer*100:>9.2f}%  {phys_eer:>9}%")
    print(f"  {'FPR':<12}  {fpr*100:>9.2f}%  {phys_fpr:>9}")
    print(f"  {'FNR':<12}  {fnr*100:>9.2f}%  {phys_fnr:>9}")
    print(f"  AUROC           : {auroc:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*65}")
    print(f"\n  Decision source breakdown (AND logic):")
    print(f"    and_fake   (both agree FAKE)         : {source_counts.get('and_fake', 0)}")
    print(f"    phys_only  (phys FAKE, ML REAL->REAL) : {source_counts.get('phys_only', 0)}")
    print(f"    ml_only    (ML FAKE, phys REAL->REAL) : {source_counts.get('ml_only', 0)}")
    print(f"    both_real  (both agree REAL)         : {source_counts.get('both_real', 0)}")
    print(f"\n  Accuracy target: > {acc_tgt_str}% — {'PASS' if accuracy > acc_target else 'FAIL'}")
    print(f"  EER target: < {eer_tgt_str}% — {'PASS' if eer < eer_target else 'FAIL'}")

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(out_csv, "a", newline="") as f:
            f.write(f"\nSUMMARY,accuracy={accuracy:.4f},EER={eer:.4f},"
                    f"FPR={fpr:.4f},FNR={fnr:.4f},"
                    f"TP={tp},TN={tn},FP={fp},FN={fn}\n")
        print(f"\nResults saved -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max",       type=int,   default=100,
                        help="Max files per class (default 100 -> 200 total)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="ML decision threshold (default 0.5)")
    parser.add_argument("--dataset",   type=str,   default="",
                        help="Path to audio directory (ASVspoof flac dir)")
    parser.add_argument("--protocol",  type=str,   default="",
                        help="Path to ASVspoof protocol .txt file")
    args = parser.parse_args()
    main(max_per_class=args.max, seed=args.seed, ml_threshold=args.threshold,
         dataset_dir=args.dataset, protocol_path=args.protocol)
