"""
Evaluate engine.py on CVoiceFake English subset.

Structure:
  en/Bonafide/                          -> real speech (Common Voice MP3)
  en/<vocoder>_generated/               -> fake speech (one folder per vocoder)
  Fake vocoders: griffin_lim, world, vctk_parallel_wavegan.v1,
                 vctk_multi_band_melgan.v2, vctk_style_melgan.v1
"""
import sys, os, csv, glob, random, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from engine import detect_from_array, _compute_eer

try:
    import librosa
except ImportError:
    sys.exit("librosa required for MP3: pip install librosa")

DATASET = "/Users/sureskr/claude_projects/data/CVoiceFake_small/en"
OUT_CSV = "results_cvoicefake.csv"

FAKE_DIRS = [
    "griffin_lim_generated",
    "vctk_multi_band_melgan.v2_generated",
    "vctk_parallel_wavegan.v1_generated",
    "vctk_style_melgan.v1_generated",
    "world_generated",
]


def collect(folder: str, label: str, n: int, rng: random.Random):
    files = glob.glob(os.path.join(folder, "*.mp3"))
    rng.shuffle(files)
    return [(f, label) for f in files[:n]]


def load_mp3(path: str):
    audio, sr = librosa.load(path, sr=None, mono=True)
    return audio.astype(np.float32), sr


def main(max_per_class: int = 100, seed: int = 42):
    rng = random.Random(seed)

    real_dir = os.path.join(DATASET, "Bonafide")
    real_files = collect(real_dir, "real", max_per_class, rng)

    # Pool all fake files across vocoders, then sample
    all_fake = []
    vocoder_counts = {}
    for vdir in FAKE_DIRS:
        path = os.path.join(DATASET, vdir)
        files = glob.glob(os.path.join(path, "*.mp3"))
        all_fake.extend([(f, "fake", vdir) for f in files])
        vocoder_counts[vdir] = len(files)

    rng.shuffle(all_fake)
    fake_sample = all_fake[:max_per_class]
    fake_files = [(f, "fake") for f, _, _ in fake_sample]
    fake_vocoders = {f: v for f, _, v in fake_sample}

    print(f"Dataset: {DATASET}")
    print(f"Structure:")
    print(f"  Bonafide: {len(glob.glob(os.path.join(real_dir,'*.mp3')))} files")
    for vdir in FAKE_DIRS:
        print(f"  {vdir}: {vocoder_counts[vdir]} files")
    print(f"\nEvaluating: {len(real_files)} real + {len(fake_files)} fake = "
          f"{len(real_files)+len(fake_files)} files\n")

    entries = real_files + fake_files
    rng.shuffle(entries)

    rows = []
    tp = tn = fp = fn = 0
    scores_real, scores_fake = [], []

    for i, (path, true_label) in enumerate(entries, 1):
        try:
            audio, sr = load_mp3(path)
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}")
            continue

        try:
            result = detect_from_array(audio, sr, audio_path=path)
        except Exception as e:
            print(f"  ERROR {os.path.basename(path)}: {e}")
            continue

        predicted_fake = result.verdict == "FAKE"
        actually_fake  = true_label == "fake"

        if   predicted_fake and     actually_fake: tp += 1
        elif not predicted_fake and not actually_fake: tn += 1
        elif predicted_fake and not actually_fake: fp += 1
        else:                                      fn += 1

        (scores_fake if actually_fake else scores_real).append(result.checks_failed)

        rows.append({
            "file":          os.path.basename(path),
            "vocoder":       fake_vocoders.get(path, "—"),
            "true":          true_label,
            "predicted":     result.verdict.lower(),
            "correct":       int(predicted_fake == actually_fake),
            "checks_failed": result.checks_failed,
            "jitter_pct":    result.jitter_shimmer.detail.get("jitter_pct", ""),
            "shimmer_db":    result.jitter_shimmer.detail.get("shimmer_db", ""),
            "mean_hnr_db":   result.subglottal.detail.get("mean_hnr_db", ""),
            "cv_jitter_pct": result.temporal_consistency.detail.get("cv_jitter_pct", ""),
            "cv_hnr_pct":    result.temporal_consistency.detail.get("cv_hnr_pct", ""),
            "flux_cv_pct":   result.spectral_flux.detail.get("flux_cv_pct", ""),
            "sba_score":     result.subband_asymmetry.detail.get("sba_score", ""),
        })

        if i % 50 == 0:
            print(f"  [{i}/{len(entries)}] processed …")

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total    if total     else 0.0
    fpr      = fp / (fp + tn)       if (fp + tn) else 0.0
    fnr      = fn / (fn + tp)       if (fn + tp) else 0.0
    eer      = _compute_eer(scores_real, scores_fake)

    print(f"\n{'='*60}")
    print(f"  CVoiceFake English  —  {total} files")
    print(f"  Accuracy  : {accuracy*100:.2f} %")
    print(f"  EER       : {eer*100:.2f} %")
    print(f"  FPR       : {fpr*100:.2f} %  (real flagged as fake)")
    print(f"  FNR       : {fnr*100:.2f} %  (fake flagged as real)")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*60}")

    # Per-vocoder breakdown
    if rows:
        from collections import defaultdict
        voc_stats = defaultdict(lambda: {"tp":0,"tn":0,"fp":0,"fn":0})
        for row in rows:
            v = row["vocoder"]
            pred_f = row["predicted"] == "fake"
            act_f  = row["true"] == "fake"
            if   pred_f and     act_f: voc_stats[v]["tp"] += 1
            elif not pred_f and not act_f: voc_stats[v]["tn"] += 1
            elif pred_f and not act_f: voc_stats[v]["fp"] += 1
            else: voc_stats[v]["fn"] += 1

        print(f"\n  Per-vocoder fake detection rate:")
        for v, s in sorted(voc_stats.items()):
            if v == "—":
                continue
            n_fake = s["tp"] + s["fn"]
            if n_fake:
                dr = 100.0 * s["tp"] / n_fake
                print(f"    {v}: {s['tp']}/{n_fake} caught ({dr:.0f} %)")

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
    parser.add_argument("--max",  type=int, default=100,
                        help="Max files per class (default 100 -> 200 total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(max_per_class=args.max, seed=args.seed)
