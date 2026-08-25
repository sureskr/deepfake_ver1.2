"""
validate_v1_2_4_perfile.py — per-file validation of the external harness against the
original v1.2.4 run that produced eval reports 01-05.

Branch `origin/eval/public-dataset-test` stores the raw v1.2.4 output for the same five
200-file samples now in test_data/ (results.json, results_asvspoof5.json, results_mlaad.json,
results_in_the_wild.json, results_codecfake.json). Each file holds `predictions` and `targets`
arrays — scores, but no filenames.

Those arrays are in the order `deployment_v1_2_4/inference.py` enumerated the files, which used
``Path.glob("*.*")`` — filesystem order, real/ then fake/. This harness enumerates with
``sorted()``. The arrays therefore cannot be compared position by position; the check below
instead matches scores within each class (real, fake) in sorted order, which is a bijection when
the two runs computed the same set of scores. Agreement to ~1e-6 across all 1,000 files means the
two scoring paths are numerically identical, which is a far stronger statement than "the
aggregate metrics matched".

Consequence worth knowing: `run_eval.py` and the `*_enriched.json` scripts on that branch
rebuilt the file list with ``sorted()`` and zipped it against the glob-ordered arrays, so their
per-file *attributions* (filename, codec, speaker) are misaligned. Class-wise aggregates (EER,
accuracy, per-class score distributions) are unaffected, because they depend only on the set of
scores within each class.

Usage:
  python validate_v1_2_4_perfile.py                      # print the check
  python validate_v1_2_4_perfile.py --write              # also record it in the results JSON
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
EVAL_BRANCH = "origin/eval/public-dataset-test"
ORIGINAL_RESULTS = {
    "deepfake-audio": "results.json",
    "asvspoof5": "results_asvspoof5.json",
    "mlaad": "results_mlaad.json",
    "in_the_wild": "results_in_the_wild.json",
    "codecfake": "results_codecfake.json",
}


def read_from_branch(path: str) -> dict:
    return json.loads(
        subprocess.check_output(["git", "show", f"{EVAL_BRANCH}:{path}"], cwd=BASE, text=True)
    )


def load_my_scores(csv_path: Path) -> dict[str, list[tuple[str, int, float]]]:
    rows: dict[str, list[tuple[str, int, float]]] = collections.defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows[r["dataset"]].append((r["filename"], int(r["label"]), float(r["score_v1_2_4"])))
    return rows


def check_dataset(original: dict, rows: list[tuple[str, int, float]]) -> dict:
    op = np.array(original["predictions"], dtype=float)
    ot = np.array(original["targets"], dtype=int)
    ms = np.array([s for _, _, s in rows], dtype=float)
    ml = np.array([l for _, l, _ in rows], dtype=int)

    result = {
        "n_original": int(len(op)),
        "n_recomputed": int(len(ms)),
        "class_counts_match": bool((ot == 0).sum() == (ml == 0).sum() and (ot == 1).sum() == (ml == 1).sum()),
        "max_abs_delta_positionwise": float(np.abs(op - ms).max()) if len(op) == len(ms) else None,
    }
    worst = 0.0
    n_positions_differing = 0
    for cls in (0, 1):
        a = np.sort(op[ot == cls])
        b = np.sort(ms[ml == cls])
        if len(a) != len(b):
            result["aligned"] = False
            return result
        worst = max(worst, float(np.abs(a - b).max()))
    # how many array positions the two enumeration orders disagree on
    for cls in (0, 1):
        oi = np.flatnonzero(ot == cls)
        mi = np.flatnonzero(ml == cls)
        o_order = oi[np.argsort(op[oi])]
        m_order = mi[np.argsort(ms[mi])]
        n_positions_differing += int(sum(1 for a, b in zip(o_order, m_order) if a != b))
    result.update({
        "max_abs_delta_after_class_alignment": worst,
        "aligned": bool(worst < 1e-4),
        "n_positions_with_different_enumeration_order": n_positions_differing,
        "original_metrics": {k: original[k] for k in ("accuracy", "eer", "eer_threshold") if k in original},
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="paper_data/external_scores.csv")
    ap.add_argument("--results", default="results_external_v1_2_6.json")
    ap.add_argument("--write", action="store_true", help="record the check under 'per_file_validation_vs_original_v1_2_4'")
    args = ap.parse_args()

    mine = load_my_scores(BASE / args.csv)
    out = {
        "what": "Per-file comparison of this harness's v1.2.4 scores against the raw v1.2.4 output "
                f"that produced eval reports 01-05 ({EVAL_BRANCH}).",
        "alignment_note": "The original arrays follow inference.py's Path.glob() enumeration; this "
                          "harness uses sorted(). Scores are therefore matched within each class in "
                          "sorted order (a bijection when both runs produced the same score set).",
        "datasets": {},
    }
    for ds, fname in ORIGINAL_RESULTS.items():
        if ds not in mine:
            continue
        res = check_dataset(read_from_branch(fname), mine[ds])
        res["original_file"] = f"{EVAL_BRANCH}:{fname}"
        out["datasets"][ds] = res
        print(f"{ds:15s} aligned={res['aligned']}  max|delta| after alignment "
              f"{res['max_abs_delta_after_class_alignment']:.2e}  "
              f"(positionwise {res['max_abs_delta_positionwise']:.3f}, "
              f"{res['n_positions_with_different_enumeration_order']}/{res['n_original']} positions "
              f"enumerated in a different order)")
    out["all_datasets_aligned"] = all(d["aligned"] for d in out["datasets"].values())
    out["max_abs_delta_overall"] = max(d["max_abs_delta_after_class_alignment"] for d in out["datasets"].values())
    print(f"\nALL ALIGNED: {out['all_datasets_aligned']}   worst delta {out['max_abs_delta_overall']:.2e}")

    if args.write:
        path = BASE / args.results
        payload = json.loads(path.read_text())
        payload["per_file_validation_vs_original_v1_2_4"] = out
        path.write_text(json.dumps(payload, indent=2))
        print(f"Recorded in {path}")


if __name__ == "__main__":
    main()
