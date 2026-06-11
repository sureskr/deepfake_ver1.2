#!/usr/bin/env python3
"""
Build or update a dataset manifest CSV from folder-based splits.

Expected layout (label subfolders):
  <root>/calibration/{real,fake}   -> split=calibration  (training)
  <root>/validation/{real,fake}    -> split=validation
  <root>/test/{real,fake}          -> split=test

Sacred RitW benchmark (outside dataset root):
  <repo>/data/test/{real,fake}     -> split=locked_test

Usage:
  py -3.11 scripts/build_manifest.py
  py -3.11 scripts/build_manifest.py --dataset-root data/v1_2_4 --out data/manifests/v1_2_4.csv
  py -3.11 scripts/build_manifest.py --verify   # fail if sha256 mismatch vs existing manifest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

# Allow running as ``python scripts/build_manifest.py`` from ml_engine/
_ML_ENGINE = Path(__file__).resolve().parent.parent
if str(_ML_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ML_ENGINE))

from training_config import repo_root  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
FIELDNAMES = [
    "file_id",
    "split",
    "label",
    "filepath",
    "source",
    "speaker_id",
    "duration_sec",
    "sample_rate",
    "sha256",
    "notes",
]

SPLIT_FOLDERS = (
    ("calibration", "calibration"),
    ("validation", "validation"),
    ("test", "test"),
)
LABEL_DIRS = ("real", "fake")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def infer_source(path: Path, dataset_root: Path) -> str:
    rel = path.relative_to(dataset_root)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] in {"calibration", "validation", "test"}:
        return parts[1] if parts[1] in {"real", "fake"} else "unknown"
    return "unknown"


def infer_speaker_id(path: Path) -> str:
    """Best-effort speaker id from filename stem (override via manifest edits)."""
    stem = path.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem[:32]


def probe_audio(path: Path) -> tuple[float | str, int | str]:
    try:
        import librosa

        dur = float(librosa.get_duration(path=str(path)))
        # librosa does not always return sr without loading; use None -> empty in CSV
        return round(dur, 3), ""
    except Exception:
        return "", ""


def collect_rows(
    dataset_root: Path,
    *,
    source_tag: str = "",
    include_locked_test: bool = True,
    locked_test_root: Path | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seq = 0

    def add_file(path: Path, split: str, label_name: str, notes: str = "") -> None:
        nonlocal seq
        seq += 1
        label = "real" if label_name == "real" else "fake"
        dur, sr = probe_audio(path)
        rows.append(
            {
                "file_id": f"{split[:3]}_{seq:06d}",
                "split": split,
                "label": label,
                "filepath": str(path.resolve()).replace("\\", "/"),
                "source": source_tag or infer_source(path, dataset_root),
                "speaker_id": infer_speaker_id(path),
                "duration_sec": str(dur) if dur != "" else "",
                "sample_rate": str(sr) if sr != "" else "",
                "sha256": sha256_file(path),
                "notes": notes,
            }
        )

    for split_name, folder_name in SPLIT_FOLDERS:
        for label_name in LABEL_DIRS:
            dir_path = dataset_root / folder_name / label_name
            if not dir_path.is_dir():
                continue
            for path in sorted(dir_path.rglob("*")):
                if path.is_file() and path.suffix.lower() in AUDIO_EXT:
                    add_file(path, split_name, label_name)

    if include_locked_test:
        lt_root = locked_test_root or (repo_root() / "data" / "test")
        for label_name in LABEL_DIRS:
            dir_path = lt_root / label_name
            if not dir_path.is_dir():
                continue
            for path in sorted(dir_path.rglob("*")):
                if path.is_file() and path.suffix.lower() in AUDIO_EXT:
                    add_file(
                        path,
                        "locked_test",
                        label_name,
                        notes="Sacred RitW benchmark — do not train or tune threshold on this split.",
                    )

    return rows


def write_manifest(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def verify_manifest(manifest_path: Path) -> int:
    """Return number of mismatched files (0 = OK)."""
    mismatches = 0
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = Path(row["filepath"])
            if not path.is_file():
                print(f"MISSING: {path}")
                mismatches += 1
                continue
            current = sha256_file(path)
            if current != row.get("sha256", ""):
                print(f"HASH MISMATCH: {path}")
                mismatches += 1
    return mismatches


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Build dataset manifest CSV.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "data" / "v1_2_4",
        help="Root with calibration/, validation/, test/ subfolders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "data" / "manifests" / "v1_2_4.csv",
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--source-tag",
        type=str,
        default="v1_2_4",
        help="Value written to the source column for all rows.",
    )
    parser.add_argument(
        "--no-locked-test",
        action="store_true",
        help="Do not scan data/test/ (locked RitW benchmark).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify sha256 hashes in existing manifest instead of rebuilding.",
    )
    args = parser.parse_args()

    if args.verify:
        if not args.out.is_file():
            raise SystemExit(f"Manifest not found: {args.out}")
        n = verify_manifest(args.out)
        if n:
            raise SystemExit(f"Verification failed: {n} issue(s).")
        print(f"OK: {args.out}")
        return

    rows = collect_rows(
        args.dataset_root.resolve(),
        source_tag=args.source_tag,
        include_locked_test=not args.no_locked_test,
    )
    if not rows:
        raise SystemExit(f"No audio files found under {args.dataset_root}")

    write_manifest(rows, args.out)
    splits: dict[str, int] = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print(f"Wrote {len(rows)} rows -> {args.out}")
    for split, count in sorted(splits.items()):
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
