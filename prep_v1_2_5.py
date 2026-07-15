"""Assemble the v1.2.5 training dataset from ASVspoof5 + MLAAD.

Produces the folder layout that ``ml_engine/train_v1_2_4_proper.py`` and
``config/training_v1_2_5.template.json`` expect::

    data/v1_2_5/
      calibration/{real,fake}/   <- training
      validation/{real,fake}/
      test/{real,fake}/          <- held-out, GENERATOR-DISJOINT from train

Sources
-------
- **ASVspoof5** (``jungjee/asvspoof5``): bonafide -> real, spoof -> fake. Real is
  split by *speaker*; spoof is split by *attack id* so no speaker or attack leaks
  across splits.
- **MLAAD** (``mueller91/MLAAD``): all fake (modern TTS). Split by *TTS engine* so
  a whole engine is held out for test — this is what proves the v1.2.5 model
  generalizes to unseen TTS instead of memorizing it (the v1.2.4 blind spot).

This script does NOT download the datasets — point it at local, already-extracted
roots. Fetch them first, e.g.::

    huggingface-cli download jungjee/asvspoof5 --repo-type dataset \\
      --local-dir data/raw/asvspoof5 --include "flac_T_*.tar" "*.tsv" "*.txt"
    # extract the flac_T_*.tar shards into one dir, e.g. data/raw/asvspoof5/flac_T
    huggingface-cli download mueller91/MLAAD --repo-type dataset \\
      --local-dir data/raw/mlaad

Then::

    python prep_v1_2_5.py \\
      --asvspoof-protocol data/raw/asvspoof5/ASVspoof5.train.metadata.txt \\
      --asvspoof-flac-dir data/raw/asvspoof5/flac_T \\
      --mlaad-root data/raw/mlaad \\
      --out-root data/v1_2_5 --link

After it finishes, build the manifest and train::

    python ml_engine/scripts/build_manifest.py --dataset-root data/v1_2_5 \\
      --out data/manifests/v1_2_5.csv --source-tag v1_2_5
    python ml_engine/train_v1_2_4_proper.py --config config/training_v1_2_5.json --train
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SPLITS = ("calibration", "validation", "test")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(text: str, maxlen: int = 48) -> str:
    s = _SAFE.sub("-", str(text)).strip("-._")
    return (s or "x")[:maxlen]


@dataclass
class Item:
    """One audio file to place, with the provenance used for disjoint splitting."""

    path: Path
    label: str          # "real" | "fake"
    source: str         # "asvspoof" | "mlaad" | "extra_real"
    group: str          # split key: speaker (real) / attack or engine (fake)
    stem: str           # original filename stem (for a readable output name)

    def out_name(self) -> str:
        return f"{self.source}_{self.label}_{sanitize(self.group)}_{sanitize(self.stem)}{self.path.suffix.lower()}"


# ── ASVspoof5 protocol parsing ───────────────────────────────────────────────

def _read_protocol_rows(protocol: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(protocol, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if parts:
                rows.append(parts)
    if not rows:
        raise SystemExit(f"No rows parsed from protocol {protocol}")
    return rows


def _detect_columns(
    rows: list[str], flac_dir: Path, overrides: dict[str, int | None]
) -> dict[str, int]:
    """Auto-detect filename/label/speaker/attack columns, honoring any overrides.

    ASVspoof5 metadata layout varies by release; rather than hardcode indices we
    detect by content and let the caller override any column with a flag.
    """
    sample = rows[: min(200, len(rows))]
    ncol = min(len(r) for r in sample)
    cols = {i: [r[i] for r in sample] for i in range(ncol)}

    def pick(name: str, scorer) -> int:
        if overrides.get(name) is not None:
            return int(overrides[name])
        best_i, best_score = None, -1.0
        for i, vals in cols.items():
            score = scorer(i, vals)
            if score > best_score:
                best_i, best_score = i, score
        if best_i is None or best_score <= 0:
            raise SystemExit(
                f"Could not auto-detect the '{name}' column in the protocol; "
                f"pass --asvspoof-{name}-col explicitly. First row: {rows[0]}"
            )
        return best_i

    label_tokens = {"bonafide", "spoof"}
    label_col = pick(
        "label",
        lambda i, vals: sum(v.lower() in label_tokens for v in vals) / len(vals),
    )

    def fname_score(i: int, vals: list[str]) -> float:
        hits = 0
        for v in vals[:40]:
            stem = Path(v).stem
            if any((flac_dir / f"{stem}{ext}").exists() for ext in (".flac", ".wav")):
                hits += 1
        return hits / min(40, len(vals))

    fname_col = pick("fname", fname_score)

    attack_re = re.compile(r"^A\d{1,2}$", re.IGNORECASE)
    attack_col = pick(
        "attack",
        lambda i, vals: 0.0
        if i in (label_col, fname_col)
        else sum(bool(attack_re.match(v)) or v == "-" for v in vals) / len(vals),
    )

    # Speaker: high-cardinality categorical that isn't fname (unique) and isn't
    # label/attack (low cardinality). Prefer the column with the most repeats.
    def speaker_score(i: int, vals: list[str]) -> float:
        if i in (label_col, fname_col, attack_col):
            return -1.0
        uniq = len(set(vals))
        if uniq <= 1 or uniq == len(vals):
            return 0.1  # constant or unique -> unlikely speaker, but allow fallback
        return uniq / len(vals) * (1.0 - uniq / len(vals)) + 0.2

    speaker_col = pick("speaker", speaker_score)
    return {"fname": fname_col, "label": label_col, "attack": attack_col, "speaker": speaker_col}


def load_asvspoof(
    protocol: Path, flac_dir: Path, overrides: dict[str, int | None]
) -> list[Item]:
    rows = _read_protocol_rows(protocol)
    idx = _detect_columns(rows, flac_dir, overrides)
    print(
        f"  ASVspoof columns -> fname={idx['fname']} label={idx['label']} "
        f"speaker={idx['speaker']} attack={idx['attack']}"
    )
    items: list[Item] = []
    missing = 0
    for r in rows:
        if len(r) <= max(idx.values()):
            continue
        stem = Path(r[idx["fname"]]).stem
        label_raw = r[idx["label"]].lower()
        if label_raw not in ("bonafide", "spoof"):
            continue
        fpath = None
        for ext in (".flac", ".wav"):
            cand = flac_dir / f"{stem}{ext}"
            if cand.exists():
                fpath = cand
                break
        if fpath is None:
            missing += 1
            continue
        if label_raw == "bonafide":
            items.append(Item(fpath, "real", "asvspoof", r[idx["speaker"]], stem))
        else:
            attack = r[idx["attack"]]
            attack = attack if re.match(r"^A\d", attack, re.I) else "Aunk"
            items.append(Item(fpath, "fake", "asvspoof", attack, stem))
    if missing:
        print(f"  [warn] {missing} protocol rows had no audio file in {flac_dir}")
    print(f"  ASVspoof loaded: {sum(i.label=='real' for i in items)} real / "
          f"{sum(i.label=='fake' for i in items)} fake")
    return items


# ── MLAAD parsing (engine = TTS architecture) ────────────────────────────────

def _mlaad_engine_for_dir(d: Path, root: Path) -> dict[str, str]:
    """Map wav filename -> engine using a sibling meta.csv if present."""
    meta = d / "meta.csv"
    out: dict[str, str] = {}
    if meta.is_file():
        try:
            with open(meta, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter="|") if "|" in f.readline() else None
            with open(meta, encoding="utf-8", errors="replace") as f:
                first = f.readline()
                delim = "|" if first.count("|") >= first.count(",") else ","
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delim)
                key = None
                for cand in ("architecture", "model_name", "tts_architecture"):
                    if reader.fieldnames and cand in reader.fieldnames:
                        key = cand
                        break
                if key:
                    for row in reader:
                        p = row.get("path") or row.get("file") or ""
                        name = Path(p).name
                        if name:
                            out[name] = row[key]
        except Exception as e:  # noqa: BLE001 - meta is best-effort
            print(f"  [warn] failed to read {meta}: {e}")
    return out


def load_mlaad(root: Path, engine_from: str) -> list[Item]:
    items: list[Item] = []
    meta_cache: dict[Path, dict[str, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXT:
            continue
        engine = None
        if engine_from == "meta":
            d = path.parent
            if d not in meta_cache:
                meta_cache[d] = _mlaad_engine_for_dir(d, root)
            engine = meta_cache[d].get(path.name)
        if not engine:
            # MLAAD v5 is flat: <lang>/<engine>/*.wav, so the engine is the
            # immediate parent folder — which is also what --holdout-engines names.
            rel = path.relative_to(root).parts
            engine = rel[-2] if len(rel) >= 2 else "mlaad"
        items.append(Item(path, "fake", "mlaad", engine, path.stem))
    engines = sorted({i.group for i in items})
    print(f"  MLAAD loaded: {len(items)} fake across {len(engines)} engines")
    return items


def load_extra_real(root: Path) -> list[Item]:
    """Optional extra real audio (LibriTTS / Common Voice). Speaker = parent dir."""
    items: list[Item] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXT:
            items.append(Item(path, "real", "extra_real", path.parent.name, path.stem))
    print(f"  extra real loaded: {len(items)} from {root}")
    return items


# ── Group-disjoint splitting ─────────────────────────────────────────────────

@dataclass
class Plan:
    per_split: dict[str, list[Item]] = field(
        default_factory=lambda: {s: [] for s in SPLITS}
    )
    groups_by_split: dict[str, set[str]] = field(
        default_factory=lambda: {s: set() for s in SPLITS}
    )


def assign_groups(
    items: list[Item],
    rng: random.Random,
    test_frac: float,
    val_frac: float,
    holdout: set[str],
) -> dict[str, str]:
    """Assign each distinct group (speaker/attack/engine) to exactly one split."""
    groups = sorted({i.group for i in items})
    rng.shuffle(groups)
    forced_test = [g for g in groups if g in holdout]
    rest = [g for g in groups if g not in holdout]
    n = len(groups)
    n_test = max(len(forced_test), round(test_frac * n)) if n else 0
    n_val = round(val_frac * n)
    assignment: dict[str, str] = {g: "test" for g in forced_test}
    for g in rest:
        if sum(v == "test" for v in assignment.values()) < n_test:
            assignment[g] = "test"
        elif sum(v == "validation" for v in assignment.values()) < n_val:
            assignment[g] = "validation"
        else:
            assignment[g] = "calibration"
    return assignment


def build_plan(items: list[Item], args, rng: random.Random) -> Plan:
    plan = Plan()
    holdout_engines = {sanitize(e) for e in args.holdout_engines}
    # Split each (source, label) stream by its own group axis, disjointly.
    streams: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for it in items:
        streams[(it.source, it.label)].append(it)

    for (source, label), stream in streams.items():
        holdout = set()
        if source == "mlaad":
            holdout = {g for g in {i.group for i in stream} if sanitize(g) in holdout_engines}
        assignment = assign_groups(stream, rng, args.test_frac, args.val_frac, holdout)
        for it in stream:
            split = assignment[it.group]
            plan.per_split[split].append(it)
            plan.groups_by_split[split].add(f"{source}:{it.group}")
    return plan


def cap_and_balance(plan: Plan, args, rng: random.Random) -> Plan:
    caps = {"calibration": args.cap_cal, "validation": args.cap_val, "test": args.cap_test}
    for split in SPLITS:
        by_label: dict[str, list[Item]] = {"real": [], "fake": []}
        for it in plan.per_split[split]:
            by_label[it.label].append(it)
        cap = caps[split]
        if args.balance:
            n = min(len(by_label["real"]), len(by_label["fake"]))
            if cap:
                n = min(n, cap)
            targets = {"real": n, "fake": n}
        else:
            targets = {lab: (min(len(v), cap) if cap else len(v)) for lab, v in by_label.items()}
        kept: list[Item] = []
        for lab, lst in by_label.items():
            rng.shuffle(lst)
            kept.extend(lst[: targets[lab]])
        plan.per_split[split] = kept
    return plan


# ── Materialize ──────────────────────────────────────────────────────────────

def place_files(plan: Plan, out_root: Path, link: bool, dry_run: bool) -> dict:
    report: dict = {"splits": {}}
    for split in SPLITS:
        counts = {"real": 0, "fake": 0}
        used: set[str] = set()
        for it in plan.per_split[split]:
            dest_dir = out_root / split / it.label
            name = it.out_name()
            base, suf = name[: -len(it.path.suffix)], it.path.suffix.lower()
            i = 1
            while name in used:
                name = f"{base}__{i}{suf}"
                i += 1
            used.add(name)
            counts[it.label] += 1
            if dry_run:
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            if link:
                os.symlink(it.path.resolve(), dest)
            else:
                shutil.copy2(it.path, dest)
        report["splits"][split] = {
            "real": counts["real"],
            "fake": counts["fake"],
            "groups": sorted(plan.groups_by_split[split]),
        }
    return report


def assert_disjoint(plan: Plan) -> None:
    seen: dict[str, str] = {}
    for split in SPLITS:
        for g in plan.groups_by_split[split]:
            if g in seen and seen[g] != split:
                raise SystemExit(
                    f"LEAK: group {g!r} appears in both {seen[g]} and {split}. "
                    f"This must never happen — aborting."
                )
            seen[g] = split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=Path("data/v1_2_5"))
    p.add_argument("--asvspoof-protocol", type=Path, help="ASVspoof5 metadata/protocol txt/tsv.")
    p.add_argument("--asvspoof-flac-dir", type=Path, help="Directory of ASVspoof5 flac files.")
    p.add_argument("--mlaad-root", type=Path, help="Root of extracted MLAAD dataset.")
    p.add_argument("--extra-real-dir", type=Path, help="Optional extra real audio (LibriTTS/CommonVoice).")
    # ASVspoof column overrides (0-based); omit to auto-detect.
    p.add_argument("--asvspoof-fname-col", type=int)
    p.add_argument("--asvspoof-label-col", type=int)
    p.add_argument("--asvspoof-speaker-col", type=int)
    p.add_argument("--asvspoof-attack-col", type=int)
    p.add_argument("--mlaad-engine-from", choices=["meta", "path"], default="path",
                   help="Derive MLAAD engine from the parent folder name (default) or meta.csv.")
    p.add_argument("--holdout-engines", default="",
                   help="Comma-separated MLAAD engines forced into the TEST split "
                        "(e.g. 'xtts_v2,bark,vits'). Held-out engines never appear in train.")
    p.add_argument("--asvspoof-max-fake", type=int, default=0,
                   help="Randomly cap ASVspoof spoof files loaded (0=all). Use to stop the "
                        "large ASVspoof spoof pool from drowning out MLAAD's modern TTS.")
    p.add_argument("--test-frac", type=float, default=0.15, help="Fraction of groups -> test.")
    p.add_argument("--val-frac", type=float, default=0.15, help="Fraction of groups -> validation.")
    p.add_argument("--cap-cal", type=int, default=0, help="Max files per class in calibration (0=all).")
    p.add_argument("--cap-val", type=int, default=2000, help="Max files per class in validation (0=all).")
    p.add_argument("--cap-test", type=int, default=2000, help="Max files per class in test (0=all).")
    p.add_argument("--balance", action="store_true", default=True, help="Trim to equal real/fake per split.")
    p.add_argument("--no-balance", dest="balance", action="store_false")
    p.add_argument("--link", action="store_true", help="Symlink instead of copy (saves disk on cloud GPUs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Print the plan; write nothing.")
    args = p.parse_args()
    args.holdout_engines = [e.strip() for e in args.holdout_engines.split(",") if e.strip()]

    rng = random.Random(args.seed)
    items: list[Item] = []

    print("Loading sources...")
    if args.asvspoof_protocol or args.asvspoof_flac_dir:
        if not (args.asvspoof_protocol and args.asvspoof_flac_dir):
            raise SystemExit("Provide BOTH --asvspoof-protocol and --asvspoof-flac-dir.")
        overrides = {
            "fname": args.asvspoof_fname_col,
            "label": args.asvspoof_label_col,
            "speaker": args.asvspoof_speaker_col,
            "attack": args.asvspoof_attack_col,
        }
        items += load_asvspoof(args.asvspoof_protocol, args.asvspoof_flac_dir, overrides)
    if args.mlaad_root:
        items += load_mlaad(args.mlaad_root, args.mlaad_engine_from)
    if args.extra_real_dir:
        items += load_extra_real(args.extra_real_dir)

    if not items:
        raise SystemExit("No sources provided. Pass --asvspoof-* and/or --mlaad-root.")

    if args.asvspoof_max_fake:
        asv_fake = [it for it in items if it.source == "asvspoof" and it.label == "fake"]
        rest = [it for it in items if not (it.source == "asvspoof" and it.label == "fake")]
        rng.shuffle(asv_fake)
        kept = asv_fake[: args.asvspoof_max_fake]
        print(f"  capping ASVspoof spoof: {len(asv_fake)} -> {len(kept)}")
        items = rest + kept

    n_real = sum(i.label == "real" for i in items)
    n_fake = sum(i.label == "fake" for i in items)
    print(f"Total pool: {n_real} real / {n_fake} fake")
    if n_real == 0 or n_fake == 0:
        raise SystemExit("Need both real and fake audio. ASVspoof bonafide (or --extra-real-dir) "
                         "supplies real; ASVspoof spoof / MLAAD supply fake.")

    plan = build_plan(items, args, rng)
    assert_disjoint(plan)
    plan = cap_and_balance(plan, args, rng)

    report = place_files(plan, args.out_root, args.link, args.dry_run)
    report["config"] = {
        "seed": args.seed, "test_frac": args.test_frac, "val_frac": args.val_frac,
        "holdout_engines": args.holdout_engines, "balance": args.balance,
        "caps": {"cal": args.cap_cal, "val": args.cap_val, "test": args.cap_test},
        "linked": args.link, "dry_run": args.dry_run,
    }

    print("\n" + "=" * 64)
    print(f"v1.2.5 dataset plan  ({'DRY RUN — nothing written' if args.dry_run else args.out_root})")
    print("=" * 64)
    for split in SPLITS:
        s = report["splits"][split]
        print(f"  {split:12s}  real={s['real']:6d}  fake={s['fake']:6d}  groups={len(s['groups'])}")
    print("=" * 64)

    if not args.dry_run:
        args.out_root.mkdir(parents=True, exist_ok=True)
        with open(args.out_root / "prep_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out_root / 'prep_report.json'}")
        print("Next:")
        print(f"  python ml_engine/scripts/build_manifest.py --dataset-root {args.out_root} "
              f"--out data/manifests/v1_2_5.csv --source-tag v1_2_5")
        print("  python ml_engine/train_v1_2_4_proper.py --config config/training_v1_2_5.json --train")


if __name__ == "__main__":
    main()
