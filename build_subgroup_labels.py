"""
build_subgroup_labels.py — recover per-file subgroup labels for the external test sets.

test_data/ stores files under anonymised names (real_000.wav / fake_012.flac / E_*.flac),
so the engine / codec / attack-ID labels needed for a per-subgroup breakdown have to be
recovered from provenance:

  deepfake-audio : SHA256-match each file against the local HuggingFace cache snapshot of
                   garystafford/deepfake-audio-detection; the original filename prefix is
                   the generator (el/sp/po/hu/hg/lv, yt = real YouTube audio).
  asvspoof5      : utterance IDs are preserved in the filenames; attack ID, codec, speaker
                   and the ground-truth label come from ASVspoof5.eval.track_1.tsv inside
                   the cached ASVspoof5_protocols.tar.
  mlaad          : the TTS engine is already in the filename; the original MLAAD filename
                   is recovered by SHA256-match against the local MLAAD cache so a future
                   session can test for literal overlap with a rebuilt training pool.
                   Real files are ASVspoof5 bonafide (same 100 files as the asvspoof5 set).
  codecfake      : codec + speaker from test_data/codecfake/manifest.json.
  in_the_wild    : speaker from test_data/in_the_wild/manifest.json.

Output: paper_data/subgroup_labels.json  (committed; the eval harness reads it).
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEST_DATA = BASE / "test_data"
HF_HUB = Path.home() / ".cache/huggingface/hub"
OUT = BASE / "paper_data/subgroup_labels.json"

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# garystafford/deepfake-audio-detection filename prefixes -> generator
GARYSTAFFORD_PREFIX = {
    "yt": "YouTube (real)",
    "el": "ElevenLabs",
    "sp": "Speechify",
    "po": "Amazon Polly",
    "hu": "Hume AI",
    "hg": "Hexgrad Kokoro",
    "lv": "Luvvoice",
}

# MLAAD engines: which were held out of v1.2.5 / v1.2.6 training and which were used.
# Source: report 06 (8 held-out engines) and the MLAAD training-engine list.
MLAAD_HELD_OUT = {
    "ElevenLabs-v3",
    "OpenAI_TTS-1_HD",
    "Gemini-3.1-Flash-TTS",
    "kokoro",
    "f5-tts",
    "sesame_csm",
}
MLAAD_SEEN = {
    "suno_bark",
    "tts_models_multilingual_multi-dataset_xtts_v2",
    "ChatTTS",
    "MeloTTS",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir(repo_dir_name: str) -> Path | None:
    root = HF_HUB / repo_dir_name / "snapshots"
    if not root.is_dir():
        return None
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    return snaps[0] if snaps else None


def iter_files(d: Path):
    if not d.is_dir():
        return
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            yield p


def build_deepfake_audio() -> dict:
    snap = snapshot_dir("datasets--garystafford--deepfake-audio-detection")
    origin_by_hash: dict[str, str] = {}
    if snap is not None:
        for sub in ("real", "fake"):
            for p in iter_files(snap / sub):
                origin_by_hash[sha256(p)] = p.name

    out = {}
    for label, sub in ((0, "real"), (1, "fake")):
        for p in iter_files(TEST_DATA / "deepfake-audio" / sub):
            orig = origin_by_hash.get(sha256(p))
            prefix = orig.split("_", 1)[0] if orig else None
            out[f"{sub}/{p.name}"] = {
                "label": label,
                "subgroup": GARYSTAFFORD_PREFIX.get(prefix, "unknown"),
                "orig_filename": orig,
            }
    return out


def load_asvspoof_protocol() -> dict[str, dict]:
    snap = snapshot_dir("datasets--jungjee--asvspoof5")
    if snap is None:
        return {}
    tar_path = snap / "ASVspoof5_protocols.tar"
    if not tar_path.is_file():
        return {}
    rows: dict[str, dict] = {}
    with tarfile.open(tar_path) as tf:
        member = next(
            (m for m in tf.getmembers() if m.name.endswith("ASVspoof5.eval.track_1.tsv")), None
        )
        if member is None:
            return {}
        fh = tf.extractfile(member)
        assert fh is not None
        for line in fh.read().decode("utf-8").splitlines():
            f = line.split()
            if len(f) < 9:
                continue
            rows[f[1]] = {
                "speaker": f[0],
                "gender": f[2],
                "codec": f[3],
                "attack": f[7],
                "protocol_label": f[8],
            }
    return rows


def build_asvspoof5(protocol: dict[str, dict]) -> dict:
    out = {}
    for label, sub in ((0, "real"), (1, "fake")):
        for p in iter_files(TEST_DATA / "asvspoof5" / sub):
            meta = protocol.get(p.stem, {})
            attack = meta.get("attack", "unknown")
            out[f"{sub}/{p.name}"] = {
                "label": label,
                "subgroup": "bonafide" if label == 0 else attack,
                "attack": attack,
                "codec": meta.get("codec"),
                "speaker": meta.get("speaker"),
                "protocol_label": meta.get("protocol_label"),
                "label_matches_protocol": (
                    None
                    if not meta
                    else (meta["protocol_label"] == ("spoof" if label == 1 else "bonafide"))
                ),
            }
    return out


def build_mlaad(protocol: dict[str, dict]) -> dict:
    snap = snapshot_dir("datasets--mueller91--MLAAD")
    origin_by_hash: dict[str, str] = {}
    if snap is not None:
        for p in sorted((snap / "fake").rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_EXT:
                origin_by_hash[sha256(p)] = str(p.relative_to(snap))

    out = {}
    for p in iter_files(TEST_DATA / "mlaad" / "real"):
        meta = protocol.get(p.stem, {})
        out[f"real/{p.name}"] = {
            "label": 0,
            "subgroup": "bonafide (ASVspoof5)",
            "contamination": "clean",
            "attack": meta.get("attack"),
            "speaker": meta.get("speaker"),
        }
    for p in iter_files(TEST_DATA / "mlaad" / "fake"):
        # mlaad_000_ElevenLabs-v3.wav -> ElevenLabs-v3
        engine = p.stem.split("_", 2)[2] if p.stem.count("_") >= 2 else "unknown"
        if engine in MLAAD_HELD_OUT:
            contamination = "held_out"
        elif engine in MLAAD_SEEN:
            contamination = "seen_in_training"
        else:
            contamination = "unknown"
        out[f"fake/{p.name}"] = {
            "label": 1,
            "subgroup": engine,
            "contamination": contamination,
            "orig_filename": origin_by_hash.get(sha256(p)),
        }
    return out


def build_from_manifest(dataset: str, field: str) -> dict:
    manifest_path = TEST_DATA / dataset / "manifest.json"
    meta_by_name: dict[str, dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for group in ("real", "fake"):
            for row in manifest.get(group, []):
                meta_by_name[row["out"]] = row

    out = {}
    for label, sub in ((0, "real"), (1, "fake")):
        for p in iter_files(TEST_DATA / dataset / sub):
            row = meta_by_name.get(p.name, {})
            out[f"{sub}/{p.name}"] = {
                "label": label,
                "subgroup": row.get(field, "unknown"),
                "orig_filename": row.get("orig") or row.get("orig_path"),
                **({"speaker": row["speaker"]} if "speaker" in row else {}),
            }
    return out


def main() -> None:
    protocol = load_asvspoof_protocol()
    print(f"ASVspoof5 eval protocol rows: {len(protocol)}")

    labels = {
        "deepfake-audio": build_deepfake_audio(),
        "asvspoof5": build_asvspoof5(protocol),
        "mlaad": build_mlaad(protocol),
        "codecfake": build_from_manifest("codecfake", "codec"),
        "in_the_wild": build_from_manifest("in_the_wild", "speaker"),
    }

    for ds, rows in labels.items():
        unknown = sum(1 for r in rows.values() if r["subgroup"] == "unknown")
        groups = sorted({r["subgroup"] for r in rows.values()})
        print(f"{ds:15s} {len(rows):4d} files, {len(groups):2d} subgroups, {unknown} unknown")

    mismatch = [
        k for k, v in labels["asvspoof5"].items() if v.get("label_matches_protocol") is False
    ]
    print(f"ASVspoof5 label/protocol mismatches: {len(mismatch)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(labels, indent=2, sort_keys=True))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
