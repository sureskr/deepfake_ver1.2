"""Sample 100 genuine + 100 spoofing from rogertseng/CodecFake.

Strategy: download one or more parquet shards locally, read with pyarrow, decode
the inline audio bytes with soundfile. Avoids the 100 GB full-dataset pull and
sidesteps the datasets-library torchcodec dependency.

CodecFake (Interspeech 2024, CC-BY-4.0) re-synthesizes VCTK utterances through 15
neural audio codecs — the family that underlies modern TTS (VALL-E, AudioLM, etc.).
This eval directly drills into the modern-TTS failure surfaced by MLAAD.
"""
import io
import json
import os
import random
import time
from collections import Counter, defaultdict

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "test_data", "codecfake")
os.makedirs(os.path.join(OUT, "real"), exist_ok=True)
os.makedirs(os.path.join(OUT, "fake"), exist_ok=True)

N_REAL = 100
N_FAKE = 100
N_FAKE_PER_CODEC = 7

REPO = "rogertseng/CodecFake"
TOTAL_SHARDS = 161


def shard_name(i):
    return f"data/train-{i:05d}-of-{TOTAL_SHARDS:05d}.parquet"


def scan_shard(shard_idx):
    print(f"\n[{time.strftime('%H:%M:%S')}] Downloading shard {shard_idx}...", flush=True)
    t0 = time.time()
    local = hf_hub_download(REPO, shard_name(shard_idx), repo_type="dataset")
    print(f"  cached at {local} ({(time.time()-t0):.0f}s)", flush=True)

    pf = pq.ParquetFile(local)
    print(f"  rows: {pf.metadata.num_rows}, columns: {pf.schema_arrow.names}", flush=True)
    table = pf.read()
    return table


real_picked = []  # list of dicts: {audio_bytes, audio_path, label, codec, speaker}
fake_per_codec = defaultdict(list)


def harvest(table):
    audio_col = table.column("audio").to_pylist()
    labels = table.column("label").to_pylist()
    codecs = table.column("codec_name").to_pylist()
    speakers = table.column("speaker_id").to_pylist()
    n = len(labels)
    print(f"  scanning {n} rows...", flush=True)
    label_counts = Counter(labels)
    codec_counts = Counter(codecs)
    print(f"  labels in shard: {dict(label_counts)}")
    print(f"  unique codecs in shard: {len(codec_counts)} -> {dict(codec_counts.most_common(5))}")

    # Shuffle indices for randomized sampling within shard
    idxs = list(range(n))
    random.shuffle(idxs)

    for i in idxs:
        if labels[i] == "genuine" and len(real_picked) < N_REAL:
            real_picked.append({
                "audio_bytes": audio_col[i]["bytes"],
                "audio_path": audio_col[i].get("path", ""),
                "label": labels[i],
                "codec": codecs[i],
                "speaker": speakers[i],
            })
        elif labels[i] == "spoofing":
            c = codecs[i]
            if len(fake_per_codec[c]) < N_FAKE_PER_CODEC:
                fake_per_codec[c].append({
                    "audio_bytes": audio_col[i]["bytes"],
                    "audio_path": audio_col[i].get("path", ""),
                    "label": labels[i],
                    "codec": codecs[i],
                    "speaker": speakers[i],
                })
        total_fake = sum(len(v) for v in fake_per_codec.values())
        if len(real_picked) >= N_REAL and total_fake >= N_FAKE and len(fake_per_codec) >= 15:
            return True
    return False


# Try shards in interleaved order so we get codec diversity faster.
# CodecFake has 707k rows across 161 shards. If sorted by codec, codecs span ~10 shards each.
# Interleaved stride attempts: 0, 11, 22, ... (covers all codecs in ~15 shards if sorted).
shard_order = list(range(0, TOTAL_SHARDS, 11)) + [i for i in range(TOTAL_SHARDS) if i % 11 != 0]

for shard_idx in shard_order:
    table = scan_shard(shard_idx)
    done = harvest(table)
    total_fake = sum(len(v) for v in fake_per_codec.values())
    print(f"  PROGRESS: real={len(real_picked)}/{N_REAL} fake={total_fake}/{N_FAKE} codecs_seen={len(fake_per_codec)}/15")
    if done:
        break

fake_picked = []
for codec, lst in fake_per_codec.items():
    fake_picked.extend(lst)
fake_picked = fake_picked[:N_FAKE]

print(f"\nFinal pick: {len(real_picked)} real / {len(fake_picked)} fake")
print("Codec distribution in fake sample:")
for c, n in Counter(p["codec"] for p in fake_picked).most_common():
    print(f"  {c}: {n}")

print("\nDecoding audio + writing wavs...")
manifest = {"real": [], "fake": []}
for i, p in enumerate(real_picked):
    audio, sr = sf.read(io.BytesIO(p["audio_bytes"]))
    out_path = os.path.join(OUT, "real", f"real_{i:03d}.wav")
    sf.write(out_path, audio, sr)
    manifest["real"].append({
        "out": f"real_{i:03d}.wav",
        "orig_path": p["audio_path"],
        "speaker": p["speaker"],
        "codec": p["codec"],
        "sr": int(sr),
        "len_s": round(len(audio) / sr, 2),
    })
for i, p in enumerate(fake_picked):
    audio, sr = sf.read(io.BytesIO(p["audio_bytes"]))
    out_path = os.path.join(OUT, "fake", f"fake_{i:03d}.wav")
    sf.write(out_path, audio, sr)
    manifest["fake"].append({
        "out": f"fake_{i:03d}.wav",
        "orig_path": p["audio_path"],
        "speaker": p["speaker"],
        "codec": p["codec"],
        "sr": int(sr),
        "len_s": round(len(audio) / sr, 2),
    })

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nDone. Files in: {OUT}")
print(f"  real/: {len(os.listdir(os.path.join(OUT, 'real')))} files")
print(f"  fake/: {len(os.listdir(os.path.join(OUT, 'fake')))} files")
