"""Sample 100 real + 100 fake from In-the-Wild zip and extract into test_data/in_the_wild/."""
import csv
import io
import os
import random
import zipfile
from collections import Counter

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--mueller91--In-The-Wild/snapshots/"
    "eee168f92c367f8c82ff2cf42b6f61e362fd6211/release_in_the_wild.zip"
)
OUT = os.path.join(BASE, "test_data", "in_the_wild")
REAL_DIR = os.path.join(OUT, "real")
FAKE_DIR = os.path.join(OUT, "fake")
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

N_REAL = 100
N_FAKE = 100

with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open("release_in_the_wild/meta.csv") as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))

print(f"meta.csv rows: {len(rows)}")
label_counts = Counter(r["label"] for r in rows)
print(f"label breakdown: {dict(label_counts)}")

real_rows = [r for r in rows if r["label"] == "bona-fide"]
fake_rows = [r for r in rows if r["label"] == "spoof"]
print(f"  bona-fide: {len(real_rows)}, spoof: {len(fake_rows)}")

random.shuffle(real_rows)
random.shuffle(fake_rows)

real_pick = real_rows[:N_REAL]
fake_pick = fake_rows[:N_FAKE]

print(f"\nSpeaker distribution in sample:")
print(f"  real speakers: {Counter(r['speaker'] for r in real_pick).most_common(5)}")
print(f"  fake speakers: {Counter(r['speaker'] for r in fake_pick).most_common(5)}")

print(f"\nExtracting {N_REAL + N_FAKE} files from zip...")
with zipfile.ZipFile(ZIP_PATH) as z:
    for i, row in enumerate(real_pick):
        src = f"release_in_the_wild/{row['file']}"
        with z.open(src) as fin:
            data = fin.read()
        with open(os.path.join(REAL_DIR, f"real_{i:03d}.wav"), "wb") as fout:
            fout.write(data)
        if (i + 1) % 25 == 0:
            print(f"  real {i+1}/{N_REAL}")
    for i, row in enumerate(fake_pick):
        src = f"release_in_the_wild/{row['file']}"
        with z.open(src) as fin:
            data = fin.read()
        with open(os.path.join(FAKE_DIR, f"fake_{i:03d}.wav"), "wb") as fout:
            fout.write(data)
        if (i + 1) % 25 == 0:
            print(f"  fake {i+1}/{N_FAKE}")

# Save manifest so we can trace back to original speakers
import json
manifest = {
    "source_zip": ZIP_PATH,
    "seed": 42,
    "real": [{"out": f"real_{i:03d}.wav", "orig": r["file"], "speaker": r["speaker"]}
             for i, r in enumerate(real_pick)],
    "fake": [{"out": f"fake_{i:03d}.wav", "orig": r["file"], "speaker": r["speaker"]}
             for i, r in enumerate(fake_pick)],
}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nDone! Test data at: {OUT}")
print(f"  real/: {len(os.listdir(REAL_DIR))} files")
print(f"  fake/: {len(os.listdir(FAKE_DIR))} files")
print(f"  manifest.json saved")
