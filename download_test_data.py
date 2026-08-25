"""Download ~200 files from garystafford/deepfake-audio-detection via HF Hub."""
import os
import random
import shutil
from huggingface_hub import HfApi, hf_hub_download

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "test_data")
REAL_DIR = os.path.join(OUT, "real")
FAKE_DIR = os.path.join(OUT, "fake")
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

REPO = "garystafford/deepfake-audio-detection"
N_REAL = 100
N_FAKE = 100

print("Listing files in dataset repo...")
api = HfApi()
files = list(api.list_repo_tree(REPO, repo_type="dataset", recursive=True))
all_files = [f.rfilename for f in files if hasattr(f, "rfilename") and f.rfilename.endswith(".flac")]

real_files = [f for f in all_files if f.startswith("real/")]
fake_files = [f for f in all_files if f.startswith("fake/")]

print(f"Found {len(real_files)} real, {len(fake_files)} fake files in repo")

random.shuffle(real_files)
random.shuffle(fake_files)

real_pick = real_files[:N_REAL]
fake_pick = fake_files[:N_FAKE]

print(f"Downloading {N_REAL} real files...")
for i, rfile in enumerate(real_pick):
    local = hf_hub_download(REPO, rfile, repo_type="dataset")
    dest = os.path.join(REAL_DIR, f"real_{i:03d}.flac")
    shutil.copy2(local, dest)
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_REAL}")

print(f"Downloading {N_FAKE} fake files...")
for i, rfile in enumerate(fake_pick):
    local = hf_hub_download(REPO, rfile, repo_type="dataset")
    dest = os.path.join(FAKE_DIR, f"fake_{i:03d}.flac")
    shutil.copy2(local, dest)
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_FAKE}")

print(f"\nDone! Test data saved to: {OUT}")
print(f"  real/: {len(os.listdir(REAL_DIR))} files")
print(f"  fake/: {len(os.listdir(FAKE_DIR))} files")
