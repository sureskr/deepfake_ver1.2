"""
ml_worker.py — persistent ML inference server (GPU)

Reads audio file paths from stdin (one per line).
Prints sigmoid scores to stdout (one float per line).
Prints "ERROR <msg>" on failure.

Paths configured via environment variables (set before launching):
  VFX_MODEL_PATH  path to best_model.pth       (required)
  VFX_W2V2_DIR    path to wav2vec2-large-960h/  (required)
  VFX_THR_PATH    path to threshold.txt         (optional, for logging only)

Run as:
  set VFX_MODEL_PATH=C:\\path\\to\\models\\best_model.pth
  set VFX_W2V2_DIR=C:\\path\\to\\wav2vec2-large-960h
  py ml_worker.py

V1.2.4 (deployment_v1_2_4):
  - Model: v1_2_4_proper — frozen Wav2Vec2-large + LinearHead (checkpoint['head_state_dict'])
  - Threshold: 0.55 (from models/threshold.txt; RitW-validated operating point)
  - Backward compatible: still loads legacy flat checkpoints with classifier.* only
  - Device: uses CUDA when available; otherwise runs on CPU (slower — see stderr warning)
  - Stderr: diagnostics only; stdout is the float protocol
"""
from __future__ import annotations

import os
import sys

# torch MUST be imported before numpy on this machine
import torch

torch.set_num_threads(1)

from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model as _Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
)

import numpy as np
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.environ.get(
    "VFX_MODEL_PATH", os.path.join(_HERE, "models", "best_model.pth")
)
_W2V2_DIR = os.environ.get(
    "VFX_W2V2_DIR", os.path.join(_HERE, "models", "wav2vec2-large-960h")
)
_W2V2_WEIGHTS = os.path.join(_W2V2_DIR, "pytorch_model.bin")
_THR_PATH = os.environ.get(
    "VFX_THR_PATH", os.path.join(_HERE, "models", "threshold.txt")
)

_SAMPLE_RATE = 16000
_TARGET_LEN = 160000  # 10 s


def _build_head(embed_dim: int = 1024):
    import torch.nn as nn

    class LinearHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(embed_dim, 1)

        def forward(self, x):
            return self.classifier(x).squeeze(-1)

    return LinearHead()


def _load_head_from_checkpoint(ckpt: dict, device: torch.device):
    """
    v1_2_4_proper / train_v1_2_4_proper: head weights in ``head_state_dict``
    (keys classifier.weight, classifier.bias).

    Legacy V1: checkpoint is a raw state_dict with the same keys.
    """
    if isinstance(ckpt, dict) and isinstance(ckpt.get("head_state_dict"), dict):
        sd = ckpt["head_state_dict"]
        if "classifier.weight" not in sd:
            raise KeyError("head_state_dict missing classifier.weight")
        embed_dim = sd["classifier.weight"].shape[1]
        head = _build_head(embed_dim)
        head.load_state_dict(sd, strict=True)
        head.eval()
        head.to(device)
        return head

    if isinstance(ckpt, dict) and "classifier.weight" in ckpt:
        embed_dim = ckpt["classifier.weight"].shape[1]
        head = _build_head(embed_dim)
        sub = {k: v for k, v in ckpt.items() if k.startswith("classifier.")}
        if len(sub) < 2:
            head.load_state_dict(ckpt, strict=True)
        else:
            head.load_state_dict(sub, strict=True)
        head.eval()
        head.to(device)
        return head

    raise KeyError(
        "Unrecognized checkpoint: need head_state_dict or flat classifier.* tensors"
    )


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.stderr.write(f"Using device: {device}\n")
    sys.stderr.flush()

    thr = (
        float(open(_THR_PATH, encoding="utf-8").read().strip())
        if os.path.exists(_THR_PATH)
        else 0.55
    )
    sys.stderr.write(f"Threshold (log): {thr}\n")
    sys.stderr.flush()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        _W2V2_DIR, local_files_only=True
    )

    config = Wav2Vec2Config.from_pretrained(_W2V2_DIR, local_files_only=True)
    backbone = _Wav2Vec2Model(config)
    raw = torch.load(_W2V2_WEIGHTS, map_location="cpu", weights_only=False)
    prefix = "wav2vec2."
    sd = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in raw.items()
        if k.startswith(prefix) or not k.startswith("lm_head.")
    }
    backbone.load_state_dict(sd, strict=False)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)

    ckpt = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
    head = _load_head_from_checkpoint(ckpt, device)

    sys.stderr.write(f"Model loaded: {_MODEL_PATH}\n")
    sys.stderr.flush()
    return processor, backbone, head, device


def score_file(processor, backbone, head, device, path):
    import librosa

    wav, _ = librosa.load(path, sr=_SAMPLE_RATE, mono=True)
    if len(wav) >= _TARGET_LEN:
        wav = wav[:_TARGET_LEN]
    else:
        wav = np.pad(wav, (0, _TARGET_LEN - len(wav)), mode="constant")

    inputs = processor(wav, sampling_rate=_SAMPLE_RATE, return_tensors="pt", padding=False)
    with torch.no_grad():
        hidden = backbone(inputs.input_values.to(device)).last_hidden_state
        pooled = hidden.mean(dim=1)
        score = torch.sigmoid(head(pooled)).item()
    return float(score)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.stderr.write(f"ML Worker starting on device: {device}\n")
    sys.stderr.flush()
    if device.type == "cpu":
        sys.stderr.write(
            "[WARN] No CUDA — running on CPU. Throughput will be much lower than GPU.\n"
        )
        sys.stderr.flush()
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    processor, backbone, head, device = load_model()
    sys.stdout.write("LOADED\n")
    sys.stdout.flush()

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            s = score_file(processor, backbone, head, device, path)
            sys.stdout.write(f"{s:.6f}\n")
        except Exception as e:
            sys.stdout.write(f"ERROR {e}\n")
        sys.stdout.flush()
