"""
combined_engine.py — Two-stage deepfake detector

AND logic: FAKE only when BOTH physics AND ML agree.

Physics engine (engine.py) — deterministic, CPU-only, no model weights.
VeriFauX ML (wav2vec2-large-960h + LinearHead) — always runs on every file.

Decision:
  physics=FAKE AND ML≥threshold → FAKE (high confidence, source=and_fake)
  physics=FAKE but ML<threshold → REAL (source=phys_only, physics overridden)
  physics=REAL but ML≥threshold → REAL (source=ml_only, ML overridden)
  both REAL                     → REAL (source=both_real)

Checkpoint:
  /Users/sureskr/claude_projects/vfx_repo/models/
  verifax_v1_2_2_fast_20250823_055109/best_model.pth
  Format: direct state dict {classifier.weight: [1,1024], classifier.bias: [1]}
  Backbone: facebook/wav2vec2-large-xlsr-53 (HuggingFace)

Score convention: sigmoid(linear(mean_pool(w2v2))) → 0.0 = real, 1.0 = fake
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_VFX_MODEL_PATH = r"C:\vdfk_engine_dev\vfx_repo\models\verifax_v1_2_2_fast_20250823_055109\best_model.pth"

import numpy as np

# ── Default thresholds ───────────────────────────────────────────────────────
# EER-calibrated threshold from VeriFauX training (val epoch 9): ~0.5285
# Using 0.5 (sigmoid midpoint) as conservative default to keep FPR < 15%.
DEFAULT_ML_THRESHOLD = 0.5

# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class CombinedResult:
    verdict: str          # "REAL" | "FAKE"
    confidence: str       # "high" | "medium" | "real"
    source: str           # "physics" | "ml" | "both_real"
    physics_checks_failed: int
    physics_verdict: str  # "REAL" | "FAKE"
    ml_score: Optional[float]     # None if ML not run
    ml_verdict: Optional[str]     # None if ML not run
    audio_path: str = ""


# ── Physics import ───────────────────────────────────────────────────────────
sys.path.insert(0, _HERE)
from engine import detect_from_array  # noqa: E402



# ── Combined detector ────────────────────────────────────────────────────────
class CombinedDetector:
    """
    Load once; call predict() or predict_from_array() per file.

    Lazy-initialises the ML backbone on first ML call so that pure-physics
    usage doesn't pay the wav2vec2-large load cost.
    """

    def __init__(
        self,
        model_path: str = _VFX_MODEL_PATH,
        ml_threshold: float = DEFAULT_ML_THRESHOLD,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"VeriFauX weights not found: {model_path}")
        self._model_path  = model_path
        self.ml_threshold = ml_threshold
        self._worker_proc = None   # persistent ml_worker.py subprocess

    # ── ML subprocess management ──────────────────────────────────────────────
    def _ensure_ml_loaded(self) -> None:
        """Start the ml_worker subprocess (loads model once, keeps running)."""
        if self._worker_proc is not None:
            return
        import subprocess, torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}', flush=True)
        worker_script = os.path.join(_HERE, "ml_worker.py")
        python = sys.executable
        print(f"[ML] Starting ml_worker subprocess …", flush=True)
        self._worker_proc = subprocess.Popen(
            [python, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent stderr so device/status messages are visible
            text=True,
            bufsize=1,
        )
        # Wait for READY then LOADED signals
        line = self._worker_proc.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"ml_worker unexpected start: {line!r}")
        line = self._worker_proc.stdout.readline().strip()
        if line != "LOADED":
            raise RuntimeError(f"ml_worker load failed: {line!r}")
        print(f"[ML] ml_worker ready.", flush=True)

    def __del__(self):
        if self._worker_proc is not None:
            try:
                self._worker_proc.stdin.close()
                self._worker_proc.wait(timeout=5)
            except Exception:
                self._worker_proc.kill()

    # ── Per-file ML score via subprocess ──────────────────────────────────────
    def _ml_score(self, audio_path: str) -> float:
        """Return sigmoid score: 0.0 = real, 1.0 = fake."""
        self._ensure_ml_loaded()
        self._worker_proc.stdin.write(audio_path + "\n")
        self._worker_proc.stdin.flush()
        response = self._worker_proc.stdout.readline().strip()
        if response.startswith("ERROR"):
            raise RuntimeError(f"ml_worker: {response}")
        return float(response)

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, audio_path: str) -> CombinedResult:
        """
        Full two-stage prediction from a file path (WAV/FLAC via soundfile).
        """
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        return self._run(audio, sr, audio_path)

    def predict_from_array(
        self, audio: np.ndarray, sr: int, audio_path: str = ""
    ) -> CombinedResult:
        """
        Two-stage prediction from a pre-loaded numpy array.
        Use this when the caller already read the audio (e.g. librosa for MP3).
        """
        return self._run(audio, sr, audio_path)

    def _run(
        self, audio: np.ndarray, sr: int, audio_path: str
    ) -> CombinedResult:
        # ── AND logic: run both stages on every file ──────────────────────────
        # FAKE only if physics=FAKE AND ML>=threshold.
        # Either system alone is insufficient — both must agree.

        # Stage 1: physics
        phys = detect_from_array(audio, sr, audio_path=audio_path)
        physics_verdict = phys.verdict

        # Stage 2: ML (always, even when physics says FAKE)
        if not audio_path or not os.path.isfile(audio_path):
            ml_score  = None
            ml_verdict = None
        else:
            ml_score   = self._ml_score(audio_path)
            ml_verdict = "FAKE" if ml_score >= self.ml_threshold else "REAL"

        # AND decision
        both_fake = (physics_verdict == "FAKE") and (ml_verdict == "FAKE")

        if both_fake:
            return CombinedResult(
                verdict              = "FAKE",
                confidence           = "high",
                source               = "and_fake",
                physics_checks_failed= phys.checks_failed,
                physics_verdict      = "FAKE",
                ml_score             = ml_score,
                ml_verdict           = "FAKE",
                audio_path           = audio_path,
            )

        # Determine why it's REAL (for diagnostics)
        if physics_verdict == "FAKE" and ml_verdict != "FAKE":
            source = "phys_only"      # physics said FAKE but ML disagrees → REAL
        elif physics_verdict == "REAL" and ml_verdict == "FAKE":
            source = "ml_only"        # ML said FAKE but physics disagrees → REAL
        else:
            source = "both_real"      # both agree REAL

        return CombinedResult(
            verdict              = "REAL",
            confidence           = "real",
            source               = source,
            physics_checks_failed= phys.checks_failed,
            physics_verdict      = physics_verdict,
            ml_score             = ml_score,
            ml_verdict           = ml_verdict,
            audio_path           = audio_path,
        )
