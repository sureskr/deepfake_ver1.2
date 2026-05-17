"""
HTTP API for local TeamsCallingBot integration.

- POST /predict  multipart field: file (WAV)
- GET  /health   { "status": "ok", "modelLoaded": bool }

Run from ml_engine root:
  python -m uvicorn windows_deployment.api_server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

_ENGINE_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("ml_api")

if str(_ENGINE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ENGINE_ROOT))


def _load_threshold() -> float:
    env = os.environ.get("ML_THRESHOLD")
    if env:
        return float(env)
    thr_file = _ENGINE_ROOT / "models" / "finetuned" / "threshold.txt"
    if thr_file.is_file():
        return float(thr_file.read_text().strip())
    return 0.5


app = FastAPI(title="ML Engine Predict API", version="1.0.0")

_inferencer = None
_model_label = ""
_threshold: float = 0.5


@app.on_event("startup")
async def _startup() -> None:
    global _inferencer, _model_label, _threshold
    _threshold = _load_threshold()
    rel = os.environ.get("ML_MODEL_PATH", "models/finetuned/best_model.pth")
    model_path = Path(rel)
    if not model_path.is_absolute():
        model_path = _ENGINE_ROOT / model_path

    if not model_path.is_file():
        LOGGER.error("Model file not found: %s", model_path)
        _inferencer = None
        _model_label = model_path.name
        return

    try:
        from inference import DLInferenceLarge

        _inferencer = DLInferenceLarge(str(model_path))
        _model_label = model_path.name
        LOGGER.info("Loaded model %s (threshold=%s)", model_path, _threshold)
    except Exception:
        LOGGER.exception("Failed to load model from %s", model_path)
        _inferencer = None
        _model_label = model_path.name


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "modelLoaded": _inferencer is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if _inferencer is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    suffix = Path(file.filename or "chunk.wav").suffix or ".wav"
    tmp_path: str | None = None
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        score = float(_inferencer.predict_single(tmp_path))
    except Exception as exc:
        LOGGER.warning("predict failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    label = "FAKE" if score >= _threshold else "REAL"
    confidence = float(max(score, 1.0 - score))
    return {
        "label": label,
        "score": score,
        "confidence": confidence,
        "model": _model_label,
    }
