# Deployment instructions — VeriFauX V1.2.4 (`deployment_v1_2_4`)

## What this is

Production-style bundle for **V1.2.4**: **frozen Wav2Vec2-large-960h** + **single linear head**, trained on the expanded v1_2_4 calibration mix (RitW + LA + MLAAD + CVoiceFake + ASVspoof5), evaluated on the **locked RitW test** (25,423 files) at **threshold 0.55**.

---

## Package layout

```
deployment_v1_2_4/
  MANIFEST.txt
  AGENT_INSTRUCTIONS.md
  ml_worker.py
  inference.py
  requirements.txt
  models/
    best_model.pth      ← v1_2_4_proper weights
    threshold.txt       ← 0.5500
```

**Not included** (must already exist on the host or be copied separately):

- `models/wav2vec2-large-960h/` (Hugging Face–style folder with `config.json`, `preprocessor_config.json`, `pytorch_model.bin`)

---

## Files to place on the inference host

| Source in this folder | On server (typical) |
|----------------------|---------------------|
| `models/best_model.pth` | `<deploy>/models/best_model.pth` |
| `models/threshold.txt` | `<deploy>/models/threshold.txt` |
| `ml_worker.py` | `<deploy>/ml_worker.py` |
| `requirements.txt` | `<deploy>/requirements.txt` |
| `inference.py` | optional; batch/offline tooling |

Environment variables (optional; defaults assume sibling `models/`):

- `VFX_MODEL_PATH` — path to `best_model.pth`
- `VFX_W2V2_DIR` — path to `wav2vec2-large-960h` directory
- `VFX_THR_PATH` — path to `threshold.txt` (logging only in worker)

---

## Pre-flight

1. GPU:

   ```bash
   py -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
   ```

2. Dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Threshold file:** open `models/threshold.txt` — must read **`0.5500`** (one line).

4. **Checkpoint format:** `torch.load(best_model.pth)` should be a `dict` with **`head_state_dict`** containing `classifier.weight` / `classifier.bias`. Legacy flat checkpoints still work in `ml_worker.py`.

---

## `ml_worker` protocol

- Stdout: `READY` → `LOADED` → one `float` per input path (or `ERROR ...`).
- Stderr: device, threshold (log), load messages.
- **Device:** uses **CUDA when available**; otherwise runs on **CPU** (stderr warns — throughput is much lower).

Decision rule in the **host application** (not inside `ml_worker`): compare score to **0.55** unless you centralize thresholds elsewhere.

---

## Headline RitW metrics (locked test, θ = 0.55)

See `MANIFEST.txt`. Summary: **Accuracy 73.30%**, **EER 27.55%**, **FPR 11.88%**, **FNR 51.75%**, **AUROC 0.792**.

---

## Rollback

Keep backups of prior `best_model.pth`, `ml_worker.py`, and `threshold.txt`. Restore previous threshold (e.g. `0.7000`) if reverting to V1 deployment behavior.

---

## Agent checklist before reporting “deployed”

- [ ] `deployment_v1_2_4/models/best_model.pth` present (size ~6–8 KB for this format)
- [ ] `deployment_v1_2_4/models/threshold.txt` contains `0.5500`
- [ ] `ml_worker.py`, `requirements.txt`, `inference.py`, `MANIFEST.txt`, `AGENT_INSTRUCTIONS.md` present
- [ ] `wav2vec2-large-960h` available at `VFX_W2V2_DIR` or `models/wav2vec2-large-960h`
- [ ] Worker starts: stderr shows `cuda` or `cpu`, stdout prints `READY` / `LOADED`
