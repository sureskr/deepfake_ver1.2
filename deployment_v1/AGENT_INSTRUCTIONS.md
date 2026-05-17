# Deployment Instructions for Internal Agent

## What This Is

Updating the deepfake detection ML engine from base model to V1 finetuned model.
Accuracy improves from 66.76% to 73.72%.

---

## Files To Deploy (in deployment_v1 folder)

1. `models/best_model.pth` → replace existing best_model.pth on server
2. `models/threshold.txt` → NEW file, add to models/ folder on server
3. `ml_worker.py` → replace existing ml_worker.py on server
4. `requirements.txt` → replace existing requirements.txt on server

---

## DO NOT Deploy

- `wav2vec2-large-960h/` folder (already on server, do not touch)
- Any training scripts
- Any dataset files
- `MANIFEST.txt` (internal reference only)

---

## Deployment Steps

1. **Backup existing server files first:**
   - Backup current `best_model.pth` as `best_model_backup.pth`
   - Backup current `ml_worker.py` as `ml_worker_backup.py`

2. **Copy new files to server**

3. **Install new requirement:**
   ```
   pip install miniaudio>=1.2
   ```

4. **Verify `threshold.txt` exists in models/ folder**
   Content should be: `0.7000`

5. **Restart inference service**

6. **Run smoke test on 5 files**
   - Expected output includes: `Using device: cuda`
   - Expected: scores vary (not all the same value)

---

## Verification Checklist

**1. Check `best_model.pth` file size = 6.0 KB**
Old model was 4 KB — size difference confirms correct file was deployed.

**2. Check `threshold.txt` exists in models/ folder**
Open file — content must be exactly: `0.7000`
If file is missing or shows `0.50` → deployment incomplete.

**3. Check `ml_worker.py` contains GPU device line near top:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Confirms new GPU-aware version was deployed.

**4. Check `requirements.txt` contains:**
```
miniaudio>=1.2
```
Confirms new FLAC decoder is included.

If all 4 checks pass → deployment successful.
No inference test needed on laptop.

---

## Rollback Plan

If anything breaks:

1. Restore `best_model_backup.pth` → `best_model.pth`
2. Restore `ml_worker_backup.py` → `ml_worker.py`
3. Delete `threshold.txt`
4. Restart service
5. System returns to previous state

---

## Key Change: Threshold

| | Old system | New system |
|---|---|---|
| Threshold value | 0.50 (hardcoded in script) | 0.70 (read from models/threshold.txt) |
| False alarm rate | Unknown | 12.37% |

Higher threshold = fewer false alarms = better for enterprise use.

---

## Expected Performance After Deployment

| Metric | Before | After |
|--------|--------|-------|
| Accuracy | 66.76% | **73.72%** |
| EER | 33.33% | **28.28%** |
| FPR | unknown | **12.37%** |

Validated on 25,423 unseen real-world files.
