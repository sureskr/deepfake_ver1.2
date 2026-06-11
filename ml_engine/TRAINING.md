# VeriFauX ML Training Guide

This document is the **canonical workflow** for training and updating the production deepfake detector (V1.2.4 recipe: frozen Wav2Vec2-large + linear head).

For legacy scripts (`finetune_calibration.py`, `train_v1_2_4.py`, `vfx_repo/`), see the mentor review in project notes — **do not use those for new data** unless you understand their risks (random splits, test leakage, etc.).

---

## Quick reference

| Role | Folder (this project) | Your mental model |
|------|----------------------|-------------------|
| Training | `data/<dataset>/calibration/{real,fake}` | `train/` |
| Validation | `data/<dataset>/validation/{real,fake}` | `validation/` |
| Dev test | `data/<dataset>/test/{real,fake}` | secondary test |
| **Locked benchmark** | `data/test/{real,fake}` | sacred `test/` — evaluate **once** per model |

**Production script:** `train_v1_2_4_proper.py`  
**Production config:** `config/training_v1_2_4_proper.json`  
**Deploy bundle:** `deployment_v1_2_4/`

---

## 1. Prerequisites

```powershell
cd C:\vdfk_engine_dev\ml_engine
pip install -r requirements.txt
```

Ensure these exist locally (gitignored):

- `ml_engine/models/wav2vec2-large-960h/` — Hugging Face Wav2Vec2-large weights
- `data/v1_2_4/calibration/` and `data/v1_2_4/validation/` — audio in `real/` and `fake/` subfolders
- Prior head checkpoint for warm-start (default: `models/v1_finetuned/best_model.pth`)

Check GPU:

```powershell
py -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. Dataset manifest (versioning)

Every dataset drop should produce a **manifest CSV** that is committed to git (audio stays in `data/`, which is gitignored).

### Schema

See `config/manifest.example.csv`:

| Column | Required | Description |
|--------|----------|-------------|
| `file_id` | yes | Stable unique ID |
| `split` | yes | `calibration`, `validation`, `test`, or `locked_test` |
| `label` | yes | `real` or `fake` |
| `filepath` | yes | Absolute or repo-relative path to audio file |
| `source` | yes | Dataset origin (e.g. `ritw`, `asvspoof2021_LA`, `cvoicefake`) |
| `speaker_id` | recommended | **Use for disjoint splits** — same speaker must not appear in train and val |
| `duration_sec` | optional | Clip duration (filled by builder) |
| `sample_rate` | optional | Native sample rate if known |
| `sha256` | yes | Content hash for reproducibility |
| `notes` | optional | Free text |

### Build manifest

```powershell
cd C:\vdfk_engine_dev\ml_engine
py -3.11 scripts/build_manifest.py `
  --dataset-root ..\data\v1_2_4 `
  --out ..\data\manifests\v1_2_4.csv `
  --source-tag v1_2_4
```

Verify files have not changed since manifest was written:

```powershell
py -3.11 scripts/build_manifest.py --out ..\data\manifests\v1_2_4.csv --verify
```

Commit `data/manifests/*.csv` to git. Re-run the builder after every data change.

---

## 3. Training configuration

All paths and hyperparameters live in JSON:

```
ml_engine/config/training_v1_2_4_proper.json
```

Paths use `{repo_root}` (auto-detected as parent of `ml_engine/`). Copy this file for a new run, e.g. `training_v1_2_5.json`, and edit:

- `run_name` — used for log file name and checkpoint metadata
- `paths.train_*` / `paths.val_*` — point at your folders
- `paths.output_dir` — **new directory** per run (never overwrite production blindly)
- `paths.head_init_checkpoint` — warm-start from previous best model
- `training.pos_weight` — keep `"auto"` to use `N_real / N_fake` from calibration folder counts

Example for a v1.2.5 experiment:

```json
{
  "run_name": "v1_2_5_proper",
  "paths": {
    "head_init_checkpoint": "{repo_root}/ml_engine/models/v1_2_4_proper/best_model.pth",
    "output_dir": "{repo_root}/ml_engine/models/v1_2_5_proper",
    "manifest_path": "{repo_root}/data/manifests/v1_2_5.csv"
  }
}
```

---

## 4. Training workflow

### Step A — Dry run (config check only)

```powershell
cd C:\vdfk_engine_dev\ml_engine
py -3.11 train_v1_2_4_proper.py
```

Prints file counts, `pos_weight`, paths, and device. **No training happens** without `--train`.

### Step B — Train

```powershell
py -3.11 train_v1_2_4_proper.py --config config/training_v1_2_4_proper.json --train
```

What happens:

1. Load frozen Wav2Vec2-large from local `models/wav2vec2-large-960h/`
2. Warm-start linear head from checkpoint
3. Preprocess: 16 kHz mono, front-crop/pad to 10 s, mean-pool → 1024-d embedding
4. Optimize head only (Adam, lr=1e-4)
5. Early stop on **validation EER** (patience=3)
6. Save best to `models/<run_name>/best_model.pth`
7. Append metrics to `logs/train_<run_name>.jsonl`

### Step C — Threshold (on validation only)

Do **not** tune threshold on the locked RitW test. Use validation scores or a dedicated calibration split:

```powershell
py -3.11 eval_v1_2_4_threshold_sweep.py
```

Production operating point for V1.2.4: **θ = 0.55** (`deployment_v1_2_4/models/threshold.txt`).

### Step D — Final evaluation (once)

Locked RitW benchmark only:

```powershell
py -3.11 eval_v1_2_4_proper_ritw.py --threshold 0.55
```

Held-out v1_2_4 test (optional, for development):

```powershell
py -3.11 eval_v1_2_4_proper_v14test_sweep.py
```

---

## 5. Adding new data (checklist)

Use this checklist for every new dataset drop:

- [ ] **QC** — listen to samples, check labels, remove corrupt/duplicate files
- [ ] **Assign split** — new files go to `calibration/` only; rebuild val if needed from **held-out speakers**
- [ ] **Update manifest** — `scripts/build_manifest.py` → commit CSV
- [ ] **New config** — copy JSON, new `run_name` and `output_dir`
- [ ] **Warm-start** — set `head_init_checkpoint` to current production `best_model.pth`
- [ ] **Train** — `--train`, monitor val EER in jsonl logs
- [ ] **Threshold** — sweep on validation; document chosen θ and rationale (FPR vs FNR trade-off)
- [ ] **Evaluate once** on `data/test/` (RitW)
- [ ] **Promote** — copy checkpoint + threshold to `deployment_v1_2_4/`, keep rollback copy

### Recommended strategy

**Fine-tune the existing linear head** (default). Do not retrain Wav2Vec2 from scratch unless you change architecture or have a massive domain shift.

Retrain from scratch only if:

- Training data is contaminated, or
- You switch to a different head/backbone architecture

Avoid `train_v1_2_4.py` for routine updates — it unfreezes backbone layers and uses a subset of the test split during model selection.

---

## 6. Split and class-balance guidelines

| Topic | Guidance |
|-------|----------|
| Ratios | ~70–80% train / 10–15% val / 10–15% dev test; RitW 25k stays fully locked |
| Stratification | Keep real/fake balance similar across splits |
| **Speaker disjointness** | Split by `speaker_id` in manifest, not random files |
| Class imbalance | `"pos_weight": "auto"` → `N_real / N_fake`; recalculate after every merge |
| Cross-validation | Single val set is fine at ~17k scale; use k-fold only if val is very small |
| Overfitting | Watch train EER ↓ while val EER ↑ → stop early (already enabled) |
| Data leakage | Never train on `data/test/`; never tune threshold on the same val set used for early stopping without a separate calibration split |

---

## 7. Checkpoint and deployment

Checkpoint payload includes:

- `head_state_dict` — linear classifier weights
- `train_config` — hyperparameters, seed, manifest path
- `val_metrics` — metrics at save time

Promote to production:

1. Compare RitW metrics vs current `deployment_v1_2_4/MANIFEST.txt` baseline
2. Copy `best_model.pth` and `threshold.txt` into `deployment_v1_2_4/models/`
3. Test worker: `py deployment_v1_2_4/ml_worker.py` (expects `READY` / `LOADED` on stderr)

Rollback: keep dated copies, e.g. `best_model_20250611.pth` and prior threshold.

---

## 8. File map

```
ml_engine/
├── TRAINING.md                          ← this guide
├── train_v1_2_4_proper.py               ← canonical training script
├── training_config.py                   ← JSON config loader
├── config/
│   ├── training_v1_2_4_proper.json      ← default training config
│   └── manifest.example.csv             ← manifest schema example
├── scripts/
│   └── build_manifest.py                ← manifest builder / verifier
├── eval_v1_2_4_proper_ritw.py           ← locked test eval
├── eval_v1_2_4_threshold_sweep.py         ← threshold sweep
├── inference.py                         ← batch inference CLI
├── models/
│   ├── wav2vec2-large-960h/             ← backbone (local copy)
│   └── v1_2_4_proper/best_model.pth     ← current production weights
└── logs/
    └── train_<run_name>.jsonl           ← per-epoch metrics

data/                                    ← gitignored
├── v1_2_4/calibration|validation|test/
├── test/                                ← locked RitW benchmark
└── manifests/*.csv                      ← commit these to git
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Empty training set | Wrong paths in config | Check `--config` paths; run dry run |
| Missing Wav2Vec2 | Weights not downloaded | Copy HF model into `models/wav2vec2-large-960h/` |
| Missing head checkpoint | No warm-start file | Train from scratch (warns) or point config at existing `.pth` |
| Val EER flat | LR too low/high or data too easy | Try lr `5e-5`; check val split is not train duplicate speakers |
| Good val, bad RitW | Overfit to val domain | More diverse calibration data; speaker-disjoint splits |

---

## 10. Scripts to avoid for routine work

| Script | Why |
|--------|-----|
| `train_v1_2_4.py` | Uses test subset for early stopping (leakage) |
| `finetune_calibration.py` | Random 80/20 split inside one folder |
| `train.py` / `vfx_repo/*` | Legacy v1.2.2 pipeline, different data layout |

Use these only for historical reproduction or experiments with full understanding of their limitations.
