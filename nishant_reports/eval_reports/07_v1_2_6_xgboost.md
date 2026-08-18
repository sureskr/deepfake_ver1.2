# Eval 07: v1.2.6 — Big Data + XGBoost Classifier

**Author:** Nishant Jain, with Claude
**Date:** 2026-08-18
**Branch:** `experiment/xgboost-v1_2_6`
**Model versions:** v1.2.6 (neural head), XGBoost v1.2.6

## Motivation

Eval 06 left v1.2.5 at 0.28 EER on modern TTS — a large improvement over v1.2.4, but
not production-grade. Two hypotheses for the ceiling:

1. **Data scale.** v1.2.5 trained on 5,266 files with real audio from a single corpus
   (~3,800 ASVspoof bonafide) and ~78 MLAAD samples per engine.
2. **Classifier capacity.** A single Linear 1024→1 head can only draw a hyperplane
   through embedding space; diverse TTS engines may occupy regions it cannot separate.

This experiment tests both, **separately**, so the effects can be told apart:

- **v1.2.6 neural** — identical recipe to v1.2.5 (frozen Wav2Vec2 + linear head,
  warm-started from v1.2.4), trained on 5× the data. Isolates the *data* effect.
- **XGBoost v1.2.6** — same frozen Wav2Vec2 features, but the head is replaced with
  gradient-boosted trees over **statistical functionals** (mean/std/min/max/p10/p50/p90
  per dimension = 7,168 features) instead of a single mean-pooled 1024-d vector.
  Isolates the *classifier* effect.

Scaling the data also resolves a methodological problem with the tree experiment: at
v1.2.5's size, 7,168 features vs 5,266 samples is p > n. At v1.2.6's size it is
26,406 samples vs 7,168 features — a fair fight.

## Dataset

All real audio is ASVspoof5 bonafide (MLAAD is fake-only). LibriTTS/CommonVoice were
deliberately **not** added, to avoid confounding "spoof detection" with "corpus
detection" in this comparison.

| Source | Available | Used |
|--------|-----------|------|
| ASVspoof5 bonafide (real) | 18,797 (all 5 train shards) | 18,797 — the binding class |
| ASVspoof5 spoof | 163,560 | capped to 8,000 (`--asvspoof-max-fake`) |
| MLAAD English (fake) | 16,441 across 32 engines | 16,441 (500/engine) |

Splits are **generator-disjoint** — real by speaker, ASVspoof spoof by attack ID, MLAAD
by TTS engine — and balanced per split:

| Split | Total | Real | Fake | Fake: MLAAD | Fake: ASVspoof |
|-------|-------|------|------|-------------|----------------|
| calibration (train) | 26,406 | 13,203 | 13,203 | 8,174 | 5,029 |
| validation | 4,000 | 2,000 | 2,000 | 1,439 | 561 |
| test | 4,000 | 2,000 | 2,000 | 1,640 | 360 |

The test set is **82% modern TTS** and uses 8 engines held out entirely from training:
ElevenLabs-v3, OpenAI TTS-1 HD, Gemini-3.1-Flash-TTS, Cartesia.ai (Sonic-3), f5-tts,
kokoro, sesame_csm, MegaTTS3. Compared to eval 06 this is 5× the training data and
3.4× the test set.

## Training

**v1.2.6 neural** — frozen Wav2Vec2-large-960h + Linear 1024→1, warm-started from
v1.2.4, Adam lr=1e-4, batch 32, pos_weight=1.0, early stop on val EER (patience 3),
seed 42, 10 s @ 16 kHz. Early-stopped at epoch 45; **best epoch 42, val EER 0.1470**.

| Epoch | Train loss | Val loss | Val f1 | Val EER |
|-------|-----------|----------|--------|---------|
| 1 | 0.5865 | 0.5034 | 0.7807 | 0.2285 |
| 9 | 0.4693 | 0.4024 | 0.8282 | 0.1795 |
| 19 | 0.4363 | 0.3746 | 0.8442 | 0.1620 |
| 29 | 0.4175 | 0.3637 | 0.8478 | 0.1535 |
| 39 | 0.4047 | 0.3572 | 0.8536 | 0.1475 |
| 42 (best) | — | — | — | **0.1470** |
| 45 (stop) | 0.3987 | 0.3504 | 0.8534 | 0.1480 |

**XGBoost v1.2.6** — 7,168 functional features, lr=0.05, max_depth=6, subsample=0.8,
colsample_bytree=0.3, early stopping on val AUC. Converged at **1,928 trees**,
best val AUC 0.9631, **val EER 0.0955**. Wav2Vec2 features are extracted once and
cached, so re-tuning the trees costs seconds.

## Results — held-out test set (4,000 files, identical for all four models)

| Model | Overall EER | Acc @0.55 | EER [mlaad] | Recall@0.55 [mlaad] | EER [asvspoof] | Recall@0.55 [asvspoof] |
|-------|------------|-----------|-------------|---------------------|----------------|------------------------|
| v1.2.4 | 0.5530 | 53.65% | 0.6235 | 6.34% | 0.2065 | 51.67% |
| v1.2.5 | 0.3010 | 71.38% | 0.2990 | 61.59% | 0.3010 | 50.56% |
| v1.2.6 neural | 0.2325 | 78.15% | 0.2325 | 69.94% | 0.2325 | 60.56% |
| **XGBoost v1.2.6** | **0.1425** | **87.43%** | **0.1595** | **77.01%** | **0.0485** | **96.94%** |

### Reading the two effects

- **More data (v1.2.5 → v1.2.6 neural):** overall EER 0.3010 → 0.2325 (−23% relative),
  modern-TTS recall 61.6% → 69.9%. Real and worthwhile, but not a step change.
- **Tree classifier (v1.2.6 neural → XGBoost):** overall EER 0.2325 → 0.1425
  (−39% relative), modern-TTS recall 69.9% → 77.0%. **Larger than the data effect.**

The most striking result is on ASVspoof: both neural heads gave up ASVspoof accuracy as
they learned modern TTS (EER 0.2065 → 0.3010 → 0.2325), while **XGBoost held both at
once** — EER 0.0485 and 96.94% recall, better than any neural head including the
ASVspoof-specialised v1.2.4. The linear head appears to be capacity-bound: it trades one
attack family against the other, whereas trees carve out separate regions for each.

### Prediction vs outcome

Before running, the stated prior was ~35–45% that XGBoost would beat the neural head
standalone, on the reasoning that the data ceiling dominated and the linear head had a
warm-start advantage. **That was wrong.** The tree skeleton was the single largest lever
in this experiment, and it won without any warm start.

## Caveats

- **Not comparable to eval 06's numbers.** This is a different, larger test set with
  different held-out attacks and engines. v1.2.4's ASVspoof EER reads 0.2065 here vs
  0.0322 in eval 06 for that reason. Only the four rows in the table above are
  mutually comparable.
- **Identical EER across subsets for the neural heads** (v1.2.6 reads 0.2325 for
  overall / mlaad / asvspoof) is plausible — all three share the same 2,000 real files,
  so the EER crossing can land at the same FPR — but it is worth a sanity check before
  the numbers are quoted externally.
- **XGBoost probabilities are not calibrated to θ=0.55.** EER is threshold-free and is
  the metric to trust; a production threshold would need its own sweep on validation.
- **Still one real corpus.** All real audio is ASVspoof bonafide, so a real/fake domain
  confound remains. The next honest test is real audio from a second corpus.
- **Deployment cost differs.** The XGBoost model is 6.9 MB / 1,928 trees vs a ~7 KB
  linear head, and inference needs the 7,168-d functional extraction rather than a mean
  pool. Still trivial next to the Wav2Vec2 forward pass, but not free.

## Next steps

1. **Score fusion** — the trees and the linear head make different errors (trees are far
   stronger on ASVspoof, both are similar on MLAAD). Fusing the two scores is the
   standard way anti-spoofing challenges are won and is now the obvious next experiment.
2. **Real-audio diversity** — add LibriTTS/CommonVoice via `--extra-real-dir` to remove
   the single-corpus confound, as a separate, clean experiment.
3. **Threshold calibration** for XGBoost, if it is to be deployed.
4. **WavLM backbone** — swapping the feature extractor remains untested and is the
   other large lever.

## Artifacts

- `ml_engine/models/v1_2_6_proper/best_model.pth` — neural head (epoch 42)
- `ml_engine/models/xgboost_v1_2_6/model.json` — XGBoost model (1,928 trees)
- `ml_engine/models/xgboost_v1_2_6/results.json` — XGBoost metrics + top features
- `results_v1_2_6_headtohead.json` — the four-way comparison
- Pipeline: `scripts/setup_pod_data.sh`, `scripts/run_v1_2_6_experiments.sh`,
  `train_xgboost_v1_2_6.py`, `ml_engine/config/training_v1_2_6.json`
