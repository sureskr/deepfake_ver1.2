# Paper Notes — index for the v1.2.x technical white paper

**Maintainer:** Nishant Jain, with Claude
**Last updated:** 2026-08-24
**Branch:** `eval/external-benchmarks-v1_2_6`

This file is the entry point for the white-paper write-up. It points at every artifact that
survives, states what each one does and does not prove, and lists the limitations the paper
must state up front rather than discover late. Everything referenced here is either committed
to this repository or reachable with a single `git show` from a named branch.

The arc the paper has to tell:

1. **v1.2.4** — frozen Wav2Vec2-large-960h + Linear 1024→1. Handles ASVspoof-family attacks,
   blind to modern TTS.
2. **v1.2.5** — same architecture and recipe, training set extended with MLAAD modern TTS.
3. **v1.2.6 neural** — same architecture and recipe again, 5× the training data. Isolates the
   *data* effect.
4. **XGBoost v1.2.6** — same frozen features, but statistical functionals (7,168-d) into
   gradient-boosted trees instead of a mean-pooled vector into a linear head. Isolates the
   *classifier* effect.
5. **External validation** (this branch) — all four models scored on five independently
   collected public benchmarks, on identical files, for the first time.

---

## 1. Training methodology

| Artifact | What it is |
|---|---|
| `prep_v1_2_5.py` | Dataset assembly. **Generator-disjoint splitting**: ASVspoof real split by *speaker*, ASVspoof spoof by *attack ID*, MLAAD fakes by *TTS engine*, so whole engines/attacks/speakers are held out for test rather than sampled across splits. Also caps the ASVspoof spoof pool (`--asvspoof-max-fake`) so it cannot drown out MLAAD, and balances real/fake per split. |
| `ml_engine/train_v1_2_4_proper.py` | The production training recipe used unchanged for v1.2.4, v1.2.5 and v1.2.6 neural: frozen backbone, linear head only, Adam, `BCEWithLogitsLoss(pos_weight=…)`, early stopping on validation EER. `load_audio_mono()` + `collate_audio_no_aug()` define the audio contract (10 s @ 16 kHz, front crop / zero pad, no augmentation). |
| `train_xgboost_v1_2_6.py` | The tree alternative. Same frozen Wav2Vec2 forward pass, but 7 functionals (mean, std, min, max, p10, p50, p90) over the time axis → 7,168 features, then `xgboost.XGBClassifier` with early stopping on validation AUC. Features are cached to disk, so re-tuning trees is seconds. |
| `ml_engine/config/training_v1_2_6.json` | The v1.2.6 run configuration (paths, audio contract, seed 42, lr 1e-4, batch 32, patience 3, `pos_weight: auto`, `default_threshold: 0.55`). `training_v1_2_5.template.json` and `training_v1_2_4_proper.json` are the earlier runs. |

**Shared audio contract for every model and every evaluation in this project:** mono, 16 kHz,
front crop or zero-pad to exactly 10 s, `Wav2Vec2FeatureExtractor` normalisation, frozen
`facebook/wav2vec2-large-960h`, no augmentation.

**Warm start:** the three neural heads were all warm-started from `v1_2_4_proper/best_model.pth`.
XGBoost was trained from scratch. See limitation 2 below.

## 2. Dataset construction and split counts

All real audio is ASVspoof5 bonafide; MLAAD is fake-only. LibriTTS/CommonVoice were deliberately
*not* mixed in, to avoid turning "spoof detection" into "corpus detection" in the comparison —
at the price of a real-vs-fake domain confound (limitation 4).

**v1.2.5** (report 06):

| Split | Total | Real | Fake | Fake = MLAAD | Fake = ASVspoof |
|---|---|---|---|---|---|
| calibration (train) | 5,266 | 2,633 | 2,633 | 1,495 | 1,138 |
| validation | 1,156 | 578 | 578 | 394 | 184 |
| test | 1,180 | 590 | 590 | 494 | 96 |

**v1.2.6** (report 07) — 5× the training data, same generator-disjoint scheme:

| Split | Total | Real | Fake | Fake = MLAAD | Fake = ASVspoof |
|---|---|---|---|---|---|
| calibration (train) | 26,406 | 13,203 | 13,203 | 8,174 | 5,029 |
| validation | 4,000 | 2,000 | 2,000 | 1,439 | 561 |
| test | 4,000 | 2,000 | 2,000 | 1,640 | 360 |

Source pools for v1.2.6: ASVspoof5 bonafide 18,797 (all five train shards, the binding class);
ASVspoof5 spoof capped to 8,000 of 163,560; MLAAD English 16,441 across 32 engines.

Eight modern engines were held out of training entirely and used only for the internal test
split: ElevenLabs-v3, OpenAI TTS-1 HD, Gemini-3.1-Flash-TTS, Cartesia.ai (Sonic-3), f5-tts,
kokoro, sesame_csm, MegaTTS3. The internal v1.2.6 test set is 82% modern TTS.

`data/v1_2_6/prep_report.json` is **gone** (built on an ephemeral machine); the counts above,
taken from report 07, are the surviving record.

## 3. Training curves

- `paper_data/v1_2_6_neural_training_curve.json` — all 45 epochs of the v1.2.6 neural run
  (train loss/F1, val loss/F1, val EER per epoch), plus the early-stop record: stopped at
  epoch 45, best epoch 42, best val EER 0.1470.
- v1.2.5's curve (13 epochs, best epoch 10, val EER 0.3097) is tabulated in report 06.
- XGBoost has no epoch curve; it early-stopped on validation AUC at **1,928 trees**, best val
  AUC 0.9631, val EER 0.0955 (`ml_engine/models/xgboost_v1_2_6/results.json`).

Note for the write-up: because the neural heads were warm-started from v1.2.4, epoch 1 of each
curve is approximately v1.2.4's own performance on that run's validation set.

## 4. Internal held-out results

- `results_v1_2_6_headtohead.json` — all four models on the **same** 4,000-file internal test
  split (overall EER, acc@0.50, acc@0.55, and EER/recall broken down by `mlaad` vs `asvspoof`).
- `ml_engine/models/xgboost_v1_2_6/results.json` — XGBoost's own metrics, hyperparameters, best
  iteration and top-20 feature indices.
- `results_v1_2_5_headtohead.json` — the earlier v1.2.4-vs-v1.2.5 comparison on the *v1.2.5*
  test split (1,180 files). Different test set: not comparable to the v1.2.6 numbers.

Headline internal numbers (4,000 files, identical for all four models):

| Model | Overall EER | Acc @0.55 | EER [mlaad] | EER [asvspoof] |
|---|---|---|---|---|
| v1.2.4 | 0.5530 | 53.65% | 0.6235 | 0.2065 |
| v1.2.5 | 0.3010 | 71.38% | 0.2990 | 0.3010 |
| v1.2.6 neural | 0.2325 | 78.15% | 0.2325 | 0.2325 |
| XGBoost v1.2.6 | **0.1425** | **87.43%** | **0.1595** | **0.0485** |

## 5. External validation (this session)

| Artifact | What it is |
|---|---|
| `eval_external_datasets.py` | The harness. One forward pass per file; the mean-pooled 1024-d vector feeds the neural heads and the 7,168-d functionals feed the trees, so all four models see identical audio and identical features. Bootstrap CIs, McNemar tests, per-subgroup breakdowns, MLAAD contamination split, and a self-check against the published v1.2.4 baselines. |
| `build_subgroup_labels.py` → `paper_data/subgroup_labels.json` | Per-file subgroup labels recovered from provenance: generator (deepfake-audio, by SHA256 match to the HF cache), attack ID + codec + speaker (ASVspoof5 official eval protocol), TTS engine + contamination flag + original MLAAD filename, codec (CodecFake manifest), speaker (In-the-Wild manifest). Zero unknown labels; ASVspoof5 labels verified against the protocol with 0 mismatches. |
| `results_external_v1_2_6.json` | Full metric set per (dataset, model): EER + threshold + 95% bootstrap CI, AUROC + CI, metrics/confusion at 0.55 and at each model's own EER threshold, per-class score distributions with deciles, ROC/DET curve points, per-subgroup breakdowns, paired model comparisons, harness validation, decode-path check. |
| `paper_data/external_scores.csv` | **Raw per-file scores** — 1,000 rows × 4 model score columns, with dataset, filename, true label, subgroup, contamination flag and original filename. Any metric or plot in the paper can be recomputed from this without re-running inference. |
| `validate_v1_2_4_perfile.py` | Per-file validation of the harness against the original v1.2.4 run stored on `origin/eval/public-dataset-test` (see §6). |
| `nishant_reports/eval_reports/08_external_benchmarks.md` | The written report: harness validation, per-dataset four-model comparison against the v1.2.4 baselines, MLAAD contamination split, calibration caveat, significance. |

Datasets (200 files each, 100 real + 100 fake): garystafford deepfake-audio (6 commercial TTS),
ASVspoof5 **eval** partition (disjoint from the train partition used for training), MLAAD
(10 engines, split held-out vs seen), In-the-Wild (deployment-quality celebrity deepfakes),
CodecFake (15 neural audio codecs).

**The one-line answer this evaluation gives:** XGBoost's large internal win does **not**
generalise. It transfers only to modern-TTS-flavoured data (deepfake-audio, MLAAD); on
ASVspoof5 eval, In-the-Wild and CodecFake, v1.2.4 — the weakest model internally — is the
strongest of the four, and no XGBoost-vs-v1.2.6-neural difference is statistically significant
on any external set. Read report 08 before writing this section.

## 6. Prior-version evaluations (v1.2.4, reports 01–05)

They live on `origin/eval/public-dataset-test` and are **not** on this branch. Read them with:

```
git show origin/eval/public-dataset-test:nishant_reports/eval_reports/01_huggingface_deepfake_audio.md
git show origin/eval/public-dataset-test:nishant_reports/eval_reports/02_asvspoof5_eval.md
git show origin/eval/public-dataset-test:nishant_reports/eval_reports/03_mlaad_eval.md
git show origin/eval/public-dataset-test:nishant_reports/eval_reports/04_in_the_wild_eval.md
git show origin/eval/public-dataset-test:nishant_reports/eval_reports/05_codecfake_eval.md
git show origin/eval/public-dataset-test:deployment_v1_2_4/inference.py   # the original scoring path
git show origin/eval/public-dataset-test:run_eval.py                      # how 01-05 were produced
```

### Prior-version raw data (available, with one caveat)

The same branch stores the **raw v1.2.4 output** for the five 200-file samples still sitting in
`test_data/`:

| File | Contents |
|---|---|
| `results.json` (deepfake-audio), `results_asvspoof5.json`, `results_mlaad.json`, `results_in_the_wild.json`, `results_codecfake.json` | `predictions` (per-file v1.2.4 scores) + `targets` (labels), 200 each, plus the aggregate metrics quoted in reports 01–05 |
| `results_all_engines.json` | deepfake-audio only: per-file rows with filename, ML score, physics verdict, checks-failed, combined verdict |
| `results_codecfake_enriched.json`, `results_in_the_wild_enriched.json` | v1.2.4 per-codec / per-speaker breakdowns and per-class score distributions |

**What can be recomputed from them:** any threshold-free or class-wise metric for v1.2.4 — EER,
AUROC, ROC/DET curves, accuracy/precision/recall/F1 and confusion at any threshold, per-class
score distributions — and paired tests against the v1.2.4 run itself.

**Caveat, verified not assumed:** the arrays follow `inference.py`'s `Path.glob("*.*")`
enumeration (filesystem order), while everything in this project since uses `sorted()`. Compared
position by position they disagree by up to 0.77. Matched within each class in sorted order, this
harness's v1.2.4 scores reproduce all 1,000 original scores to **≤ 8.2 × 10⁻⁶**
(`validate_v1_2_4_perfile.py`, recorded in
`results_external_v1_2_6.json → per_file_validation_vs_original_v1_2_4`). Two consequences:

1. The v1.2.4 column of `paper_data/external_scores.csv` **is** the original v1.2.4 run, correctly
   attributed to filenames (confirmed against a standalone re-implementation of
   `inference.py` on individual files). Use it rather than the raw arrays — it has filenames,
   subgroup labels, and the other three models on the same rows.
2. **The per-file attributions on the eval branch are misaligned** — `run_eval.py` and the
   enrichment scripts zipped `sorted()` filenames onto glob-ordered scores.
   `results_all_engines.json` disagrees with a correct recomputation on 199 of 200 filenames, and
   the `*_enriched.json` per-codec / per-speaker tables inherit the same error (it is also the
   root cause of the eval-04 per-speaker correction in report 08 §5). Do **not** reuse those
   subgroup breakdowns as v1.2.4 baselines; the correctly-attributed versions are in
   `results_external_v1_2_6.json`. Class-wise aggregates in reports 01–05 are unaffected and
   reproduce exactly.

Reports 01–05 also cover a deterministic "physics engine" (jitter/shimmer, formant velocity,
HNR, spectral flux) and an AND-combined engine. Only the **ML engine** rows are comparable to
anything in this project's later work; the v1.2.4 ML baselines are reproduced exactly by the
external harness (report 08, §Harness validation).

## 7. Reproducibility facts

- **Seed:** 42 everywhere — dataset splitting (`prep_v1_2_5.py`), training
  (`training_v1_2_6.json`), test-set sampling (`prep_in_the_wild.py`, `prep_codecfake.py`,
  `download_test_data.py`), and bootstrap resampling in the external harness.
- **Backbone:** `facebook/wav2vec2-large-960h`, frozen. None of the four checkpoints contains
  fine-tuned backbone weights (verified: the harness refuses to run if a checkpoint carries a
  `backbone_state_dict`).
- **Neural head hyperparameters** (identical for v1.2.4/v1.2.5/v1.2.6): Linear 1024→1, Adam
  lr = 1e-4, batch 32, max 50 epochs, early stopping patience 3 on val EER, `pos_weight` auto
  (= 1.0 on balanced splits), no augmentation, 10 s @ 16 kHz mean-pooled.
- **XGBoost hyperparameters:** n_estimators 2000 (early-stopped at 1,928), lr 0.05, max_depth 6,
  subsample 0.8, colsample_bytree 0.3, min_child_weight 5, reg_lambda 1.0,
  `objective=binary:logistic`, `eval_metric=auc`, `tree_method=hist`, early stopping 50 rounds
  on val AUC. 7,168 input features (verified against the saved booster).
- **Library versions used for the external evaluation:** Python 3.12.13 on macOS (arm64),
  torch 2.12.0, transformers 5.8.1, librosa 0.11.0, scikit-learn 1.8.0, numpy 2.4.5,
  scipy 1.17.1, xgboost 3.4.1. Recorded in `results_external_v1_2_6.json → meta.library_versions`.
- **Commit SHAs:**
  - `f5d4c03` — v1.2.6 models, XGBoost, report 07, training curve (branch `experiment/xgboost-v1_2_6`)
  - `90c2e82` — reports 01–05 and the original v1.2.4 eval path (branch `eval/public-dataset-test`)
  - the external evaluation commit on this branch is recorded in
    `results_external_v1_2_6.json → meta.git_commit`
- **Known environment quirk:** on macOS, torch and xgboost link different OpenMP runtimes and
  deadlock in a single process. The harness scores the trees in a subprocess for this reason —
  a packaging detail, not a modelling one.

## 8. Known limitations the paper must state

1. **The internal v1.2.6 test set no longer exists.** It was built on an ephemeral machine, so
   internal per-file scores and DET curves cannot be regenerated without rebuilding the dataset
   from `prep_v1_2_5.py` with seed 42 and the counts in §2. Only summary metrics survive for the
   internal evaluation. Every per-file artifact in this project is external
   (`paper_data/external_scores.csv`).
2. **The classifier comparison is not perfectly controlled.** The three neural heads were
   warm-started from v1.2.4; XGBoost was trained from scratch. This disadvantaged XGBoost, which
   still won internally — so the internal conclusion is conservative in that direction. It also
   means "trees beat linear heads" and "training from scratch beats warm starting" are not
   separated by this experiment.
3. **Reports 06 and 07 used different test sets** and their numbers are not comparable to each
   other. The external evaluation in report 08 is the first time all four models were scored on
   identical files, and it should be the backbone of the comparison in the paper.
4. **All real training audio came from a single corpus** (ASVspoof5 bonafide), so a real-vs-fake
   domain confound remains: a model can score "sounds like ASVspoof bonafide" rather than "sounds
   human". The external results are consistent with this having happened — see the calibration
   collapse in report 08 (all three post-v1.2.4 models flag 74–100% of genuine external audio at
   the production threshold).
5. **n = 200 per external dataset is small.** All external EERs carry 95% bootstrap CIs roughly
   ±0.07–0.10 wide, so differences under ~0.10 EER are generally not resolvable. Report CIs,
   avoid ranking models by point estimates, and do not over-claim.
6. **MLAAD is partially contaminated for v1.2.5/v1.2.6** — 4 of the 10 external MLAAD engines
   were training engines. Report 08 always splits MLAAD into held-out vs seen engines. Literal
   file-level overlap could not be checked because the training file lists are gone with the
   ephemeral machine; `paper_data/subgroup_labels.json` records the original MLAAD filename of
   every external MLAAD file so a rebuilt training pool can be diffed against it later.
7. **The XGBoost probabilities are not calibrated** to the production threshold of 0.55, and
   neither, it turns out, are the v1.2.5/v1.2.6 neural heads on external audio. EER is the
   primary metric throughout; accuracy at 0.55 is reported but should never be quoted alone.

## 9. Things a careful reader will ask

- *Is the external harness the same scoring path that produced reports 01–05?* Yes, at per-file
  resolution: all 1,000 original v1.2.4 scores reproduce to ≤ 8.2 × 10⁻⁶, and all five published
  EERs and accuracies reproduce (report 08, §1). The FLAC decode path (miniaudio vs librosa) was
  also checked and changes no score at 1e-6 resolution.
- *Are the external MLAAD and ASVspoof5 sets independent?* No. The 100 real files are the same
  100 ASVspoof5 bonafide files in both, so the real-side statistics of those two datasets are
  identical by construction, and the two rows should not be treated as independent evidence.
- *Why is In-the-Wild the most informative set?* Its fakes are deployment-quality deepfakes that
  already fooled human audiences, and no version trained on anything like them. CodecFake plays
  the same role for neural-codec artifacts.
