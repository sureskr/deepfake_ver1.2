# Eval 08: External Public Benchmarks — All Four Models on Identical Files

**Author:** Nishant Jain, with Claude
**Date:** 2026-08-24
**Branch:** `eval/external-benchmarks-v1_2_6`
**Model versions:** v1.2.4, v1.2.5, v1.2.6 neural, XGBoost v1.2.6
**Engine tested:** ML engine only (no physics engine)

## Motivation

Eval 07 showed XGBoost over Wav2Vec2 functionals beating every linear head on the internal
v1.2.6 held-out split — 0.1425 EER vs 0.2325 for the v1.2.6 neural head. That test set was
built by the same pipeline as the training set, from the same two corpora (ASVspoof5 train
partition + MLAAD English). The question this eval answers is narrow and important:

> **Does the XGBoost win survive on data nobody in this project assembled?**

Five public benchmarks, 200 files each (100 real + 100 fake), all four models scored on
**identical files with identical features**. This is also the first time v1.2.5 and v1.2.6 have
been measured against the same external benchmarks v1.2.4 was measured on in evals 01–05.

## Methodology

**One forward pass, two feature views.** All four models share the frozen
`facebook/wav2vec2-large-960h` front end, so each file is decoded once (mono, 16 kHz, front
crop / zero-pad to 10 s), pushed through the backbone once, and both feature views are derived
from the same hidden states:

- neural heads → mean-pooled 1024-d vector;
- XGBoost → 7 functionals (mean, std, min, max, p10, p50, p90) over the time axis = 7,168 dims,
  the exact construction in `train_xgboost_v1_2_6.py`.

The "mean" functional *is* the mean-pooled vector, so feature parity is structural rather than
best-effort. None of the three neural checkpoints carries fine-tuned backbone weights (the
harness aborts if one does), so the shared forward pass is valid for all four.

**Thresholds.** XGBoost was never calibrated to the production threshold of 0.55 — and, as §6
shows, neither v1.2.5 nor v1.2.6 neural still is on external audio. **EER is the primary metric
throughout.** Accuracy is reported at 0.55 *and* at each model's own EER threshold so that no
comparison is an artifact of threshold choice.

**Statistics.** 2,000 class-stratified bootstrap resamples per dataset (the 100/100 design is
preserved in each resample), the same resample indices reused across models so model
differences are *paired*. Reported: 95% CIs on EER and AUROC, paired CIs on EER differences,
and exact McNemar tests on per-file correctness at both threshold conventions.

**Subgroup labels** were recovered from provenance, not guessed (`build_subgroup_labels.py`):
generator for deepfake-audio by SHA256 match against the HuggingFace cache; attack ID, codec and
speaker for ASVspoof5 from the official `ASVspoof5.eval.track_1.tsv`; TTS engine for MLAAD from
the filename plus the original MLAAD path by SHA256 match; codec and speaker from the CodecFake
and In-the-Wild manifests. Zero files were left unlabelled, and all 200 ASVspoof5 labels agree
with the official protocol (0 mismatches).

## 1. Harness validation

Before any new model was scored, v1.2.4 was re-run through this harness and compared against
reports 01–05, which used `deployment_v1_2_4/inference.py`. **All five EERs reproduce exactly**;
accuracy reproduces within 0.02 (reports 01/03 quote accuracy at 0.55, reports 02/04/05 quote
`inference.py`'s accuracy at 0.50 — both conventions are shown).

| Dataset | Reported acc | Reproduced @0.55 | Reproduced @0.50 | Reported EER | Reproduced EER | Result |
|---|---|---|---|---|---|---|
| deepfake-audio | 29.0% | 29.0% | 25.0% | 0.850 | **0.850** | PASS |
| asvspoof5 | 75.5% | 76.5% | 75.5% | 0.250 | **0.250** | PASS |
| mlaad | 44.5% | 46.5% | 44.5% | 0.620 | **0.620** | PASS |
| in_the_wild | 72.0% | 72.0% | 72.5% | 0.260 | **0.260** | PASS |
| codecfake | 54.5% | 54.5% | 56.0% | 0.410 | **0.410** | PASS |

### Per-file agreement with the original v1.2.4 run

`origin/eval/public-dataset-test` also stores the **raw per-file v1.2.4 output** behind reports
01–05 (`results.json`, `results_asvspoof5.json`, `results_mlaad.json`, `results_in_the_wild.json`,
`results_codecfake.json`), which upgrades the check above from "the aggregates match" to "the
scores match".

Those arrays carry no filenames, and they are in the order `inference.py` enumerated files —
`Path.glob("*.*")`, i.e. filesystem order — whereas this harness enumerates with `sorted()`.
Compared position by position they disagree by up to 0.77, purely from ordering. Matched within
each class in sorted order (a bijection when both runs produced the same set of scores):

| Dataset | max abs Δ after class alignment | positions enumerated differently |
|---|---|---|
| deepfake-audio | 8.2 × 10⁻⁶ | 199/200 |
| ASVspoof5 eval | 8.6 × 10⁻⁷ | 200/200 |
| MLAAD | 8.6 × 10⁻⁷ | 198/200 |
| In-the-Wild | 1.2 × 10⁻⁶ | 198/200 |
| CodecFake | 1.7 × 10⁻⁶ | 198/200 |

**All 1,000 v1.2.4 scores reproduce to ≤ 8.2 × 10⁻⁶** (the original files store 4-decimal or
float32 values, so this is rounding, not drift). The two scoring paths are numerically the same
path. A consequence worth stating for the paired tests in §3: the v1.2.4 column in this eval *is*
the original v1.2.4 run, so every model-vs-v1.2.4 comparison here is already a paired comparison
against the published baseline.

The mapping direction was confirmed independently: four files were re-scored by a standalone
reimplementation of `deployment_v1_2_4/inference.py` (no harness code), and each agreed with the
harness's filename→score mapping to < 5 × 10⁻⁷.

> **Do not reuse the eval branch's per-file breakdowns.** `run_eval.py` and the
> `*_enriched.json` scripts rebuilt the file list with `sorted()` and zipped it against the
> glob-ordered score arrays, so their **per-file attributions are misaligned** —
> `results_all_engines.json` disagrees with a correct recomputation on 199 of 200 filenames, and
> the per-codec / per-speaker tables derived the same way inherit the error. This is the root
> cause of the eval-04 per-speaker discrepancy noted in §5. Class-wise aggregates (EER, accuracy,
> per-class score distributions) are **unaffected**, because they depend only on the set of
> scores within each class — which is why reports 01–05's headline numbers reproduce exactly.
> Subgroup baselines for v1.2.4 should be taken from `results_external_v1_2_6.json`, where they
> are recomputed from provenance-recovered labels.

**Decode-path check.** v1.2.4's original eval decoded FLAC with librosa; v1.2.5/v1.2.6/XGBoost
training used miniaudio. Both paths were run over all three FLAC datasets: **no score differs at
1e-6 resolution and every EER is identical**, so the decode path is not a confound
(`results_external_v1_2_6.json → decode_path_check`).

Together these mean the numbers below are produced by the same scoring path that produced the
published v1.2.4 baselines, and any difference between models is a difference between models.

## 2. Headline results

EER with 95% bootstrap CI. Lower is better; 0.500 is chance.

| Dataset | v1.2.4 | v1.2.5 | v1.2.6 neural | XGBoost v1.2.6 |
|---|---|---|---|---|
| deepfake-audio (6 commercial TTS) | 0.850 [0.800, 0.900] | 0.570 [0.490, 0.650] | 0.340 [0.270, 0.420] | **0.310 [0.250, 0.380]** |
| ASVspoof5 eval | **0.250 [0.180, 0.310]** | 0.330 [0.250, 0.400] | 0.340 [0.270, 0.410] | 0.430 [0.360, 0.510] |
| MLAAD (10 engines, blended) | 0.620 [0.540, 0.690] | 0.470 [0.400, 0.560] | 0.470 [0.400, 0.540] | **0.430 [0.360, 0.520]** |
| In-the-Wild | **0.260 [0.200, 0.340]** | 0.430 [0.360, 0.500] | 0.500 [0.420, 0.570] | 0.520 [0.440, 0.590] |
| CodecFake | **0.410 [0.330, 0.470]** | 0.450 [0.400, 0.540] | 0.580 [0.490, 0.640] | 0.470 [0.400, 0.550] |
| **macro-average EER** | 0.478 | 0.450 | 0.446 | **0.432** |

AUROC (threshold-free ranking quality; 0.5 is chance, below 0.5 means the model ranks fakes as
*more* real than real audio):

| Dataset | v1.2.4 | v1.2.5 | v1.2.6 neural | XGBoost v1.2.6 |
|---|---|---|---|---|
| deepfake-audio | 0.079 | 0.385 | 0.652 | **0.745** |
| ASVspoof5 eval | **0.824** | 0.761 | 0.756 | 0.618 |
| MLAAD (blended) | 0.341 | 0.528 | 0.578 | **0.601** |
| In-the-Wild | **0.811** | 0.610 | 0.524 | 0.463 |
| CodecFake | **0.637** | 0.544 | 0.450 | 0.528 |

Accuracy at both threshold conventions (the 0.55 column is **uncalibrated** for all three
post-v1.2.4 models — see §6):

| Dataset | | v1.2.4 | v1.2.5 | v1.2.6 | XGBoost |
|---|---|---|---|---|---|
| deepfake-audio | @0.55 / @EER | 29.0 / 15.5 | 47.5 / 43.5 | 50.0 / 66.0 | 50.0 / **69.0** |
| ASVspoof5 | @0.55 / @EER | 76.5 / **75.0** | 57.5 / 66.5 | 58.0 / 66.5 | 59.5 / 56.5 |
| MLAAD | @0.55 / @EER | 46.5 / 38.0 | 46.0 / 53.0 | 48.5 / 53.0 | 54.0 / **56.5** |
| In-the-Wild | @0.55 / @EER | 72.0 / **73.5** | 52.0 / 57.0 | 53.5 / 50.0 | 51.0 / 48.0 |
| CodecFake | @0.55 / @EER | 54.5 / **59.0** | 50.0 / 54.5 | 50.5 / 42.5 | 50.5 / 53.0 |

**Reading it:** the modern-TTS training paid off exactly where modern TTS is the attack
(deepfake-audio: 0.850 → 0.310 EER, a ~3× reduction; MLAAD 0.620 → 0.430). Everywhere else it
cost accuracy, and on the two datasets that best simulate deployment — In-the-Wild and CodecFake
— **v1.2.4 is still the best of the four models.** The macro-average hides this: it improves
monotonically across versions while three of five individual datasets get worse.

## 3. Does the XGBoost win generalise?

**No — not measurably.** XGBoost vs the v1.2.6 neural head, paired bootstrap on identical files:

| Dataset | ΔEER (neural − XGB) | 95% CI | Bootstrap p | McNemar p (@own EER thr) | Significant? |
|---|---|---|---|---|---|
| deepfake-audio | +0.030 | [−0.050, +0.100] | 0.600 | 0.504 | no |
| ASVspoof5 eval | −0.090 | [−0.160, −0.020] | 0.015 | 0.003 | **yes — XGBoost is worse** |
| MLAAD | +0.040 | [−0.040, +0.090] | 0.546 | 0.324 | no |
| In-the-Wild | −0.020 | [−0.110, +0.070] | 0.809 | 0.724 | no |
| CodecFake | +0.110 | [−0.010, +0.190] | 0.101 | 0.040 | borderline |

On four of five external datasets the tree-vs-linear difference is inside the noise at n = 200.
The only clearly significant difference runs **against** XGBoost (ASVspoof5 eval). Internally the
same comparison was 0.2325 → 0.1425, a −39% relative improvement, and XGBoost's internal
ASVspoof EER was 0.0485 versus 0.4300 here.

XGBoost vs v1.2.4 (the shipped production model) is the comparison that matters commercially:

| Dataset | ΔEER (v1.2.4 − XGB) | 95% CI | Significant? | Who wins |
|---|---|---|---|---|
| deepfake-audio | +0.540 | [+0.440, +0.620] | **yes** | XGBoost |
| MLAAD | +0.190 | [+0.060, +0.290] | **yes** | XGBoost |
| ASVspoof5 eval | −0.180 | [−0.280, −0.090] | **yes** | v1.2.4 |
| In-the-Wild | −0.260 | [−0.350, −0.140] | **yes** | v1.2.4 |
| CodecFake | −0.060 | [−0.170, +0.030] | no | — |

**Two significant wins, two significant losses, one tie.** The honest summary is not "XGBoost is
better" or "worse" but *differently specialised*: it moved the model's competence from
ASVspoof-style attacks and real-world voice clones towards modern TTS output.

### The most informative datasets

- **In-the-Wild** (deployment-quality deepfakes that fooled real audiences): v1.2.4 0.260 →
  v1.2.5 0.430 → v1.2.6 0.500 → XGBoost 0.520. Every version made it worse, monotonically, and
  the v1.2.4-vs-anything gaps are all significant. XGBoost's AUROC of 0.463 is below chance —
  on this data it carries no usable ranking signal at all.
- **CodecFake** (15 neural codecs, an attack family nobody trained on): v1.2.4 0.410 remains the
  best; v1.2.6 neural degrades significantly to 0.580 (below chance); XGBoost 0.470 is not
  significantly different from v1.2.4 either way. Codec-resynthesis artifacts were not learned
  by any version, and the newer models actively lost ground.

This is the finding to lead with in the paper: **the internal improvement from v1.2.4 → v1.2.6
did not transfer to independently collected data, and on deployment-like audio it inverted.**

## 4. MLAAD split: held-out vs seen engines

v1.2.5/v1.2.6 trained on MLAAD, so a blended MLAAD number would be partly a memorisation
measurement. Of the 10 external engines, 6 were held out (ElevenLabs v3, OpenAI TTS-1 HD,
Gemini 3.1 Flash, kokoro, f5-tts, sesame CSM) and 4 were training engines (suno bark, XTTS v2,
ChatTTS, MeloTTS). Both views share the same 100 real files.

| View | n | v1.2.4 | v1.2.5 | v1.2.6 | XGBoost |
|---|---|---|---|---|---|
| held-out engines (60 fakes) | 160 | 0.670 | 0.480 | 0.470 | **0.450** |
| seen (training) engines (40 fakes) | 140 | 0.520 | 0.470 | 0.470 | **0.430** |
| *gap (seen − held-out)* | | −0.150 | −0.010 | 0.000 | −0.020 |

**Contamination did not inflate the result.** For v1.2.5/v1.2.6/XGBoost the seen-engine EER is
within 0.02 of the held-out EER — well inside the CI width (±0.10). The modern-TTS gain is
generalisation to engine *families*, not recall of specific training files. (Note the sign of
v1.2.4's gap: it is *worse* on held-out engines, which is what an uncontaminated model looks
like when held-out engines are simply newer and harder.)

Literal file-level overlap could not be tested: `data/raw/mlaad` no longer exists locally and
the training file lists were lost with the machine that built them. As a hedge, the original
MLAAD filename of every external MLAAD file is recorded in `paper_data/subgroup_labels.json`,
so a rebuilt training pool can be diffed against it later.

## 5. Per-subgroup breakdown

### Per generator — deepfake-audio (EER vs all 100 real files)

| Generator | n | v1.2.4 | v1.2.5 | v1.2.6 | XGBoost |
|---|---|---|---|---|---|
| Speechify | 32 | 0.860 | 0.600 | 0.360 | **0.230** |
| ElevenLabs | 20 | 0.720 | 0.520 | **0.310** | 0.380 |
| Luvvoice | 14 | 0.870 | 0.570 | 0.270 | **0.170** |
| Hume AI | 13 | 0.910 | 0.600 | **0.360** | **0.360** |
| Amazon Polly | 12 | 0.950 | 0.630 | 0.190 | **0.220** |
| Hexgrad Kokoro | 9 | 0.870 | 0.650 | 0.570 | **0.550** |

Every generator improves monotonically across versions — the cleanest evidence in this eval that
the v1.2.5/v1.2.6 training programme did what it was designed to do. Kokoro remains the hardest.

### Per TTS engine — MLAAD (EER vs all real; ★ = training engine for v1.2.5/v1.2.6)

| Engine | v1.2.4 | v1.2.5 | v1.2.6 | XGBoost |
|---|---|---|---|---|
| f5-tts | 0.590 | 0.150 | 0.210 | **0.150** |
| MeloTTS ★ | 0.620 | 0.400 | 0.300 | **0.180** |
| ChatTTS ★ | 0.350 | **0.230** | 0.340 | 0.270 |
| kokoro | 0.890 | 0.750 | 0.470 | **0.290** |
| ElevenLabs v3 | 0.770 | 0.400 | 0.440 | **0.330** |
| Gemini 3.1 Flash | 0.880 | **0.440** | 0.470 | 0.450 |
| sesame CSM | **0.550** | 0.480 | 0.640 | 0.610 |
| suno bark ★ | **0.520** | 0.640 | 0.620 | 0.610 |
| XTTS v2 ★ | **0.630** | 0.680 | 0.640 | 0.610 |
| OpenAI TTS-1 HD | **0.660** | 0.690 | 0.610 | 0.620 |

Two of the four *training* engines (bark, XTTS v2) are still handled worse than by v1.2.4 —
further evidence against a memorisation explanation, and a sign that MLAAD engine coverage is
not what drives the gains. At n = 10 fakes per engine these rows are individually very noisy;
read them as a pattern, not as measurements.

### Per attack — ASVspoof5 eval (EER vs all real, attacks with n ≥ 6)

| Attack | n | v1.2.4 | v1.2.5 | v1.2.6 | XGBoost |
|---|---|---|---|---|---|
| A27 | 6 | **0.010** | 0.010 | 0.000 | 0.320 |
| A32 | 9 | **0.010** | 0.010 | 0.020 | 0.340 |
| A19 | 8 | 0.040 | **0.020** | 0.030 | 0.370 |
| A26 | 11 | 0.190 | 0.170 | **0.160** | 0.320 |
| A25 | 6 | **0.070** | 0.270 | 0.340 | 0.500 |
| A21 | 10 | **0.320** | 0.510 | 0.580 | 0.470 |
| A29 | 7 | **0.350** | 0.400 | 0.450 | 0.430 |
| A24 | 7 | 0.590 | 0.600 | **0.410** | 0.530 |
| A28 | 6 | 0.770 | 0.740 | **0.620** | 0.620 |
| A22 | 8 | **0.190** | 0.400 | 0.460 | 0.620 |

This is the sharpest single result in the eval. The three linear heads solve several ASVspoof
eval attacks almost perfectly (A27, A32, A19 at 0.00–0.04 EER); **XGBoost solves none of them**,
sitting at 0.32–0.37 on exactly those attacks while reporting 0.0485 EER on ASVspoof fakes in
the internal test. The trees learned the *train-partition* attack conditions, not the attack
family.

### Per codec — CodecFake (EER vs all real, 7 files per codec)

| Codec | v1.2.4 | v1.2.5 | v1.2.6 | XGBoost |
|---|---|---|---|---|
| encodec_24khz | **0.080** | 0.300 | 0.470 | 0.470 |
| audiodec_24k_320d | **0.210** | 0.310 | 0.520 | 0.410 |
| descript-audio-codec-44khz | **0.230** | 0.410 | 0.460 | 0.410 |
| funcodec en_libritts-16k-nq32ds640 | **0.230** | 0.440 | 0.400 | 0.330 |
| SpeechTokenizer (n=2) | **0.230** | 0.300 | 0.770 | 0.330 |
| academicodec_hifi_16k_320d_large_uni | **0.330** | 0.550 | 0.510 | 0.330 |
| funcodec en_libritts-16k-gr1nq32ds320 | **0.300** | 0.550 | 0.630 | 0.430 |
| academicodec_hifi_24k_320d | **0.410** | 0.610 | 0.580 | 0.560 |
| descript-audio-codec-16khz | **0.480** | 0.640 | 0.640 | 0.650 |
| descript-audio-codec-24khz | 0.870 | 0.730 | **0.490** | 0.710 |

v1.2.4 is best or tied-best on 9 of 10 codecs shown. Eval 05's conclusion that the model is
"blind to the entire codec paradigm" needs a small amendment: v1.2.4 had *some* codec signal
(Encodec 0.08, AudioDec 0.21), and the later versions lost it.

### Per speaker — In-the-Wild

Per-speaker groups are small (largest fake group n = 17), so only the aggregate is trustworthy;
the full table is in `results_external_v1_2_6.json`. One pattern is stark: v1.2.4 detects the
Alec Guinness fakes at 0.07 EER (17 files, the largest fake group) while v1.2.5/v1.2.6/XGBoost
sit at 0.49/0.57/0.58 on the same files.

> **Correction to eval 04.** Report 04's corrected per-speaker false-positive table does not
> survive a recomputation from `test_data/in_the_wild/manifest.json`. The cause is now known and
> is the ordering artifact documented in §1: the enrichment script paired `sorted()` filenames
> with glob-ordered scores. The aggregate is exactly right (7 false positives at 0.55, accuracy
> 72.0%) and the speaker composition matches, but the per-speaker attribution does not: the seven v1.2.4 false positives are Ayn Rand ×2, and one
> each for Calvin Coolidge, Louis Farrakhan, 2Pac, Barack Obama and Nick Offerman. **Donald
> Trump is 0/17 and Bernie Sanders 0/8**, not 3/17 and 2/8 as report 04 states. Report 04's
> "no political-speech bias" conclusion holds — more strongly, in fact.

## 6. Calibration: the production threshold is no longer usable

Fraction of **genuine** external audio flagged as fake at the production threshold of 0.55:

| Dataset | v1.2.4 | v1.2.5 | v1.2.6 | XGBoost |
|---|---|---|---|---|
| deepfake-audio | 43% | 98% | 96% | 99% |
| ASVspoof5 eval | 19% | 75% | 76% | 74% |
| MLAAD (same real files as above) | 19% | 75% | 76% | 74% |
| In-the-Wild | 7% | 94% | 92% | 96% |
| CodecFake | 1% | 100% | 99% | 99% |

Median score assigned to real audio:

| Dataset | v1.2.4 | v1.2.5 | v1.2.6 | XGBoost |
|---|---|---|---|---|
| deepfake-audio | 0.504 | 0.921 | 0.970 | 1.000 |
| ASVspoof5 eval | 0.336 | 0.704 | 0.794 | 0.998 |
| In-the-Wild | 0.279 | 0.912 | 0.985 | 1.000 |
| CodecFake | 0.249 | 0.953 | 0.998 | 1.000 |

The expected caveat was that XGBoost is uncalibrated. What the data actually shows is that **all
three post-v1.2.4 models are uncalibrated on external audio** — their score distributions have
migrated so far up that at 0.55 they classify nearly everything as fake, and the fact that
CodecFake accuracy sits at 50.5% is only because they flag all 100 fakes too. Any deployment of
v1.2.5, v1.2.6 or XGBoost needs a fresh threshold sweep on data resembling its input, and
accuracy at 0.55 must never be quoted for them without this caveat. XGBoost is the extreme case:
on CodecFake every one of the 100 fakes scores exactly 1.000 (std 0.000).

## 7. Statistical power — what this eval can and cannot resolve

At n = 200 (100 real + 100 fake) the 95% bootstrap CI on a single EER is roughly ±0.07–0.10 wide,
and on a *paired* EER difference roughly ±0.08. Practically:

- Differences **> 0.15 EER** are resolvable. All four v1.2.4-vs-newer gaps on deepfake-audio,
  MLAAD, ASVspoof5 and In-the-Wild fall in this class and are significant.
- Differences **< 0.10 EER** are not. This includes every XGBoost-vs-v1.2.6-neural comparison
  except ASVspoof5 — so the internal headline result (trees > linear head) is neither confirmed
  nor refuted externally; it is simply not detectable at this sample size.
- Per-subgroup rows (n = 2–32 fakes) are indicative only. No significance is claimed for them.

To resolve a 0.05 EER difference at this design would need roughly 1,500–2,000 files per dataset.
That is the single cheapest methodological upgrade available for the next round.

## 8. Limitations

1. **n = 200 per dataset.** See §7. Point-estimate rankings within ~0.10 EER are not meaningful.
2. **ASVspoof5 and MLAAD share their 100 real files** (MLAAD is fake-only, so its real audio is
   ASVspoof5 bonafide). Their real-side statistics are identical by construction and the two
   datasets are not independent evidence.
3. **MLAAD is partly contaminated** for v1.2.5/v1.2.6 — handled by the split in §4, but literal
   file-level overlap is unverifiable (§4).
4. **The classifier comparison is not perfectly controlled.** The neural heads were warm-started
   from v1.2.4; XGBoost was trained from scratch. Internally this disadvantaged XGBoost, which
   won anyway; externally it means "tree vs linear" and "scratch vs warm start" cannot be
   separated.
5. **One real corpus in training.** All real training audio is ASVspoof5 bonafide, so the models
   may have learned "sounds like ASVspoof bonafide" rather than "sounds human". The calibration
   collapse in §6 — where genuine YouTube, VCTK and In-the-Wild audio is flagged 92–100% of the
   time — is exactly what that failure mode looks like.
6. **The internal v1.2.6 test set is gone** (ephemeral machine), so internal and external results
   cannot be compared per-file, only summary-to-summary.

## 9. What to do next

1. **Add real audio from multiple corpora** and re-train. This is now the top priority, ahead of
   any further classifier work: §6 says the current models fail on genuine audio from unseen
   corpora, which is a data problem, not a classifier problem.
2. **Re-sweep the production threshold** for any model past v1.2.4 before it goes anywhere near
   deployment.
3. **Score fusion** (eval 07's next step) is still worth testing, but it must be evaluated
   externally from the start — v1.2.4 and XGBoost make errors on genuinely different subsets
   (v1.2.4 wins ASVspoof5/In-the-Wild, XGBoost wins deepfake-audio/MLAAD), which is exactly the
   profile fusion exploits.
4. **Grow the external sets to ~1,500 files each** so that 0.05-EER differences become
   resolvable.
5. **Add CodecFake-style codec resynthesis to training** — still absent, still undetected, and
   the newer models are worse at it than v1.2.4.

## Artifacts

- `eval_external_datasets.py` — the harness (any checkpoints × any `test_data/` subset)
- `validate_v1_2_4_perfile.py` — per-file check against the original v1.2.4 run on the eval branch
- `build_subgroup_labels.py` → `paper_data/subgroup_labels.json` — provenance-recovered labels
- `results_external_v1_2_6.json` — full metric set incl. ROC/DET points, CIs, comparisons,
  harness validation, decode-path check
- `paper_data/external_scores.csv` — raw per-file scores, 1,000 rows × 4 models
- `paper_data/PAPER_NOTES.md` — index for the white-paper session
- Test data (untracked, local): `test_data/{deepfake-audio,asvspoof5,mlaad,in_the_wild,codecfake}/`

Reproduce:

```
pip install xgboost            # macOS also needs: brew install libomp
python build_subgroup_labels.py
python eval_external_datasets.py --bootstrap 2000
python validate_v1_2_4_perfile.py --write     # per-file check vs the original v1.2.4 run
# decode-path check
python eval_external_datasets.py --datasets deepfake-audio asvspoof5 mlaad --decode librosa \
  --out /tmp/results_librosa.json --csv /tmp/scores_librosa.csv
```
