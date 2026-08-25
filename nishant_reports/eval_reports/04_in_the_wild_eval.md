# Eval 04: In-the-Wild (Real-World Audio Deepfakes)

**Date:** 2026-06-05
**Conducted by:** Nishant Jain, with Claude
**Branch:** `eval/public-dataset-test`
**Model version:** v1.2.4
**Engine tested:** ML Engine only

## Dataset

**Source:** [mueller91/In-The-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) on HuggingFace

**License:** CC-BY-SA 4.0 (commercial use allowed with attribution + share-alike). Unlike AV-Deepfake1M (academic-only EULA), this license fits a commercial product benchmark.

In-the-Wild contains 31,779 audio clips of 58 celebrities and politicians, collected from publicly available YouTube videos, podcasts, and social media. Total 37.9 hours (20.7 real / 17.2 fake). The dataset accompanies the paper *"Does Audio Deepfake Detection Generalize?"* ([arXiv:2203.16263](https://arxiv.org/abs/2203.16263)).

Unlike ASVspoof5 (lab-clean studio TTS) and MLAAD (standard TTS-engine output), In-the-Wild fakes are **deployment-quality** — deepfakes that already fooled real human audiences on the internet. Many target deceased figures (Alec Guinness, Ronald Reagan, JFK) or political figures with abundant training audio (Obama, Trump, Sanders).

We randomly sampled 200 files (100 bona-fide + 100 spoof), seed 42. Sample composition:
- **Real top speakers:** Barack Obama (19), Donald Trump (17), Bernie Sanders (8), Ayn Rand (7), Ronald Reagan (7)
- **Fake top speakers:** Alec Guinness (17), Bernie Sanders (10), Ayn Rand (8), Bill Clinton (7), Christopher Hitchens (5)

Verification: label correctness checked against the dataset's `meta.csv` for all 200 files (zero mismatches), plus SHA256 byte-identity spot-check on 10 random files vs. the source zip — all match.

## Methodology

Ran the ML engine (v1.2.4) on the 200 files using `deployment_v1_2_4/inference.py` with `models/best_model.pth` (frozen Wav2Vec2-large-960h + linear classification head). Inference on CPU, ~80 seconds total.

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **72.0%** |
| **Precision** | 87.9% |
| **Recall** | 51.0% |
| **F1-Score** | 64.6% |
| **EER** | 26.0% |
| **EER Threshold** | 0.389 |

At production threshold (0.55):
- 51/100 spoof files correctly detected
- 7/100 bona-fide files falsely flagged (FPR 7.0%)
- 49/100 spoofs missed (FNR 49.0%)

Score distributions:
- **Bona-fide:** mean 0.317, median 0.279, std 0.125
- **Spoof:** mean 0.534, median 0.561, std 0.200

**Score directionality is correct** — fakes score higher than reals on average, unlike MLAAD (where the model inverted) or HuggingFace garystafford eval. However, the fake distribution straddles the 0.55 threshold (median 0.561, std 0.200), which is why recall collapses to 51% — half the fakes land just below.

## Per-Speaker Detection Rate (threshold 0.55)

> **Correction notice (2026-08-24, supersedes 2026-06-05):** the per-speaker tables in this report have now been wrong twice, both times for the same underlying reason. `deployment_v1_2_4/inference.py` enumerated files with `Path.glob()` (filesystem order), while `run_eval.py` and the enriched-metrics scripts rebuilt the file list with `sorted()` and zipped it onto those arrays — so per-file scores were attributed to the wrong filenames. The 2026-06-05 correction was itself derived from `results_in_the_wild_enriched.json`, which inherits the same defect, and so remained incorrect.
>
> **The tables below are recomputed from provenance-recovered per-file attributions in `results_external_v1_2_6.json`**, whose scores were validated per-file against the original v1.2.4 run to within 8.2e-6. Overall metrics — accuracy, EER, precision/recall, score distributions — depend only on the *set* of scores within each class and were never affected; they are unchanged and reproduce exactly. Do not use `results_in_the_wild_enriched.json` or `results_all_engines.json` for per-file or per-speaker analysis.

**Fakes — detection rate by speaker (n ≥ 4):**

| Speaker | Detected | Mean Score |
|---------|----------|------------|
| Alec Guinness | 14/17 (82%) | 0.641 |
| Bernie Sanders | 3/10 (30%) | 0.483 |
| Ayn Rand | 1/8 (12%) | 0.394 |
| Bill Clinton | 0/7 (0%) | 0.340 |
| Ronald Reagan | 3/5 (60%) | 0.532 |
| Christopher Hitchens | 3/5 (60%) | 0.584 |
| The Notorious B.I.G. | 2/5 (40%) | 0.413 |
| Mark Zuckerberg | 4/5 (80%) | 0.663 |
| Alan Watts | 2/4 (50%) | 0.579 |

**Reals — false-flag rate by speaker (n ≥ 4):**

| Speaker | Flagged | Mean Score |
|---------|---------|------------|
| Barack Obama | 1/19 (5%) | 0.279 |
| Donald Trump | 0/17 (0%) | 0.245 |
| Bernie Sanders | 0/8 (0%) | 0.286 |
| Ayn Rand | 2/7 (29%) | 0.412 |
| Ronald Reagan | 0/7 (0%) | 0.337 |
| Bill Clinton | 0/5 (0%) | 0.310 |
| Alec Guinness | 0/5 (0%) | 0.295 |
| Louis Farrakhan | 1/4 (25%) | 0.373 |

**Real political speech is clean.** Donald Trump (0/17), Bernie Sanders (0/8), Ronald Reagan (0/7), Bill Clinton (0/5) and Alec Guinness (0/5) all have zero false flags. The non-zero groups at n >= 4 are Ayn Rand (2/7, 29%), Louis Farrakhan (1/4, 25%) and Barack Obama (1/19, 5%); the remaining three false positives fall in speakers with n < 4 (2Pac 1/1, Calvin Coolidge 1/2, Nick Offerman 1/3). All seven are scattered across the pool rather than concentrated in any one group.

**Detection is strongest on the highest-profile cloning targets.** Alec Guinness (deceased 2000) fakes are detected 82% and Mark Zuckerberg 80% — both well above the overall 51% recall. The weak groups are Bill Clinton (0/7) and Ayn Rand (1/8), where the model catches almost nothing, and the spread across speakers (0% to 82%) is far wider than the headline recall suggests.

## Comparison Across All Evals

| Eval | Dataset | License | Accuracy | EER | Score directionality |
|------|---------|---------|----------|-----|---------------------|
| 01 | HuggingFace deepfake-audio | Open | 29.0% | 85.0% | Inverted |
| 02 | ASVspoof5 | Research | 75.5% | 25.0% | Correct |
| 03 | MLAAD | CC-BY-NC | 44.5% | 62.0% | Inverted |
| **04** | **In-the-Wild** | **CC-BY-SA 4.0** | **72.0%** | **26.0%** | **Correct** |

## Takeaways

1. **The model generalizes to real-world audio** at the EER level — 26.0% EER on In-the-Wild is essentially identical to 25.0% on ASVspoof5. This is the first cross-domain eval where the model behaves consistently with its in-distribution numbers.

2. **High-precision, low-recall regime.** At threshold 0.55, the model catches only 51% of fakes but flags real audio incorrectly just 7% of the time (precision 87.9%). Useful as a **fake-confirmer** (when it says fake, trust it) but not as a **fake-detector** (it misses half).

3. **No political-speech bias.** The seven false positives are scattered rather than concentrated in political speech: the two most-sampled political speakers, Trump (0/17) and Bernie Sanders (0/8), have zero false flags, as do Reagan (0/7) and Bill Clinton (0/5). The highest single-speaker rate at n >= 4 is Ayn Rand (2/7). The model's FPR is low and broadly uniform across speaker groups — production-threshold conservatism, not bias.

4. **The MLAAD failure was specifically modern TTS, not real-world audio.** In-the-Wild's fakes are mostly 2018-2022-era voice clones (Tortoise, ElevenLabs v1, RVC, etc.). The model handles these directionally correctly. The MLAAD result (44.5%, inverted scores) was specifically a failure against 2024-era commercial TTS (ElevenLabs v3, OpenAI TTS-1 HD, Gemini Flash TTS). The retraining priority is modern TTS, not real-world audio coverage.

5. **License-fit for commercial benchmarking.** CC-BY-SA 4.0 is the most permissive of the four datasets evaluated and the only one usable in external/marketing-facing materials (with attribution + cite Müller et al. 2022).

## Files

- Test data: `test_data/in_the_wild/{real,fake}/` (200 wav files)
- Provenance manifest: `test_data/in_the_wild/manifest.json` (original filenames + speakers)
- Raw inference output: `results_in_the_wild.json`
- Enriched metrics + per-speaker breakdown: `results_external_v1_2_6.json` (correctly attributed; see correction notice)
- ~~`results_in_the_wild_enriched.json`~~ — **per-file attribution is misaligned; class-wise aggregates only**
- Prep script: `prep_in_the_wild.py`

## Citation

```bibtex
@article{muller2022does,
  title={Does audio deepfake detection generalize?},
  author={M{\"u}ller, Nicolas M and Czempin, Pavel and Dieckmann, Franziska and Froghyar, Adam and B{\"o}ttinger, Konstantin},
  journal={arXiv preprint arXiv:2203.16263},
  year={2022}
}
```
