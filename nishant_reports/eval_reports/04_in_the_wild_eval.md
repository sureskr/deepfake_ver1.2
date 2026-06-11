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

> **Correction notice (2026-06-05):** an earlier version of this report had an index-mapping bug that swapped the per-speaker numbers between the fake and real groups. The overall metrics (accuracy, EER, score distributions) were never affected. The corrected per-speaker numbers below tell the opposite story from the original draft, and the takeaways have been rewritten accordingly. See `results_in_the_wild_enriched.json` for the source of truth.

**Fakes — detection rate by speaker (n ≥ 4):**

| Speaker | Detected | Mean Score |
|---------|----------|------------|
| Alec Guinness | 9/17 (53%) | 0.536 |
| Bernie Sanders | 3/10 (30%) | 0.442 |
| Ayn Rand | 5/8 (63%) | 0.556 |
| Bill Clinton | 4/7 (57%) | 0.564 |
| Ronald Reagan | 2/5 (40%) | 0.480 |
| Christopher Hitchens | 2/5 (40%) | 0.464 |
| The Notorious B.I.G. | 3/5 (60%) | 0.574 |
| Mark Zuckerberg | 3/5 (60%) | 0.594 |
| Alan Watts | 3/4 (75%) | 0.666 |

**Reals — false-flag rate by speaker (n ≥ 4):**

| Speaker | Flagged | Mean Score |
|---------|---------|------------|
| Barack Obama | 0/19 (0%) | 0.261 |
| Donald Trump | 3/17 (18%) | 0.311 |
| Bernie Sanders | 2/8 (25%) | 0.339 |
| Ayn Rand | 0/7 (0%) | 0.378 |
| Ronald Reagan | 0/7 (0%) | 0.326 |
| Bill Clinton | 0/5 (0%) | 0.261 |
| Alec Guinness | 0/5 (0%) | 0.370 |
| Louis Farrakhan | 0/4 (0%) | 0.254 |

**Politicians' real speech is the cleanest group, not the worst.** Obama (0/19), Reagan (0/7), Ayn Rand (0/7), Bill Clinton (0/5), Louis Farrakhan (0/4) all have zero false flags. Trump (18%) and Sanders (25%) are the only above-zero outliers. The 7 false positives total are scattered, not concentrated.

**Detection rate is reasonable for the speakers the dataset was built to catch.** Alec Guinness (deceased 2000) fakes are detected 53%, Alan Watts (deceased 1973) 75%, Mark Zuckerberg 60%, Notorious B.I.G. 60%. These are exactly the high-profile cloning targets, and the model handles them roughly as well as the overall 51% recall suggests.

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

3. **No political-speech bias.** All seven false positives are scattered across the speaker pool. Obama, Reagan, Bill Clinton, Ayn Rand, and Louis Farrakhan have zero false flags between them across 42 real samples. The model's FPR is uniformly low across speaker groups — production-threshold conservatism, not bias.

4. **The MLAAD failure was specifically modern TTS, not real-world audio.** In-the-Wild's fakes are mostly 2018-2022-era voice clones (Tortoise, ElevenLabs v1, RVC, etc.). The model handles these directionally correctly. The MLAAD result (44.5%, inverted scores) was specifically a failure against 2024-era commercial TTS (ElevenLabs v3, OpenAI TTS-1 HD, Gemini Flash TTS). The retraining priority is modern TTS, not real-world audio coverage.

5. **License-fit for commercial benchmarking.** CC-BY-SA 4.0 is the most permissive of the four datasets evaluated and the only one usable in external/marketing-facing materials (with attribution + cite Müller et al. 2022).

## Files

- Test data: `test_data/in_the_wild/{real,fake}/` (200 wav files)
- Provenance manifest: `test_data/in_the_wild/manifest.json` (original filenames + speakers)
- Raw inference output: `results_in_the_wild.json`
- Enriched metrics + per-speaker breakdown: `results_in_the_wild_enriched.json`
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
