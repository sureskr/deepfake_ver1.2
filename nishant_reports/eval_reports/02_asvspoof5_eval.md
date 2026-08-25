# Eval 02: ASVspoof5 Evaluation Set

**Date:** 2026-05-18
**Conducted by:** Nishant Jain, with Claude
**Branch:** `eval/public-dataset-test`
**Model version:** v1.2.4
**Engine tested:** ML Engine only

## Dataset

**Source:** [jungjee/asvspoof5](https://huggingface.co/datasets/jungjee/asvspoof5) on HuggingFace

ASVspoof5 is a large-scale spoofing and deepfake detection challenge dataset with ~680K evaluation utterances across 2,000+ speakers. The eval set uses disjoint speakers and separate attack generators from the training set, making it a proper held-out benchmark even if ASVspoof5 train data was used during model training.

We randomly sampled 200 files (100 bonafide, 100 spoof) from the first evaluation archive (`flac_E_aa.tar`, containing 68,188 files: 14,171 bonafide + 54,017 spoof). Audio is FLAC format. Labels come from the official Track 1 protocol (`ASVspoof5.eval.track_1.tsv`).

- **Bonafide audio:** Crowdsourced real speech from diverse speakers and acoustic conditions
- **Spoof audio:** Generated using 20+ attack types (codec-processed, adversarial, TTS/VC), labeled with attack IDs (A21-A32) and adaptation conditions

## Methodology

We ran the ML engine (v1.2.4) on the 200 files using `deployment_v1_2_4/inference.py`. The model uses frozen Wav2Vec2-large-960h as a feature extractor with a linear classification head. Inference was run on CPU.

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **75.5%** |
| **Precision** | 75.8% |
| **Recall** | 75.0% |
| **F1-Score** | 75.4% |
| **EER** | 25.0% |
| **EER Threshold** | 0.4997 |

At production threshold (0.55):
- 72/100 spoof files correctly detected
- 19/100 bonafide files falsely flagged (FPR 19%)

Score distributions:
- **Bonafide:** mean 0.377, median 0.336
- **Spoof:** mean 0.678, median 0.727

## Comparison with Reported Metrics

| Metric | Reported (RitW, 25K files) | This eval (ASVspoof5, 200 files) |
|--------|---------------------------|----------------------------------|
| Accuracy | 73.3% | 75.5% |
| EER | 27.6% | 25.0% |
| FPR | 11.9% | 19.0% (at 0.55) |

Accuracy and EER are consistent with the model's reported performance. FPR is higher here, likely due to the small sample size and the different acoustic conditions in ASVspoof5 (codec-processed audio, diverse recording environments).

## Takeaways

The ML engine performs as expected on ASVspoof5 eval data. The score separation between real (mean 0.38) and fake (mean 0.68) is clear, unlike the previous eval on the HuggingFace dataset where scores were inverted. This confirms the model generalizes to unseen speakers and attack variants within the ASVspoof family, but struggles with out-of-distribution TTS engines (ElevenLabs, Speechify, etc.) as shown in Eval 01.
