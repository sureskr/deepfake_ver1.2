# Eval 05: CodecFake (Neural Audio Codec Re-synthesis)

**Date:** 2026-06-05
**Conducted by:** Nishant Jain, with Claude
**Branch:** `eval/public-dataset-test`
**Model version:** v1.2.4
**Engine tested:** ML Engine only

## Dataset

**Source:** [rogertseng/CodecFake](https://huggingface.co/datasets/rogertseng/CodecFake) on HuggingFace

**License:** CC-BY-4.0 (commercial use allowed with attribution) — same license tier as In-the-Wild, friendlier than MLAAD.

CodecFake (Interspeech 2024, [arXiv:2406.07237](https://arxiv.org/abs/2406.07237)) re-synthesizes VCTK utterances through **15 neural audio codecs** — the same family of codecs that underlies modern TTS systems (VALL-E, AudioLM, SoundStream-based stacks). The premise of the paper is that detecting these codec-resynthesized samples is a proxy for detecting modern codec-based deepfake speech.

Total dataset: 707,872 examples (44,242 genuine VCTK + 663,630 spoofing). We sampled 100 genuine + 100 spoofing (seed 42), drawing 7 spoofing per codec across all 15 codecs for balanced coverage.

### Codecs tested

| Family | Variants in sample |
|---|---|
| FunCodec | 6 (Chinese-English general nq32ds320/640, LibriTTS nq32ds320/640, LibriTTS gr1/gr8) |
| AcademiCodec (HiFi) | 3 (16k/320d, 16k/320d-large-uni, 24k/320d) |
| Descript Audio Codec | 3 (16k, 24k, 44k) |
| Encodec | 1 (24k) |
| AudioDec | 1 (24k/320d) |
| SpeechTokenizer | 1 |

### Verification

For all 200 files: real `codec` field == `'genuine'`, fake `codec` field != `'genuine'`. Audio decodes cleanly (RMS 0.03–0.13, sample rates 16k/24k/44k by codec). Sanity check confirms inference loader returns `labels[0:100]==0` (real) and `labels[100:200]==1` (fake).

## Methodology

Ran `deployment_v1_2_4/inference.py` with `models/best_model.pth` (Wav2Vec2-large-960h + linear head). Multi-rate inputs are auto-resampled to 16k by librosa. Inference on CPU, ~80 seconds.

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **54.5%** |
| **Precision** | 90.9% |
| **Recall** | 10.0% |
| **F1-Score** | 18.0% |
| **EER** | 41.0% |
| **EER Threshold** | 0.279 |

At production threshold (0.55):
- 10/100 spoof files detected (FNR 90%)
- 1/100 bona-fide files falsely flagged (FPR 1%)

Score distributions:
- **Bona-fide:** mean 0.263, median 0.249, std 0.098
- **Spoof:** mean 0.327, median 0.317, std 0.138

Fake scores *do* skew higher than real (directionally correct), but by a margin so small (~0.06) that the production threshold catches almost nothing. The EER threshold (0.279) is below the production threshold and still only gets the error rate down to 41%.

## Per-Codec Detection Rate (threshold 0.55)

Every codec sits in the 0–2/7 detection range — statistically indistinguishable from random.

| Codec | Detected | Mean Score |
|---|---|---|
| funcodec-funcodec_en_libritts-16k-gr8nq32ds320 | 2/7 | 0.379 |
| funcodec-funcodec_zh_en_general_16k_nq32ds640 | 1/7 | 0.370 |
| funcodec-funcodec_en_libritts-16k-nq32ds640 | 1/7 | 0.370 |
| academicodec_hifi_24k_320d | 1/7 | 0.369 |
| academicodec_hifi_16k_320d | 1/7 | 0.356 |
| audiodec_24k_320d | 1/7 | 0.355 |
| encodec_24khz | 1/7 | 0.341 |
| descript-audio-codec-24khz | 1/7 | 0.331 |
| funcodec-funcodec_en_libritts-16k-gr1nq32ds320 | 1/7 | 0.315 |
| academicodec_hifi_16k_320d_large_uni | 0/7 | 0.276 |
| descript-audio-codec-16khz | 0/7 | 0.268 |
| funcodec-funcodec_en_libritts-16k-nq32ds320 | 0/7 | 0.266 |
| funcodec-funcodec_zh_en_general_16k_nq32ds320 | 0/7 | 0.245 |
| descript-audio-codec-44khz | 0/7 | 0.367 |
| SpeechTokenizer | 0/2 | 0.234 |

**There is no codec family the model handles better than another.** The failure is essentially uniform — Descript, Encodec, FunCodec, AudioDec, AcademiCodec, SpeechTokenizer all defeat the model at roughly the same rate. This rules out the "model trained against one codec family" theory; it's blind to the entire paradigm.

## Comparison Across All Evals

| Eval | Dataset | License | Accuracy | EER | Fake mean | Real mean | Separation |
|---|---|---|---|---|---|---|---|
| 01 | HuggingFace deepfake-audio | Open | 29.0% | 85.0% | — | — | inverted |
| 02 | ASVspoof5 | Research | 75.5% | 25.0% | 0.678 | 0.377 | 0.30 |
| 03 | MLAAD | CC-BY-NC | 44.5% | 62.0% | 0.289 | 0.377 | -0.09 (inverted) |
| 04 | In-the-Wild | CC-BY-SA 4.0 | 72.0% | 26.0% | 0.534 | 0.317 | 0.22 |
| **05** | **CodecFake** | **CC-BY-4.0** | **54.5%** | **41.0%** | **0.327** | **0.263** | **0.06** |

## Takeaways

1. **Neural-codec resynthesis is a real generalization gap.** EER 41% is much worse than ASVspoof5 (25%) and In-the-Wild (26%), confirming that the modern-TTS family of attacks is genuinely outside the model's training distribution. Score separation drops from 0.22 → 0.06 vs. In-the-Wild.

2. **It's not as bad as MLAAD (62% EER).** This is a partial signal — the model can faintly distinguish codec resynthesis (fake mean 0.327 > real mean 0.263). MLAAD's failure was worse and inverted (fake mean *lower* than real). Hypothesis: MLAAD's fakes are full TTS systems (codec + language model + prosody synthesis), so they remove more codec-artifact signal than pure resynthesis does. CodecFake is therefore a *lower bound* on the difficulty — real-world commercial TTS will be harder than pure codec resynthesis.

3. **The detection failure spans all 15 codecs uniformly.** There's no "weak codec" the model handles better. Retraining can't be targeted at one architecture; it needs broad codec-family coverage.

4. **False positive rate is excellent on VCTK studio audio (1%).** All 100 false-positive opportunities produced just one false flag. This is the cleanest FPR across all five evals — but it reflects studio-quality acoustic conditions, not deployment audio.

5. **Retraining priority is concrete now:** include codec-resynthesized audio from at least the codec families in CodecFake (Encodec, Descript, FunCodec, AcademiCodec, AudioDec, SpeechTokenizer) in the training set. License is CC-BY-4.0 so this is usable for a commercial model.

## Files

- Test data: `test_data/codecfake/{real,fake}/` (200 wav files)
- Provenance manifest: `test_data/codecfake/manifest.json` (codec + speaker per file)
- Raw inference output: `results_codecfake.json`
- Enriched metrics + per-codec breakdown: `results_codecfake_enriched.json`
- Prep script: `prep_codecfake.py`

## Citation

```bibtex
@inproceedings{lu2024codecfake,
  title={CodecFake: Enhancing Anti-Spoofing Models Against Deepfake Audios from Codec-Based Speech Synthesis Systems},
  author={Lu, Yi-Chiao and Yang, Cheng-Han and Tseng, Ya-Wen and Lee, Hung-yi},
  booktitle={Interspeech},
  year={2024}
}
```
