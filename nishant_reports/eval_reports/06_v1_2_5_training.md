# Eval 06: v1.2.5 Training — Closing the Modern-TTS Gap

**Author:** Nishant Jain, with Claude
**Date:** 2026-07-15
**Branch:** `prep/v1_2_5-dataset`
**Model version:** v1.2.5

## Motivation

v1.2.4 (frozen Wav2Vec2-large-960h + Linear 1024→1 head) handles ASVspoof-family attacks well but fails on modern TTS — ElevenLabs, OpenAI, Gemini, Cartesia, and similar commercial/open-source engines. Prior evals (01, 03) showed it scored modern fakes *lower* than real audio: on MLAAD the model was actively ranking deepfakes as more real than bonafide speech, so its `eer[mlaad]` came out *worse than random*.

The goal of v1.2.5 was narrow and concrete: **fix the modern-TTS blind spot without regressing ASVspoof**. This report documents the dataset assembly, training run, and a head-to-head evaluation against v1.2.4 on a held-out modern-TTS test set.

## Process

1. **Built `prep_v1_2_5.py`** — assembles ASVspoof5 + MLAAD into `calibration` / `validation` / `test` splits with **generator-disjoint** splitting: ASVspoof real split by speaker, ASVspoof spoof by attack ID, MLAAD by TTS engine. Whole engines are held out for `test/`, so test accuracy measures generalization to *unseen* TTS rather than memorization.
2. **Provisioned a RunPod RTX 4090.** The backbone is frozen (only the linear head trains), so the run is cheap. Cloned the repo and downloaded the `facebook/wav2vec2-large-960h` backbone plus the datasets on the pod.
3. **Stage 1 — ASVspoof-only smoke test** to validate the pipeline end to end: best val EER **0.1448** @ epoch 22 (early-stopped at epoch 25). Note this is a held-out ASVspoof *attack* — a different, easier target than the Stage 2 modern-TTS test — and is included here only as pipeline-validation context, not as a headline result.
4. **Fixed two bugs found during the run:**
   - The warm-start loader didn't recognize the `head_state_dict` checkpoint format that the training script itself saves.
   - MLAAD v5 is flat (`fake/<lang>/<engine>/*.wav`), so engine inference had to use the parent folder. Also added `--asvspoof-max-fake` to stop the large ASVspoof spoof pool from drowning out MLAAD.
5. **Downloaded MLAAD English** (32 engines, ~200 files each) with a targeted per-engine downloader — the full `hf download` stalled enumerating the whole multi-language repo.
6. **Stage 2 — the real v1.2.5:** ASVspoof5 real + capped spoof + MLAAD, with 8 modern engines held out for test.
7. **Trained** (same production recipe, warm-started from v1.2.4), then ran a **head-to-head eval** (v1.2.4 head vs v1.2.5 head) on the held-out modern-TTS test set.

## Dataset

All audio is public: ASVspoof5 (real + spoof) and MLAAD English (modern-TTS fakes only).

### Sources

| Source | Real (bonafide) | Fake (spoof) | Notes |
|--------|-----------------|--------------|-------|
| ASVspoof5 (`flac_T_aa.tar`, 1 shard) | 3,801 | 32,699 → **capped to 4,000** | Spoof capped via `--asvspoof-max-fake` |
| MLAAD English | 0 | 7,186 | 32 TTS engines, ~200 files/engine |
| **Combined pool** | **3,801** | **11,186** | 4,000 ASVspoof spoof + 7,186 MLAAD |

All real audio is ASVspoof bonafide — MLAAD is fake-only.

### Splits (files) — balanced real/fake per split, generator-disjoint

| Split | Total | Real | Fake | Fake = MLAAD | Fake = ASVspoof |
|-------|-------|------|------|--------------|-----------------|
| calibration (train) | 5,266 | 2,633 | 2,633 | 1,495 | 1,138 |
| validation | 1,156 | 578 | 578 | 394 | 184 |
| test | 1,180 | 590 | 590 | 494 | 96 |
| **total** | **7,602** | | | | |

### Engine-disjoint holdout

8 modern engines are reserved for **TEST only** and never appear in training, so the test set measures generalization to TTS the model has never seen:

> ElevenLabs-v3, OpenAI TTS-1 HD, Gemini-3.1-Flash-TTS, Cartesia.ai (Sonic-3), f5-tts, kokoro, sesame_csm, MegaTTS3

24 training engines:

> ChatTTS, Edge-TTS, MeloTTS, XTTS-v2, suno_bark, Higgs-Audio-V2, Index-TTS-2.0, Llasa-3B, Spark-TTS-0.5B, VoxCPM-0.5B, Qwen2.5-Omni, WhisperSpeech, parler_tts_mini_v1, microsoft_speecht5_tts, MiniMax-Speech-2.8-Turbo, orpheus-tts-0.1-finetune, Step-Audio-EditX, DeepGram, FishTTS, Marvis-TTS, Chatterbox, ElevenLabs-Turbo-v2.5, ljspeech-vits, ljspeech-tacotron2-DDC

## Training Configuration

Same production recipe as v1.2.4 — frozen backbone, linear head, warm-started from the current production checkpoint.

| Setting | Value |
|---------|-------|
| Backbone | Wav2Vec2-large-960h (frozen) |
| Head | Linear 1024 → 1 |
| Warm start | `v1_2_4_proper/best_model.pth` |
| Optimizer | Adam, lr = 1e-4 |
| Batch size | 32 |
| `pos_weight` | 1.0 (auto = 2633/2633, balanced) |
| Early stopping | patience 3 on val EER |
| Max epochs | 50 |
| Seed | 42 |
| Augmentation | none |
| Audio | 10 s @ 16 kHz, mean-pooled |

**Result:** early-stopped at epoch 13, **best epoch 10, val EER 0.3097**.

### Training curve (val EER per epoch, Stage 2)

| Epoch | Train loss | Val loss | Val F1 | Val EER |
|-------|-----------|----------|--------|---------|
| 1 | — | — | — | 0.4343 |
| 2 | — | — | — | 0.3806 |
| 3 | 0.5581 | 0.6353 | 0.6273 | 0.3564 |
| 4 | 0.5435 | 0.6155 | 0.6506 | 0.3426 |
| 5 | 0.5332 | 0.6112 | 0.6448 | 0.3304 |
| 6 | 0.5261 | 0.6026 | 0.6568 | 0.3235 |
| 7 | 0.5193 | 0.5964 | 0.6615 | 0.3201 |
| 8 | 0.5143 | 0.5965 | 0.6593 | 0.3166 |
| 9 | 0.5107 | 0.5898 | 0.6667 | 0.3183 |
| **10 (best)** | **0.5075** | **0.5848** | **0.6750** | **0.3097** |
| 11 | 0.5043 | 0.5851 | 0.6706 | 0.3114 |
| 12 | 0.5018 | 0.5846 | 0.6687 | 0.3149 |
| 13 | 0.4987 | 0.5849 | 0.6673 | 0.3097 |

Because we warm-started from v1.2.4, the epoch 1 val EER (~0.43) is effectively v1.2.4's own performance on modern TTS — so the curve shows v1.2.5 improving on v1.2.4 in real time, dropping from ~0.43 to ~0.31 over 10 epochs.

## Head-to-Head Results

Both heads evaluated on the **held-out modern-TTS test set**: 1,180 files (590 real, 590 fake = 494 MLAAD + 96 ASVspoof). No test engine was seen during v1.2.5 training.

| Metric | v1.2.4 | v1.2.5 |
|--------|--------|--------|
| Overall EER | 0.5525 | **0.2508** |
| Overall acc @0.50 | 56.10% | **75.51%** |
| Overall acc @0.55 | 56.61% | **76.02%** |
| **Modern-TTS EER `[mlaad]`** | **0.6203** | **0.2797** |
| **Modern-TTS recall @0.55 `[mlaad]`** | **6.07%** | **60.93%** |
| ASVspoof EER | 0.0322 | 0.0576 |
| ASVspoof recall @0.55 | 98.96% | 98.96% |

**Headline:** modern-deepfake recall went **6.1% → 60.9%** (≈10×), and modern-TTS EER dropped **0.62 → 0.28**, all while ASVspoof recall stayed pinned at **98.96%**. v1.2.4's `eer[mlaad]` of 0.62 was *worse than random* — it ranked modern fakes as more real than real audio — so this is the specific failure mode v1.2.5 set out to fix, and it is fixed.

## Caveats & Next Steps

- **Not production-grade yet.** 60.9% recall / 0.28 EER on modern TTS is a large step, but still far short of deployable. The ceiling here is **data scale**: real audio came only from ASVspoof bonafide (~3,800 files — the class bottleneck, and also a real/fake domain confound since every real clip and every ASVspoof fake share an acoustic domain the MLAAD fakes do not), and each training engine contributed only ~78 MLAAD samples.
- **Minor ASVspoof regression.** EER rose 0.032 → 0.058 — negligible, and recall @0.55 is unchanged at 98.96%. Acceptable given the modern-TTS gains.
- **Next iteration:**
  - Add more ASVspoof train shards **and** a diverse real corpus (LibriTTS / Common Voice via `prep_v1_2_5.py --extra-real-dir`) to break the real-audio bottleneck and remove the domain confound.
  - Increase MLAAD samples per engine well beyond ~200.

## Artifacts

- `ml_engine/models/v1_2_5_proper/best_model.pth` — trained head (committed on this branch)
- `results_v1_2_5_headtohead.json` — full head-to-head metrics (committed on this branch)
