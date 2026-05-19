# ML Engine: Improvement Recommendations

**Author:** Nishant Jain, with Claude
**Date:** 2026-05-18
**Based on:** Evals 01-03 (see `eval_reports/`)

## Summary of Findings

We tested the v1.2.4 ML engine across three public datasets:

| Eval | Dataset | Accuracy | EER | Fake Sources |
|------|---------|----------|-----|--------------|
| 01 | HuggingFace deepfake-audio | 29.0% | 85.0% | ElevenLabs, Speechify, Polly, Hume, Kokoro, Luvvoice |
| 02 | ASVspoof5 eval | 75.5% | 25.0% | ASVspoof-family attacks (codec, adversarial, TTS/VC) |
| 03 | MLAAD | 44.5% | 62.0% | ElevenLabs v3, OpenAI, Gemini, bark, XTTS v2, f5-tts, ChatTTS, sesame, MeloTTS, kokoro |

The model works on ASVspoof-style attacks but fails on modern TTS. On MLAAD, the model scored fakes *lower* than reals (mean 0.29 vs 0.38) — it's actively classifying modern deepfakes as real with high confidence. No commercial TTS engine (ElevenLabs v3, OpenAI TTS-1 HD, Gemini Flash TTS) was detected at all (0/10 each).

## 1. Expand the Training Data with Modern TTS

**Problem:** The training mix (RitW + ASVspoof2021 LA + MLAAD + CVoiceFake + ASVspoof5) used older MLAAD versions. The TTS engines that dominate the current threat landscape — ElevenLabs, OpenAI, Google/Gemini, Cartesia, Hume — are absent from training.

**Fix:** MLAAD now has **116 English TTS engines** on HuggingFace (`mueller91/MLAAD`), each with 1000 samples. Priority engines to add to training:

- **Commercial APIs** (highest real-world threat): ElevenLabs v3, OpenAI TTS-1 HD, Gemini 3.1 Flash TTS, DeepGram, Edge-TTS, Cartesia Sonic-3
- **Open-source models** (growing threat): f5-tts, kokoro, ChatTTS, sesame CSM, MeloTTS, Index-TTS, Chatterbox
- **Older engines already likely in training** (for continuity): Tacotron2, VITS, XTTS v2, bark

Even adding just the top 20 engines (20K new fake samples) would dramatically improve coverage.

## 2. Increase Training Set Size

**Problem:** The v1.2.4 calibration set was ~17K files (6,685 real + 10,361 fake). This is small for a Wav2Vec2-based detector.

**Fix:** Scale to 50-100K+ samples. Sources for additional real audio:
- ASVspoof5 bonafide (138K eval files available)
- LibriSpeech / LibriTTS (public, high quality)
- Common Voice (multilingual, diverse speakers)

For fake audio, MLAAD provides 116K English samples alone. Combined with the existing training sources, a 100K+ balanced dataset is achievable without any new recording or generation.

## 3. Add Audio Augmentation to Simulate Real-World Conditions

**Problem:** The model is trained on clean audio but deployed on Teams calls, which introduce codec compression, packet loss, background noise, and resampling. Our ASVspoof5 eval (which includes codec-processed audio) showed higher FPR (19%) than the reported RitW FPR (11.9%), suggesting the model is sensitive to codec artifacts.

**Fix:** Add augmentations during training:
- **Codec simulation:** Opus, AAC, MP3 at various bitrates (Teams uses Opus/SILK)
- **Noise injection:** Office noise, room tone, microphone hiss at realistic SNR levels (15-30 dB)
- **Resampling:** Downsample to 8kHz/22kHz and back up to 16kHz
- **Room impulse responses:** Simulate different room acoustics

These can be applied on-the-fly during training using `torchaudio.transforms` or the `audiomentations` library, with no additional data collection needed.

## 4. Unfreeze More Backbone Layers

**Problem:** The current approach freezes the entire Wav2Vec2 backbone and only fine-tunes encoder layers 23-24 plus the linear head. This limits the model's ability to learn new discriminative features for TTS engines not seen during Wav2Vec2's pre-training.

**Fix:** Gradually unfreeze more encoder layers (e.g., layers 18-24) with a discriminative learning rate schedule — lower LR for early layers, higher for later layers. This is standard practice for transfer learning and the codebase already supports partial backbone unfreezing (`backbone_state_dict` in the checkpoint format). Start with unfreezing 4 layers instead of 2 and measure the impact.

## 5. Upgrade the Classification Head

**Problem:** The production model uses a single linear layer (1024 → 1). While simple, this limits the decision boundary to a hyperplane in embedding space — it can only learn a single linear separation between real and fake.

**Fix:** The codebase already has `MLPHeadLarge` (1024 → 256 → 1 with ReLU and dropout). Eval scripts reference it but the deployed model uses the linear head. Switching to the MLP head would allow the model to learn non-linear decision boundaries, which matters when the fake audio comes from diverse TTS architectures that occupy different regions of embedding space. This is a one-line config change in the training script.

## 6. Implement Curriculum or Multi-Stage Training

**Problem:** The model was trained on a single mixed dataset. When the mix includes both easy fakes (older TTS) and hard fakes (modern commercial TTS), the model may converge on features that only work for the easy cases.

**Fix:** Train in stages:
1. **Stage 1:** Train on the full mix (current approach) to learn general real-vs-fake separation
2. **Stage 2:** Fine-tune on hard examples only — samples where the Stage 1 model scores incorrectly or with low confidence. This forces the model to learn discriminative features for the engines it currently misses.

Alternatively, use a weighted loss where hard-to-detect TTS engines get higher sample weight during training.

## Priority Order

| Priority | Action | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| 1 | Add modern TTS to training data (MLAAD) | Low — data is free, scripts exist | High — directly addresses the core failure |
| 2 | Increase dataset size to 50K+ | Low — combine existing public sources | Medium — more data improves generalization |
| 3 | Switch to MLP head | Trivial — already implemented | Medium — better decision boundaries |
| 4 | Add audio augmentations | Medium — need augmentation pipeline | Medium — improves robustness to codecs/noise |
| 5 | Unfreeze more backbone layers | Low — config change + longer training | Medium — learns deeper features |
| 6 | Multi-stage training | Medium — new training pipeline | High — targets specific failure modes |
