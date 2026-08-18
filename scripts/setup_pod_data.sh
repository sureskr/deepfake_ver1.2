#!/bin/bash
# setup_pod_data.sh — download + assemble the big-data v1.2.6 dataset on a fresh pod.
#
# Pulls all 5 ASVspoof5 train shards (real bonafide is the binding constraint) +
# MLAAD English @1000/engine, then builds data/v1_2_6 with generator-disjoint
# splits (8 modern engines held out for test). ASVspoof-only real audio.
#
# Prereqs: run from repo root; HF_TOKEN exported; deps installed
#   (pip install -r ml_engine/requirements.txt miniaudio xgboost).
#
# Usage:  bash scripts/setup_pod_data.sh
set -euo pipefail

RAW_ASV=data/raw/asvspoof5
RAW_MLAAD=data/raw/mlaad
OUT=data/v1_2_6
SHARDS=(flac_T_aa.tar flac_T_ab.tar flac_T_ac.tar flac_T_ad.tar flac_T_ae.tar)

echo "== 1/5 backbone =="
if [ ! -d ml_engine/models/wav2vec2-large-960h ]; then
  hf download facebook/wav2vec2-large-960h --local-dir ml_engine/models/wav2vec2-large-960h
fi

echo "== 2/5 ASVspoof5: protocols + all 5 train shards =="
hf download jungjee/asvspoof5 ASVspoof5_protocols.tar "${SHARDS[@]}" \
  --repo-type dataset --local-dir "$RAW_ASV"
( cd "$RAW_ASV" && tar xf ASVspoof5_protocols.tar )
mkdir -p "$RAW_ASV/flac_T"
for t in "${SHARDS[@]}"; do
  echo "  extracting $t ..."
  tar xf "$RAW_ASV/$t" -C "$RAW_ASV/flac_T"
done
echo "  ASVspoof flac files present: $(find "$RAW_ASV/flac_T" -name '*.flac' | wc -l)"

echo "== 3/5 MLAAD English @1000/engine =="
python scripts/download_mlaad_en.py --dest "$RAW_MLAAD" --n-per-engine 1000

echo "== 4/5 assemble data/v1_2_6 (ASVspoof-only real; disjoint splits) =="
# ASVspoof spoof capped so real (~19k) is roughly the binding class after MLAAD.
python prep_v1_2_5.py \
  --asvspoof-protocol "$RAW_ASV/ASVspoof5.train.tsv" \
  --asvspoof-flac-dir "$RAW_ASV/flac_T/flac_T" \
  --asvspoof-fname-col 1 --asvspoof-label-col 8 --asvspoof-speaker-col 0 --asvspoof-attack-col 7 \
  --asvspoof-max-fake 8000 \
  --mlaad-root "$RAW_MLAAD" \
  --holdout-engines "ElevenLabs-v3,OpenAI TTS-1 HD,Gemini-3.1-Flash-TTS,Cartesia.ai (Sonic-3),f5-tts,kokoro,sesame_csm,MegaTTS3" \
  --out-root "$OUT" --link

echo "== 5/5 manifest =="
python ml_engine/scripts/build_manifest.py --dataset-root "$OUT" \
  --out data/manifests/v1_2_6.csv --source-tag v1_2_6 --no-locked-test

echo
echo "DONE. Dataset at $OUT. Next: bash scripts/run_v1_2_6_experiments.sh"
