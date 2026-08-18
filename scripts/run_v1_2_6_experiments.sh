#!/bin/bash
# run_v1_2_6_experiments.sh — train + compare three models on the big-data v1.2.6 set.
#
# 1. Neural head retrained on big data (isolates the "more data" effect vs v1.2.5).
# 2. XGBoost on Wav2Vec2 functionals (tests the tree skeleton now that n >> p).
# 3. Head-to-head eval of v1.2.4, v1.2.5, and v1.2.6 neural heads on the SAME
#    v1.2.6 held-out modern-TTS test set.
#
# Run after scripts/setup_pod_data.sh. Long pole is the neural retrain (~2-3 h on
# a 4090 at this data scale); XGBoost extracts features once (~45 min) then fits
# in seconds. Launch inside tmux.
#
# Usage:  bash scripts/run_v1_2_6_experiments.sh
set -euo pipefail
CFG=ml_engine/config/training_v1_2_6.json

echo "===== 1/3 neural head (big data) ====="
python ml_engine/train_v1_2_4_proper.py --config "$CFG" --train 2>&1 | tee v1_2_6_neural.log

echo "===== 2/3 XGBoost on functionals ====="
python train_xgboost_v1_2_6.py --config "$CFG" --train 2>&1 | tee v1_2_6_xgb.log

echo "===== 3/3 head-to-head (v1.2.4 vs v1.2.5 vs v1.2.6 neural) on v1.2.6 test ====="
python eval_v1_2_5_headtohead.py \
  --config "$CFG" \
  --heads \
    v1_2_4=ml_engine/models/v1_2_4_proper/best_model.pth \
    v1_2_5=ml_engine/models/v1_2_5_proper/best_model.pth \
    v1_2_6=ml_engine/models/v1_2_6_proper/best_model.pth \
  --out results_v1_2_6_headtohead.json 2>&1 | tee v1_2_6_eval.log

echo
echo "===== SUMMARY ====="
echo "--- neural heads (eval_v1_2_5_headtohead) ---"
cat results_v1_2_6_headtohead.json
echo
echo "--- XGBoost (its own results) ---"
cat ml_engine/models/xgboost_v1_2_6/results.json
echo
echo "Artifacts: results_v1_2_6_headtohead.json, ml_engine/models/xgboost_v1_2_6/results.json"
echo "Copy models off the pod before terminating:"
echo "  runpodctl send ml_engine/models/v1_2_6_proper/best_model.pth"
echo "  runpodctl send ml_engine/models/xgboost_v1_2_6/model.json"
