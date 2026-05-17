# VeriFauX v1.2.2 Training & Evaluation Usage

## Training with MLP Head

```bash
python3 train_verifax_precompute.py \
  --use_precomputed \
  --train_dir /home/surya/vfx/data/train \
  --val_dir /home/surya/vfx/data/val \
  --embed_cache_dir /home/surya/vfx/cache/emb \
  --head mlp \
  --mlp_hidden 256 \
  --dropout 0.2 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --epochs 6 \
  --patience 3 \
  --save_path /home/surya/vfx/models/run_X/model.pth
```

## Calibrate Threshold on Validation Set

```bash
python3 train_verifax_precompute.py \
  --use_precomputed \
  --val_dir /home/surya/vfx/data/val \
  --embed_cache_dir /home/surya/vfx/cache/emb \
  --head mlp \
  --mlp_hidden 256 \
  --dropout 0.2 \
  --eval_only \
  --split val \
  --weights /home/surya/vfx/models/run_X/model.pth \
  --out_csv /home/surya/vfx/models/run_X/val_logits.csv
```

This will:
- Print the best threshold and F1 score: `[calib] best_threshold=0.320 best_f1=0.7282`
- Save the threshold to `threshold.txt` next to the CSV output
- Generate predictions CSV with columns: `filepath,prob_fake,label_pred,label_true`

## Inference on Test Set Using Calibrated Threshold

```bash
# Read the calibrated threshold
THRESH=$(cat /home/surya/vfx/models/run_X/threshold.txt)

# Run inference on test set
python3 train_verifax_precompute.py \
  --use_precomputed \
  --test_dir /home/surya/vfx/data/test \
  --embed_cache_dir /home/surya/vfx/cache/emb \
  --head mlp \
  --mlp_hidden 256 \
  --dropout 0.2 \
  --eval_only \
  --split test \
  --weights /home/surya/vfx/models/run_X/model.pth \
  --threshold "$THRESH" \
  --out_csv /home/surya/vfx/models/run_X/test_preds.csv
```

## Key Features

- **MLP Head**: Multi-layer perceptron with GELU activation and dropout for better performance than linear heads
- **Threshold Calibration**: Automatically finds optimal threshold by sweeping [0.05, 0.95] and maximizing F1 score
- **CSV Output**: Structured predictions with probabilities and binary labels
- **Auto-detection**: Embedding dimensions are auto-detected from cached files
- **Checkpoint Tracking**: Best checkpoint path is saved to `best_checkpoint.txt` after training

## Model Performance

Recent results with MLP head (256 hidden, 0.2 dropout):
- **Training F1**: 0.6937 (69.37%)
- **Validation F1**: 0.6954 (69.54%) 
- **Calibrated F1**: 0.7282 (72.82%)
- **Best Threshold**: 0.320
- **Training Time**: ~16.4 minutes (3 epochs with precomputed embeddings)
