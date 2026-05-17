#!/usr/bin/env python3
"""
VeriFauX v1.2.2 Cloud Training Script - Based on Working Production Script
Optimized for GPU training with the exact architecture that achieved 71.8% F1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import librosa
from pathlib import Path
import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np
from tqdm import tqdm
import logging
import os
import hashlib
import random
from datetime import datetime
import time
import multiprocessing

# Set multiprocessing start method for CUDA compatibility
multiprocessing.set_start_method("spawn", force=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return float(eer), float(eer_threshold)

class W2V2LargeEmbeddingDataset(Dataset):
    """Dataset class from working v1.2.2 training"""
    
    def __init__(self, data_dir: str, cache_dir: str, target_length: float = 10.0, max_samples: int = None, 
                 sample_rate: int = 16000, processor: Wav2Vec2Processor = None, 
                 backbone: Wav2Vec2Model = None):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.max_samples = max_samples
        self.target_samples = int(sample_rate * target_length)
        self.processor = processor
        self.backbone = backbone

        self.samples: list[Path] = []
        self.labels: list[int] = []

        # Load real samples
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for p in real_dir.glob("*.wav"):
                self.samples.append(p)
                self.labels.append(0)
            for p in real_dir.glob("*.mp3"):
                self.samples.append(p)
                self.labels.append(0)
            for p in real_dir.glob("*.flac"):
                self.samples.append(p)
                self.labels.append(0)
                
        # Load fake samples
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for p in fake_dir.glob("*.wav"):
                self.samples.append(p)
                self.labels.append(1)
            for p in fake_dir.glob("*.mp3"):
                self.samples.append(p)
                self.labels.append(1)
            for p in fake_dir.glob("*.flac"):
                self.samples.append(p)
                self.labels.append(1)

        logger.info(f"Found {len(self.samples)} samples: {sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real")
        
        # Apply sample limiting if specified
        if hasattr(self, "max_samples") and self.max_samples is not None:
            random.seed(42)  # For reproducibility
            if len(self.samples) > self.max_samples:
                # Ensure class balance when possible
                real_indices = [i for i, label in enumerate(self.labels) if label == 0]
                fake_indices = [i for i, label in enumerate(self.labels) if label == 1]
                
                # Calculate samples per class
                samples_per_class = self.max_samples // 2
                
                # Sample from each class
                if len(real_indices) >= samples_per_class and len(fake_indices) >= samples_per_class:
                    selected_real = random.sample(real_indices, samples_per_class)
                    selected_fake = random.sample(fake_indices, samples_per_class)
                    selected_indices = selected_real + selected_fake
                else:
                    # If one class has fewer samples, take what we can and fill with the other
                    selected_indices = real_indices + fake_indices
                    if len(selected_indices) > self.max_samples:
                        selected_indices = random.sample(selected_indices, self.max_samples)
                
                # Reorder samples and labels
                self.samples = [self.samples[i] for i in selected_indices]
                self.labels = [self.labels[i] for i in selected_indices]
                
                logger.info(f"Limited to {len(self.samples)} samples: {sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real")

    def __len__(self):
        return len(self.samples)

    def _audio_to_embedding(self, path: Path) -> np.ndarray:
        """Convert audio to wav2vec2 embeddings with caching"""
        # Cache key per absolute path and target_length
        key = stable_hash(str(path.resolve()) + f"|{self.target_length}")
        npy_path = self.cache_dir / f"{key}.npy"
        
        # Load and process audio
        try:
            wav, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            # Return zero features if loading fails
            return np.zeros(1024, dtype=np.float32)

        if len(wav) > self.target_samples:
            wav = wav[:self.target_samples]
        else:
            pad = self.target_samples - len(wav)
            if pad > 0:
                wav = np.pad(wav, (0, pad), mode="constant")

        # Process through wav2vec2
        inputs = self.processor(
            wav, sampling_rate=self.sample_rate, return_tensors="pt", padding=False
        )
        
        with torch.no_grad():
            outputs = self.backbone(inputs.input_values.to(next(self.backbone.parameters()).device))
            hidden = outputs.last_hidden_state  # (1, time, 1024)
            pooled = hidden.mean(dim=1).squeeze(0).detach().cpu().numpy()  # (1024,)

        np.save(npy_path, pooled)
        return pooled

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        emb = self._audio_to_embedding(path)
        return torch.tensor(emb, dtype=torch.float32), label

class LinearHeadLarge(nn.Module):
    """Simple linear head that achieved 71.8% F1 in v1.2.2"""
    
    def __init__(self, in_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)

def run_epoch(model: nn.Module, dataloader: DataLoader, device: str, 
              criterion: nn.Module, optimizer: optim.Optimizer = None) -> dict:
    """Run one epoch of training or validation"""
    model.train() if optimizer else model.eval()
    
    losses = []
    predictions = []
    targets = []
    
    with torch.no_grad() if not optimizer else torch.enable_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training" if optimizer else "Validation")):
            data, target = data.to(device), target.to(device)
            
            if optimizer:
                optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target.float())
            
            if optimizer:
                loss.backward()
                optimizer.step()
            
            losses.append(loss.item())
            predictions.extend(torch.sigmoid(output).detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    pred_binary = (predictions > 0.5).astype(int)
    
    acc = accuracy_score(targets, pred_binary)
    prec = precision_score(targets, pred_binary, zero_division=0)
    rec = recall_score(targets, pred_binary, zero_division=0)
    f1 = f1_score(targets, pred_binary, zero_division=0)
    eer, eer_threshold = calculate_eer(targets, predictions)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold)
    }

def main():
    parser = argparse.ArgumentParser(description="VeriFauX v1.2.2 Cloud Training - GPU Optimized")
    parser.add_argument("--train_dir", required=True, help="Training data directory")
    parser.add_argument("--val_dir", required=True, help="Validation data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (increased for GPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--save_path", type=str, default="models/verifax_v1_2_2_cloud_trained.pth", help="Model save path")
    parser.add_argument("--cache_dir", type=str, default="models/verifax_v1_2_2_cache", help="Cache directory")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-large-960h", help="Wav2Vec2 model name")
    parser.add_argument("--target_length", type=float, default=10.0, help="Audio target length in seconds")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--cuda", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum training samples (None for all)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Maximum validation samples (None for all)")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device setup
    # Setup checkpoint directories
    run_dir = Path(args.save_path).parent
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Checkpoint dir: {ckpt_dir}")
    if args.cuda == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.cuda)
    
    logger.info(f"🚀 Starting VeriFauX v1.2.2 Cloud Training")
    logger.info(f"📱 Device: {device}")
    logger.info(f"📂 Train dir: {args.train_dir}")
    logger.info(f"📂 Val dir: {args.val_dir}")
    logger.info(f"🎯 Seed: {args.seed}")
    logger.info(f"⏱️  Max epochs: {args.epochs}")
    logger.info(f"📦 Batch size: {args.batch_size}")
    logger.info(f"📈 Learning rate: {args.lr}")
    logger.info(f"🔄 Gradient accumulation: {args.grad_accum}")
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load processor and backbone
    logger.info(f"📥 Loading {args.model_name}...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(args.model_name)
        backbone = Wav2Vec2Model.from_pretrained(args.model_name)
        logger.info(f"✅ Successfully loaded {args.model_name}")
    except Exception as e:
        logger.error(f"❌ Error loading {args.model_name}: {e}")
        logger.info("🔄 Trying alternative model: facebook/wav2vec2-large-960h")
        args.model_name = "facebook/wav2vec2-large-960h"
        processor = Wav2Vec2Processor.from_pretrained(args.model_name)
        backbone = Wav2Vec2Model.from_pretrained(args.model_name)
    
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    
    # Create datasets
    train_cache = Path(args.cache_dir) / "train"
    val_cache = Path(args.cache_dir) / "val"
    
    train_ds = W2V2LargeEmbeddingDataset(
        args.train_dir, str(train_cache), args.target_length, args.max_train_samples, 16000, 
        processor, backbone
    )
    val_ds = W2V2LargeEmbeddingDataset(
        args.val_dir, str(val_cache), args.target_length, args.max_val_samples, 16000, 
        processor, backbone
    )
    
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Infer input dimension
    first_emb, _ = train_ds[0]
    in_dim = int(first_emb.numel())
    logger.info(f"🔢 Input dimension: {in_dim}")
    
    # Create model (same architecture as v1.2.2)
    model = LinearHeadLarge(in_dim)
    model.to(device)
    logger.info(f"🏗️  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if specified
    if args.resume_path and Path(args.resume_path).is_file():
        logger.info(f"🔄 Resuming from checkpoint: {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict")))
        optimizer.load_state_dict(ckpt.get("optimizer", ckpt.get("optimizer_state_dict")))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        logger.info(f"📅 Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val = float("inf")
    
    # Early stopping setup
    best_f1 = 0.0
    best_eer = float('inf')
    patience_counter = 0
    history = {"train": [], "val": []}
    start_time = time.time()
    
    logger.info(f"🎯 Starting training for up to {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Training
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        
        # Validation
        val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None)
        
        # Logging
        epoch_time = time.time() - epoch_start
        logger.info(f"📊 Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        logger.info(f"  🚂 Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, EER: {train_metrics['eer']:.4f}")
        logger.info(f"  ✅ Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, EER: {val_metrics['eer']:.4f}")
        
        # Save history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        
        # Save periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            save = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "args": vars(args),
            }
            torch.save(save, ckpt_dir / f"epoch_{epoch:02d}.pth")
            torch.save(save, ckpt_dir / "latest.pth")
            logger.info(f"💾 Saved checkpoint for epoch {epoch+1}")
        
        # Early stopping check
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_eer = val_metrics["eer"]
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "input_dim": in_dim,
                "model_name": args.model_name,
                "training_args": vars(args),
                "timestamp": datetime.now().isoformat(),
                "best_f1": best_f1,
                "best_eer": best_eer,
                "seed": args.seed
            }
            
            torch.save(checkpoint, args.save_path)
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            logger.info(f"💾 Saved best model (F1: {best_f1:.4f}, EER: {best_eer:.4f}) to {args.save_path}")
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement for {patience_counter} epochs (patience: {args.patience})")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    logger.info(f"🎉 Training completed in {total_time/60:.1f} minutes")
    logger.info(f"🏆 Best validation F1: {best_f1:.4f}")
    logger.info(f"🏆 Best validation EER: {best_eer:.4f}")
    
    # Save final history
    history_file = args.save_path.replace('.pth', '_history.json')
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"📊 Training history saved to {history_file}")
    
    # Performance comparison with v1.2.2
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE COMPARISON WITH v1.2.2")
    logger.info(f"{'='*60}")
    logger.info(f"v1.2.2 Production: F1: 71.8%, EER: 28.7%")
    logger.info(f"Cloud Training:     F1: {best_f1:.1%}, EER: {best_eer:.1%}")
    
    if best_f1 > 0.718:
        logger.info(f"🎉 IMPROVEMENT: +{(best_f1 - 0.718)*100:.1f}% F1")
    else:
        logger.info(f"📉 REGRESSION: {(best_f1 - 0.718)*100:.1f}% F1")

if __name__ == "__main__":
    main()

