#!/usr/bin/env python3
"""
VeriFauX Linear Head Training Script v1.0.0
Clean, simple linear classifier for binary audio classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torchaudio.functional
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
import contextlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_linear.log'),
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

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return float(eer), float(eer_threshold)

# Simple Linear Head for Binary Classification
class LinearHead(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
        # Initialize weights for better training
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        return self.classifier(x)

# Dataset for precomputed embeddings
class EmbeddingFileDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        # Look for PT files in subdirectories (fake/ and real/)
        self.files = []
        for subdir in ["fake", "real"]:
            subdir_path = self.cache_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.pt"))
                self.files.extend(files)
        
        logger.info(f"📁 Found {len(self.files)} precomputed embeddings in {cache_dir}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Extract label from parent directory (fake/ = 1, real/ = 0)
        if "fake" in str(file_path.parent):
            label = 1
        elif "real" in str(file_path.parent):
            label = 0
        else:
            # Fallback: try to infer from filename
            label = 1 if "fake" in file_path.name else 0
            
        # Load precomputed embedding
        embedding_data = torch.load(file_path, map_location='cpu')
        
        # Handle different embedding formats
        if isinstance(embedding_data, dict):
            # If it's a dictionary, extract the embedding tensor
            if 'embedding' in embedding_data:
                embedding = embedding_data['embedding']
            elif 'features' in embedding_data:
                embedding = embedding_data['features']
            elif 'hidden_states' in embedding_data:
                embedding = embedding_data['hidden_states']
            else:
                # Try to find any tensor in the dict
                for key, value in embedding_data.items():
                    if isinstance(value, torch.Tensor):
                        embedding = value
                        break
                else:
                    raise ValueError(f"Could not find embedding tensor in {file_path}")
        else:
            # If it's directly a tensor
            embedding = embedding_data
            
        # Ensure embedding is 1D (1024 features)
        if embedding.dim() > 1:
            embedding = embedding.squeeze()
            
        return embedding, label, str(file_path)

def collate_emb(batch):
    """Collate function for embeddings"""
    embeddings = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    paths = [item[2] for item in batch]
    return embeddings, labels, paths

def run_epoch(model, data_loader, device, criterion, optimizer=None, use_precomputed=True):
    """Run one epoch of training or validation"""
    model.train() if optimizer else model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad() if not optimizer else contextlib.nullcontext():
        for batch_idx, (embeddings, labels, paths) in enumerate(tqdm(data_loader, desc="Processing")):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(embeddings)
            scores = torch.sigmoid(logits).squeeze()
            
            # Calculate loss
            if optimizer:
                loss = criterion(logits.squeeze(), labels.float())
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = criterion(logits.squeeze(), labels.float())
                total_loss += loss.item()
            
            # Store predictions and labels
            predictions = (scores >= 0.5).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.detach().cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eer": eer,
        "eer_threshold": eer_threshold
    }

def main():
    parser = argparse.ArgumentParser(description="Train VeriFauX Linear Head")
    parser.add_argument("--train_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation data directory")
    parser.add_argument("--save_path", type=str, required=True, help="Model save path")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Cache directory")
    parser.add_argument("--embed_cache_dir", type=str, default=None, help="Precomputed embeddings directory")
    parser.add_argument("--use_precomputed", action="store_true", help="Use precomputed embeddings")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--checkpoint_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", type=str, default="auto", help="CUDA device")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device setup
    if args.cuda == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.cuda)
    
    logger.info(f"🚀 Starting VeriFauX Linear Head Training v1.0.0")
    logger.info(f"📱 Device: {device}")
    logger.info(f"💾 Model will be saved to: {args.save_path}")
    
    # Setup embedding cache directory
    if args.embed_cache_dir is None:
        args.embed_cache_dir = str(Path(args.cache_dir) / "emb")
    emb_root = Path(args.embed_cache_dir)
    
    # Create output directories
    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    
    # Load precomputed embeddings
    logger.info(f"📁 Using precomputed embeddings from {emb_root}")
    train_ds = EmbeddingFileDataset(str(emb_root / "train"))
    val_ds = EmbeddingFileDataset(str(emb_root / "val"))
    
    if len(train_ds) == 0:
        raise RuntimeError(f"Precomputed training dataset is empty at {emb_root}/train")
    
    # Create data loaders for embeddings
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True, 
                              collate_fn=collate_emb)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True, 
                            collate_fn=collate_emb)
    
    input_dim = 1024  # Wav2Vec2 large outputs 1024 features
    logger.info(f"🔢 Using precomputed embeddings with dimension: {input_dim}")
    
    # Create LINEAR model (not MLP)
    model = LinearHead(input_dim)
    model.to(device)
    logger.info(f"🏗️  LINEAR model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Setup checkpoint directories
    run_dir = Path(args.save_path).parent
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Early stopping setup
    best_f1 = 0.0
    best_eer = float('inf')
    patience_counter = 0
    history = {"train": [], "val": []}
    start_time = time.time()
    
    logger.info(f"🎯 Starting LINEAR training for up to {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Training
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, 
                                  use_precomputed=True)
        
        # Validation
        val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None, 
                                use_precomputed=True)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        
        # Logging
        epoch_time = time.time() - epoch_start
        logger.info(f"📊 Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        logger.info(f"  🚂 Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, EER: {train_metrics['eer']:.4f}")
        logger.info(f"  ✅ Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, EER: {val_metrics['eer']:.4f}")
        logger.info(f"  📈 LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        
        # Save periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            save = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
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
                "input_dim": input_dim,
                "model_name": "LinearHead",
                "training_args": vars(args),
                "timestamp": datetime.now().isoformat(),
                "best_f1": best_f1,
                "best_eer": best_eer,
                "seed": args.seed
            }
            
            torch.save(checkpoint, args.save_path)
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            
            # Save best checkpoint info
            with open(run_dir / "best_checkpoint.txt", "w") as f:
                f.write(str(args.save_path))
            
            logger.info(f"💾 Saved best LINEAR model (F1: {best_f1:.4f}, EER: {best_eer:.4f}) to {args.save_path}")
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement for {patience_counter} epochs (patience: {args.patience})")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    logger.info(f"🎉 LINEAR training completed in {total_time/60:.1f} minutes")
    logger.info(f"🏆 Best validation F1: {best_f1:.4f}")
    logger.info(f"🏆 Best validation EER: {best_eer:.4f}")
    
    # Save final history
    history_file = args.save_path.replace('.pth', '_history.json')
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"📊 Training history saved to {history_file}")
    
    # Save model summary
    model_summary = {
        "architecture": "LinearHead",
        "input_dim": input_dim,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_f1": best_f1,
        "best_eer": best_eer,
        "training_time_minutes": total_time/60,
        "epochs_trained": epoch + 1,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = run_dir / "model_summary.json"
    with open(summary_file, "w") as f:
        json.dump(model_summary, f, indent=2)
    logger.info(f"📋 Model summary saved to {summary_file}")

if __name__ == "__main__":
    main()
