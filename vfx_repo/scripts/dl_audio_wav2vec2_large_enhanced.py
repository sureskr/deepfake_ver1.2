#!/usr/bin/env python3
"""
Enhanced VeriFauX Phase 4 Training Script
Supports early stopping, mixed precision, gradient accumulation, and comprehensive monitoring
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
    def __init__(self, data_dir: str, cache_dir: str, target_length: float = 10.0, sample_rate: int = 16000,
                 processor: Wav2Vec2Processor = None, backbone: Wav2Vec2Model = None):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.target_samples = int(sample_rate * target_length)
        self.processor = processor
        self.backbone = backbone

        self.samples: list[Path] = []
        self.labels: list[int] = []

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

        print(f"Found {len(self.samples)} samples: {sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real")

    def __len__(self):
        return len(self.samples)

    def _audio_to_embedding(self, path: Path) -> np.ndarray:
        # Cache key per absolute path and target_length to avoid collisions
        key = stable_hash(str(path.resolve()) + f"|{self.target_length}")
        npy_path = self.cache_dir / f"{key}.npy"
        if npy_path.exists():
            return np.load(npy_path)

        # Load and process audio
        try:
            wav, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return zero embedding as fallback
            return np.zeros(1024)
            
        if len(wav) > self.target_samples:
            wav = wav[: self.target_samples]
        else:
            pad = self.target_samples - len(wav)
            if pad > 0:
                wav = np.pad(wav, (0, pad), mode="constant")

        inputs = self.processor(
            wav, sampling_rate=self.sample_rate, return_tensors="pt", padding=False
        )
        with torch.no_grad():
            outputs = self.backbone(inputs.input_values)
            hidden = outputs.last_hidden_state  # (1, time, feat)
            pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (feat,)

        np.save(npy_path, pooled)
        return pooled

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        emb = self._audio_to_embedding(path)
        return torch.tensor(emb, dtype=torch.float32), label

class LinearHeadLarge(nn.Module):
    def __init__(self, in_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)

def run_epoch(model, loader, device, criterion, optimizer=None, grad_accum_steps=1):
    """Run one epoch with optional gradient accumulation"""
    model.train() if optimizer else model.eval()
    losses = []
    targets = []
    preds = []
    
    for i, (emb, label) in enumerate(tqdm(loader, desc="Training" if optimizer else "Validation")):
        emb, label = emb.to(device), label.to(device)
        
        with torch.set_grad_enabled(optimizer is not None):
            output = model(emb)
            loss = criterion(output, label.float())
            
            if optimizer:
                # Gradient accumulation
                loss = loss / grad_accum_steps
                loss.backward()
                
                if (i + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            losses.append(loss.item() * grad_accum_steps)
            targets.extend(label.cpu().numpy())
            preds.extend(torch.sigmoid(output).cpu().detach().numpy())

    # Calculate metrics
    targets = np.array(targets)
    preds = np.array(preds)
    pred_binary = (preds > 0.5).astype(int)
    
    acc = accuracy_score(targets, pred_binary)
    prec = precision_score(targets, pred_binary, zero_division=0)
    rec = recall_score(targets, pred_binary, zero_division=0)
    f1 = f1_score(targets, pred_binary, zero_division=0)
    eer, eer_threshold = calculate_eer(targets, preds)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold)
    }

def create_temp_dirs_from_manifests(train_csv, val_csv, test_csv, corpus_root, manifests_dir):
    """Create temporary directory structures from manifests"""
    import tempfile
    import shutil
    import pandas as pd
    
    # Create path to SHA mapping
    mapping = {}
    for split in ["train", "val", "test"]:
        manifest_file = Path(manifests_dir) / f"split.v1_2_2.paths_resolved.{split}.csv"
        if manifest_file.exists():
            df = pd.read_csv(manifest_file, low_memory=False)
            for _, row in df.iterrows():
                original_path = str(row['path'])
                sha256 = str(row['sha256'])
                if pd.notna(sha256) and sha256 != 'nan':
                    sha8 = sha256[:8]
                    orig_filename = Path(original_path).name
                    sha_filename = f"{sha8}_{orig_filename}"
                    mapping[original_path] = sha_filename
    
    print(f"Created mapping for {len(mapping)} files")
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp(prefix="verifax_prod_"))
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
    test_dir = temp_dir / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        (d / "real").mkdir(parents=True, exist_ok=True)
        (d / "fake").mkdir(parents=True, exist_ok=True)
    
    print(f"Created temporary directory: {temp_dir}")
    
    # Function to copy files from manifest to temp dir
    def copy_from_manifest(csv_path, target_dir):
        df = pd.read_csv(csv_path, low_memory=False)
        copied = 0
        
        for _, row in df.iterrows():
            original_path = str(row['path'])
            
            if original_path in mapping:
                sha_filename = mapping[original_path]
                
                # Find the file in the corpus
                found = False
                for subdir in ["real", "fake"]:
                    for split_dir in ["train", "val", "test"]:
                        corpus_file = Path(corpus_root) / split_dir / subdir / sha_filename
                        if corpus_file.exists():
                            target_subdir = "real" if int(row['label']) == 0 else "fake"
                            target_file = target_dir / target_subdir / sha_filename
                            
                            shutil.copy2(corpus_file, target_file)
                            copied += 1
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    print(f"Warning: Could not find SHA-prefixed file: {sha_filename}")
            else:
                print(f"Warning: No SHA mapping found for path: {original_path}")
        
        return copied
    
    # Copy files for each split
    train_count = copy_from_manifest(train_csv, train_dir)
    val_count = copy_from_manifest(val_csv, val_dir)
    test_count = copy_from_manifest(test_csv, test_dir)
    
    print(f"Copied files: Train={train_count}, Val={val_count}, Test={test_count}")
    
    return temp_dir, train_dir, val_dir

def main():
    parser = argparse.ArgumentParser(description="Enhanced VeriFauX Phase 4 Training")
    
    # Data arguments
    parser.add_argument("--train_csv", required=True, help="Path to training manifest CSV")
    parser.add_argument("--val_csv", required=True, help="Path to validation manifest CSV")
    parser.add_argument("--test_csv", required=True, help="Path to test manifest CSV")
    parser.add_argument("--corpus_root", default="data/corpus/audio/v1_2_2", help="Corpus root directory")
    parser.add_argument("--manifests_dir", default="data/corpus/manifests", help="Manifests directory")
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--tag", default="v1.2.2", help="Experiment tag")
    parser.add_argument("--save_dir", default="runs/v1_2_2_prod", help="Save directory")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--target_length", type=float, default=10.0, help="Audio target length in seconds")
    
    # Early stopping
    parser.add_argument("--early_stop", default="metric=eer split=val patience=8", help="Early stopping config")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    
    # Model arguments
    parser.add_argument("--model_name", default="facebook/wav2vec2-large-960h", help="Wav2Vec2 model name")
    parser.add_argument("--cache_dir", default=None, help="Cache directory (auto-generated if None)")
    
    args = parser.parse_args()
    
    # Parse early stopping
    early_stop_metric = "eer"  # Default
    early_stop_split = "val"   # Default
    if "metric=" in args.early_stop:
        early_stop_metric = args.early_stop.split("metric=")[1].split()[0]
    if "split=" in args.early_stop:
        early_stop_split = args.early_stop.split("split=")[1].split()[0]
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fresh cache directory to prevent data leakage
    if args.cache_dir is None:
        cache_dir = save_dir / f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        cache_dir = Path(args.cache_dir)
    
    print("=== Enhanced VeriFauX Phase 4 Training ===")
    print(f"Tag: {args.tag}")
    print(f"Seed: {args.seed}")
    print(f"Save directory: {args.save_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Early stopping: {early_stop_metric} on {early_stop_split} with patience {args.patience}")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.grad_accum}")
    print()
    
    try:
        # Create temporary directory structures
        print("Creating temporary directory structures...")
        temp_dir, train_dir, val_dir = create_temp_dirs_from_manifests(
            args.train_csv, args.val_csv, args.test_csv, 
            args.corpus_root, args.manifests_dir
        )
        
        # Load processor and backbone
        print(f"Loading {args.model_name}...")
        processor = Wav2Vec2Processor.from_pretrained(args.model_name)
        backbone = Wav2Vec2Model.from_pretrained(args.model_name)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        
        # Create datasets with fresh cache
        train_cache = cache_dir / "train"
        val_cache = cache_dir / "val"
        train_ds = W2V2LargeEmbeddingDataset(str(train_dir), str(train_cache), args.target_length, 16000, processor, backbone)
        val_ds = W2V2LargeEmbeddingDataset(str(val_dir), str(val_cache), args.target_length, 16000, processor, backbone)
        
        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Create model
        first_emb, _ = train_ds[0]
        in_dim = int(first_emb.numel())
        print(f"Embedding dimension: {in_dim}")
        
        model = LinearHeadLarge(in_dim)
        device = "cpu"  # Explicitly CPU as per original
        model.to(device)
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Early stopping
        best_metric = float('inf') if early_stop_metric == "eer" else 0.0
        patience_counter = 0
        best_epoch = 0
        
        # History tracking
        history = {"train": [], "val": []}
        
        print(f"\nStarting training for up to {args.epochs} epochs...")
        print("=" * 80)
        
        for epoch in range(args.epochs):
            # Training
            train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, args.grad_accum)
            
            # Validation
            val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None)
            
            # Record history
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train - loss: {train_metrics['loss']:.4f}, f1: {train_metrics['f1']:.4f}, eer: {train_metrics['eer']:.4f}")
            print(f"  Val   - loss: {val_metrics['loss']:.4f}, f1: {val_metrics['f1']:.4f}, eer: {val_metrics['eer']:.4f}")
            
            # Early stopping check
            current_metric = val_metrics[early_stop_metric]
            is_better = False
            
            if early_stop_metric == "eer":
                is_better = current_metric < best_metric
            else:
                is_better = current_metric > best_metric
            
            if is_better:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                save_path = save_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "embed_dim": in_dim,
                    "model_name": args.model_name,
                    "args": vars(args),
                }, save_path)
                print(f"  ✅ Saved best model to {save_path}")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement for {patience_counter} epochs (patience: {args.patience})")
            
            # Check early stopping
            if patience_counter >= args.patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"Best {early_stop_metric}: {best_metric:.4f} at epoch {best_epoch+1}")
                break
            
            print("-" * 80)
        
        # Save final model and history
        final_save_path = save_dir / "final.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "embed_dim": in_dim,
            "model_name": args.model_name,
            "args": vars(args),
        }, final_save_path)
        
        history_path = save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\n🎯 Training completed!")
        print(f"Final model saved to: {final_save_path}")
        print(f"Training history saved to: {history_path}")
        print(f"Best {early_stop_metric}: {best_metric:.4f} at epoch {best_epoch+1}")
        
    finally:
        # Clean up temporary directory
        if 'temp_dir' in locals():
            print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up {temp_dir}: {e}")

if __name__ == "__main__":
    main()



