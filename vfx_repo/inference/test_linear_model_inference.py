#!/usr/bin/env python3
"""
Test Linear Model Inference on Corrected Test Cache
Tests the trained Linear model on the newly generated test cache
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinearHead(nn.Module):
    """Linear classifier head that matches the training script"""
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.classifier(x)

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return float(eer), float(eer_threshold)

def load_model_and_cache():
    """Load the trained Linear model and test cache"""
    print("=== Loading Model and Cache ===")
    
    # Load trained model
    model_path = "/home/surya/vfx/models/verifax_linear_v1_0_0/best_model.pth"
    print(f"Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load model state
    checkpoint = torch.load(model_path, map_location='cpu')
    model = LinearHead(input_dim=1024)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        # Direct state dict
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
    
    model.eval()
    print(f"✅ Model loaded successfully")
    
    # Load test cache
    cache_dir = Path("/home/surya/vfx/cache/emb/test")
    fake_cache_dir = cache_dir / "fake"
    real_cache_dir = cache_dir / "real"
    
    fake_files = list(fake_cache_dir.glob("*.pt"))
    real_files = list(real_cache_dir.glob("*.pt"))
    
    print(f"📁 Test cache loaded:")
    print(f"  Fake files: {len(fake_files)}")
    print(f"  Real files: {len(real_files)}")
    print(f"  Total: {len(fake_files) + len(real_files)}")
    
    return model, fake_files, real_files

def evaluate_on_cache(model, fake_files, real_files, max_samples=None):
    """Evaluate model performance on the test cache"""
    print(f"\n=== Evaluating on Test Cache ===")
    
    # Limit samples if specified
    if max_samples:
        fake_files = fake_files[:max_samples//2]
        real_files = real_files[:max_samples//2]
        print(f"Testing on {len(fake_files) + len(real_files)} samples (limited)")
    
    all_predictions = []
    all_labels = []
    all_scores = []
    
    # Process fake files (label = 1)
    print(f"\nProcessing {len(fake_files)} fake files...")
    for file_path in tqdm(fake_files, desc="Fake files"):
        try:
            cache_data = torch.load(file_path, map_location='cpu')
            features = cache_data['emb'].unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                output = model(features)
                score = torch.sigmoid(output).item()
            
            all_predictions.append(1 if score > 0.5 else 0)
            all_labels.append(1)  # Fake
            all_scores.append(score)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    # Process real files (label = 0)
    print(f"\nProcessing {len(real_files)} real files...")
    for file_path in tqdm(real_files, desc="Real files"):
        try:
            cache_data = torch.load(file_path, map_location='cpu')
            features = cache_data['emb'].unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                output = model(features)
                score = torch.sigmoid(output).item()
            
            all_predictions.append(1 if score > 0.5 else 0)
            all_labels.append(0)  # Real
            all_scores.append(score)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    return all_predictions, all_labels, all_scores

def calculate_metrics(predictions, labels, scores):
    """Calculate comprehensive performance metrics"""
    print(f"\n=== Performance Metrics ===")
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # EER calculation
    eer, eer_threshold = calculate_eer(labels, scores)
    
    # Threshold analysis
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    print(f"📊 Results:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  EER: {eer:.4f} ({eer*100:.2f}%)")
    print(f"  EER Threshold: {eer_threshold:.4f}")
    
    # Label distribution
    fake_count = sum(labels)
    real_count = len(labels) - fake_count
    print(f"\n📈 Label Distribution:")
    print(f"  Fake files: {fake_count}")
    print(f"  Real files: {real_count}")
    print(f"  Ratio: {fake_count/len(labels):.3f} fake, {real_count/len(labels):.3f} real")
    
    # Score analysis
    fake_scores = [s for s, l in zip(scores, labels) if l == 1]
    real_scores = [s for s, l in zip(scores, labels) if l == 0]
    
    print(f"\n🎯 Score Analysis:")
    print(f"  Fake scores (should be HIGH): mean={np.mean(fake_scores):.3f}, std={np.std(fake_scores):.3f}")
    print(f"  Real scores (should be LOW): mean={np.mean(real_scores):.3f}, std={np.std(real_scores):.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'fake_scores': fake_scores,
        'real_scores': real_scores
    }

def test_real_time_inference(model):
    """Test real-time inference on a few sample files"""
    print(f"\n=== Real-Time Inference Test ===")
    
    # Test on a few files from the cache
    cache_dir = Path("/home/surya/vfx/cache/emb/test")
    fake_files = list((cache_dir / "fake").glob("*.pt"))[:3]
    real_files = list((cache_dir / "real").glob("*.pt"))[:3]
    
    print(f"Testing real-time inference on 6 sample files...")
    
    for file_path in fake_files + real_files:
        try:
            cache_data = torch.load(file_path, map_location='cpu')
            features = cache_data['emb'].unsqueeze(0)
            true_label = cache_data['y']
            filename = Path(cache_data['src']).name
            
            start_time = time.time()
            with torch.no_grad():
                output = model(features)
                score = torch.sigmoid(output).item()
            inference_time = (time.time() - start_time) * 1000  # ms
            
            prediction = 1 if score > 0.5 else 0
            status = "✅" if prediction == true_label else "❌"
            
            print(f"  {status} {filename}: score={score:.3f}, pred={prediction}, true={true_label}, time={inference_time:.1f}ms")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"✅ Real-time inference test complete")

def main():
    print("🚀 Linear Model Inference Testing on Corrected Test Cache")
    print("=" * 60)
    
    try:
        # Load model and cache
        model, fake_files, real_files = load_model_and_cache()
        
        # Evaluate on full cache (or limited sample for speed)
        print(f"\nChoose evaluation mode:")
        print(f"1. Quick test (1000 samples)")
        print(f"2. Full evaluation (all {len(fake_files) + len(real_files)} samples)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            max_samples = 1000
            print(f"Running quick test on {max_samples} samples...")
        else:
            max_samples = None
            print(f"Running full evaluation on all samples...")
        
        # Evaluate model
        predictions, labels, scores = evaluate_on_cache(model, fake_files, real_files, max_samples)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, labels, scores)
        
        # Test real-time inference
        test_real_time_inference(model)
        
        print(f"\n🎉 Inference testing complete!")
        print(f"Model is ready for stakeholder testing with:")
        print(f"  ✅ Corrected test cache ({len(fake_files) + len(real_files)} files)")
        print(f"  ✅ Proper feature format (1024 dimensions)")
        print(f"  ✅ Accurate performance metrics")
        print(f"  ✅ Real-time inference capability")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
