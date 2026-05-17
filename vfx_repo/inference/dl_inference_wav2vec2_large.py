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

def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold

class LinearHeadLarge(nn.Module):
    def __init__(self, in_dim: int = 1024):  # wav2vec2-large has 1024 dimensions
        super().__init__()
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)

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
            for p in real_dir.glob("*.*"):
                if p.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                    self.samples.append(p)
                    self.labels.append(0)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for p in fake_dir.glob("*.*"):
                if p.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
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
        wav, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
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

class DLInferenceLarge:
    def __init__(self, model_path, device=None):
        # Force CPU to avoid CUDA compatibility issues
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Set audio processing parameters
        self.sample_rate = 16000
        self.target_length = 10.0
        self.target_samples = int(self.sample_rate * self.target_length)
        
        # Load the checkpoint to get model info
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.embed_dim = checkpoint.get('embed_dim', 1024)  # Default to 1024 for large model
        self.model_name = checkpoint.get('model_name', 'facebook/wav2vec2-large-xlsr-53')
        
        # Load processor and backbone
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(self.model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Load the model
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path):
        model = LinearHeadLarge(self.embed_dim)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        return model
    
    def predict_single(self, audio_path):
        """Predict single audio file"""
        try:
            # Load and process audio directly
            wav, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            if len(wav) > self.target_samples:
                wav = wav[:self.target_samples]
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
                pooled = hidden.mean(dim=1).squeeze(0).unsqueeze(0).to(self.device)  # (1, feat)
                
                logits = self.model(pooled)
                score = torch.sigmoid(logits).item()
            
            return score
        except Exception as e:
            logger.error(f"DL prediction failed for {audio_path}: {e}")
            return 0.5
    
    def evaluate_dataset(self, test_dir):
        """Evaluate entire dataset"""
        print(f"Loading dataset from: {test_dir}")
        
        # Create cache directory for test
        test_cache = Path("models/w2v2_large_cache/test")
        test_cache.mkdir(parents=True, exist_ok=True)
        
        dataset = W2V2LargeEmbeddingDataset(test_dir, str(test_cache), 
                                          processor=self.processor, backbone=self.backbone)
        
        if len(dataset) == 0:
            print("No samples found!")
            return {}
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        all_predictions = []
        all_labels = []
        
        print("Running inference...")
        with torch.no_grad():
            for batch_idx, (emb, target) in enumerate(tqdm(dataloader, desc="Inference")):
                emb = emb.to(self.device)
                logits = self.model(emb)
                
                predictions = torch.sigmoid(logits).cpu().numpy()
                targets = target.numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(targets)
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_scores = np.array(all_predictions)
        
        # Calculate metrics
        y_pred = (y_scores > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        eer, eer_threshold = calculate_eer(y_true, y_scores)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'num_samples': len(y_true),
            'num_fake': int(sum(y_true)),
            'num_real': int(len(y_true) - sum(y_true)),
            'predictions': y_scores.tolist(),
            'targets': y_true.tolist()
        }
        
        print(f"\nResults (wav2vec2-large-xlsr-53):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"EER: {eer:.4f}")
        print(f"EER Threshold: {eer_threshold:.4f}")
        print(f"Total samples: {len(y_true)}")
        print(f"Fake samples: {sum(y_true)}")
        print(f"Real samples: {len(y_true) - sum(y_true)}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Inference with wav2vec2-large-xlsr-53 model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test directory')
    parser.add_argument('--out_json', type=str, help='Path to save metrics JSON')
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        return
    
    # Initialize inference
    inference = DLInferenceLarge(args.model_path)
    
    # Run evaluation
    metrics = inference.evaluate_dataset(args.test_dir)
    
    # Save results
    if args.out_json:
        os.makedirs(Path(args.out_json).parent, exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.out_json}")

if __name__ == "__main__":
    main()
