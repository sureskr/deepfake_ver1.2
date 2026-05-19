import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
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

logger = logging.getLogger(__name__)


def _extract_linear_head_state_dict(checkpoint: object) -> dict:
    """
    Supports:
      - Full train checkpoint: checkpoint['model_state_dict']
      - Raw state dict: torch.save(model.state_dict(), ...) as used in finetune_v2 / finetune_continue / finetune_calibration
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)}")
    for key in ("model_state_dict", "state_dict", "model"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict) and inner and any(torch.is_tensor(v) for v in inner.values()):
            return inner
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    keys_preview = ", ".join(str(k) for k in list(checkpoint.keys())[:20])
    raise KeyError(
        "No weight dict found (expected model_state_dict, state_dict, model, or a raw state_dict). "
        f"Top-level keys: {keys_preview}"
    )


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


class MLPHeadLarge(nn.Module):
    """Same topology as train_v1_2_4.ClassifierHead (fc1→ReLU→fc2 + dropout)."""

    def __init__(
        self,
        in_dim: int = 1024,
        hidden: int = 256,
        dropout: float = 0.3,
        dropout_mid: float = 0.1,
    ):
        super().__init__()
        self.dropout_p_in = dropout
        self.dropout_p_mid = dropout_mid
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout_p_in, self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_p_mid, self.training)
        return self.fc2(x).squeeze(-1)


def _strip_head_prefix(state: dict) -> dict:
    """Allow state dicts saved as head.* or raw fc1./classifier. keys."""
    if not state:
        return state
    if not any(str(k).startswith("head.") for k in state):
        return state
    return {str(k)[len("head.") :]: v for k, v in state.items() if str(k).startswith("head.")}


def _build_head_from_checkpoint(checkpoint: dict, embed_dim: int) -> nn.Module:
    """
    train_v1_2_4 saves ``head_state_dict`` with fc1/fc2.
    Older runs use a flat dict with classifier.* only.
    """
    train_cfg = checkpoint.get("train_config") or {}
    hidden = int(train_cfg.get("head_hidden", 256))
    dropout = float(train_cfg.get("dropout", 0.3))
    dropout_mid = float(train_cfg.get("dropout_head_mid", 0.1))

    head_sd = checkpoint.get("head_state_dict")
    if isinstance(head_sd, dict) and head_sd:
        head_sd = _strip_head_prefix(head_sd)
        keys = set(head_sd.keys())
        if {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"} <= keys:
            head = MLPHeadLarge(
                embed_dim, hidden=hidden, dropout=dropout, dropout_mid=dropout_mid
            )
            head.load_state_dict(head_sd, strict=True)
            return head
        if {"classifier.weight", "classifier.bias"} <= keys:
            head = LinearHeadLarge(embed_dim)
            head.load_state_dict(head_sd, strict=True)
            return head
        raise ValueError(f"Unrecognized head_state_dict keys (sample): {list(keys)[:8]}")

    # Legacy: single linear inside model_state_dict / state_dict / top-level tensor dict
    state = _strip_head_prefix(_extract_linear_head_state_dict(checkpoint))
    if {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"} <= set(state.keys()):
        head = MLPHeadLarge(
            embed_dim, hidden=hidden, dropout=dropout, dropout_mid=dropout_mid
        )
        sub_mlp = {k: v for k, v in state.items() if k.startswith(("fc1.", "fc2."))}
        head.load_state_dict(sub_mlp, strict=True)
        return head
    if {"classifier.weight", "classifier.bias"} <= set(state.keys()):
        head = LinearHeadLarge(embed_dim)
        head.load_state_dict(
            {k: v for k, v in state.items() if k.startswith("classifier")}, strict=True
        )
        return head
    # Whole-model dict with many keys: try classifier only
    sub = {k: v for k, v in state.items() if k.startswith("classifier.")}
    if sub:
        head = LinearHeadLarge(embed_dim)
        head.load_state_dict(sub, strict=True)
        return head
    sub_mlp = {k: v for k, v in state.items() if k.startswith(("fc1.", "fc2."))}
    if len(sub_mlp) >= 4:
        head = MLPHeadLarge(
            embed_dim, hidden=hidden, dropout=dropout, dropout_mid=dropout_mid
        )
        head.load_state_dict(sub_mlp, strict=True)
        return head
    raise KeyError(
        "Could not find head weights (expected head_state_dict or classifier./fc* tensors)."
    )

class W2V2LargeEmbeddingDataset(Dataset):
    def __init__(self, data_dir: str, cache_dir: str, target_length: float = 10.0, sample_rate: int = 16000,
                 processor: Wav2Vec2FeatureExtractor | None = None, backbone: Wav2Vec2Model | None = None):
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
        # Fix local-only model names that aren't valid HF repo IDs
        if self.model_name.endswith('-local'):
            self.model_name = self.model_name.removesuffix('-local')

        # FeatureExtractor only (no Wav2Vec2Processor tokenizer) — avoids HF tokenizer/vocab NoneType errors on some transformers versions.
        # Set ML_WAV2VEC_LOCAL to a folder with config.json, preprocessor_config.json, and pytorch_model.bin / model.safetensors for offline use.
        local_w2v = os.environ.get("ML_WAV2VEC_LOCAL", "").strip()
        if local_w2v:
            local_root = str(Path(local_w2v).resolve())
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                local_root, local_files_only=True
            )
            self.backbone = Wav2Vec2Model.from_pretrained(
                local_root, local_files_only=True
            )
        else:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.backbone = Wav2Vec2Model.from_pretrained(self.model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Load the model (head + optional v1.2.4 fine-tuned encoder layers 23–24 / indices 22–23)
        self.model = self._load_model(checkpoint)
        self.model.eval()

    def _load_model(self, checkpoint: dict) -> nn.Module:
        model = _build_head_from_checkpoint(checkpoint, self.embed_dim)
        model = model.to(self.device)

        backbone_sd = checkpoint.get("backbone_state_dict")
        if backbone_sd is not None:
            filtered = {
                k: v
                for k, v in backbone_sd.items()
                if any(f"encoder.layers.{i}" in k for i in (22, 23))
            }
            if filtered:
                self.backbone.load_state_dict(filtered, strict=False)
                print("Loaded fine-tuned backbone layers 23-24")
            else:
                print(
                    "[WARN] backbone_state_dict present but no encoder.layers.22/23 tensors found; "
                    "using pretrained backbone only."
                )

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
