#!/usr/bin/env python3
"""
Custom script to precompute embeddings for codec-mixed validation set only
"""
import os
import torch
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodecValDataset(Dataset):
    def __init__(self, data_dir, target_length=16000*10, sample_rate=16000):
        self.data_dir = Path(data_dir)
        self.target_length = target_length
        self.sample_rate = sample_rate
        
        # Find all audio files
        self.files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.files.extend(list(self.data_dir.rglob(ext)))
        
        logger.info(f"Found {len(self.files)} audio files in {data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Load audio
        try:
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            # Return dummy audio if loading fails
            audio = np.zeros(self.target_length // self.sample_rate)
            sr = self.sample_rate
        
        # Pad or truncate to target length
        if len(audio) < self.target_length // self.sample_rate:
            audio = np.pad(audio, (0, self.target_length // self.sample_rate - len(audio)))
        else:
            audio = audio[:self.target_length // self.sample_rate]
        
        return {
            'audio': audio,
            'path': str(file_path),
            'sr': sr
        }

def precompute_codec_val(val_dir, cache_dir, target_length=16000*10, sample_rate=16000, batch_size=32):
    """Precompute embeddings for codec-mixed validation set"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load processor and model
    logger.info("Loading Wav2Vec2 processor and model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
    backbone.to(device)
    backbone.eval()
    
    # Create dataset and dataloader
    dataset = CodecValDataset(val_dir, target_length, sample_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create cache directory
    cache_path = Path(cache_dir) / "val"
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Precompute embeddings
    logger.info(f"Precomputing embeddings for {len(dataset)} files...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
        
        audio = batch['audio'].to(device)
        paths = batch['path']
        
        with torch.no_grad():
            # Process audio through Wav2Vec2 - fix tensor shapes
            # audio should be [batch_size, samples]
            if audio.dim() == 3:
                audio = audio.squeeze(1)  # Remove extra dimension if present
            
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Extract embeddings
            outputs = backbone(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        
        # Save embeddings
        for i, (emb, path) in enumerate(zip(embeddings, paths)):
            # Determine label from path
            if '/fake/' in path.lower():
                label = 1
            elif '/real/' in path.lower():
                label = 0
            else:
                label = -1  # Unknown
            
            # Create filename based on path hash
            rel_path = Path(path).relative_to(val_dir)
            cache_file = cache_path / f"{hash(str(rel_path)) % 1000000:06d}.pt"
            
            # Save embedding with metadata
            torch.save({
                'emb': emb.cpu(),
                'y': label,
                'src': str(rel_path)
            }, cache_file)
    
    logger.info(f"✅ Precompute completed! Saved to {cache_path}")
    logger.info(f"📊 Files processed: {len(dataset)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute embeddings for codec-mixed validation")
    parser.add_argument("--val_dir", required=True, help="Directory with codec-mixed validation files")
    parser.add_argument("--cache_dir", required=True, help="Directory to save embeddings")
    parser.add_argument("--target_length", type=int, default=16000*10, help="Target audio length in samples")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    precompute_codec_val(
        val_dir=args.val_dir,
        cache_dir=args.cache_dir,
        target_length=args.target_length,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size
    )
