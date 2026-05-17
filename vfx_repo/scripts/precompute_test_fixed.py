#!/usr/bin/env python3
"""
Regenerate precomputed embeddings for test dataset
Optimized version for large datasets (30K+ files)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import hashlib
import json
from tqdm import tqdm
import gc

def load_audio_file(file_path, target_sr=16000, min_duration=0.5):
    """Load audio file with proper error handling and minimum duration check"""
    try:
        # Try librosa first
        audio, sr = librosa.load(str(file_path), sr=target_sr)
        
        # Check if audio is too short
        duration = len(audio) / target_sr
        if duration < min_duration:
            print(f"Warning: {file_path.name} is too short ({duration:.3f}s), skipping")
            return None
            
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
            
        return audio
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features_batch(audio_batch, processor, model, device, max_length=160000):
    """Extract features from a batch of audio files"""
    try:
        processed_audio = []
        
        for audio in audio_batch:
            if audio is None:
                processed_audio.append(None)
                continue
                
            # Ensure audio is the right shape and type
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Truncate or pad to max_length
            if len(audio) > max_length:
                audio = audio[:max_length]
            elif len(audio) < max_length:
                # Pad with zeros
                audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
            
            processed_audio.append(audio)
        
        # Filter out None values
        valid_audio = [a for a in processed_audio if a is not None]
        if not valid_audio:
            return [None] * len(audio_batch)
        
        # Convert to tensor
        audio_tensor = torch.tensor(valid_audio, dtype=torch.float32)
        audio_tensor = audio_tensor.to(device)
        
        # Process with Wav2Vec2
        with torch.no_grad():
            inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Extract features
            outputs = model(**inputs, output_hidden_states=True)
            
            # Use the last hidden state
            features = outputs.hidden_states[-1].mean(dim=1)  # Average over time
            features = features.cpu().numpy()
            
            # Map back to original batch
            result = [None] * len(audio_batch)
            valid_idx = 0
            for i, audio in enumerate(processed_audio):
                if audio is not None:
                    result[i] = features[valid_idx]
                    valid_idx += 1
            
            return result
            
    except Exception as e:
        print(f"Error extracting features: {e}")
        return [None] * len(audio_batch)

def process_directory(input_dir, output_dir, processor, model, device, batch_size=32):
    """Process files in a directory in batches"""
    files = list(input_dir.glob("*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in ['.mp3', '.wav', '.flac', '.m4a']]
    
    successful = 0
    total_files = len(files)
    
    print(f"Processing {total_files} files in {input_dir.name} directory...")
    
    # Process in batches
    for i in tqdm(range(0, total_files, batch_size), desc=f"Processing {input_dir.name}"):
        batch_files = files[i:i + batch_size]
        
        # Load audio for this batch
        audio_batch = []
        for file_path in batch_files:
            audio = load_audio_file(file_path)
            audio_batch.append(audio)
        
        # Extract features for this batch
        features_batch = extract_features_batch(audio_batch, processor, model, device)
        
        # Save features for this batch
        for j, (file_path, features) in enumerate(zip(batch_files, features_batch)):
            if features is not None:
                output_path = output_dir / f"{file_path.stem}.pt"
                torch.save(features, output_path)
                successful += 1
        
        # Clear memory
        del audio_batch, features_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return successful

def main():
    print("=== Regenerating Test Cache (Optimized) ===")
    
    # Setup paths
    base_dir = Path("/home/surya/vfx")
    test_dir = base_dir / "data" / "test"
    cache_dir = base_dir / "cache" / "emb" / "test"
    
    # Clean up existing test cache
    if cache_dir.exists():
        print(f"Removing existing test cache: {cache_dir}")
        import shutil
        shutil.rmtree(cache_dir)
    
    # Create fresh cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    fake_cache_dir = cache_dir / "fake"
    real_cache_dir = cache_dir / "real"
    fake_cache_dir.mkdir(exist_ok=True)
    real_cache_dir.mkdir(exist_ok=True)
    
    # Load model and processor
    print("Loading Wav2Vec2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    
    # Process fake files
    print("\nProcessing fake files...")
    successful_fake = process_directory(
        test_dir / "fake", 
        fake_cache_dir, 
        processor, 
        model, 
        device, 
        batch_size=16  # Smaller batch size for memory efficiency
    )
    
    # Process real files
    print("\nProcessing real files...")
    successful_real = process_directory(
        test_dir / "real", 
        real_cache_dir, 
        processor, 
        model, 
        device, 
        batch_size=16
    )
    
    print(f"\n=== Cache Generation Complete ===")
    print(f"Successful fake files: {successful_fake}")
    print(f"Successful real files: {successful_real}")
    print(f"Total cached: {successful_fake + successful_real}")
    
    # Verify cache
    fake_cache_files = list(fake_cache_dir.glob("*.pt"))
    real_cache_files = list(real_cache_dir.glob("*.pt"))
    
    print(f"Cache verification:")
    print(f"  Fake cache: {len(fake_cache_files)} files")
    print(f"  Real cache: {len(real_cache_files)} files")
    
    if fake_cache_files and real_cache_files:
        # Check first file to verify format
        sample_features = torch.load(fake_cache_files[0])
        print(f"  Feature shape: {sample_features.shape}")
        print(f"  Feature type: {type(sample_features)}")
        print(f"  Feature dtype: {sample_features.dtype}")

if __name__ == "__main__":
    main()
