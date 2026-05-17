#!/usr/bin/env python3
"""
Run inference on unseen test dataset using trained VeriFauX Phase 4 model
"""

import tempfile
import shutil
import pandas as pd
from pathlib import Path
import subprocess

def create_test_dir_for_inference(test_csv, corpus_root, manifests_dir):
    """Create temporary directory with test audio files for inference"""
    
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
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="verifax_test_inference_"))
    test_dir = temp_dir / "test"
    (test_dir / "real").mkdir(parents=True, exist_ok=True)
    (test_dir / "fake").mkdir(parents=True, exist_ok=True)
    
    print(f"Created temporary directory: {temp_dir}")
    
    # Copy test files
    df = pd.read_csv(test_csv, low_memory=False)
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
                        target_file = test_dir / target_subdir / sha_filename
                        
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
    
    print(f"Copied {copied} test files")
    return temp_dir, test_dir

def main():
    test_csv = "data/corpus/manifests/split.v1_2_2.paths_resolved.test.csv"
    corpus_root = "data/corpus/audio/v1_2_2"
    manifests_dir = "data/corpus/manifests"
    model_path = "runs/v1_2_2_prod/best.pt"
    
    print("=== VeriFauX Phase 4 Test Inference ===")
    print(f"Test CSV: {test_csv}")
    print(f"Corpus root: {corpus_root}")
    print(f"Model path: {model_path}")
    print()
    
    try:
        # Create temporary directory with test audio files
        temp_dir, test_dir = create_test_dir_for_inference(test_csv, corpus_root, manifests_dir)
        
        # Run inference
        cmd = [
            "python", "dl_inference_wav2vec2_large.py",
            "--model_path", model_path,
            "--test_dir", str(test_dir),
            "--out_json", "runs/v1_2_2_prod/test_inference_results.json"
        ]
        
        print(f"Running inference: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Inference completed successfully!")
        print("STDOUT:", result.stdout)
        
        # Show results
        results_file = "runs/v1_2_2_prod/test_inference_results.json"
        if Path(results_file).exists():
            print(f"\nResults saved to: {results_file}")
            
            # Display key metrics
            import json
            with open(results_file, 'r') as f:
                metrics = json.load(f)
            
            print("\n" + "="*50)
            print("🎯 UNSEEN TEST SET RESULTS")
            print("="*50)
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"EER: {metrics['eer']:.4f}")
            print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
            print(f"Total samples: {metrics['num_samples']}")
            print(f"Fake samples: {metrics['num_fake']}")
            print(f"Real samples: {metrics['num_real']}")
            print("="*50)
        
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'temp_dir' in locals():
            print(f"\nCleaning up: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up {temp_dir}: {e}")

if __name__ == "__main__":
    main()



