# 🚀 VeriFauX v1.2.2 Cloud Training - Clean & Focused

## 🎯 **What This Is**

This directory contains **clean, focused scripts** to retrain your **working VeriFauX v1.2.2 model** on cloud GPU. 

**Goal**: Improve 71.8% F1 → 75-80% F1 using the exact same architecture that worked.

## 📁 **Clean File Structure**

```
cloud_staging/
├── ✅ dl_audio_wav2vec2_large_enhanced.py  # WORKING v1.2.2 training script (71.8% F1)
├── ✅ dl_inference_wav2vec2_large.py       # WORKING v1.2.2 inference script
├── ✅ run_test_inference.py                # Test inference orchestrator
├── 🧪 train_verifax_v1_2_2_cloud.py       # Cloud-optimized training (alternative)
├── 🚀 deploy_v1_2_2_cloud.py              # Cloud deployment automation
├── 📦 requirements.txt                     # Dependencies
└── 📚 README.md                            # This file
```

## 🔍 **Previous Training Analysis (CLEAR CONCLUSION)**

### **✅ What Worked in v1.2.2**
- **Script**: `dl_audio_wav2vec2_large_enhanced.py`
- **Architecture**: Simple `LinearHeadLarge` (1024 → 1)
- **Backbone**: `wav2vec2-large-960h` (frozen)
- **Pooling**: Global average pooling (1024 features)
- **Results**: 71.8% F1, 28.7% EER

### **📊 Dataset Used**
- **Train**: 286,098 samples (143,049 REAL + 143,049 FAKE)
- **Val**: 61,306 samples (30,653 REAL + 30,653 FAKE)
- **Test**: 61,308 samples (30,654 REAL + 30,654 FAKE)

### **🎯 Why Retrain on Cloud GPU?**
- **Current**: CPU training (slower, less efficient)
- **Target**: GPU training (10-50x faster, better optimization)
- **Expected**: 71.8% → 75-80% F1, 28.7% → 20-25% EER

## 🎯 **Which Scripts to Use for Retraining?**

### **Option A: Use EXACT Working Script (Recommended)**
```bash
# Use the EXACT script that achieved 71.8% F1
python dl_audio_wav2vec2_large_enhanced.py \
  --train_csv /path/to/split.v1_2_2.paths_resolved.train.csv \
  --val_csv /path/to/split.v1_2_2.paths_resolved.val.csv \
  --test_csv /path/to/split.v1_2_2.paths_resolved.test.csv \
  --corpus_root /path/to/v1_2_2 \
  --epochs 50 \
  --batch_size 32 \
  --seed 1337
```

### **Option B: Use Cloud-Optimized Script (Alternative)**
```bash
# Use cloud-optimized version with GPU improvements
python train_verifax_v1_2_2_cloud.py \
  --train_dir /path/to/train \
  --val_dir /path/to/val \
  --epochs 50 \
  --batch_size 64 \
  --device cuda \
  --seed 1337
```

## 🚀 **Quick Start**

### **1. Analyze Previous Training**
```bash
cd cloud_staging
python deploy_v1_2_2_cloud.py
```

### **2. Create Data Preparation Script**
```bash
python deploy_v1_2_2_cloud.py --create-data-prep
```

### **3. Create Cloud Deployment Scripts**
```bash
python deploy_v1_2_2_cloud.py --create-cloud-scripts --project-id YOUR_GCP_PROJECT_ID
```

## 📋 **Files to Move to Cloud**

### **Essential Training Files (Choose One Approach)**

**Option A: EXACT Working Scripts**
1. **`dl_audio_wav2vec2_large_enhanced.py`** - EXACT working training script
2. **`dl_inference_wav2vec2_large.py`** - EXACT working inference script
3. **`run_test_inference.py`** - Test orchestrator
4. **Manifest CSVs**: `split.v1_2_2.paths_resolved.{train,val,test}.csv`
5. **Audio data**: `v1_2_2/` directory structure

**Option B: Cloud-Optimized Scripts**
1. **`train_verifax_v1_2_2_cloud.py`** - Cloud-optimized training script
2. **`cloud_training_data/v1_2_2/`** - Prepared train/val directories

**Always Include**
1. **`requirements_v1_2_2_cloud.txt`** - Dependencies

### **Generated Cloud Scripts**
1. **`v1_2_2_cloud_startup.sh`** - VM startup script
2. **`deploy_v1_2_2_to_gcp.sh`** - GCP deployment script

## 🎯 **Training Command (Cloud)**

```bash
python train_verifax_v1_2_2_cloud.py \
  --train_dir v1_2_2/train \
  --val_dir v1_2_2/val \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001 \
  --seed 1337 \
  --patience 8 \
  --device cuda
```

## 📊 **Expected Results**

| Metric | v1.2.2 (CPU) | Target (GPU) | Improvement |
|--------|---------------|--------------|-------------|
| **F1-Score** | 71.8% | **75-80%** | +3-8% |
| **EER** | 28.7% | **20-25%** | -8-9% |
| **Training Time** | 2-4 hours | **30-60 min** | 10-50x faster |

## 💡 **Key Benefits**

1. **✅ Proven Architecture**: Uses exact working v1.2.2 setup
2. **🚀 GPU Acceleration**: 10-50x faster training
3. **📊 Clear Baseline**: Direct comparison with 71.8% F1
4. **🎯 Specific Targets**: 75-80% F1, 20-25% EER
5. **🔄 Same Data**: Identical dataset, better training

## 🆘 **Support**

- **Analysis**: `python deploy_v1_2_2_cloud.py`
- **Data Prep**: `python deploy_v1_2_2_cloud.py --create-data-prep`
- **Cloud Scripts**: `python deploy_v1_2_2_cloud.py --create-cloud-scripts --project-id YOUR_ID`

---

**🎉 You're ready to retrain your working v1.2.2 model on cloud GPU!**
