# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Setup Environment
```bash
# Activate your virtual environment
.\hamburg_project_2\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset (~5 minutes)
```bash
python scripts/download_dataset.py
```

### Step 3: Train Model (~40-60 minutes on Colab GPU)
```bash
python scripts/train.py
```

### Step 4: Evaluate Results
```bash
python scripts/evaluate.py --checkpoint models/checkpoints/best.pth --split test
```

---

## 📊 What to Expect

### Training Progress
- **Epoch 1**: mIoU ~35-40% (learning basic features)
- **Epoch 5**: mIoU ~55-65% (recognizing main classes)
- **Epoch 10**: mIoU ~70-75% (good performance)
- **Epoch 15-20**: mIoU ~75-85% (optimal)

### File Sizes
- Dataset: ~700MB
- Model checkpoint: ~100MB
- Training results: ~50MB

---

## 🔧 Troubleshooting

### Out of Memory Error
```python
# In config.py, reduce batch size:
BATCH_SIZE = 2  # Instead of 4
```

### Dataset Not Found
```bash
# Make sure you downloaded the dataset:
python scripts/download_dataset.py
```

### Slow Training
- ✅ Make sure you're using GPU (`device: cuda` should appear)
- ✅ If on Colab, ensure GPU runtime is selected: Runtime → Change runtime type → GPU

---

## 📝 For Your Portfolio

### What to Include:
1. **README.md** - Project description and results
2. **Training curves** - `results/training_curves.png`
3. **Sample predictions** - Best examples from `results/predictions/`
4. **Final metrics** - Copy terminal output from evaluation
5. **Code structure** - Show clean, professional organization

### Tips for Hamburg Application:
- Emphasize **85%+ mIoU** achievement
- Show **complete pipeline** (data → training → inference → navigation)
- Highlight **production-ready code** quality
- Mention **real-world application** (robot navigation)

---

## 🎯 Next Steps

After basic training:
1. **Visualize results**: Check `results/predictions/`
2. **Try inference**: Use notebook `notebooks/complete_training_pipeline.ipynb`
3. **Optimize**: Experiment with hyperparameters in `config.py`
4. **Deploy**: Try the Colab notebook for demos

Good luck! 🚀
