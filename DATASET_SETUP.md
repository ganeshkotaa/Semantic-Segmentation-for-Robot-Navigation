# CamVid Dataset Setup Guide

## Quick Setup (Recommended)

### Step 1: Download from Kaggle
1. Visit: **https://www.kaggle.com/datasets/carlolepelaars/camvid**
2. Click **"Download"** button (requires free Kaggle account)
3. Save `camvid.zip` to your Downloads folder

### Step 2: Extract to Project
Extract the downloaded ZIP file to:
```
C:\Users\jskota\OneDrive - Hexagon\Desktop\ham\data\raw\camvid\
```

### Step 3: Verify Structure
After extraction, your folder structure should look like:
```
data/raw/camvid/
├── train/          (367 images: .png files)
├── trainannot/     (367 labels: .png files)
├── val/            (101 images)
├── valannot/       (101 labels)
├── test/           (233 images)
└── testannot/      (233 labels)
```

### Step 4: Verify Installation
Run this command to verify the dataset:
```powershell
python scripts/verify_dataset.py
```

## Expected Output
```
✓ Found 367 training images
✓ Found 367 training labels
✓ Found 101 validation images
✓ Found 101 validation labels
✓ Found 233 test images
✓ Found 233 test labels

Dataset is ready! Total: 701 images
```

## Alternative: Use Automated Script
If the Kaggle download doesn't work, try:
```powershell
python scripts/download_dataset.py
```

## Troubleshooting

### Problem: Wrong folder structure
If you extracted and have `camvid/camvid/train/` (double nested), move the inner folder:
```powershell
Move-Item "data\raw\camvid\camvid\*" "data\raw\camvid\" -Force
Remove-Item "data\raw\camvid\camvid" -Recurse
```

### Problem: Missing folders
Create the required directory:
```powershell
New-Item -ItemType Directory -Force -Path "data\raw\camvid"
```

### Problem: Can't access Kaggle
Try these alternative mirrors:
- Google Drive: https://drive.google.com/drive/folders/0B0ZXjo_p8lHBfms3X0JJbGFYQ2M1WWxNWEpJN0VfY3B4YlJuaGFnQUJvNDBBMWM1NVJxeWc
- GitHub Mirror: https://github.com/mostafaizz/camvid
- Official: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

## Next Steps
Once dataset is ready:
1. ✅ Visualize samples: `python scripts/visualize_dataset.py`
2. ✅ Start training: `python scripts/train.py`
3. ✅ Run inference: `python scripts/inference.py`

---
**Need Help?** Check that all 6 folders (train, trainannot, val, valannot, test, testannot) exist and contain .png files.
