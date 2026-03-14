import sys
sys.path.append('.')
from utils.dataset import CamVidDataset, get_train_transform
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load ONE sample with new fix
dataset = CamVidDataset(
    root_dir='data/raw/camvid',
    split='train',
    transform=get_train_transform(),
    img_size=(360, 480)
)

sample = dataset[0]
image = sample['image']
label = sample['label']

print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")
print(f"Label dtype: {label.dtype}")
print(f"\nUnique classes in label: {torch.unique(label).tolist()}")
print(f"\nClass distribution:")
for c in torch.unique(label):
    count = (label == c).sum().item()
    pct = count / label.numel() * 100
    class_name = ['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'Sign', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Unlabelled'][c]
    print(f"  Class {c} ({class_name}): {pct:.1f}%")

# Check if Road class exists
has_road = 3 in torch.unique(label)
print(f"\n{'✓' if has_road else '✗'} Road class (3) detected: {has_road}")
print(f"Result: {'WILL WORK!' if has_road else 'PROBLEM!'}")
