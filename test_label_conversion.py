import sys
sys.path.append('.')
from utils.dataset import rgb_to_class_index
import numpy as np
from PIL import Image

# Load a real label
label = np.array(Image.open('data/raw/camvid/trainannot/0001TP_009210_L.png'))
print(f"Original label shape: {label.shape}")
print(f"Original unique values: {np.unique(label)}")

# Convert RGB to class
class_label = rgb_to_class_index(label)
print(f"\nConverted label shape: {class_label.shape}")
print(f"Converted unique classes: {np.unique(class_label)}")
print(f"\nClass distribution:")
for c in np.unique(class_label):
    count = (class_label == c).sum()
    pct = count / class_label.size * 100
    print(f"  Class {c}: {count} pixels ({pct:.1f}%)")
