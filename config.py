"""
Configuration file for Semantic Segmentation for Robot Navigation
All hyperparameters and paths are defined here for easy modification
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Results paths
RESULTS_DIR = BASE_DIR / "results"
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, 
                  TENSORBOARD_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Dataset choice: 'camvid' (recommended for Colab), 'cityscapes', or 'custom'
DATASET_NAME = "camvid"

# CamVid Dataset URLs - Using reliable mirror
# Original source was github.com/alexgkendall/SegNet-Tutorial but links are broken
# Alternative: Download manually from Kaggle if automated download fails
CAMVID_URLS = {
    "train": "https://github.com/davidtvs/kaggle-camvid/raw/master/data/train.zip",
    "val": "https://github.com/davidtvs/kaggle-camvid/raw/master/data/val.zip", 
    "test": "https://github.com/davidtvs/kaggle-camvid/raw/master/data/test.zip",
    "trainannot": "https://github.com/davidtvs/kaggle-camvid/raw/master/data/trainannot.zip",
    "valannot": "https://github.com/davidtvs/kaggle-camvid/raw/master/data/valannot.zip",
    "testannot": "https://github.com/davidtvs/kaggle-camvid/raw/master/data/testannot.zip"
}

# CamVid class labels (32 classes originally, but we'll use 12 main classes)
CAMVID_CLASSES = {
    0: 'Sky',
    1: 'Building',
    2: 'Pole',
    3: 'Road',
    4: 'Pavement',
    5: 'Tree',
    6: 'SignSymbol',
    7: 'Fence',
    8: 'Car',
    9: 'Pedestrian',
    10: 'Bicyclist',
    11: 'Unlabelled'
}

# Simplified navigation-focused classes (binary or 3-class segmentation)
NAVIGATION_CLASSES = {
    0: 'Non-Navigable',  # obstacles, buildings, poles, fences, cars, pedestrians
    1: 'Navigable',      # roads, pavements
    2: 'Caution'         # trees, signs (might need caution)
}

# Number of classes for segmentation
NUM_CLASSES = len(CAMVID_CLASSES)  # Use 12 for full CamVid, or 3 for simplified

# Image dimensions
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
INPUT_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# ═══════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Model architecture
MODEL_NAME = "deeplabv3plus_resnet50"
BACKBONE = "resnet50"
PRETRAINED_BACKBONE = True  # Use ImageNet pretrained weights

# Output stride (8 or 16) - lower = higher resolution but more memory
OUTPUT_STRIDE = 16

# ═══════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Training settings
BATCH_SIZE = 4  # Adjust based on GPU memory (Colab free tier: 4-8)
NUM_EPOCHS = 20  # 20 epochs should be enough for good results
NUM_WORKERS = 2  # Data loading workers (0 on Windows, 2-4 on Linux/Colab)

# Optimizer settings
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Learning rate scheduler
LR_SCHEDULER = "poly"  # Options: 'poly', 'step', 'cosine'
LR_POWER = 0.9  # For poly scheduler
LR_STEP_SIZE = 10  # For step scheduler
LR_GAMMA = 0.1  # For step scheduler

# Loss function
LOSS_FUNCTION = "cross_entropy"  # Options: 'cross_entropy', 'focal'
CLASS_WEIGHTS = None  # Set to list for weighted loss, None for uniform

# ═══════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

# Training augmentation
TRAIN_AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "random_crop": False,  # Use resize instead
    "color_jitter": True,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "random_rotation": 5,  # degrees
    "normalize": True
}

# Validation augmentation (minimal)
VAL_AUGMENTATION = {
    "normalize": True
}

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ═══════════════════════════════════════════════════════════════════
# TRAINING SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Checkpoint settings
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQUENCY = 5  # Save every N epochs
SAVE_BEST_ONLY = True  # Save only when validation mIoU improves

# Early stopping
EARLY_STOPPING = True
PATIENCE = 7  # Stop if no improvement for N epochs

# Mixed precision training (faster on modern GPUs)
USE_AMP = True  # Automatic Mixed Precision

# Gradient clipping
CLIP_GRAD_NORM = 1.0

# ═══════════════════════════════════════════════════════════════════
# EVALUATION SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Metrics to compute
METRICS = ["miou", "pixel_accuracy", "dice"]

# Evaluation frequency
EVAL_FREQUENCY = 1  # Evaluate every N epochs

# ═══════════════════════════════════════════════════════════════════
# ROBOT NAVIGATION SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Cost map generation
COST_MAP_SIZE = (100, 100)  # Downsampled grid size for path planning

# Cost values for different classes
COST_VALUES = {
    'Road': 1,        # Low cost - easily navigable
    'Pavement': 2,    # Low cost - sidewalks
    'Building': 100,  # High cost - obstacle
    'Pole': 100,      # High cost - obstacle
    'Car': 100,       # High cost - obstacle
    'Pedestrian': 50, # Medium-high cost - avoid but not solid obstacle
    'Tree': 30,       # Medium cost - might be passable
    'Fence': 100,     # High cost - obstacle
    'Sky': 5,         # Low cost (if mistakenly classified)
    'SignSymbol': 20, # Low-medium cost
    'Bicyclist': 50,  # Medium-high cost
    'Unlabelled': 10  # Default cost
}

# A* path planning settings
ALLOW_DIAGONAL = True  # Allow diagonal movement
HEURISTIC = "euclidean"  # Options: 'euclidean', 'manhattan', 'octile'

# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Color map for visualization (RGB colors for each class)
CLASS_COLORS = [
    [128, 128, 128],  # Sky - gray
    [128, 0, 0],      # Building - dark red
    [192, 192, 128],  # Pole - tan
    [128, 64, 128],   # Road - purple
    [60, 40, 222],    # Pavement - blue
    [128, 128, 0],    # Tree - olive
    [192, 128, 128],  # SignSymbol - pink
    [64, 64, 128],    # Fence - dark blue
    [64, 0, 128],     # Car - purple-blue
    [64, 64, 0],      # Pedestrian - dark olive
    [0, 128, 192],    # Bicyclist - cyan
    [0, 0, 0]         # Unlabelled - black
]

# Visualization settings
SAVE_PREDICTIONS = True
PREDICTION_OPACITY = 0.6  # Overlay opacity (0-1)
SAVE_FORMAT = "png"

# ═══════════════════════════════════════════════════════════════════
# RANDOM SEED (for reproducibility)
# ═══════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Will be set automatically in training script
DEVICE = "cuda"  # or "cpu"

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

# TensorBoard logging
USE_TENSORBOARD = True
LOG_INTERVAL = 10  # Log every N batches

# Console logging
VERBOSE = True
