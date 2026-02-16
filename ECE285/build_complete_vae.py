#!/usr/bin/env python3
"""Build complete optimized VAE notebook"""
import json

def create_complete_notebook():
    # All cell contents
    cells_data = [
        ("markdown", "# VAE for Chest X-Ray Generation\n**ECE 285 - Deep Generative Models**\n\nOptimized implementation with:\n- Spatial latent space for preserving structure\n- Perceptual loss (VGG) for sharp images\n- GroupNorm + SiLU for stability\n- Low beta (0.0001-0.001) for reconstruction quality\n- Two-phase training strategy"),
        
        ("code", """import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import cv2
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')"""),
        
        ("markdown", "## Configuration"),
        
        ("code", """CONFIG = {
    'img_size': 128,
    'latent_spatial': (4, 4),
    'latent_channels': 64,
    'batch_size': 32,
    'epochs_phase1': 50,
    'epochs_phase2': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'beta_start': 0.0,
    'beta_end': 0.001,
    'beta_warmup': 50,
    'num_workers': 2,
    'l1_weight': 1.0,
    'perceptual_weight': 0.1,
    'ssim_weight': 0.5,
}

DATA_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray'
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/results', exist_ok=True)

print('Configuration:')
for k, v in CONFIG.items():
    print(f'  {k}: {v}')"""),
        
        ("markdown", "## Dataset"),
        
        ("code", """def apply_clahe(img_np, clip_limit=2.0, tile_size=(8, 8)):
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img_np)

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, img_size=128, split='train'):
        self.img_size = img_size
        self.is_training = (split == 'train')
        self.image_paths = []
        
        split_dir = os.path.join(root_dir, split)
        for category in ['NORMAL', 'PNEUMONIA']:
            cat_path = os.path.join(split_dir, category)
            if os.path.exists(cat_path):
                for ext in ['*.jpeg', '*.jpg', '*.png']:
                    self.image_paths.extend(glob.glob(os.path.join(cat_path, ext)))
        
        print(f'{split}: {len(self.image_paths)} images')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('L')
            img_np = apply_clahe(np.array(img))
            img = Image.fromarray(img_np)
            img = TF.resize(img, (self.img_size, self.img_size))
            
            if self.is_training and np.random.random() > 0.5:
                img = TF.hflip(img)
            
            return TF.to_tensor(img)
        except:
            return self.__getitem__((idx + 1) % len(self))

train_dataset = ChestXRayDataset(DATA_DIR, CONFIG['img_size'], 'train')
val_dataset = ChestXRayDataset(DATA_DIR, CONFIG['img_size'], 'val')
test_dataset = ChestXRayDataset(DATA_DIR, CONFIG['img_size'], 'test')

train_loader = DataLoader(train_dataset, CONFIG['batch_size'], shuffle=True, 
                         num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, CONFIG['batch_size'], shuffle=False,
                       num_workers=CONFIG['num_workers'], pin_memory=True)
test_loader = DataLoader(test_dataset, CONFIG['batch_size'], shuffle=False,
                        num_workers=CONFIG['num_workers'], pin_memory=True)"""),
        
        ("code", """# Visualize samples
sample = next(iter(train_loader))
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.suptitle('Sample Training Images')
plt.tight_layout()
plt.savefig('/kaggle/working/results/sample_data.png', dpi=150)
plt.show()

print(f'Stats - Mean: {sample.mean():.3f}, Std: {sample.std():.3f}')"""),
    ]
    
    # Convert to notebook format
    cells = []
    for cell_type, content in cells_data:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [content]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Save notebook
    nb_path = '/Users/raghu/coding/scratchpad/ECE285/vae_chest_xray.ipynb'
    with open(nb_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✓ Created notebook with {len(cells)} cells at {nb_path}")

if __name__ == '__main__':
    create_complete_notebook()

