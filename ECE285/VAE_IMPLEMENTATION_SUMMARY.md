# Optimized VAE for Chest X-Ray Generation - Implementation Summary

## ✓ Complete Notebook Created
**Location:** `vae_chest_xray.ipynb` (25 cells total)

---

## Key Improvements Implemented

### 1. **Spatial Latent Space** (4×4×64 channels)
- Preserves spatial structure better than flattened vectors
- Total latent size: 1,024 dimensions
- Maintains anatomical relationships in chest X-rays

### 2. **Advanced Loss Function**
- **L1 Loss** (weight: 1.0) - Sharper edges than MSE
- **Perceptual Loss** (weight: 0.1) - VGG features from layers 3, 8, 15
- **SSIM Loss** (weight: 0.5) - Preserves luminance, contrast, structure
- **KL Divergence** (beta: 0.0001 → 0.001) - Very low for sharp images

### 3. **Architecture Enhancements**
- **GroupNorm** instead of BatchNorm (more stable for small batches)
- **SiLU/Swish activation** instead of LeakyReLU (smoother gradients)
- **Attention blocks** at 16×16 resolution for global context
- **Bilinear upsampling** instead of transposed convolutions (no checkerboard artifacts)
- **Skip connections** in ResBlocks for better gradient flow

### 4. **Two-Phase Training Strategy**
- **Phase 1** (50 epochs): Reconstruction only (beta=0)
  - Learns to encode/decode accurately
  - No latent space regularization
- **Phase 2** (100 epochs): VAE training with KL
  - Beta anneals from 0 to 0.001 over 50 epochs
  - Maintains reconstruction quality while regularizing latent space

### 5. **Training Optimizations**
- **AdamW optimizer** with weight decay (1e-5)
- **Learning rate**: 1e-4 (phase 1), 5e-5 (phase 2)
- **Gradient clipping** (max_norm=1.0)
- **Cosine annealing** schedule with warm restarts
- **ReduceLROnPlateau** for phase 2

### 6. **Data Preprocessing**
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
  - Essential for X-ray contrast normalization
  - Clip limit: 2.0, Tile size: 8×8
- **Minimal augmentation**: Horizontal flip only
- **No normalization**: Keep pixel values in [0, 1]

---

## Architecture Details

### Encoder Path (128 → 4)
```
Input (128×128)
  ↓ Conv 7×7, stride 2, GroupNorm, SiLU → (64×64×32)
  ↓ ResBlock, stride 2 → (32×32×64)
  ↓ ResBlock, stride 2 → (16×16×128)
  ↓ ResBlock, stride 2 + Attention → (8×8×256)
  ↓ ResBlock, stride 2 → (4×4×512)
  ↓ Convs → mu, logvar (4×4×64)
```

### Decoder Path (4 → 128)
```
Latent z (4×4×64)
  ↓ Conv → (4×4×512)
  ↓ Upsample + ResBlock → (8×8×256)
  ↓ Upsample + ResBlock + Attention → (16×16×128)
  ↓ Upsample + ResBlock → (32×32×64)
  ↓ Upsample + ResBlock → (64×64×32)
  ↓ Upsample + Convs + Sigmoid → (128×128×1)
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image Size | 128×128 | Balance between detail and compute |
| Latent Channels | 64 | Sufficient for medical image representation |
| Batch Size | 32 | GPU memory efficient |
| Epochs Phase 1 | 50 | Learn reconstruction first |
| Epochs Phase 2 | 100 | Fine-tune with regularization |
| Learning Rate | 1e-4 | Conservative, stable |
| Beta (KL weight) | 0.001 | **Very low** for sharp images |
| L1 Weight | 1.0 | Primary reconstruction loss |
| Perceptual Weight | 0.1 | High-level structure |
| SSIM Weight | 0.5 | Structural similarity |

---

## Why These Changes Matter

### 1. **Spatial Latent Space**
- **Problem**: Flattened vectors lose spatial relationships
- **Solution**: Keep 4×4 spatial structure
- **Result**: Better anatomical coherence in generated X-rays

### 2. **Very Low Beta (0.001)**
- **Problem**: Standard beta (0.5-1.0) over-regularizes, causing blurriness
- **Solution**: Use beta=0.001, prioritizing reconstruction
- **Result**: Sharp, detailed images while maintaining valid latent space

### 3. **Perceptual Loss**
- **Problem**: MSE/L1 alone average over pixel-level uncertainty
- **Solution**: Match high-level VGG features
- **Result**: Perceptually realistic images with correct global structure

### 4. **L1 instead of MSE**
- **Problem**: MSE penalizes outliers quadratically → over-smoothing
- **Solution**: L1 loss is less sensitive to outliers
- **Result**: Sharper edges and finer details

### 5. **Two-Phase Training**
- **Problem**: Simultaneous reconstruction + regularization is unstable
- **Solution**: Learn reconstruction first, then add KL gradually
- **Result**: Better convergence, higher quality outputs

### 6. **GroupNorm + SiLU**
- **Problem**: BatchNorm unstable with small batches, ReLU has dead neurons
- **Solution**: GroupNorm (batch-independent), SiLU (smooth, no dead regions)
- **Result**: More stable training, better gradients

---

## Notebook Structure (25 Cells)

1. **Header** - Overview and key improvements
2. **Imports** - All required libraries
3. **Configuration** - Hyperparameters and setup
4. **Dataset** - CLAHE preprocessing, ChestXRayDataset class, dataloaders
5. **Visualization** - Sample images and statistics
6. **Model Architecture** - ResBlock, AttentionBlock, Encoder, Decoder, SpatialVAE
7. **Loss Functions** - PerceptualLoss, SSIMLoss, VAELoss
8. **Initialize Model** - Create model, test forward pass
9. **Training Functions** - get_beta, train_epoch, validate
10. **Phase 1 Training** - Reconstruction-only (50 epochs)
11. **Phase 2 Training** - With KL regularization (100 epochs)
12. **Loss Curves** - Visualize training progress
13. **Generate Images** - Sample from latent space, reconstructions
14. **Summary** - Final results and metrics

---

## Expected Results

### Training Metrics
- **Reconstruction Loss**: Should decrease steadily in Phase 1
- **KL Loss**: Starts at 0, gradually increases in Phase 2
- **Validation Loss**: Monitor for overfitting

### Generated Images
- **Sharp edges**: Due to L1 + Perceptual loss
- **Anatomically coherent**: Due to spatial latent + attention
- **Realistic contrast**: Due to CLAHE preprocessing + SSIM
- **Diverse samples**: Due to proper KL regularization (even at low beta)

### FID Score Target
- **Good VAE**: FID < 100 for medical images
- **Excellent VAE**: FID < 50

---

## Usage Instructions

1. **Upload to Kaggle**
   - Create new notebook
   - Copy all cells from `vae_chest_xray.ipynb`
   
2. **Add Dataset**
   - Add "Chest X-Ray Pneumonia" dataset
   - Path: `/kaggle/input/chest-xray-pneumonia/chest_xray`

3. **Run All Cells**
   - Phase 1: ~1-2 hours (50 epochs)
   - Phase 2: ~2-4 hours (100 epochs)
   - Total: ~3-6 hours on Kaggle GPU

4. **Monitor Progress**
   - Check generated samples every 10 epochs
   - Watch loss curves for convergence
   - Verify reconstructions look sharp

5. **Adjust if Needed**
   - If blurry: Decrease beta further (0.0005)
   - If unstable: Decrease learning rate
   - If slow: Reduce image size to 64×64

---

## Code Optimizations

✓ **Vectorized operations** - Uses native PyTorch operations
✓ **No unnecessary loops** - Batch processing throughout
✓ **Minimal comments** - Clean, readable code
✓ **Efficient attention** - Matrix multiplication instead of loops
✓ **GPU optimized** - pin_memory, efficient data loading

---

## Comparison to Original Implementation

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Latent Space | Flattened (256-d) | Spatial (4×4×64) |
| Activation | LeakyReLU | SiLU |
| Normalization | BatchNorm | GroupNorm |
| Recon Loss | MSE | L1 + Perceptual + SSIM |
| Beta | 0.5 | 0.001 |
| Training | Single phase | Two-phase |
| Upsampling | Transposed Conv | Bilinear |
| Attention | None | 16×16 resolution |

---

## References & Inspiration

- **Beta-VAE**: Higgins et al. - Understanding beta in VAE
- **Perceptual Loss**: Johnson et al. - Perceptual Losses for Real-Time Style Transfer
- **MedVAE** (2026): Efficient medical image autoencoders with spatial latents
- **SSIM**: Wang et al. - Image Quality Assessment
- **GroupNorm**: Wu & He - Group Normalization

---

## Notes

- **No excessive docstrings**: Code is self-explanatory with strategic comments
- **Optimized for Kaggle**: Uses `/kaggle/input` and `/kaggle/working` paths
- **GPU efficient**: Batch size 32 works on most Kaggle GPUs
- **Reproducible**: Set random seeds if needed for consistency

---

**Created**: February 2026  
**Purpose**: ECE 285 Deep Generative Models Assignment  
**Goal**: Generate clear, distinctive chest X-ray images with minimal distortion

