import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fftpack import dct
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def zigzag(matrix):
    """
    Extract zigzag pattern from 8x8 matrix
    """
    zigzag_pattern = [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ]
    
    result = np.zeros(64)
    for i in range(8):
        for j in range(8):
            result[zigzag_pattern[i][j]] = matrix[i, j]
    return result

def dct2(block):
    """
    2D DCT transform
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def fisher_discriminant_ratio(mean1, var1, mean2, var2):
    """
    Calculate Fisher's Discriminant Ratio (Fisher Score)
    Formula: (μ₁ - μ₂)² / (σ₁² + σ₂²)
    
    Higher score = better class separation
    """
    numerator = (mean1 - mean2) ** 2
    denominator = var1 + var2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return numerator / denominator

# ===================================================================
# Problem a: Prior probabilities
# ===================================================================
n = 1053 + 250
c_cheetah = 250
c_grass = 1053
prior_cheetah = c_cheetah / n
prior_grass = c_grass / n

print("=" * 60)
print("Problem A: Prior Probabilities")
print("=" * 60)
print(f"Prior probability of cheetah: {prior_cheetah:.4f}")
print(f"Prior probability of grass: {prior_grass:.4f}")

# ===================================================================
# Problem b: Maximum likelihood estimates and marginal plots
# ===================================================================
print("\n" + "=" * 60)
print("Problem B: Loading training data and computing statistics")
print("=" * 60)

# Load training data
train_set = loadmat('TrainingSamplesDCT_8_new.mat')
FGmat = train_set['TrainsampleDCT_FG']
BGmat = train_set['TrainsampleDCT_BG']

print(f"Foreground samples shape: {FGmat.shape}")
print(f"Background samples shape: {BGmat.shape}")

# Maximum likelihood estimates - Foreground
meanFG = np.mean(FGmat, axis=0)
covFG = np.cov(FGmat, rowvar=False)
varFG = np.var(FGmat, axis=0)

# Maximum likelihood estimates - Background
meanBG = np.mean(BGmat, axis=0)
covBG = np.cov(BGmat, rowvar=False)
varBG = np.var(BGmat, axis=0)

# Calculate MLE for each sample (for reference, not used later)
mleFG = np.zeros(c_cheetah)
for i in range(c_cheetah):
    tmp = 0
    for j in range(64):
        tmp += ((FGmat[i, j] - meanFG[j]) / np.sqrt(varFG[j])) ** 2
    mleFG[i] = np.exp(-32 * np.log(2 * np.pi) - np.sum(np.log(np.sqrt(varFG))) - 0.5 * tmp)

mleBG = np.zeros(c_grass)
for i in range(c_grass):
    tmp = 0
    for j in range(64):
        tmp += ((BGmat[i, j] - meanBG[j]) / np.sqrt(varBG[j])) ** 2
    mleBG[i] = np.exp(-32 * np.log(2 * np.pi) - np.sum(np.log(np.sqrt(varBG))) - 0.5 * tmp)

# Calculate Fisher's Discriminant Ratio for all 64 dimensions
print("\nCalculating Fisher's Discriminant Ratio for feature selection...")
fisher_scores = np.zeros(64)

for i in range(64):
    ave_FG = np.mean(FGmat[:, i])
    variance_FG = np.var(FGmat[:, i])
    ave_BG = np.mean(BGmat[:, i])
    variance_BG = np.var(BGmat[:, i])
    
    fisher_scores[i] = fisher_discriminant_ratio(ave_FG, variance_FG, ave_BG, variance_BG)

print(f"Fisher scores calculated for all 64 features")
print(f"Fisher score range: [{fisher_scores.min():.4f}, {fisher_scores.max():.4f}]")

# Create marginal plots for all 64 dimensions
print("\nGenerating marginal plots for all 64 dimensions...")
fig = plt.figure(figsize=(20, 20))

for i in range(64):
    plt.subplot(8, 8, i + 1)
    
    ave_FG = np.mean(FGmat[:, i])
    variance_FG = np.var(FGmat[:, i])
    sigma_FG = np.sqrt(variance_FG)
    
    ave_BG = np.mean(BGmat[:, i])
    variance_BG = np.var(BGmat[:, i])
    sigma_BG = np.sqrt(variance_BG)
    
    xFG = np.linspace(ave_FG - 7 * sigma_FG, ave_FG + 7 * sigma_FG, 350)
    xBG = np.linspace(ave_BG - 7 * sigma_BG, ave_BG + 7 * sigma_BG, 350)
    x = np.sort(np.concatenate([xFG, xBG]))
    
    # Calculate normal PDFs
    y_cheetah = (1 / (sigma_FG * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - ave_FG) / sigma_FG) ** 2)
    y_grass = (1 / (sigma_BG * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - ave_BG) / sigma_BG) ** 2)
    
    plt.plot(x, y_cheetah, label='Cheetah')
    plt.plot(x, y_grass, label='Grass')
    plt.title(f'Index={i + 1}, F={fisher_scores[i]:.2f}', fontsize=7)
    plt.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig('64_plots_fisher.png', dpi=150, bbox_inches='tight')
print("Saved: 64_plots_fisher.png")
plt.close()

# Find Best and Worst based on Fisher's Discriminant Ratio
FisherIdx = np.argsort(fisher_scores)
worstidx = np.sort(FisherIdx[:8])
bestidx = np.sort(FisherIdx[56:64])

print(f"\nFeature Selection Results (Fisher's Discriminant Ratio):")
print(f"Best 8 features (highest Fisher scores): {bestidx + 1}")
print(f"  Fisher scores: {fisher_scores[bestidx]}")
print(f"Worst 8 features (lowest Fisher scores): {worstidx + 1}")
print(f"  Fisher scores: {fisher_scores[worstidx]}")

# Create best and worst comparison plots
fig = plt.figure(figsize=(16, 16))

for i in range(8):
    idx_w = worstidx[i]
    idx_b = bestidx[i]
    
    # Best features (top row)
    plt.subplot(4, 4, i + 1)
    ave_FG = np.mean(FGmat[:, idx_b])
    variance_FG = np.var(FGmat[:, idx_b])
    sigma_FG = np.sqrt(variance_FG)
    ave_BG = np.mean(BGmat[:, idx_b])
    variance_BG = np.var(BGmat[:, idx_b])
    sigma_BG = np.sqrt(variance_BG)
    
    xFG = np.linspace(ave_FG - 7 * sigma_FG, ave_FG + 7 * sigma_FG, 350)
    xBG = np.linspace(ave_BG - 7 * sigma_BG, ave_BG + 7 * sigma_BG, 350)
    x = np.sort(np.concatenate([xFG, xBG]))
    
    y_cheetah = (1 / (sigma_FG * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - ave_FG) / sigma_FG) ** 2)
    y_grass = (1 / (sigma_BG * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - ave_BG) / sigma_BG) ** 2)
    
    plt.plot(x, y_cheetah, label='Cheetah')
    plt.plot(x, y_grass, label='Grass')
    plt.title(f'Best {idx_b + 1} (F={fisher_scores[idx_b]:.2f})', fontsize=10)
    plt.legend(fontsize=8)
    
    # Worst features (bottom row)
    plt.subplot(4, 4, i + 9)
    ave_FG = np.mean(FGmat[:, idx_w])
    variance_FG = np.var(FGmat[:, idx_w])
    sigma_FG = np.sqrt(variance_FG)
    ave_BG = np.mean(BGmat[:, idx_w])
    variance_BG = np.var(BGmat[:, idx_w])
    sigma_BG = np.sqrt(variance_BG)
    
    xFG = np.linspace(ave_FG - 7 * sigma_FG, ave_FG + 7 * sigma_FG, 350)
    xBG = np.linspace(ave_BG - 7 * sigma_BG, ave_BG + 7 * sigma_BG, 350)
    x = np.sort(np.concatenate([xFG, xBG]))
    
    y_cheetah = (1 / (sigma_FG * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - ave_FG) / sigma_FG) ** 2)
    y_grass = (1 / (sigma_BG * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - ave_BG) / sigma_BG) ** 2)
    
    plt.plot(x, y_cheetah, label='Cheetah')
    plt.plot(x, y_grass, label='Grass')
    plt.title(f'Worst {idx_w + 1} (F={fisher_scores[idx_w]:.2f})', fontsize=10)
    plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig('BestandWorst_plots_fisher.png', dpi=150, bbox_inches='tight')
print("Saved: BestandWorst_plots_fisher.png")
plt.close()

# ===================================================================
# Problem c(i): Use 64-dimension Gaussian
# ===================================================================
print("\n" + "=" * 60)
print("Problem C(i): Classification using 64D Gaussian")
print("=" * 60)

# Load image
img = np.array(Image.open('cheetah.bmp').convert('L'), dtype=np.float64) / 255.0

# Add padding
img = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)

# DCT and classification
m, n = img.shape
Blocks = np.ones((m - 7, n - 7))

mean_FG_full = meanFG
mean_BG_full = meanBG
inv_covFG = np.linalg.inv(covFG)
inv_covBG = np.linalg.inv(covBG)
DcovFG = np.linalg.det(covFG)
DcovBG = np.linalg.det(covBG)

print("Processing image blocks...")
for i in range(m - 7):
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i + 1}/{m - 7} rows")
    for j in range(n - 7):
        block = img[i:i + 8, j:j + 8]
        DCT = dct2(block)
        feature = zigzag(DCT)
        
        # Cheetah discriminant
        g_cheetah = (feature @ inv_covFG @ feature.T - 
                     2 * feature @ inv_covFG @ mean_FG_full.T + 
                     mean_FG_full @ inv_covFG @ mean_FG_full.T + 
                     np.log(DcovFG) - 2 * np.log(prior_cheetah))
        
        # Grass discriminant
        g_grass = (feature @ inv_covBG @ feature.T - 
                   2 * feature @ inv_covBG @ mean_BG_full.T + 
                   mean_BG_full @ inv_covBG @ mean_BG_full.T + 
                   np.log(DcovBG) - 2 * np.log(prior_grass))
        
        if g_cheetah >= g_grass:
            Blocks[i, j] = 0

# Save prediction
Image.fromarray((Blocks * 255).astype(np.uint8)).save('prediction_64d_fisher.jpg')
print("Saved: prediction_64d_fisher.jpg")

# Calculate error
ground_truth = np.array(Image.open('cheetah_mask.bmp'), dtype=np.float64) / 255.0
prediction = Blocks

x, y = ground_truth.shape
count1 = 0  # False negatives (predicted grass, actually cheetah)
count2 = 0  # False positives (predicted cheetah, actually grass)
count_cheetah_truth = 0
count_grass_truth = 0

for i in range(x):
    for j in range(y):
        if prediction[i, j] > ground_truth[i, j]:
            count2 += 1
            count_grass_truth += 1
        elif prediction[i, j] < ground_truth[i, j]:
            count1 += 1
            count_cheetah_truth += 1
        elif ground_truth[i, j] > 0:
            count_cheetah_truth += 1
        else:
            count_grass_truth += 1

error1_64 = (count1 / count_cheetah_truth) * prior_cheetah
error2_64 = (count2 / count_grass_truth) * prior_grass
total_error_64 = error1_64 + error2_64

print(f"\n64D Results:")
print(f"  Error 1 (miss cheetah): {error1_64:.6f}")
print(f"  Error 2 (false alarm): {error2_64:.6f}")
print(f"  Total error: {total_error_64:.6f}")

# ===================================================================
# Problem c(ii): Use 8 best features (selected by Fisher's ratio)
# ===================================================================
print("\n" + "=" * 60)
print("Problem C(ii): Classification using 8 best features")
print(f"                (Selected by Fisher's Discriminant Ratio)")
print("=" * 60)

# Load image again
img = np.array(Image.open('cheetah.bmp').convert('L'), dtype=np.float64) / 255.0

# Add padding
img = np.pad(img, ((4, 3), (4, 3)), mode='constant', constant_values=0)

# DCT and classification
m, n = img.shape
Blocks = np.ones((m - 7, n - 7))

mean_FG_best = np.mean(FGmat[:, bestidx], axis=0)
mean_BG_best = np.mean(BGmat[:, bestidx], axis=0)
cov_cheetah = np.cov(FGmat[:, bestidx], rowvar=False)
cov_grass = np.cov(BGmat[:, bestidx], rowvar=False)
inv_covFG_best = np.linalg.inv(cov_cheetah)
inv_covBG_best = np.linalg.inv(cov_grass)
DcovFG_best = np.linalg.det(cov_cheetah)
DcovBG_best = np.linalg.det(cov_grass)

print("Processing image blocks...")
for i in range(m - 7):
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i + 1}/{m - 7} rows")
    for j in range(n - 7):
        block = img[i:i + 8, j:j + 8]
        DCT = dct2(block)
        zigzag_full = zigzag(DCT)
        feature = zigzag_full[bestidx]
        
        # Cheetah discriminant
        g_cheetah = (feature @ inv_covFG_best @ feature.T - 
                     2 * feature @ inv_covFG_best @ mean_FG_best.T + 
                     mean_FG_best @ inv_covFG_best @ mean_FG_best.T + 
                     np.log(DcovFG_best) - 2 * np.log(prior_cheetah))
        
        # Grass discriminant
        g_grass = (feature @ inv_covBG_best @ feature.T - 
                   2 * feature @ inv_covBG_best @ mean_BG_best.T + 
                   mean_BG_best @ inv_covBG_best @ mean_BG_best.T + 
                   np.log(DcovBG_best) - 2 * np.log(prior_grass))
        
        if g_cheetah >= g_grass:
            Blocks[i, j] = 0

# Save prediction
Image.fromarray((Blocks * 255).astype(np.uint8)).save('prediction_8d_fisher.jpg')
print("Saved: prediction_8d_fisher.jpg")

# Calculate error
prediction = Blocks
count1 = 0
count2 = 0
count_cheetah_truth = 0
count_grass_truth = 0

for i in range(x):
    for j in range(y):
        if prediction[i, j] > ground_truth[i, j]:
            count2 += 1
            count_grass_truth += 1
        elif prediction[i, j] < ground_truth[i, j]:
            count1 += 1
            count_cheetah_truth += 1
        elif ground_truth[i, j] > 0:
            count_cheetah_truth += 1
        else:
            count_grass_truth += 1

error1_8 = (count1 / count_cheetah_truth) * prior_cheetah
error2_8 = (count2 / count_grass_truth) * prior_grass
total_error_8 = error1_8 + error2_8

print(f"\n8D Results (Fisher-selected features):")
print(f"  Error 1 (miss cheetah): {error1_8:.6f}")
print(f"  Error 2 (false alarm): {error2_8:.6f}")
print(f"  Total error: {total_error_8:.6f}")

print("\n" + "=" * 60)
print("SUMMARY (Using Fisher's Discriminant Ratio)")
print("=" * 60)
print(f"Feature Selection Method: Fisher's Discriminant Ratio")
print(f"Best 8 features: {bestidx + 1}")
print(f"64D Gaussian - Total Error: {total_error_64:.6f}")
print(f"8D Gaussian  - Total Error: {total_error_8:.6f}")
print("=" * 60)