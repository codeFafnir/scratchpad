import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


class InceptionV3Features(nn.Module):
    # Extract features from InceptionV3 for FID calculation
    def __init__(self, device):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        # Remove final layers, keep up to avgpool
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.to(device)
        self.eval()
        self.device = device
    
    @torch.no_grad()
    def forward(self, x):
        # Inception expects 3-channel 299x299 images
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.blocks(x)
        return x.view(x.size(0), -1)


def get_activations(images, model, batch_size=32):
    # Get InceptionV3 activations for a set of images
    model.eval()
    activations = []
    
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in tqdm(loader, desc='Computing activations'):
        batch = batch[0].to(model.device)
        act = model(batch)
        activations.append(act.cpu().numpy())
    
    return np.concatenate(activations, axis=0)


def calculate_fid(real_activations, fake_activations):
    # Calculate Frechet Inception Distance
    mu_real = np.mean(real_activations, axis=0)
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    sigma_fake = np.cov(fake_activations, rowvar=False)
    
    diff = mu_real - mu_fake
    
    # Product of covariances
    try:
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    except Exception as e:
        print(f"FID calculation error: {e}")
        fid = float('inf')
    
    return fid


def calculate_inception_score(images, model, batch_size=32, splits=10):
    # Calculate Inception Score
    model.eval()
    
    # Get predictions from Inception
    inception = models.inception_v3(pretrained=True).to(model.device)
    inception.eval()
    
    preds = []
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Computing IS'):
            batch = batch[0].to(model.device)
            if batch.size(1) == 1:
                batch = batch.repeat(1, 3, 1, 1)
            batch = nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            pred = nn.functional.softmax(inception(batch), dim=1)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Calculate IS
    scores = []
    split_size = preds.shape[0] // splits
    
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0)
        kl_divs = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl_div = np.mean(np.sum(kl_divs, axis=1))
        scores.append(np.exp(kl_div))
    
    return np.mean(scores), np.std(scores)


def evaluate_vae(vae_model, real_loader, device, num_samples=1000):
    # Evaluate VAE using FID and IS
    print("Evaluating VAE...")
    
    # Generate fake images
    print(f"Generating {num_samples} images...")
    vae_model.eval()
    fake_images = vae_model.generate(num_samples, device)
    
    # Collect real images
    print("Collecting real images...")
    real_images = []
    for batch in real_loader:
        real_images.append(batch)
        if len(real_images) * batch.size(0) >= num_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Initialize Inception model
    print("Loading InceptionV3...")
    inception_model = InceptionV3Features(device)
    
    # Calculate FID
    print("Calculating FID...")
    try:
        real_acts = get_activations(real_images, inception_model)
        fake_acts = get_activations(fake_images.cpu(), inception_model)
        fid_score = calculate_fid(real_acts, fake_acts)
        print(f"FID Score: {fid_score:.4f}")
    except Exception as e:
        print(f"FID calculation failed: {e}")
        fid_score = None
    
    # Calculate IS
    print("Calculating Inception Score...")
    try:
        is_mean, is_std = calculate_inception_score(fake_images.cpu(), inception_model)
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        print(f"IS calculation failed: {e}")
        is_mean, is_std = None, None
    
    return {
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std,
        'fake_images': fake_images,
        'real_images': real_images
    }


def visualize_samples(images, title='Generated Samples', nrow=4, ncol=4, save_path=None):
    # Visualize a grid of images
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()


def visualize_reconstructions(model, data_loader, device, num_samples=8, save_path=None):
    # Show original vs reconstructed images
    model.eval()
    
    batch = next(iter(data_loader))[:num_samples].to(device)
    with torch.no_grad():
        recon, _, _ = model(batch)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        axes[0, i].imshow(batch[i].cpu().squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        axes[1, i].imshow(recon[i].cpu().squeeze().numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()


def interpolate_latent(model, device, num_steps=10, save_path=None):
    # Interpolate between two random latent vectors
    model.eval()
    
    z1 = torch.randn(1, model.latent_dim).to(device)
    z2 = torch.randn(1, model.latent_dim).to(device)
    
    interpolations = []
    for alpha in np.linspace(0, 1, num_steps):
        z = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            img = model.decoder(z)
        interpolations.append(img)
    
    interpolations = torch.cat(interpolations, dim=0)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
    for i, ax in enumerate(axes):
        ax.imshow(interpolations[i].cpu().squeeze().numpy(), cmap='gray')
        ax.axis('off')
    
    plt.suptitle('Latent Space Interpolation', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()

