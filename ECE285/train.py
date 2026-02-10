import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class VAETrainer:
    def __init__(self, model, device, lr=1e-4, beta=1.0):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Track losses
        self.train_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, data in enumerate(pbar):
            try:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                recon, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, self.beta)
                
                # Normalize by batch size
                loss = loss / data.size(0)
                recon_loss = recon_loss / data.size(0)
                kl_loss = kl_loss / data.size(0)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        
        self.train_losses.append(avg_loss)
        self.recon_losses.append(avg_recon)
        self.kl_losses.append(avg_kl)
        
        return avg_loss, avg_recon, avg_kl
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        for data in val_loader:
            try:
                data = data.to(self.device)
                recon, mu, logvar = self.model(data)
                loss, _, _ = vae_loss(recon, data, mu, logvar, self.beta)
                total_loss += loss.item() / data.size(0)
            except Exception as e:
                print(f"Validation error: {e}")
                continue
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader=None, epochs=50, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss, recon_loss, kl_loss = self.train_epoch(train_loader, epoch)
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Recon={recon_loss:.4f}, KL={kl_loss:.4f}')
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f'Validation Loss: {val_loss:.4f}')
                self.scheduler.step(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'), epoch)
            else:
                self.scheduler.step(train_loss)
                if train_loss < best_loss:
                    best_loss = train_loss
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'), epoch)
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'), epoch)
        
        # Save final model
        self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'), epochs)
        return self.train_losses, self.recon_losses, self.kl_losses, self.val_losses
    
    def save_checkpoint(self, path, epoch):
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'recon_losses': self.recon_losses,
                'kl_losses': self.kl_losses,
                'val_losses': self.val_losses,
            }, path)
            print(f'Checkpoint saved: {path}')
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.recon_losses = checkpoint.get('recon_losses', [])
            self.kl_losses = checkpoint.get('kl_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
            return checkpoint['epoch']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (BCE)
    recon_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def plot_losses(train_losses, recon_losses, kl_losses, val_losses=None, save_path='loss_curves.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(train_losses, label='Train', color='blue')
    if val_losses:
        axes[0].plot(val_losses, label='Validation', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Reconstruction loss
    axes[1].plot(recon_losses, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].grid(True)
    
    # KL loss
    axes[2].plot(kl_losses, color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Loss curves saved to {save_path}')

