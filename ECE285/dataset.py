import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, img_size=128, split='train'):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        # Collect all images from both NORMAL and PNEUMONIA folders
        self.image_paths = []
        split_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")
        
        for category in ['NORMAL', 'PNEUMONIA']:
            category_path = os.path.join(split_dir, category)
            if os.path.exists(category_path):
                patterns = ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
                for pattern in patterns:
                    self.image_paths.extend(glob.glob(os.path.join(category_path, pattern)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {split_dir}")
        
        print(f"Found {len(self.image_paths)} images in {split} split")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a random valid image instead
            return self.__getitem__((idx + 1) % len(self))


def get_dataloaders(data_dir, img_size=128, batch_size=32, num_workers=2):
    try:
        train_dataset = ChestXRayDataset(data_dir, img_size, split='train')
        val_dataset = ChestXRayDataset(data_dir, img_size, split='val')
        test_dataset = ChestXRayDataset(data_dir, img_size, split='test')
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise


def get_all_data_loader(data_dir, img_size=128, batch_size=32, num_workers=2):
    # Combine train, val, test for maximum training data
    all_paths = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            for category in ['NORMAL', 'PNEUMONIA']:
                category_path = os.path.join(split_dir, category)
                if os.path.exists(category_path):
                    patterns = ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
                    for pattern in patterns:
                        all_paths.extend(glob.glob(os.path.join(category_path, pattern)))
    
    print(f"Total images across all splits: {len(all_paths)}")
    
    class CombinedDataset(Dataset):
        def __init__(self, paths, img_size):
            self.paths = paths
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            try:
                image = Image.open(self.paths[idx]).convert('RGB')
                return self.transform(image)
            except Exception as e:
                print(f"Error loading {self.paths[idx]}: {e}")
                return self.__getitem__((idx + 1) % len(self))
    
    dataset = CombinedDataset(all_paths, img_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return loader

