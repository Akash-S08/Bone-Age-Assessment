import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, Any, Optional
import cv2

class BoneAgeDataset(Dataset):
    """Dataset for bone age assessment"""
    
    def __init__(self, data_dir: str, csv_file: str, transform: Optional[A.Compose] = None, 
                 image_size: int = 224):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        
        # Load CSV with image names and ages
        self.df = pd.read_csv(csv_file)
        self.df['age_months'] = self.df['age_months'].astype(float)
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, row['image_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get age in months
        age = float(row['age_months'])
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(age, dtype=torch.float32)

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Training augmentations"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    data_dir = config['paths']['data_dir']
    
    # Create datasets
    train_dataset = BoneAgeDataset(
        data_dir=os.path.join(data_dir, 'train'),
        csv_file=os.path.join(data_dir, 'train_labels.csv'),
        transform=get_train_transforms(config['data']['image_size'])
    )
    
    val_dataset = BoneAgeDataset(
        data_dir=os.path.join(data_dir, 'val'),
        csv_file=os.path.join(data_dir, 'val_labels.csv'),
        transform=get_val_transforms(config['data']['image_size'])
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if config['training']['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True if config['training']['device'] == 'cuda' else False
    )
    
    return train_loader, val_loader

# Sample data generator for testing
def generate_sample_data(output_dir: str, num_samples: int = 100):
    """Generate synthetic data for testing"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    train_data = []
    val_data = []
    
    # Generate synthetic X-ray-like images
    for i in range(num_samples):
        # Create synthetic hand X-ray (grayscale with hand-like shape)
        img = np.random.randint(20, 200, (400, 300), dtype=np.uint8)
        # Add hand-like structure
        cv2.ellipse(img, (150, 200), (80, 120), 0, 0, 360, 255, -1)
        cv2.ellipse(img, (150, 200), (60, 100), 0, 0, 360, 100, -1)
        
        # Convert to 3-channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Random age between 0-216 months (0-18 years)
        age = np.random.uniform(0, 216)
        
        if i < num_samples * 0.8:  # 80% train
            folder = 'train'
            img_name = f'train_{i:04d}.png'
            train_data.append({'image_name': img_name, 'age_months': age})
        else:  # 20% val
            folder = 'val'
            img_name = f'val_{i:04d}.png'
            val_data.append({'image_name': img_name, 'age_months': age})
        
        # Save image
        cv2.imwrite(os.path.join(output_dir, folder, img_name), img)
    
    # Save CSV files
    pd.DataFrame(train_data).to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
    pd.DataFrame(val_data).to_csv(os.path.join(output_dir, 'val_labels.csv'), index=False)
    
    print(f"Generated {len(train_data)} training and {len(val_data)} validation samples")