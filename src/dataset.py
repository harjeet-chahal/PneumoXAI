import os
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config import TRAIN_DIR, TEST_DIR, VAL_DIR, BATCH_SIZE

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Walk through the directory to get all images
        # Structure: root_dir/NORMAL/*.jpeg, root_dir/PNEUMONIA/*.jpeg
        for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = os.path.join(root_dir, class_name)
            # Support both jpg and jpeg
            for ext in ['*.jpeg', '*.jpg', '*.png']:
                paths = glob.glob(os.path.join(class_dir, ext))
                self.image_paths.extend(paths)
                self.labels.extend([label] * len(paths))
                


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        # Ensure image was read correctly
        if image is None:
             raise ValueError(f"Failed to load image: {img_path}")

        # Convert to grayscale for CLAHE
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(gray)
        
        # Convert back to RGB (Albumentations expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = self.labels[idx]
        return image, label

def get_transforms(phase='train'):
    if phase == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=20, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def get_data_loaders(batch_size=BATCH_SIZE):
    # Create datasets
    train_dataset = PneumoniaDataset(
        root_dir=TRAIN_DIR,
        transform=get_transforms('train')
    )
    val_dataset = PneumoniaDataset(
        root_dir=VAL_DIR,
        transform=get_transforms('val')
    )
    test_dataset = PneumoniaDataset(
        root_dir=TEST_DIR,
        transform=get_transforms('test')
    )

    # Calculate weights for WeightedRandomSampler (Train only)
    targets = train_dataset.labels
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=2
    )
    # Shuffle val/test is optional, but usually False for validation consistency
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    return train_loader, val_loader, test_loader
