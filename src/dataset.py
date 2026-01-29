import os
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupShuffleSplit
import re
from .config import TRAIN_DIR, TEST_DIR, VAL_DIR, BATCH_SIZE

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir=None, file_paths=None, labels=None, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if file_paths is not None and labels is not None:
            self.image_paths = file_paths
            self.labels = labels
        elif root_dir is not None:
            self.root_dir = root_dir
            # Walk through the directory to get all images
            # Structure: root_dir/NORMAL/*.jpeg, root_dir/PNEUMONIA/*.jpeg
            for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
                class_dir = os.path.join(root_dir, class_name)
                # Support both jpg and jpeg
                for ext in ['*.jpeg', '*.jpg', '*.png']:
                    paths = glob.glob(os.path.join(class_dir, ext))
                    self.image_paths.extend(paths)
                    self.labels.extend([label] * len(paths))
        else:
             raise ValueError("Must provide either root_dir or (file_paths and labels)")
                


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

def get_all_filepaths_and_labels():
    """Helper to gather all files from all splits to re-split them."""
    all_paths = []
    all_labels = []
    
    # We will look into all directories
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                continue
            for ext in ['*.jpeg', '*.jpg', '*.png']:
                paths = glob.glob(os.path.join(class_dir, ext))
                all_paths.extend(paths)
                all_labels.extend([label] * len(paths))
    return np.array(all_paths), np.array(all_labels)

def extract_patient_id(filename):
    r"""
    Extracts patient ID from filename.
    PNEUMONIA: person(\d+)_...
    NORMAL: IM-(\d+)-...
    """
    basename = os.path.basename(filename)
    # Check for Pneumonia pattern
    match = re.search(r'person(\d+)_', basename)
    if match:
        return f"P_{match.group(1)}"
    
    # Check for Normal pattern
    match = re.search(r'IM-(\d+)', basename)
    if match:
        return f"N_{match.group(1)}"
    
    # Fallback if no pattern matches (shouldn't happen with this dataset but good for safety)
    return basename

def get_data_loaders(batch_size=BATCH_SIZE):
    # 1. Gather ALL data
    X, y = get_all_filepaths_and_labels()
    
    if len(X) == 0:
        raise ValueError("No images found in data directories!")

    # 2. Extract Patient IDs
    patient_ids = np.array([extract_patient_id(p) for p in X])
    
    print(f"Total images: {len(X)}")
    print(f"Total unique patients: {len(np.unique(patient_ids))}")

    # 3. Split: Train (70%) vs Temp (30%)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss.split(X, y, groups=patient_ids))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    groups_temp = patient_ids[temp_idx]
    
    # 4. Split Temp: Val (50% of Temp -> 15% total) vs Test (50% of Temp -> 15% total)
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val.split(X_temp, y_temp, groups=groups_temp))
    
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    
    print(f"Train: {len(X_train)} images")
    print(f"Val:   {len(X_val)} images")
    print(f"Test:  {len(X_test)} images")

    # Verify no leakage
    train_patients = set(patient_ids[train_idx])
    val_patients = set(groups_temp[val_idx])
    test_patients = set(groups_temp[test_idx])
    
    assert train_patients.isdisjoint(val_patients), "Train and Val sets share patients!"
    assert train_patients.isdisjoint(test_patients), "Train and Test sets share patients!"
    assert val_patients.isdisjoint(test_patients), "Val and Test sets share patients!"
    print("Patient-wise split verification passed: No leakage.")

    # 5. Create Datasets
    train_dataset = PneumoniaDataset(
        file_paths=X_train,
        labels=y_train,
        transform=get_transforms('train')
    )
    val_dataset = PneumoniaDataset(
        file_paths=X_val,
        labels=y_val,
        transform=get_transforms('val')
    )
    test_dataset = PneumoniaDataset(
        file_paths=X_test,
        labels=y_test,
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
