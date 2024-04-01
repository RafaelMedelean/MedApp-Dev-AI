import os
import numpy as np
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images, labels = zip(*batch)
    images = torch.utils.data.dataloader.default_collate(images)
    labels = torch.utils.data.dataloader.default_collate(labels)
    return images, labels

def get_training_augmentations():
    # Common augmentations
    resize_transform = tio.CropOrPad((32, 32, 32))
    rescale_intensity = tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5)) #Intensity Augumentations
    flip = tio.RandomFlip(axes=(0, 1, 2), p=0.5)
    random_transforms = tio.Compose([
            tio.RandomElasticDeformation(max_displacement=(0,4,4), p=0.5),
            tio.RandomNoise(mean=0, std=0.02, p=0.3),
            tio.RandomBlur(std=1, p=0.3)
        ], p=0.8)

    affine = tio.RandomAffine(
        scales=(0,0.15,0.15),
        degrees=(15, 0, 0),
        translation=(2,2,2),
        default_pad_value=-1000, 
        p=0.8
    )

    combined_transforms = tio.Compose([affine, resize_transform, rescale_intensity, flip, random_transforms])
    return combined_transforms

def get_validation_augmentations():
    resize_transform = tio.CropOrPad((32, 32, 32))
    rescale_intensity = tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5))
    combined_transforms = tio.Compose([resize_transform, rescale_intensity])
    return combined_transforms

class CustomDataset(Dataset):
    def __init__(self, data, path_images, augmentations=None):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            print(f"Loading CSV from: {data}")
            self.data = pd.read_csv(data) 
            
        #self.data = self.data[:70]
        self.augmentations = augmentations
        self.path_images = path_images
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            image_path = self.data.iloc[idx, 0]
            label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
            
            full_path = self.path_images + image_path
        except Exception as e:
            print(f"An error occurred while processing file path_images={self.path_images}, image_path={image_path}: {e}")
            return None, None
        if not os.path.exists(full_path):
            print(f"Error: File {full_path} not found. Skipping this sample.")
            return None, None 
        
        image = np.load(full_path)
        #image = torch.clamp(torch.tensor(image, dtype=torch.float32).unsqueeze(0), min=-1000, max=400)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        # Apply augmentations if provided
        if callable(self.augmentations):
            if "training" in str(self.augmentations):
                image = get_training_augmentations()(image)
            else:
                image = get_validation_augmentations()(image)
        return image, label

def create_balanced_split(csv_path, validation_size=0.075, pos_ratio=0.1):
    # Load the dataset
    data = pd.read_csv(csv_path)
    
    # Separate positive and negative samples
    positives = data[data['class_label'] == 1]
    negatives = data[data['class_label'] == 0]
    
    # Calculate the number of positives needed based on the desired ratio
    total_samples = len(data)
    total_val_samples = int(total_samples * validation_size)
    val_positives = int(total_val_samples * pos_ratio)
    val_negatives = total_val_samples - val_positives
    print(val_positives)
    print(val_negatives)
    # Split positives and negatives into training and validation sets
    pos_train, pos_val = train_test_split(positives, test_size=val_positives, random_state=42)
    neg_train, neg_val = train_test_split(negatives, test_size=val_negatives, random_state=42)
    
    # Combine the splits back into training and validation datasets
    train_data = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42)  # Shuffle the dataset
    val_data = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=42)  # Shuffle the dataset
    
    return train_data, val_data

# Load the entire dataset
path_to_csv = '/sdb/ImageRetrievalVest/csvs/all_patches(crops)_names_labels_smaller_bigger_csv_completed.csv'#'/sdb/ImageRetrievalVest/csvs/all_patches(crops)_names_labels_smaller_bigger_csv_completed.csv'#'/sdb/LUNA16/all_patches(crops)_names_labels_smaller_csv_completed.csv'
train_data, val_data = create_balanced_split(path_to_csv)

# Define path to images
path_images = '/sdb/LUNA16/64x45x45_all_patches_smaller_but_bigger_csv/'#'/sdb/LUNA16/64x45x45_all_patches_smaller_csv_completed/'

# Now, you can proceed to create your datasets and dataloaders
train_dataset = CustomDataset(train_data, path_images, augmentations=get_training_augmentations())
val_dataset = CustomDataset(val_data, path_images, augmentations=get_validation_augmentations())

train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True, num_workers=7, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=300, shuffle=False, num_workers=7, collate_fn=custom_collate_fn)