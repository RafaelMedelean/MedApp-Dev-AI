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

    images, labels, original_x, original_y, original_z, resample_x, resample_y, resample_z, image_path = zip(*batch)
    images = torch.utils.data.dataloader.default_collate(images)
    labels = torch.utils.data.dataloader.default_collate(labels)
    return images, labels, original_x, original_y, original_z, resample_x, resample_y, resample_z, image_path

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
            #print(f"Loading CSV from: {data}")
            self.data = pd.read_csv(data) 
            
        #self.data = self.data[:5]
        self.augmentations = augmentations
        self.path_images = path_images
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            image_path = self.data.iloc[idx, 0]
            label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
            original_x = self.data.iloc[idx,2]
            original_y = self.data.iloc[idx,3]
            original_z = self.data.iloc[idx,4]
            resample_x = self.data.iloc[idx,5]
            resample_y = self.data.iloc[idx,6]
            resample_z = self.data.iloc[idx,7]
            
            full_path = self.path_images + image_path
        except Exception as e:
            print(image_path,label,original_x,original_y,original_z)
            print(f"An error occurred while processing file path_images={self.path_images}, image_path={image_path}: {e}")
            return None, None
        
        if not os.path.exists(full_path):
            print(f"Error: File {full_path} not found. Skipping this sample.")
            return None, None 
        
        image = np.load(full_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = get_validation_augmentations()(image)
        return image , label, original_x, original_y, original_z, resample_x, resample_y, resample_z, image_path

# Load the entire dataset
path_to_csv = '/workspaces/MedApp-Dev-AI/sdb2/all_patches_balanced_candidates.csv'#'/sdb/ImageRetrievalVest/csvs/all_patches(crops)_names_labels_smaller_bigger_csv_completed.csv'#'/sdb/LUNA16/all_patches(crops)_names_labels_smaller_csv_completed.csv'

# Define path to images
path_images = '/workspaces/MedApp-Dev-AI/sdb2/balanced_candidates/'#'/sdb/LUNA16/64x45x45_all_patches_smaller_csv_completed/'

dataset = CustomDataset(path_to_csv, path_images, augmentations=get_validation_augmentations())

data_loader = DataLoader(dataset, batch_size=300, shuffle=True, num_workers=7, collate_fn=custom_collate_fn)