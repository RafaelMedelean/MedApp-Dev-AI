import os
import numpy as np
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

SEED = 1337
os.environ['PYTHONHASHSEED'] = str(SEED)
#random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

def get_augumentations_dinamic_densenet():
    resize_transform = tio.Resize(target_shape=(32, 256, 256))
    #crop = tio.CropOrPad((64, 224, 224))
    rescale_intensity = tio.RescaleIntensity((0, 1), percentiles=(0.5,99.5))
    combined_transforms = tio.Compose([resize_transform, rescale_intensity])
    return combined_transforms
'''
def preprocess_and_save_validation_data(val_csv, path_images, preprocess_path, augmentations_func, config):
    data = pd.read_csv(val_csv)
    for idx, row in data.iterrows():
        image_path = row['patch_name']
        full_path = os.path.join(path_images, image_path)
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue

        image = np.load(full_path)
        # Ensure image is a 4D tensor: (1, depth, height, width)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        #length = row['length']
        image = get_augumentations_dinamic_densenet()(image)

        preprocess_full_path = os.path.join(preprocess_path, image_path)
        # Save the numpy array without the batch dimension
        np.save(preprocess_full_path, image.squeeze(0).numpy())
'''
def preprocess_and_save_npy_data(input_folder, output_folder, augmentations_func):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all .npy files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            full_path = os.path.join(input_folder, filename)
            image = np.load(full_path)

            # Ensure image is a 4D tensor: (1, depth, height, width)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

            # Apply the augmentations
            image = get_augumentations_dinamic_densenet()(image)

            # Prepare the full path for the preprocessed file
            preprocess_full_path = os.path.join(output_folder, filename)

            # Save the numpy array without the batch dimension
            np.save(preprocess_full_path, image.squeeze(0).numpy())

from tqdm import tqdm  # Import tqdm for the progress bar

def preprocess_and_save_validation_data(val_csv, path_images, preprocess_path):
    data = pd.read_csv(val_csv)
    # Wrap data.iterrows() with tqdm for a progress bar
    for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing images"):
        image_path = row['seriesuid']
        full_path = os.path.join(path_images, image_path)
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue

        image = np.load(full_path)
        # Ensure image is a 4D tensor: (1, depth, height, width)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = get_augumentations_dinamic_densenet()(image)

        preprocess_full_path = os.path.join(preprocess_path, image_path)
        # Save the numpy array without the batch dimension
        np.save(preprocess_full_path, image.squeeze(0).numpy())

# Define the directory for preprocessed validation images in your config
path_to_save_preprocessed_files = '/sdb/LUNA16/balanced_candidates_augumented'
#os.makedirs(path_to_save_preprocessed_files, exist_ok=True)

# Example usage
input_folder = '/sdb/LUNA16/balanced_candidates'  # Change this to your input directory
#preprocess_and_save_validation_data(csv_missing, input_folder, path_to_save_preprocessed_files)
preprocess_and_save_npy_data(input_folder, path_to_save_preprocessed_files, get_augumentations_dinamic_densenet)