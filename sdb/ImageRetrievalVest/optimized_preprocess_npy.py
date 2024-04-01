import os
import numpy as np
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm  # Make sure to import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
    resize_transform = tio.Resize(target_shape=(16, 256, 256))
    crop = tio.CropOrPad((16, 100, 100))
    rescale_intensity = tio.RescaleIntensity((0, 1), percentiles=(0.5,99.5))
    combined_transforms = tio.Compose([crop, resize_transform, rescale_intensity])
    return combined_transforms

def process_file(filename, input_folder, output_folder):
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
    return filename  # Return filename to track progress

def preprocess_and_save_npy_data(input_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # List all .npy files in the input directory
    filenames = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    # Create a partial function with fixed arguments for parallel execution
    process_func = partial(process_file, input_folder=input_folder, output_folder=output_folder)

    # Use ProcessPoolExecutor to parallelize file processing
    with ProcessPoolExecutor() as executor:
        # Wrap executor.map with tqdm for a progress bar
        results = list(tqdm(executor.map(process_func, filenames), total=len(filenames), desc="Processing files"))

# Example usage
input_folder = '/sdb/LUNA16/balanced_candidates'  # Change this to your input directory
path_to_save_preprocessed_files = '/sdb/LUNA16/balanced_candidates_augmented'
os.makedirs(path_to_save_preprocessed_files, exist_ok=True)
preprocess_and_save_npy_data(input_folder, path_to_save_preprocessed_files)
