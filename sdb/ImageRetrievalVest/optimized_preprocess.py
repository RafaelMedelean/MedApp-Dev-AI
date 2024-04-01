import os
import numpy as np
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm  # Import tqdm for the progress bar
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
    resize_transform = tio.Resize(target_shape=(32, 256, 256))
    #crop = tio.CropOrPad((64, 224, 224))
    rescale_intensity = tio.RescaleIntensity((0, 1), percentiles=(0.5,99.5))
    combined_transforms = tio.Compose([resize_transform, rescale_intensity])
    return combined_transforms

def process_image(row, path_images, preprocess_path):
    image_path = row['seriesuid']
    full_path = os.path.join(path_images, image_path)
    if not os.path.exists(full_path):
        return f"File not found: {full_path}"

    image = np.load(full_path)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    image = get_augumentations_dinamic_densenet()(image)

    preprocess_full_path = os.path.join(preprocess_path, image_path)
    np.save(preprocess_full_path, image.squeeze(0).numpy())
    return f"Processed: {image_path}"

def preprocess_and_save_validation_data(val_csv, path_images, preprocess_path):
    data = pd.read_csv(val_csv)
    process_func = partial(process_image, path_images=path_images, preprocess_path=preprocess_path)
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_func, [row for _, row in data.iterrows()]), total=len(data), desc="Processing images"))
    
    for result in results:
        print(result)
# Define the directory for preprocessed validation images in your config

path_to_save_preprocessed_files = '/sdb/LUNA16/balanced_candidates_augumented'
input_folder = '/sdb/LUNA16/balanced_candidates' 

os.makedirs(path_to_save_preprocessed_files, exist_ok=True)

#preprocess_and_save_validation_data(csv_missing, input_folder, path_to_save_preprocessed_files)
preprocess_and_save_npy_data(input_folder, path_to_save_preprocessed_files, get_augumentations_dinamic_densenet)