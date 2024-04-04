import numpy as np
import os
from PIL import Image
from tqdm import tqdm  # Import tqdm for the progress bar

def npy_to_jpeg(input_folder, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .npy files in the input folder
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    # Wrap npy_files with tqdm for a progress bar
    for npy_file in tqdm(npy_files, desc="Converting .npy files to JPEG"):
        # Load the numpy array (.npy file)
        ct_array = np.load(os.path.join(input_folder, npy_file))

        # Create a new directory for this .npy file's JPEGs
        ct_dir_name = npy_file.replace('.npy', '')
        ct_dir_path = os.path.join(output_folder, ct_dir_name)
        if not os.path.exists(ct_dir_path):
            os.makedirs(ct_dir_path)

        # Convert each slice to JPEG and save
        num_slices = ct_array.shape[0]
        for i in tqdm(range(num_slices), desc=f"Converting slices in {npy_file}", leave=False):
            # Normalize the pixel values to be between 0 and 255
            slice_normalized = (ct_array[i] - np.min(ct_array[i])) / (np.max(ct_array[i]) - np.min(ct_array[i])) * 255
            slice_img = Image.fromarray(slice_normalized.astype(np.uint8))

            # Save the slice as JPEG
            slice_img.save(os.path.join(ct_dir_path, f'slice_{i:03d}.jpeg'))

if __name__ == "__main__":
    input_folder = "/sdb/LUNA16/balanced_candidates_augmented"
    output_folder = "/sdb/LUNA16/balanced_candidates_augmented_jpegs"
    npy_to_jpeg(input_folder, output_folder)
