import os
import numpy as np
import shutil

def find_common_numpy_files(numpy_folder, subfolders_parent_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the numpy_folder
    numpy_files = [f for f in os.listdir(numpy_folder) if f.endswith('.npy')]

    for numpy_file in numpy_files:
        numpy_file_path = os.path.join(numpy_folder, numpy_file)
        
        # Check if a subfolder with the same name as the numpy file (without extension) exists
        subfolder_name = os.path.splitext(numpy_file)[0]
        subfolder_path = os.path.join(subfolders_parent_folder, subfolder_name)
        
        if os.path.isdir(subfolder_path):
            # If both exist, copy the numpy file to the output folder
            output_path = os.path.join(output_folder, numpy_file)
            shutil.copy(numpy_file_path, output_path)
            print(f'Copied {numpy_file} to {output_folder}')

# Example usage
numpy_folder = '/workspaces/MedApp-Dev-AI/sdb2/balanced_candidates'  # Replace with the path to your NumPy files folder
subfolders_parent_folder = '/workspaces/MedApp-Dev-AI/sdb2/balanced_candidates_augmented_jpegs'  # Replace with the path to the folder containing subfolders
output_folder = '/workspaces/MedApp-Dev-AI/sdb2/verifiedNumpy'  # Replace with the path to the folder where you want to save the common files

find_common_numpy_files(numpy_folder, subfolders_parent_folder, output_folder)
