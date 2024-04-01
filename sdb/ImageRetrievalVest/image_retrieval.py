import torch
import pandas as pd
import numpy as np
import torchio as tio
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from models import generate_densenet_feature_extraction
from data_loaders_features import data_loader
import random

class FeatureExtractor:
    def __init__(self, model_depth, pretrained_weights_path, device):
        self.model = generate_densenet_feature_extraction(model_depth)
        self.pretrained_weights = pretrained_weights_path
        self.load_pretrained_weights(pretrained_weights_path)
        self.model = self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        self.device = device

    def load_pretrained_weights(self, weights_path):
        pretrained_dict = torch.load(weights_path)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        #
        # 
        # print(f'weights loaded: {self.pretrained_weights}')

    def extract_features(self, x):
        x = x.float().to(self.device)
        with torch.no_grad():
            features = self.model(x).squeeze()
        return features.cpu().numpy()  # Convert to NumPy array for easier handling

    def process_data_loader(self, data_loader):
        all_data = []
        for batch in tqdm(data_loader, desc="Extracting Features"):
            images, labels, orig_x, orig_y, orig_z, resample_x, resample_y, resample_z, image_paths = batch
            features = self.extract_features(images)
            for i in range(len(labels)):
                all_data.append([
                    labels[i].item(), orig_x[i], orig_y[i], orig_z[i], 
                    resample_x[i], resample_y[i], resample_z[i], image_paths[i],
                    features[i].tolist()  # Convert the feature tensor to a list for storage
                ])
        return all_data

    def save_features_to_csv(self, data_loader, output_csv_path):
        all_data = self.process_data_loader(data_loader)
        df = pd.DataFrame(all_data, columns=[
            'label', 'original_x', 'original_y', 'original_z', 
            'resample_x', 'resample_y', 'resample_z', 'image_path', 'feature_vector'
        ])
        df.to_csv(output_csv_path, index=False)

    def find_closest_images(self, feature_vector, feature_csv_path, output_csv_path, top_k=2):
        # Load the feature vectors CSV
        df = pd.read_csv(feature_csv_path)
        
        # Convert feature vectors from string back to numpy arrays
        df['feature_vector'] = df['feature_vector'].apply(lambda x: np.array([float(num) for num in x.strip('[]').split(',')]))
        
        # Calculate distances
        distances = df['feature_vector'].apply(lambda x: euclidean(x, feature_vector))
        df['distance'] = distances
        
        # Sort by distance
        closest_images_df = df.sort_values('distance').head(top_k)
        
        # Extract the 'image_path' column as a list of strings
        closest_image_paths_numpys = closest_images_df['image_path'].tolist()
        
        # Create a new list where each element has the '.npy' part removed
        closest_folder_image_paths = [path.replace('.npy', '') for path in closest_image_paths_numpys]
        
        # Return the list of closest image paths without '.npy'
        #print(closest_folder_image_paths)
        return closest_image_paths_numpys[0], closest_folder_image_paths
    
    def get_validation_augmentations(self):
        resize_transform = tio.CropOrPad((32, 32, 32))
        rescale_intensity = tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5))
        combined_transforms = tio.Compose([resize_transform, rescale_intensity])
        return combined_transforms

    def process_single_image(self, img_path, feature_csv_path, output_csv_path = ''):
        # Load and preprocess the image
        image = np.load(img_path)
        #print(image)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch and channel dimensions
        image = self.get_validation_augmentations()(image)
        image = image.squeeze(0).numpy()
        patch_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Extract feature vector
        feature_vector = self.extract_features(patch_tensor)
        #print(feature_vector)

        # Find and save closest images
        closest_image_numpys, closest_images_jpegs = self.find_closest_images(feature_vector, feature_csv_path, output_csv_path)
        print(closest_images_jpegs[0])
        return (closest_image_numpys[0],closest_images_jpegs[0])

# Usage example
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feature_extractor = FeatureExtractor(
#     model_depth=169,
#     pretrained_weights_path='/sdb/ImageRetrievalVest/saving_models/Densenet169_bigger_smaller_first_training_Loss(0.8,2)_optim(0.0001,0.001)_StepLR(50,0.8)/best_model_f1=0.884437596302003_epoch=44.pth',
#     device=device
# )

# def return_new_path(numpy_name, path_to_folder = "/sdb/ImageRetrievalVest/balanced_candidates", feature_vector_csv="/sdb/ImageRetrievalVest/saving_features/saving_features_balanced_candidates.csv"):
#     full_image_path1 = path_to_folder + numpy_name
#     output_csv_path='/sdb/ImageRetrievalVest/top_k results csv/results_32x100x100_csv.csv'
#     listx = feature_extractor.process_single_image(full_image_path1, feature_vector_csv, output_csv_path)
#     return listx

def select_random_path(path_to_csv="/sdb/ImageRetrievalVest/csvs/all_patches_balanced_candidates.csv", path_to_jpegs_folder="/sdb/LUNA16/balanced_candidates_augmented_jpegs"):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path_to_csv)
    
    # Generate a random index based on the length of the DataFrame
    index = random.randint(0, len(df) - 1)  # Make sure the index is within the range of the DataFrame
    
    # Return only the value from the 'numpy_filename' column at the randomly selected index
    return df.iloc[index]['numpy_filename']
    
# return_new_path("1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405_227.04500000000002_142.18987854_215.75640800000002.npy","/sdb/ImageRetrievalVest/test_files/")
print(select_random_path())
# Assuming data_loader is defined elsewhere and properly loaded
#feature_extractor.save_features_to_csv(data_loader, '/sdb/ImageRetrievalVest/saving_features/saving_features_balanced_candidates.csv')
