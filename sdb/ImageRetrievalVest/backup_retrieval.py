import torch
import pandas as pd
from tqdm import tqdm
from models import generate_densenet_feature_extraction
from data_loaders_features import data_loader

class FeatureExtractor:
    def __init__(self, model_depth, pretrained_weights_path, device):
        self.model = generate_densenet_feature_extraction(model_depth)
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
        
    def get_validation_augmentations():
        resize_transform = tio.CropOrPad((32, 32, 32))
        rescale_intensity = tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5))
        combined_transforms = tio.Compose([resize_transform, rescale_intensity])
        return combined_transforms
    
    def compute_feature_vector(self, img_path):
        if not os.path.exists(img_path):
            print(f"Error: File {img_path} not found. Skipping this sample.")
            return
        
        image = np.load(full_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        image = get_validation_augmentations()(image)
        feature_vector = self.extract_features(image)
        return feature_vector
    
    def find_closest_images(self, feature_vector, feature_csv_path, output_csv_path, top_k=10):
        # Load the feature vectors CSV
        df = pd.read_csv(feature_csv_path)
        
        # Convert feature vectors from string back to numpy arrays
        df['feature_vector'] = df['feature_vector'].apply(lambda x: np.fromstring(x[1:-1], sep=','))
        
        # Calculate distances
        distances = df['feature_vector'].apply(lambda x: euclidean(x, feature_vector))
        df['distance'] = distances
        
        # Sort by distance
        closest_images = df.sort_values('distance').head(top_k)
        
        # Save to new CSV
        closest_images.to_csv(output_csv_path, index=False)
        return closest_images

    def process_single_image(self, img_path, feature_csv_path, output_csv_path, top_k=10):
        feature_vector = self.compute_feature_vector(img_path)
        closest_images = self.find_closest_images(feature_vector, feature_csv_path, output_csv_path, top_k=top_k)
        return closest_images
        

# Usage
device = torch.device("cpu")
feature_extractor = FeatureExtractor(
    model_depth=169,
    pretrained_weights_path='/sdb/ImageRetrievalVest/saving_models/Densenet169_smaller_csv/best_model_0.8421052631578948.pth',
    device=device
)

# Assuming data_loader is defined elsewhere and properly loaded
feature_extractor.save_features_to_csv(data_loader, '/sdb/ImageRetrievalVest/saving_features/saving_features_try.csv')
