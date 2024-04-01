from fastapi import FastAPI
from image_retrieval import FeatureExtractor
import torch
import pandas as pd
import random
from fastapi.middleware.cors import CORSMiddleware
import os
app = FastAPI()
origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path_to_folder = "/sdb/LUNA16/balanced_candidates/"
feature_vector_csv = "/sdb/ImageRetrievalVest/saving_features/saving_features_balanced_candidates.csv"
path_to_csv = "/sdb/ImageRetrievalVest/csvs/all_patches_balanced_candidates.csv" 
path_to_jpegs_folder="/sdb/LUNA16/balanced_candidates_augmented_jpegs"
path_to_jpges_local="/public/"

@app.get("/next_image")
def ai(image: str | None = None):
    final_image = ''
    without_npy = []
    if image:
        full_image_path1 = path_to_folder + image
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = FeatureExtractor(
            model_depth=169,
            pretrained_weights_path='InteliMed.AI\sdb\ImageRetrievalVest\saving_models\Densenet169_bigger_smaller_first_training_Loss(0.8,2)_optim(0.0001,0.001)_StepLR(50,0.8)\best_model_f1=0.884437596302003_epoch=44.pth',
            device=device
        )
        
        listx = feature_extractor.process_single_image(full_image_path1, feature_vector_csv)
        final_image = listx[1]
        without_npy_extension = final_image.replace(".npy","")
        final_path_to_folder = path_to_jpegs_folder + "/" + without_npy_extension
            # Iterate through all files in the specified folder
        for filename in os.listdir(final_path_to_folder):
            if filename.endswith(".jpeg"):  # Check if the file is a JPEG
                full_path = os.path.join(path_to_jpges_local+without_npy_extension, filename)
                without_npy.append(full_path)  # Add the full path to the list
        print("AI")
        # If there are no JPEG images, return an error message
        if not without_npy:
            return {'error': 'No JPEG images found in the specified folder'}
        without_npy.sort()
    else:
                # Read the CSV file into a DataFrame
        df = pd.read_csv(path_to_csv)
        
        # Generate a random index based on the length of the DataFrame
        index = random.randint(0, len(df) - 1)  # Make sure the index is within the range of the DataFrame
        print("random")
        # Return only the value from the 'numpy_filename' column at the randomly selected index
        final_image = df.iloc[index]['numpy_filename']
        without_npy_extension = final_image.replace(".npy","")
        final_path_to_folder = path_to_jpegs_folder + "/" + without_npy_extension
            # Iterate through all files in the specified folder
        for filename in os.listdir(final_path_to_folder):
            if filename.endswith(".jpeg"):  # Check if the file is a JPEG
                full_path = os.path.join(path_to_jpges_local+without_npy_extension, filename)
                without_npy.append(full_path)  # Add the full path to the list

        # If there are no JPEG images, return an error message
        if not without_npy:
            return {'error': 'No JPEG images found in the specified folder'}
        without_npy.sort()
    return {'image': without_npy, 'npy': final_image}

@app.get("/check_image")
def check(image: str, answer: int):
    df = pd.read_csv(path_to_csv)
    
    # Return only the value from the 'numpy_filename' column at the randomly selected index
    row = df.loc[df['numpy_filename'] == image]
    if not row.empty and row['class_label'].dtype == 'int64':
        label = int(row['class_label'].iloc[0])
        
    print(f'label={label}')
    print(f'answer={answer}')
    correct = label == answer
    
    return {'correct': correct}
