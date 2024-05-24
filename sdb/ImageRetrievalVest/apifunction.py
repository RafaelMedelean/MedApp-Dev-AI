# import random
# import pandas as pd
# import torch
# from pymongo import MongoClient
# import gridfs
# import io
# from PIL import Image
# import base64
# from image_retrieval import FeatureExtractor

# # MongoDB connection
# client = MongoClient('mongodb+srv://maleticimiroslav2016:N2oIudjEPZInvp34@noduleimages.uvkc9zc.mongodb.net/?retryWrites=true&w=majority&appName=NoduleImages')
# db = client['NoduleImages']
# fs = gridfs.GridFS(db)

# # Paths to required files
# path_to_folder = "/workspaces/MedApp-Dev-AI/sdb2/balanced_candidates/"
# feature_vector_csv = "/workspaces/MedApp-Dev-AI/sdb2/saving_features_balanced_candidates.csv"
# path_to_csv = "/workspaces/MedApp-Dev-AI/sdb2/working_sample_jpegs.csv"

# def get_image(image: str | None = None):
#     final_image = ''
#     without_npy = []
#     if image:
#         full_image_path1 = path_to_folder + image
#         device = torch.device("cpu")
#         feature_extractor = FeatureExtractor(
#             model_depth=169,
#             pretrained_weights_path='/workspaces/MedApp-Dev-AI/sdb/ImageRetrievalVest/saving_models/Densenet169_bigger_smaller_first_training_Loss(0.8,2)_optim(0.0001,0.001)_StepLR(50,0.8)/best_model_f1=0.88_epoch=41.pth',
#             device=device
#         )
        
#         listx = feature_extractor.process_single_image(full_image_path1, feature_vector_csv)
#         final_image = listx[1]
#         without_npy_extension = final_image.replace(".npy", "")
        
#         # Fetch images from MongoDB GridFS
#         for file in db.fs.files.find({"filename": {"$regex": without_npy_extension}}):
#             file_id = file['_id']
#             image_data = fs.get(file_id).read()
#             image = Image.open(io.BytesIO(image_data))
#             buffered = io.BytesIO()
#             image.save(buffered, format="JPEG")
#             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#             without_npy.append(f"data:image/jpeg;base64,{img_str}")
        
#         if not without_npy:
#             return {'error': 'No JPEG images found in the specified folder1'}
#     else:
#         # Read the CSV file into a DataFrame
#         df = pd.read_csv(path_to_csv)
        
#         # Generate a random index based on the length of the DataFrame
#         index = random.randint(0, len(df) - 1)  # Make sure the index is within the range of the DataFrame
#         print("random")
#         # Return only the value from the 'numpy_filename' column at the randomly selected index
#         final_image = df.iloc[index]['numpy_filename']
#         without_npy_extension = final_image.replace(".npy", "")
#         print("without_npy_extension: ", without_npy_extension)
#         # Fetch images from MongoDB GridFS
#         for file in db.fs.files.find({"filename": {"$regex": without_npy_extension}}):
#             file_id = file['_id']
#             image_data = fs.get(file_id).read()
#             image = Image.open(io.BytesIO(image_data))
#             buffered = io.BytesIO()
#             image.save(buffered, format="JPEG")
#             img_str = base64.b64encode(buffered.getvalue()).decode("uxtf-8")
#             without_npy.append(f"data:image/jpeg;base64,{img_str}")
        
#         if not without_npy:
#             return {'error': 'No JPEG images found in the specified folder2'}
    
#     return {'image': without_npy, 'npy': final_image}

# def check_image(image: str, answer: int):
#     df = pd.read_csv(path_to_csv)
    
#     # Return only the value from the 'numpy_filename' column at the randomly selected index
#     row = df.loc[df['numpy_filename'] == image]
#     if not row.empty and row['class_label'].dtype == 'int64':
#         label = int(row['class_label'].iloc[0])
        
#     print(f'label={label}')
#     print(f'answer={answer}')
#     correct = label == answer
    
#     return {'correct': correct}

# if __name__ == "__main__":
#     # Example usage:
#     # Fetch the next image (randomly or specified)
#     result = get_image()
#     print("result: ", result)
#     print("\n\n\n\n\n")
#     # Example usage:
#     # Check the image label
#     image_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.118140393257625250121502185026_82.89933760000002_277.75874035_127.23653476999999.npy'  # replace with the actual image name to check
#     # # user_answer = 1  # replace with the user's answer
#     # # check_result = check_image(image_name, user_answer)
#     # # print(check_result)
#     newimage=get_image(image_name)
#     print(newimage)







# #v2
# import random
# import pandas as pd
# import torch
# from pymongo import MongoClient
# import gridfs
# import io
# import numpy as np
# from PIL import Image
# import base64
# from image_retrieval import FeatureExtractor

# # MongoDB connection
# client = MongoClient('mongodb+srv://maleticimiroslav2016:N2oIudjEPZInvp34@noduleimages.uvkc9zc.mongodb.net/?retryWrites=true&w=majority&appName=NoduleImages')
# db = client['NoduleImages']
# fs = gridfs.GridFS(db)

# # Paths to required files
# path_to_folder = "/workspaces/MedApp-Dev-AI/sdb2/balanced_candidates/"
# feature_vector_csv = "/workspaces/MedApp-Dev-AI/sdb2/saving_features_balanced_candidates.csv"
# path_to_csv = "/workspaces/MedApp-Dev-AI/sdb2/working_sample_jpegs.csv"

# def get_image(image: str | None = None):
#     final_image = ''
#     without_npy = []
#     numpy_array = None
    
#     if image:
#         full_image_path1 = path_to_folder + image
#         device = torch.device("cpu")
#         feature_extractor = FeatureExtractor(
#             model_depth=169,
#             pretrained_weights_path='/workspaces/MedApp-Dev-AI/sdb/ImageRetrievalVest/saving_models/Densenet169_bigger_smaller_first_training_Loss(0.8,2)_optim(0.0001,0.001)_StepLR(50,0.8)/best_model_f1=0.88_epoch=41.pth',
#             device=device
#         )
        
#         listx = feature_extractor.process_single_image(full_image_path1, feature_vector_csv)
#         final_image = listx[1]
#         without_npy_extension = final_image.replace(".npy", "")
        
#         # Fetch the .npy file from MongoDB GridFS
#         for file in db.fs.files.find({"filename": {"$regex": without_npy_extension}}):
#             if file["filename"].endswith(".npy"):
#                 file_id = file['_id']
#                 numpy_data = fs.get(file_id).read()
#                 numpy_array = np.load(io.BytesIO(numpy_data))
#                 print(f"Numpy array for {file['filename']}: {numpy_array}")
#                 break
        
#         if numpy_array is None:
#             return {'error': 'No Numpy file found in the specified folder'}
        
#     else:
#         # Read the CSV file into a DataFrame
#         df = pd.read_csv(path_to_csv)
        
#         # Generate a random index based on the length of the DataFrame
#         index = random.randint(0, len(df) - 1)  # Make sure the index is within the range of the DataFrame
#         print("random")
#         # Return only the value from the 'numpy_filename' column at the randomly selected index
#         final_image = df.iloc[index]['numpy_filename']
#         without_npy_extension = final_image.replace(".npy", "")
#         print("without_npy_extension: ", without_npy_extension)
        
#         # Fetch the .npy file from MongoDB GridFS
#         for file in db.fs.files.find({"filename": {"$regex": without_npy_extension}}):
#             if file["filename"].endswith(".npy"):
#                 file_id = file['_id']
#                 numpy_data = fs.get(file_id).read()
#                 numpy_array = np.load(io.BytesIO(numpy_data))
#                 print(f"Numpy array for {file['filename']}: {numpy_array}")
#                 break
        
#         if numpy_array is None:
#             return {'error': 'No Numpy file found in the specified folder'}
    
#     return {'npy': final_image, 'numpy_array': numpy_array.tolist()}

# def check_image(image: str, answer: int):
#     df = pd.read_csv(path_to_csv)
    
#     # Return only the value from the 'numpy_filename' column at the randomly selected index
#     row = df.loc[df['numpy_filename'] == image]
#     if not row.empty and row['class_label'].dtype == 'int64':
#         label = int(row['class_label'].iloc[0])
        
#     print(f'label={label}')
#     print(f'answer={answer}')
#     correct = label == answer
    
#     return {'correct': correct}

# if __name__ == "__main__":
#     # Example usage:
#     # Fetch the next image (randomly or specified)
#     result = get_image()
#     print("result: ", result)
#     print("\n\n\n\n\n")
#     # Example usage:
#     # Check the image label
#     image_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.118140393257625250121502185026_82.89933760000002_277.75874035_127.23653476999999.npy'  # replace with the actual image name to check
#     newimage = get_image(image_name)
#     print(newimage)







#v3


import random
import pandas as pd
import torch
from pymongo import MongoClient
import gridfs
import io
import numpy as np
from PIL import Image
import os
from image_retrieval import FeatureExtractor

# MongoDB connection
client = MongoClient('mongodb+srv://maleticimiroslav2016:N2oIudjEPZInvp34@noduleimages.uvkc9zc.mongodb.net/?retryWrites=true&w=majority&appName=NoduleImages')
db = client['NoduleImages']
fs = gridfs.GridFS(db)

# Paths to required files
path_to_folder = "/workspaces/MedApp-Dev-AI/sdb2/balanced_candidates/"
feature_vector_csv = "/workspaces/MedApp-Dev-AI/sdb2/saving_features_balanced_candidates.csv"
path_to_csv = "/workspaces/MedApp-Dev-AI/sdb2/working_sample_jpegs.csv"
save_npy_folder = "/workspaces/MedApp-Dev-AI/sdb2/saved_npy_files/"  # Folder to save the .npy files

# Ensure the save folder exists
os.makedirs(save_npy_folder, exist_ok=True)

def get_image(image: str | None = None):
    final_image = ''
    without_npy = []
    numpy_array = None
    
    if image:

        full_image_path1 = path_to_folder + image
        device = torch.device("cpu")
        feature_extractor = FeatureExtractor(
            model_depth=169,
            pretrained_weights_path='/workspaces/MedApp-Dev-AI/sdb/ImageRetrievalVest/saving_models/Densenet169_bigger_smaller_first_training_Loss(0.8,2)_optim(0.0001,0.001)_StepLR(50,0.8)/best_model_f1=0.88_epoch=41.pth',
            device=device
        )
        
        listx = feature_extractor.process_single_image(full_image_path1, feature_vector_csv)
        final_image = listx[1]
        final_image+=".npy"
        print("final_image: ", final_image)
        # without_npy_extension = final_image.replace(".npy", "")
        # print("without_npy_extension: ", without_npy_extension)
        # Fetch the .npy file from MongoDB GridFS
        for file in db.fs.files.find({"filename": {"$regex": final_image}}):
            if file["filename"].endswith(".npy"):
                file_id = file['_id']
                numpy_data = fs.get(file_id).read()
                numpy_array = np.load(io.BytesIO(numpy_data))
                print(f"Numpy array for {file['filename']}: {numpy_array}")
                
                # Save the numpy array to a file
                save_path = os.path.join(save_npy_folder, file['filename'])
                with open(save_path, 'wb') as f:
                    f.write(numpy_data)
                print(f"Numpy file saved to {save_path}")
                break
        
        if numpy_array is None:
            return {'error': 'No Numpy file found in the specified folder pe if '}
    
        
    else:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(path_to_csv)
        
        # Generate a random index based on the length of the DataFrame
        index = random.randint(0, len(df) - 1)  # Make sure the index is within the range of the DataFrame
        print("random")
        # Return only the value from the 'numpy_filename' column at the randomly selected index
        final_image = df.iloc[index]['numpy_filename']
        without_npy_extension = final_image.replace(".npy", "")
        print("without_npy_extension: ", without_npy_extension)
        
        # Fetch the .npy file from MongoDB GridFS
        for file in db.fs.files.find({"filename": {"$regex": without_npy_extension}}):
            if file["filename"].endswith(".npy"):
                file_id = file['_id']
                numpy_data = fs.get(file_id).read()
                numpy_array = np.load(io.BytesIO(numpy_data))
                print(f"Numpy array for {file['filename']}: {numpy_array}")
                
                # Save the numpy array to a file
                save_path = os.path.join(save_npy_folder, file['filename'])
                with open(save_path, 'wb') as f:
                    f.write(numpy_data)
                print(f"Numpy file saved to {save_path}")
                break
        
        if numpy_array is None:
            return {'error': 'No Numpy file found in the specified folder'}
    
    return {'npy': final_image, 'numpy_array': numpy_array.tolist()}

def check_image(image: str, answer: int):
    df = pd.read_csv(path_to_csv)
    
    # Return only the value from the 'numpy_filename' column at the randomly selected index
    row = df.loc[df['numpy_filename'] == image]
    if not row.empty and row['class_label'].dtype == 'int64':
        label = int(row['class_label'].iloc[0])
        
    print(f'label={label}')
    print(f'answer={answer}')
    correct = label == answer
    
    return {'correct': correct}

if __name__ == "__main__":
    # Example usage:
    # Fetch the next image (randomly or specified)
    result = get_image()
    # print("result: ", result)
    print("\n\n\n\n\n")
    # Example usage:
    # Check the image label
    # image_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.118140393257625250121502185026_82.89933760000002_277.75874035_127.23653476999999.npy'  # replace with the actual image name to check
    print("result: ", result["npy"])
    newimage = get_image(result["npy"])
    print(newimage)



