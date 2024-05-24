# import os
# from pymongo import MongoClient
# import gridfs

# # Conectează-te la MongoDB
# client = MongoClient('mongodb+srv://maleticimiroslav2016:N2oIudjEPZInvp34@noduleimages.uvkc9zc.mongodb.net/?retryWrites=true&w=majority&appName=NoduleImages')
# db = client['ImaginiMongoDB']

# # Initializează GridFS
# fs = gridfs.GridFS(db)

# # Funcția pentru a încărca o imagine în GridFS
# def upload_image(file_path):
#     with open(file_path, 'rb') as f:
#         image_data = f.read()
#         fs.put(image_data, filename=os.path.basename(file_path))
#         print(f'Imaginea {file_path} a fost încărcată cu succes.')

# # Specifică directorul unde sunt stocate imaginile
# image_directory = 'calea/catre/directorul/cu/imagini'

# # Încarcă toate imaginile din director
# for filename in os.listdir(image_directory):
#     if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#         file_path = os.path.join(image_directory, filename)
#         upload_image(file_path)

# print("Toate imaginile au fost încărcate.")

import os
from pymongo import MongoClient
import gridfs

# Conectează-te la MongoDB
client = MongoClient('mongodb+srv://maleticimiroslav2016:N2oIudjEPZInvp34@noduleimages.uvkc9zc.mongodb.net/?retryWrites=true&w=majority&appName=NoduleImages')
db = client['NoduleImages']

# Initializează GridFS
fs = gridfs.GridFS(db)

# Funcția pentru a încărca o imagine în GridFS
def upload_image(file_path):
    with open(file_path, 'rb') as f:
        image_data = f.read()
        file_id = fs.put(image_data, filename=os.path.basename(file_path))
        return file_id

# Funcția pentru a încărca toate imaginile dintr-un subdirector și a crea documentul corespunzător
def upload_directory(directory_path):
    image_ids = []
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(directory_path, filename)
            file_id = upload_image(file_path)
            image_ids.append(file_id)
    if image_ids:
        document = {
            "path": directory_path,
            "images": image_ids
        }
        db.imagini.insert_one(document)
        print(f'Directorul {directory_path} a fost încărcat cu succes.')

# Specifică directorul principal unde sunt stocate subdirectoarele
main_directory = '/workspaces/MedApp-Dev-AI/sdb2/balanced_candidates_augmented_jpegs'

# Parcurge toate subdirectoarele și încarcă imaginile
for subdirectory in os.listdir(main_directory):
    subdirectory_path = os.path.join(main_directory, subdirectory)
    if os.path.isdir(subdirectory_path):
        upload_directory(subdirectory_path)

print("Toate imaginile au fost încărcate.")
