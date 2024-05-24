import os
import numpy as np
from pymongo import MongoClient
import gridfs

# Conectează-te la MongoDB
client = MongoClient('mongodb+srv://maleticimiroslav2016:N2oIudjEPZInvp34@noduleimages.uvkc9zc.mongodb.net/?retryWrites=true&w=majority&appName=NoduleImages')
db = client['NoduleImages']

# Initializează GridFS
fs = gridfs.GridFS(db)

# Funcția pentru a încărca un fișier numpy în GridFS și a crea un document corespunzător
def upload_numpy(file_path):
    with open(file_path, 'rb') as f:
        numpy_data = f.read()
        file_id = fs.put(numpy_data, filename=os.path.basename(file_path))
        
        document = {
            "filename": os.path.basename(file_path),
            "numpy_buffer": file_id
        }
        db.numpy_files.insert_one(document)
        print(f'Fișierul {file_path} a fost încărcat cu succes.')

# Funcția pentru a încărca toate fișierele numpy dintr-un director și a crea documentele corespunzătoare
def upload_numpy_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory_path, filename)
            upload_numpy(file_path)

# Specifică directorul principal unde sunt stocate fișierele numpy
main_directory = '/workspaces/MedApp-Dev-AI/sdb2/verifiedNumpy'  # Înlocuiește cu calea către directorul principal

# Încarcă toate fișierele numpy din directorul principal
upload_numpy_files_in_directory(main_directory)

print("Toate fișierele numpy au fost încărcate.")
