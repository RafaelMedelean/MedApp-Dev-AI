import random
import numpy as np
from pymongo import MongoClient
import gridfs
import io

# Conectează-te la MongoDB
client = MongoClient('mongodb+srv://maleticimiroslav2016:N2oIudjEPZInvp34@noduleimages.uvkc9zc.mongodb.net/?retryWrites=true&w=majority&appName=NoduleImages')
db = client['NoduleImages']

# Initializează GridFS
fs = gridfs.GridFS(db)

# Funcția pentru a descărca un fișier numpy aleatoriu din GridFS și a-l încărca într-o variabilă NumPy
def download_random_numpy():
    # Obține toate documentele din colecția numpy_files
    documents = list(db.numpy_files.find())
    
    if not documents:
        print("Nu există fișiere numpy în baza de date.")
        return
    
    # Alege un document aleatoriu
    random_document = random.choice(documents)
    file_id = random_document['numpy_buffer']
    
    # Descarcă fișierul din GridFS
    numpy_data = fs.get(file_id).read()
    
    # Încarcă datele într-un array NumPy folosind un obiect BytesIO
    numpy_array = np.load(io.BytesIO(numpy_data))  # Using BytesIO to correctly interpret the npy file
    
    print(f'Numele fișierului: {random_document["filename"]}')
    print(f'Datele numpy: \n{numpy_array}')
    return numpy_array

# Descărcați și afișați un fișier numpy aleatoriu
random_numpy_array = download_random_numpy()
