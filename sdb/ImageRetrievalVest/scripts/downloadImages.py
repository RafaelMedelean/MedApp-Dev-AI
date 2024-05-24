import random
from pymongo import MongoClient
import gridfs
import os

# Conectează-te la MongoDB local
client = MongoClient('mongodb+srv://maleticimiroslav2016:N2oIudjEPZInvp34@noduleimages.uvkc9zc.mongodb.net/?retryWrites=true&w=majority&appName=NoduleImages')
db = client['NoduleImages']

# Initializează GridFS
fs = gridfs.GridFS(db)

# Funcția pentru a extrage o imagine aleatorie din GridFS
def get_random_image():
    # Obține toate documentele din colecția imagini
    documents = list(db.imagini.find())
    if not documents:
        print("Nu există imagini în baza de date.")
        return
    
    # Alege un document aleatoriu
    random_document = random.choice(documents)
    
    # Alege o imagine aleatorie din documentul selectat
    random_image_id = random.choice(random_document['images'])
    
    # Descarcă imaginea din GridFS
    image_data = fs.get(random_image_id).read()
    
    # Salvează imaginea pe disc
    output_path = os.path.join('/workspaces/MedApp-Dev-AI/mere', f'random_image_{random_image_id}.jpg')
    with open(output_path, 'wb') as f:
        f.write(image_data)
    
    print(f'Imaginea a fost salvată la: {output_path}')

# Extrage și salvează o imagine aleatorie
get_random_image()
