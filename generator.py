from PIL import Image
import numpy as np
import os

def crea_immagini_cartella(nome_classe, num_immagini, output_dir):
    classe_path = os.path.join(output_dir, nome_classe)
    os.makedirs(classe_path, exist_ok=True)

    for i in range(num_immagini):
        array = np.random.randint(0, 256, (1080, 1080, 3), dtype=np.uint8)
        img = Image.fromarray(array)
        img.save(os.path.join(classe_path, f"{nome_classe}_{i}.jpg"))

# Esempio
output_dir = "dataset/train"
crea_immagini_cartella("classe1", 10, output_dir)
crea_immagini_cartella("classe2", 10, output_dir)

output_dir = "dataset/val"
crea_immagini_cartella("classe1", 3, output_dir)
crea_immagini_cartella("classe2", 3, output_dir)

print("Immagini generate.")
