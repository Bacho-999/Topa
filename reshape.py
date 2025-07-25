from PIL import Image
import os

def resize_dataset(input_folder, output_folder, size=(32, 32)):
    os.makedirs(output_folder, exist_ok=True)

    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                input_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, rel_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                img = Image.open(input_path).convert("RGB")
                img_resized = img.resize(size)
                img_resized.save(output_path)

    print(f"Tutte le immagini sono state ridimensionate a {size[0]}x{size[1]}.")