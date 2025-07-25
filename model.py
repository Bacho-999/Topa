import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from reshape import resize_dataset



if __name__ == "__main__":
    input_dir = "dataset_originale"
    output_dir = "dataset_32x32"
    resize_dataset(input_dir, output_dir, size=(32, 32))

# === Modello base ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
"""
# Definizione della rete neurale base
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 8, 8]
        x = x.view(-1, 32 * 8 * 8)            # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#"""

# Inizializzazione del modello (non addestrato)
model = SimpleCNN(num_classes=10)

# Funzione di preprocessamento immagine
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Aggiunge batch dimension

# Esempio di utilizzo con un'immagine
if __name__ == "__main__":
    image_path = "esempio.jpg"  # Sostituisci con un path valido
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"Classe predetta: {predicted_class}")
