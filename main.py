import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN
from reshape import resize_dataset

# === STEP 1: RIDIMENSIONA LE IMMAGINI ===
original_data_dir = "./dataset"
resized_data_dir = "dataset_32x32"

if not os.path.exists(resized_data_dir):
    print("Ridimensionamento immagini in corso...")
    resize_dataset(original_data_dir, resized_data_dir, size=(32, 32))

# === STEP 2: PARAMETRI ===
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# === STEP 3: DATASET ===
transform = transforms.ToTensor()

train_dataset = datasets.ImageFolder(os.path.join(resized_data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(resized_data_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

num_classes = len(train_dataset.classes)
print(f"Classi rilevate: {train_dataset.classes}")

# === STEP 4: MODELLO ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes).to(device)

# === STEP 5: LOSS & OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === STEP 6: TRAINING LOOP ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"[Epoca {epoch+1}/{num_epochs}] Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # VALIDAZIONE
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"  -> Val Accuracy: {val_acc:.2f}%\n")

# === STEP 7: SALVA MODELLO ===
torch.save(model.state_dict(), "modello_addestrato.pth")
print("✅ Modello salvato in modello_addestrato.pth")
