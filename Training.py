import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import LabelImages
from StegoDetector import StegoDetector
import LabelImages

# Prepare dataset
train_normal_folder = "preprocessed_images/train_images_after_preprcessing/"
train_stego_folder = "preprocessed_images/train_stagoimages_after_preprcessing/"
test_normal_folder = "preprocessed_images/test_images_after_preprcessing/"
test_stego_folder = "preprocessed_images/test_stagoimages_after_preprcessing/"

train_normal = [os.path.join(train_normal_folder, f) for f in os.listdir(train_normal_folder)]
train_stego = [os.path.join(train_stego_folder, f) for f in os.listdir(train_stego_folder)]
test_normal = [os.path.join(test_normal_folder, f) for f in os.listdir(test_normal_folder)]
test_stego = [os.path.join(test_stego_folder, f) for f in os.listdir(test_stego_folder)]

train_dataset = LabelImages.LabelImages(train_normal, train_stego)
test_dataset = LabelImages.LabelImages(test_normal, test_stego)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Training setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StegoDetector().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    
# Testing    
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")