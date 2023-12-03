# VGG-5 classifier for EMNIST dataset
# Gian Favero, 2023
# Training script

# Project imports
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import math
import torch.nn.functional as F

from vgg5 import VGG

class Hyperparameters():
    batch_size = 64
    gpu = 1
    epochs = 150
    learning_rate = 0.001
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_data = transforms.ToPILImage()(sample_data)

        if self.transform:
            sample_data = self.transform(sample_data)

        return sample_data, self.targets[idx]

HP = Hyperparameters()

# setup device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Read image data and their label into a Dataset 
data = pickle.load(open('data/Train.pkl', 'rb' ))
targets = np.genfromtxt('data/Train_labels.csv', delimiter=',')

data = torch.from_numpy(data)
targets = torch.from_numpy(targets)

# Remove first row and column from targets
targets = targets[1:]
targets = targets[:,1]

dataset = TensorDataset(data,targets)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomPerspective(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 10% translation
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = CustomDataset(train_dataset[:][0], train_dataset[:][1], transform=train_transforms)
val_dataset = CustomDataset(val_dataset[:][0], val_dataset[:][1], transform=val_transforms)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=HP.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=HP.batch_size, shuffle=True)
    
# Create model
model = VGG().to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=HP.learning_rate)

# Train model
train_loss = []
train_acc = []
val_loss = []
val_acc = []

for epoch in range(HP.epochs):
    # Train model
    model.train()
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = HP.criterion(outputs.float(), labels.long())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{HP.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Decay learning rate
    if (epoch+1) > 20 == 0:
        HP.learning_rate /= 3
        optimizer = torch.optim.Adam(model.parameters(), lr=HP.learning_rate)

    # Evaluate model
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        epoch_loss = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = HP.criterion(outputs.float(), labels.long())
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(val_loader)
        val_loss.append(epoch_loss)
        val_acc.append(correct / total)

        print(f'Epoch [{epoch+1}/{HP.epochs}], Val Loss: {epoch_loss:.4f}, Val Acc: {100 * correct / total:.4f}')

torch.save(model.state_dict(), 'models/vgg5.ckpt')