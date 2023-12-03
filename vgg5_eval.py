# VGG-5 classifier for EMNIST dataset
# Gian Favero, 2023
# Evaluation script

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
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_data = transforms.ToPILImage()(sample_data)

        if self.transform:
            sample_data = self.transform(sample_data)

        return sample_data

HP = Hyperparameters()

# setup device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Read image data into a Dataset 
data = pickle.load(open('data/Test.pkl', 'rb' ))
data = torch.from_numpy(data)

# Define transforms
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = CustomDataset(data, transform=test_transforms)

# Create dataloaders
test_loader = DataLoader(test_dataset, batch_size=HP.batch_size, shuffle=False)
    
# Create model
model = VGG().to(device)

# Load model
model.load_state_dict(torch.load('models/vgg5.ckpt'))

# Generate labels for test set
model.eval()

with torch.no_grad():
    test_pred = torch.LongTensor()
    for i, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)

# Save predictions to csv with ID and label
test_pred = test_pred.numpy()
test_pred = np.insert(test_pred, 0, np.arange(0, len(test_pred)), axis=1)
np.savetxt("data/Test_labels.csv", test_pred, delimiter=",", fmt='%d', header="id,class", comments='')
