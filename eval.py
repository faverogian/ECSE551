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
from spinal_vgg5 import SpinalVGG

class Hyperparameters():
    batch_size = 64
    gpu = 1
    epochs = 150
    learning_rate = 0.001
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    model = 'ensemble'                  # vgg5, spinal-vgg5, ensemble 

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
if HP.model == 'spinal-vgg5':
    model = SpinalVGG().to(device)
    model.load_state_dict(torch.load('models/spinal-vgg5.ckpt'))
    model.eval()
    model = [model]
elif HP.model == 'vgg5':
    model = VGG().to(device)
    model.load_state_dict(torch.load('models/vgg5.ckpt'))
    model.eval()
    model = [model]
elif HP.model == 'ensemble':
    model = [VGG().to(device), SpinalVGG().to(device)]
    model[0].load_state_dict(torch.load('models/vgg5.ckpt'))
    model[1].load_state_dict(torch.load('models/spinal-vgg5.ckpt'))
    model[0].eval()
    model[1].eval()

with torch.no_grad():
    test_pred = torch.LongTensor()
    for i, data in enumerate(test_loader):
        data = data.to(device)

        # Get the max probability and index of the max probability
        output = []
        for i, m in enumerate(model):
            output.append(m(data))
        
        if HP.model == 'ensemble':
            # Get preds for each model
            model1_pred = output[0].cpu().data.max(1, keepdim=True)
            model2_pred = output[1].cpu().data.max(1, keepdim=True)

            # Fill in a new tensor with the max probability and index of the max probability
            pred = torch.zeros(model1_pred[0].shape)
            for i in range(len(pred)):
                if model1_pred[0][i] > model2_pred[0][i]:
                    pred[i] = model1_pred[1][i]
                else:
                    pred[i] = model2_pred[1][i]
            test_pred = torch.cat((test_pred, pred), dim=0)
        else:
            output = output.squeeze()  
            pred = output.cpu().data.max(1, keepdim=True)[1]
            test_pred = torch.cat((test_pred, pred), dim=0)

# Save predictions to csv with ID and label
test_pred = test_pred.numpy()
test_pred = np.insert(test_pred, 0, np.arange(0, len(test_pred)), axis=1)
np.savetxt("data/Test_labels.csv", test_pred, delimiter=",", fmt='%d', header="id,class", comments='')
