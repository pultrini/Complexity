# %%
import pandas as pd
import numpy as np
import os
import re
from PIL import Image
import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale, ToPILImage
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import get_confusion_matrix
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
# %%
df_subset = pd.read_csv("src/Patient_Sex.csv")


# %%
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx,0]
        image = Image.open(img_path)
        image = image.resize((224,224))
        image = image.convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
transform = Compose([
    Resize((224,224)),
    #Grayscale(),
    ToTensor()
])

dataset = CustomImageDataset(annotations_file='src/Patient_Sex.csv', transform=transform)


training_data, test_data = train_test_split(dataset, test_size=0.3, shuffle=True, random_state=21)

#val_data, test_data = train_test_split(dataset, test_size=0.5, shuffle=True, random_state=21)

#%% 
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

# %%
class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv32_1 = nn.Conv2d(3,32, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv32_2 = nn.Conv2d(32,32, kernel_size=3, stride=1, padding='same', bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv64_1 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding='same', bias = False )
        self.conv64_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv32resid = nn.Conv2d(32, 32, kernel_size=1, stride=2, bias=False)
        self.conv64resid = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False)
        self.avg = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv32_1(x)
        x = self.relu(x)
        y = self.maxpool(x)
        x = self.conv32_2(y)
        x = self.relu(x)
        x = self.maxpool(x)
        residual = self.conv32resid(y)
        residual = self.relu(residual)
        y = residual + x

        x = self.conv64_1(y)
        x = self.relu(x)
        x = self.conv64_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        residual = self.conv64resid(y)
        residual = self.relu(residual)
        y = residual + x
        

        x = torch.flatten(y, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.out(x)
        x = self.sigmoid(x)
        
        return x
    

model = ResNet()

model.to(device)

# %%
loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# %%
import numpy as np

def normalize_data(data):
    """ Normalize the data to the range [0, 1]. """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def calculate_probabilities(data, num_bins):
    """ Calculate the probability distribution of the data. """
    counts, _ = np.histogram(data, bins=num_bins, density=True)
    probabilities = counts/np.sum(counts)
    return probabilities[probabilities > 0]  # Remove zero probabilities

def shannon_entropy(probabilities):
    """ Calculate the Shannon entropy. """
    return -np.sum(probabilities * np.log(probabilities))

def disequilibrium(probabilities, num_bins):
    """ Calculate the disequilibrium. """
    equi_prob = 1.0 / num_bins
    return np.sqrt(np.sum((probabilities - equi_prob)**2))

def lmc_complexity(data, num_bins=100):
    """ Calculate the LMC complexity of the data. """
    normalized_data = normalize_data(data)
    probabilities = calculate_probabilities(normalized_data, num_bins)
    H = shannon_entropy(probabilities)
    D = disequilibrium(probabilities, num_bins)
    C = H * D
    return H, D, C

# %%
from sklearn.metrics import accuracy_score

epochs = 80
epoch_max_complexity = 0
max_valid_accuracy = 0.0
min_valid_loss = float('inf')
max_valid_complex = 0.0
loss_values_train = []
loss_values_val = []

acc_values_train = []
acc_values_val = []

single_complexity = list()
single_entropy = list()
single_disequilibrium = list()

for e in range(epochs):
    y_pred_train = []
    y_true_train = []

    train_loss = 0.0
    model.train()     
    for data, labels in train_dataloader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        target = model(data)
        loss = loss_function(target,labels)
        #loss.requires_grad = True
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0) 
        
        _, preds = torch.max(target, dim=1)
        y_pred_train.append(preds.tolist())
        y_true_train.append(labels.tolist())

    y_pred_train = [item for sublist in y_pred_train for item in sublist]
    y_true_train = [item for sublist in y_true_train for item in sublist]
    train_acc = accuracy_score(y_true_train, y_pred_train)
    acc_values_train.append(train_acc)
    loss_values_train.append(train_loss / len(train_dataloader))

    myp = torch.nn.utils.parameters_to_vector(model.parameters())
    myp = myp.cpu().detach().numpy()
    entropia, disequilibrio, complexidade = lmc_complexity(myp)
    single_complexity.append(complexidade)
    single_entropy.append(entropia)
    single_disequilibrium.append(disequilibrio)

    
    y_pred_val = []
    y_true_val = []
    
    valid_loss = 0.0
    model.eval()     
    for data, labels in test_dataloader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        target = model(data)
        loss = loss_function(target,labels)
        valid_loss += loss.item() * data.size(0)
        
        _, preds = torch.max(target, dim=1)
        y_pred_val.append(preds.tolist())
        y_true_val.append(labels.tolist())
        
    y_pred_val = [item for sublist in y_pred_val for item in sublist]
    y_true_val = [item for sublist in y_true_val for item in sublist]
    val_acc = accuracy_score(y_true_val, y_pred_val)
    acc_values_val.append(val_acc)
    loss_values_val.append(valid_loss / len(test_dataloader))
    
    if val_acc  > max_valid_accuracy:
        max_valid_accuracy = val_acc 
    if valid_loss  < min_valid_loss:
        min_valid_loss = valid_loss 
        torch.save(model.state_dict(), '/home/users/u12559743/DAVI/IC/Complexity/results/min_loss.pth')
    if complexidade  > max_valid_complex:
        max_valid_complex = complexidade 
        torch.save(model.state_dict(), '/home/users/u12559743/DAVI/IC/Complexity/results/max_complexity.pth')
        epoch_max_complexity = e
    
    
    print(f'Epoch {e+1}: \n Training Loss: {train_loss/len(train_dataloader)} \t Training Acc: {train_acc} \n Validation Loss: {valid_loss/len(test_dataloader)} \t Validation Acc: {val_acc}')
    print(f'complexidade {complexidade}')
    '''
    if min_valid_loss > valid_loss:
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'resnet_model.pth')'
    '''
    
# %%
import json

metrics = {
    'Max_accuracy' : max_valid_accuracy,
    'Min_loss' : min_valid_loss,
    'Max_complexity': max_valid_complex,
    'Epoch_max_complexity' : epoch_max_complexity
}

output_file = 'results/metrics/metrics_model1.json'
with open(output_file, 'w') as file:
    json.dump(metrics, file, indent=4)

torch.cuda.empty_cache()