# %%
import pandas as pd
import numpy as np
import os
import re
from PIL import Image
import torch
from torch import nn
import torchvision
from torchvision.transforms.v2 import ToDtype, ToImage, Compose, Grayscale, ToPILImage
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from medmnist import TissueMNIST, BloodMNIST, INFO
import mlflow

# %%
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
DATA_ROOT = '/home/users/u12559743/Documentos/complexity_antiga/Complexity/data/'
transform =  Compose([
        Grayscale(),
        ToImage(),
        ToDtype(torch.float32, scale=True)
    ])
dataset_train = BloodMNIST(split='train', download=True,
                                root=DATA_ROOT, transform=transform)
dataset_val = BloodMNIST(split='val', download=True,
                                root=DATA_ROOT, transform=transform)

train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=256,
        shuffle=True,
        drop_last=True
    )   
val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=128,
        shuffle=True,
        drop_last=True
    )

model = torchvision.models.resnet50(weights=None)

original_conv = model.conv1
model.conv1 = nn.Conv2d(
    in_channels=1,  
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=False
)

info = INFO['bloodmnist']
n_classes = len(info['label'])
original_num_features = model.fc.in_features
model.fc = nn.Linear(
    in_features=original_num_features,
    out_features=n_classes
)
weigths_path = '/home/users/u12559743/Documentos/complexity_antiga/Complexity/results/max_complexity.pth'
model.load_state_dict(torch.load(weigths_path))

model.to(device)

#%%


# %%
# Initialize the loss function
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# %%
import numpy as np
mlflow.set_tracking_uri("http://127.0.0.1:4252")
mlflow.set_experiment(experiment_id=820686148083765719)
def normalize_data(data):
    """ Normalize the data to the range [0, 1]. """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def calculate_probabilities(data, num_bins):
    """ Calculate the probability distribution of the data. """
    counts, _ = np.histogram(data, bins=num_bins, density=False)
    total = np.sum(counts)
    if total == 0:
        return np.zeros(num_bins)
    probabilities = counts/np.sum(counts)
    return probabilities[probabilities > 0]  # Remove zero probabilities

def shannon_entropy(probabilities):
    """ Calculate the Shannon entropy. """
    p_nonzero = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log(p_nonzero))

def disequilibrium(probabilities, num_bins):
    """ Calculate the disequilibrium. """
    equi_prob = 1.0 / num_bins
    return np.sqrt(np.sum((probabilities - equi_prob)**2))

def lmc_complexity(data, num_bins=100):
    """ Calculate the LMC complexity of the data. """
    normalized_data = normalize_data(data)
    probabilities = calculate_probabilities(normalized_data, num_bins)
    H = shannon_entropy(probabilities)
    D = disequilibrium(probabilities, num_bins=100)
    C = H * D
    return H, D, C

# %%
from sklearn.metrics import accuracy_score

with mlflow.start_run(run_name='complexity run') as run:
    epochs = 10
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
        for data, labels in train_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            labels = labels.squeeze(1).long()
            
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
        mlflow.log_metric('acurácia treino max complexity',f'{train_acc:2f}', step=e)
        acc_values_train.append(train_acc)
        loss_values_train.append(train_loss / len(train_loader))
        mlflow.log_metric('perda treino max complexity',train_loss, step=e)

        myp = torch.nn.utils.parameters_to_vector(model.parameters())
        myp = myp.cpu().detach().numpy()
        entropia, disequilibrio, complexidade = lmc_complexity(myp)
        single_complexity.append(complexidade)
        mlflow.log_metric('complexidade max complexity',complexidade, step=e)
        mlflow.log_metric('disequilibrio max complexity',disequilibrio, step=e)
        mlflow.log_metric('entropia max complexity', entropia, step=e)
        single_entropy.append(entropia)
        single_disequilibrium.append(disequilibrio)
        print(f'época {e+1} Complexity {complexidade}, Entropy {entropia}, disequilibrio {disequilibrio}')

        
        y_pred_val = []
        y_true_val = []
        
        valid_loss = 0.0
        model.eval()     
        for data, labels in val_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            labels = labels.squeeze(1).long()
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
        mlflow.log_metric('acuracia validacao max complexity', val_acc, step=e)
        loss_values_val.append(valid_loss / len(val_loader))
        mlflow.log_metric('perda validacao max complexity', valid_loss, step=e)
        
        print(f'loss {train_loss/len(train_loader)}, acc {train_acc}')
        # print(f'Epoch {e+1}: \n Training Loss: {train_loss/len(train_loader)} \t Training Acc: {train_acc} \n Validation Loss: {valid_loss/len(val_loader)} \t Validation Acc: {val_acc}')
        # print(f'complexidade {complexidade}')
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

output_file ='results/metrics/metrics_model2_loss.json'
with open(output_file, 'w') as file:
    json.dump(metrics, file, indent=4)
# %%
torch.cuda.empty_cache()