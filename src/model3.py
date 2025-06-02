# %%
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

print_config()
# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
# %%
resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
md5 = "0bc7306e7427e00ad1c5526a6677552d"

compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
data_dir = os.path.join(root_dir, "MedNIST")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)
# %%

set_determinism(seed=0)

# %%
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])
num_total = len(image_class)
image_width, image_height = PIL.Image.open(image_files_list[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")
# %%
'''
plt.subplots(3, 3, figsize=(8, 8))
for i, k in enumerate(np.random.randint(num_total, size=9)):
    im = PIL.Image.open(image_files_list[k])
    arr = np.array(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
plt.tight_layout()
plt.show()
'''
# %%
val_frac = 0.1
test_frac = 0.1
length = len(image_files_list)
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")

# %%
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ],  

)

val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])

# %%
class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_ds = MedNISTDataset(train_x, train_y, train_transforms)
train_dataloader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

val_ds = MedNISTDataset(val_x, val_y, val_transforms)
val_dataloader = DataLoader(val_ds, batch_size=300, num_workers=10)

test_ds = MedNISTDataset(test_x, test_y, val_transforms)
test_dataloader = DataLoader(test_ds, batch_size=300, num_workers=10)

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
from torch import nn
import torch.nn.functional as F
class ResNet(nn.Module):
    def __init__(self, num_classes=num_class):
        super().__init__()
        self.conv32_1 = nn.Conv2d(1,32, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv32_2 = nn.Conv2d(32,32, kernel_size=3, stride=1, padding='same', bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv64_1 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding='same', bias = False )
        self.conv64_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv32resid = nn.Conv2d(32, 32, kernel_size=1, stride=2, bias=False)
        self.conv64resid = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False)
        self.avg = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(4096, 128)
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
    

pretrained_dict = torch.load('/home/users/u12559743/DAVI/IC/Complexity/results/max_complexity.pth')
model = ResNet()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and model_dict[k].size() == v.size()}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.to(device)



# pesos_fixos = torch.ones((512, 28*28)) * 0.01  
# bias_fixos = torch.zeros(512)  

# with torch.no_grad():  
#     model.linear_relu_stack[0].weight = nn.Parameter(pesos_fixos)
#     model.linear_relu_stack[0].bias = nn.Parameter(bias_fixos)

#print(model)


# %%
# Initialize the loss function
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

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

def shannon_entropy(data, num_bins=40):
    """ Calculate the Shannon entropy. """
    normalized_data = normalize_data(data)
    probabilities = calculate_probabilities(normalized_data, num_bins)
    return -np.sum(probabilities * np.log2(probabilities))

def disequilibrium(data, num_bins=40):
    """ Calculate the disequilibrium. """
    normalized_data = normalize_data(data)
    probabilities = calculate_probabilities(normalized_data, num_bins)
    equi_prob = 1.0 / num_bins
    return np.sqrt(np.sum((probabilities - equi_prob)**2))

def lmc_complexity(data, num_bins=40):
    """ Calculate the LMC complexity of the data. """
    normalized_data = normalize_data(data)
    probabilities = calculate_probabilities(normalized_data, num_bins)
    H = shannon_entropy(probabilities)
    D = disequilibrium(probabilities, num_bins)
    C = H * D
    return H, D, C

# %%
from sklearn.metrics import accuracy_score

epochs = 4
max_valid_accuracy = 0.0
epoch_max_complexity = 0 
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
        torch.save(model.state_dict(), 'min_loss_model3.pth')
    if complexidade  > max_valid_complex:
        max_valid_complex = complexidade 
        torch.save(model.state_dict(), 'max_complexity_model3.pth')
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
plt.figure("train", (12, 6))
plt.subplot(1,2,1)

plt.title("Loss function")
x = [i + 1 for i in range(len(loss_values_train))]
y = loss_values_train
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1,2,2)
plt.title("Iteration Accuracy")
x = [i + 1 for i in range(len(acc_values_train))]
y = acc_values_train
plt.xlabel("Iteration")
plt.plot(x, y)
plt.savefig("accuracy_numberMNIST.png")
plt.close()
#plt.show()


# %%
plt.figure("train", (18, 6))
plt.subplot(1,3,1)
plt.title("Complexity")
x = [i + 1 for i in range(len(single_complexity))]
y = single_complexity
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1,3,2)
plt.title("Disequilibrium")
x = [1 * (i + 1) for i in range(len(single_disequilibrium))]
y = single_disequilibrium
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1,3,3)
plt.title("Entropy")
x = [1 * (i + 1) for i in range(len(single_entropy))]
y = single_entropy
plt.xlabel("Iteration")
plt.plot(x, y)
plt.savefig("complexity_numberMNIST.png")
#plt.savefig('Classification Complexity')
#plt.show()


# %%
import json

metrics = {
    'Max_accuracy' : max_valid_accuracy,
    'Min_loss' : min_valid_loss,
    'Max_complexity': max_valid_complex,
    'Epoch_max_complexity' : epoch_max_complexity
}

output_file ='results/metrics/metrics_model3.json'
with open(output_file, 'w') as file:
    json.dump(metrics, file, indent=4)
# %%
torch.cuda.empty_cache()