import sys
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import Net


def round_tensor(x):
    return torch.round(x)

# Define transforms to apply to the dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # convert to grayscale
    transforms.Resize((128, 128)),  # resize to 128x128
    transforms.ToTensor()  # convert to tensor
])

transform.transforms.append(transforms.Lambda(round_tensor))

# Load the dataset
data_path = 'data'
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Define the train, validation, and test sets
train_size = int(0.1 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train size:", train_size)
print("Validate size:", val_size)
print("Test size:", test_size)

# Instantiate the CNN and define the loss function and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the CNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
epochs = 1

for epoch in range(epochs):
    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print('Epoch %d, validation accuracy: %f %%' % (epoch+1, accuracy))

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    print('Test accuracy: %f %%' % (accuracy))

# Save the model
PATH = "models/model_v1.pt"
torch.save(model.state_dict(), PATH)
