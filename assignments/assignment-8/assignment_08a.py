###############################################################################
# File Name: assignment_08a.py
#
# Description: This script trains an autoencoder on only the “good” images to 
# learn normal patterns, then detects anomalies in test images by comparing 
# reconstruction MSE to a threshold. It plots the reconstruction error for 
# each image, showing actual and predicted labels, where defective images 
# have higher MSE.
#
# Record of Revisions (Date | Author | Change):
# 11/02/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import pandas as pd
import os

# Define AutoEncoder with linear layers
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=4096, encoding_dim=128):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

device = torch.device("mps" if torch.mps.is_available() else "cpu") # Set device

# Initialize model, loss, and optimizer
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set Path
trainPath = 'data/train'
testPath = 'data/test'

# Transformations
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(), 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Load Datasets and Dataloaders
trainDataset = datasets.ImageFolder(trainPath, transform = transform)
goodIndices = [i for i, (_, label) in enumerate(trainDataset) if label == 0]
trainDataset = Subset(trainDataset, goodIndices)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testDataset = datasets.ImageFolder(testPath, transform = transform)
testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)

# Train on normal images only
for epoch in range(100):
    for batch, _ in trainLoader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Initialize lists
mseScores = []
goodScores = []
labels = []
stringLabels = []
predictions = []
file = []
threshold = 0.017

# Detect anomalies
with torch.no_grad():
    for batch, label in testLoader:
        batch = batch.to(device)
        recon = model(batch)
        mse = torch.mean((batch - recon)**2, dim=1)
        mseScores.extend(mse.cpu().numpy())
        labels.extend(label.cpu().numpy())

# Convert labels and set threshold
for index, label in enumerate(labels):
    if label == 1:
        stringLabels.append('Good')
        goodScores.append(mseScores[index])
    else:
        stringLabels.append('Defective')

# Set threshold
avgGoodScore = sum(goodScores) / len(goodScores)
threshold = avgGoodScore + avgGoodScore * 0.05

# Make predictions
for score in mseScores:
    if score < threshold:
        predictions.append('Good')
    else:
        predictions.append('Defective')

# Save file names
for path, _ in testDataset.imgs:
    filename = os.path.basename(path)
    file.append(filename)

# Create dataframe
testDf = pd.DataFrame({'File Name': file, 'Actual State': stringLabels, 'Reconstruction MSE': mseScores, 'Predicted State': predictions})

# Map actual labels to marker style
marker_map = {'Good': 'o', 'Defective': 's'}

# Map predicted labels to color
color_map = {'Good': 'green', 'Defective': 'red'}

# Plot data
for index in range(len(mseScores)):
    plt.scatter(index, mseScores[index], marker=marker_map[stringLabels[index]], color=color_map[predictions[index]], alpha=0.6)

plt.xlabel("Image Index")
plt.ylabel("Reconstruction MSE")
plt.title("Actual and Predicted States")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Actual Good', markerfacecolor='gray', markersize=8),
    Line2D([0], [0], marker='s', color='w', label='Actual Defective', markerfacecolor='gray', markersize=8),
    Line2D([0], [0], marker='o', color='green', label='Predicted Good', markersize=8),
    Line2D([0], [0], marker='o', color='red', label='Predicted Defective', markersize=8)
]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()