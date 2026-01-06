###############################################################################
# File Name: assignment_04a.py
#
# Description: This program finetunes a pretrained MONAI DenseNet-121 
# convolutional neural network in PyTorch on grayscale CT scan image data using 
# data augmentation, evaluates its performance on a validation set across 
# multiple epochs, and saves the trained model for future inference.
# 
# Note: Dataset not included due to GitHub size constraints.
#
# Record of Revisions (Date | Author | Change):
# 09/12/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer

# Transforms (Same as MONAI pretrained model)
trainTransforms = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.Grayscale(num_output_channels=1),  # Convert to Grayscale
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal Flip
    transforms.RandomVerticalFlip(p=0.5),    # Vertical Flip
    transforms.RandomRotation(degrees=90),  # Rotates Randomly between -90° and +90°
    transforms.ToTensor()  # Convert to Tensor
])

valTransforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1), # Convert to Grayscale
    transforms.ToTensor() # Convert to Tensor
])

# Loading Data
trainData = datasets.ImageFolder('data/train-20240112T210350Z-001/train', transform = trainTransforms)
valData = datasets.ImageFolder('data/test-20240112T210346Z-001/test', transform = valTransforms)

# Set up Dataloaders
trainLoader = DataLoader(trainData, batch_size = 32, shuffle = True)
testLoader = DataLoader(valData, batch_size = 32)

classNames = trainData.classes # Pull Class Names
outputShape = len(classNames) # Set Output Shape to Number of Classes

device = 'mps' if torch.mps.is_available() else 'cpu' # Use MPS gpu else cpu
torch.manual_seed(42) # Set Seed for Reproducibility

lossFunc = nn.CrossEntropyLoss() # Loss Function

# Build Model
model = models.densenet121(weights = None).to(device) # Initialize Model

model.features.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False).to(device) # Change Input Layer to Accept Grayscale

# Load Model Weights
state_dict = torch.load('data/pretrainedModel.pt', map_location=device)

# Apply Weights
model.load_state_dict(state_dict, strict=False)

# for param in model.features.parameters(): # Freeze Feature Layers
#     param.requires_grad = False

# model.classifier = nn.Linear(in_features=1024, out_features=outputShape).to(device) # Change Classifier to Match Output Shape

optimizer = optim.Adam(model.parameters(), lr=0.001) # Set Optimizer

# Train Model
start_time = timer() # Start the timer

epochs = 15 # Set number of epochs

# Create empty results dictionary
results = {"train_loss": [],
           "train_acc": [],
           "test_loss": [],
           "test_acc": []
}

model.train() # Put model in train mode

# Train for n epochs
for epoch in range(epochs):
    train_loss, train_acc = 0, 0 # Setup train loss and train accuracy values

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(trainLoader):
        X, y = X.to(device), y.to(device) # Send data to target device
    
        y_pred = model(X) # Forward Pass
    
        loss = lossFunc(y_pred, y) # Calculate and accumulate loss

        train_loss += loss.item() 
        
        optimizer.zero_grad() # Optimizer zero grad
    
        loss.backward() # Loss backward
    
        optimizer.step() # Optimizer step
    
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += ((y_pred_class == y).sum().item()/len(y_pred))

    # Test loop
    model.eval() # Put model in eval mode

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(testLoader):
            X, y = X.to(device), y.to(device) # Send data to target device

            test_pred_logits = model(X) # Forward Pass

            loss = lossFunc(test_pred_logits, y) # Calculate and accumulate loss
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(trainLoader)
    train_acc = train_acc / len(trainLoader)
    test_loss = test_loss / len(testLoader)
    test_acc = test_acc / len(testLoader)
    
    print(f"Epoch:{epoch + 1} Average Train Loss: {train_loss:.3f} Average Train acc: {train_acc:.2f} Average Test Loss: {test_loss:.3f} Average Test Acc: {test_acc:.2f}")

    # Update results dictionary - this will be used later to plot the loss curve
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

# End the timer and print out how long it took
end_time = timer()
print(f"\nTotal training time: {end_time-start_time:.3f} seconds")

torch.save(model.state_dict(), "output/covidCTModel.pth")
