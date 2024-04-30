import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch import nn
from torch.utils.data import random_split
from indian_pines_dataset import IndianPinesDataset
from indian_pines_network import *
from torch.utils.data import DataLoader

# Figure out what type of device we have
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Load the Indian Pines data
ds = IndianPinesDataset()

#print(ds)
#print(len(ds))
#print(ds[0])

# 80/20 split? Why not
training_size = int(0.8 * len(ds))
testing_size = len(ds) - training_size
training_data, testing_data = random_split(ds, [training_size, testing_size])

#**********************************************************
#
# These are the key parameters that seem I have been
# tweaking to get better performance. The settings
# committed to the repo get about 75% accuracy.

# Bigger batch sizes run faster, but need more epochs
# to converge. Lots of epochs probably means we need
# to add some dropout to prevent overfitting.
batch_size=32
epochs=1000

# With more epochs, reducing the learning rate seems
# to help
learning_rate=1e-4

# This loads different networks
model = IndianPinesLeakyNetwork(0.1).to(device)
#model = IndianPinesLeakyNetwork(0.01).to(device)
#model = IndianPinesReLUNetwork().to(device)

#**********************************************************

# Create data loaders to handle minibatches
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define lists to store training and validation losses
train_losses = []
val_losses = []

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        logits = model(X)
        loss = loss_fn(logits, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    train_loss = running_loss / len(dataloader)
    train_losses.append(train_loss)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    val_losses.append(test_loss)
    correct /= size
    model.accuracy=(round((100*correct), 2))
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

    ## Find best accuracy model
    if(os.path.exists("./BestModel.pth") & os.path.exists("./BestModelAccuracy.pth")):
        bestAccuracy=torch.load("./BestModelAccuracy.pth")
        if(model.accuracy>bestAccuracy):
            torch.save(model.state_dict(), "./BestModel.pth")
            torch.save(model.accuracy, "./BestModelAccuracy.pth") # save accuracy
    else:
        torch.save(model.state_dict(), "./BestModel.pth") # save inital model
        torch.save(model.accuracy, "./BestModelAccuracy.pth") # save accuracy


for t in range(epochs):
    print(f"Epoch {t+1} - ", end='')
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

bestAccuracy=torch.load("./BestModelAccuracy.pth")
print(f"Best model accuracy: {bestAccuracy}")

# Plotting the loss curves
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.savefig(f"loss_curves.png")
plt.close()

print("Done!")