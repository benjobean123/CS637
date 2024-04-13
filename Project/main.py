import torch
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
epochs=200

# With more epochs, reducing the learning rate seems
# to help
learning_rate=1e-5

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

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        logits = model(X)
        loss = loss_fn(logits, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if batch % size == 0:
            #loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

for t in range(epochs):
    print(f"Epoch {t+1} - ", end='')
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")