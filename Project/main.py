# CS637 Final Project Spring 2024
# Ben Davis and Brad Hester

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import sys
import os
from torch import nn
from torch.utils.data import random_split
from indian_pines_dataset import IndianPinesDataset
from indian_pines_network import *
from torch.utils.data import DataLoader

if len(sys.argv) == 3:
    model_arg = int(sys.argv[1])
    epoch_arg = int(sys.argv[2])
elif len(sys.argv) == 2:
    model_arg = int(sys.argv[1])
    epoch_arg = 1000
elif len(sys.argv) == 1:
    model_arg = 3
    epoch_arg = 1000
else:
    print("main.py [model [epochs]]")
    print("model  = 0-3 (3 by default)")
    print("epochs = positive integer (1000 by default)")
    sys.exit(1)

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
epochs=epoch_arg

# With more epochs, reducing the learning rate seems
# to help
learning_rate=1e-4

# This loads different networks
if model_arg == 0:
    model = IndianPinesReLUNetwork().to(device)
    model_name = 'IP_ReLU'
elif model_arg == 1:
    model = IndianPinesLeakySmallNetwork(0.1).to(device)
    model_name = 'IP_SmallLeaky'
elif model_arg == 2:
    model = IndianPinesLeakyLargeNetwork(0.1).to(device)
    model_name = 'IP_LargeLeaky'
elif model_arg == 3:
    model = IndianPinesLeakyFinalNetwork(0.1).to(device)
    model_name = 'IP_FinalLeaky'
else:
    print("No model selected. Should be between 0-3")
    sys.exit(1)

#**********************************************************

# Create data loaders to handle minibatches
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define lists to store training and validation losses
train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss, correct = 0, 0
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
        correct += (logits.argmax(1) == y).type(torch.float).sum().item()

    train_loss = running_loss / len(dataloader)
    train_losses.append(train_loss)

    correct /= size
    accuracy = 100*correct
    train_accuracies.append(accuracy)

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
    test_losses.append(test_loss)

    correct /= size
    accuracy = 100*correct
    test_accuracies.append(accuracy)
    model.accuracy=(round(accuracy, 2))
    print(f"Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")

    ## Find best accuracy model
    if(os.path.exists("./BestModel.pth") & os.path.exists("./BestModelAccuracy.pth")):
        bestAccuracy=torch.load("./BestModelAccuracy.pth")
        if(model.accuracy>bestAccuracy):
            torch.save(model.state_dict(), "./BestModel.pth")
            torch.save(model.accuracy, "./BestModelAccuracy.pth") # save accuracy
    else:
        torch.save(model.state_dict(), "./BestModel.pth") # save inital model
        torch.save(model.accuracy, "./BestModelAccuracy.pth") # save accuracy


print(f'Starting Training of {model_name} over {epochs} Epochs\n')

for t in range(epochs):
    print(f"Epoch {t+1} - ", end='')
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print(f'\nFinished Training of {model_name} over {epochs} Epochs')

bestAccuracy=torch.load("./BestModelAccuracy.pth")
print(f"Best model accuracy: {bestAccuracy}")

# Create the images folder if it does not exist
if not os.path.exists('./images'):
    os.mkdir('./images')

# Plotting the loss curves
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), test_losses, label='Validation Loss')
plt.xlabel('Epoch')
# Format the x-axis ticks to display integers without decimal places
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.ylabel('Loss')
plt.title(f'{model_name} over {epochs} Epochs - Loss Curves')
plt.legend()
plt.savefig(f"./images/{model_name}_{epochs}E_loss_curves.png")
plt.close()

# Plotting the accuracy curves
plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs+1), test_accuracies, label='Validation Accuracy')

# Find the highest validation accuracy
max_val_acc = max(test_accuracies)
max_val_epoch = test_accuracies.index(max_val_acc) + 1  # adding 1 to match the epoch number

plt.xlabel('Epoch')
# Format the x-axis ticks to display integers without decimal places
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.ylabel('Accuracy')
plt.title(f'{model_name} over {epochs} Epochs - Accuracy Curves')
plt.legend([f'Training Accuracy', f'Validation Accuracy (Max: {max_val_acc:.2f}%)'])
plt.savefig(f"./images/{model_name}_{epochs}E_accuracy_curves.png")
plt.close()