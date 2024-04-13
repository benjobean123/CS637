from torchvision import datasets
from torchvision.transforms import ToTensor

def get_training_data():
    # Download training data from open datasets.
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

def get_testing_data():
    # Download test data from open datasets.
    return datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )