import os
import scipy.io
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class IndianPinesDataset(Dataset):
    def __init__(self):
        mat = scipy.io.loadmat('Indian_pines_corrected.mat')
        raw = mat['indian_pines_corrected']
        self.data = raw.reshape(-1, raw.shape[2])

        mat = scipy.io.loadmat('Indian_pines_gt.mat')
        raw = mat['indian_pines_gt']
        self.truth = raw.flatten()

    def __len__(self):
        return len(self.truth)

    def __getitem__(self, idx):
        return self.data[idx], self.truth[idx]