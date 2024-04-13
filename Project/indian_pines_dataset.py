import os
import scipy.io
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class IndianPinesDataset(Dataset):
    def __init__(self):
        mat = scipy.io.loadmat('Indian_pines_gt.mat')
        raw = mat['indian_pines_gt']
        flat_truth = raw.flatten().astype(np.int64)

        mat = scipy.io.loadmat('Indian_pines_corrected.mat')
        raw = mat['indian_pines_corrected']
        flat_data = raw.reshape(-1, raw.shape[2]).astype(np.float32)

        self.truth = []
        self.data = []
        for i, val in enumerate(flat_truth):
            if val == 0:
                continue
            self.truth.append(val-1)
            self.data.append(flat_data[i])

    def __len__(self):
        return len(self.truth)

    def __getitem__(self, idx):
        return self.data[idx], self.truth[idx]