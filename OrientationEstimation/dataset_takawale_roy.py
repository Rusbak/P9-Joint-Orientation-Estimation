import h5py
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from helper_scripts import features_takawale_roy

class TakawaleRoyDataset(Dataset):
    '''
    Since the binaural dataset is to large to have in RAM we instead grab a signal, compute its features, and return that along with the target label.
    '''
    def __init__(self, file_path, split, fs):
        self.file_path = file_path
        self.split = split
        self.fs = fs

        with h5py.File(file_path, "r") as f:
            self.num_samples = f[f"{split}/signals"].shape[0]

        self.file = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Open HDF5 file lazily
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        # Load one binaural signal: [2, T]
        signal = self.file[f"{self.split}/signals"][idx]

        # Load labels: [4]
        labels = self.file[f"{self.split}/labels"][:, idx]

        # Extract five Takawale and Roy features: [5, F]
        features = features_takawale_roy.extract_takawale_roy_features(
            signal, self.fs
        )

        # Select relative orientations
        targets = np.array([
            labels[1],
            labels[3]
        ], dtype=np.float32)

        # Convert to sine/cosine representation
        targets = np.array([
            np.sin(targets[0]),
            np.cos(targets[0]),
            np.sin(targets[1]),
            np.cos(targets[1])
        ], dtype=np.float32)

        return (
            torch.from_numpy(features),
            torch.from_numpy(targets)
        )