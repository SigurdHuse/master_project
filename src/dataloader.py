from torch.utils.data import Dataset
import numpy as np
import torch
from typing import Tuple


class DataLoaderEuropean(Dataset):
    """Class to load dataset for the inverse problem"""

    def __init__(self, filename: str, training_noise: float = 0.0, seed=2024):
        """Constructor

        Args:
            filename (str): Filename to load data from
            training_noise (float, optional): Variance of normally distributed noise with mean 0 to add to loaded data. Defaults to 0.0.
            seed (int, optional): Seed of RNG used to generate noise. Defaults to 2024.
        """
        super(DataLoaderEuropean, self).__init__()
        self.filename = filename

        self.X = torch.from_numpy(np.load(filename))

        if training_noise > 0.0:
            rng = np.random.default_rng(seed=seed)
            self.noise = rng.normal(
                loc=0.0, scale=training_noise, size=self.X.shape[0])
        else:
            self.noise = np.zeros(self.X.shape[0])

    def __len__(self):
        """Get length of data set"""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.array, float]:
        """Retrieve data

        Args:
            idx (int): Index of data to retrieve

        Returns:
            Tuple[np.array, float]: Returns input point and target for given index.
        """
        x = self.X[idx, :2]
        y = self.X[idx, -1] + self.noise[idx]
        return x, y


if __name__ == "__main__":
    dataloader = DataLoaderEuropean("data/european_one_dimensional.npy")
    print(dataloader[1])
