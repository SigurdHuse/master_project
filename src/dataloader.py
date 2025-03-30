from torch.utils.data import Dataset
import numpy as np
import torch


class DataLoaderEuropean(Dataset):
    def __init__(self, filename: str, training_noise: float = 0.0, seed=2024):
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
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx, :2]
        y = self.X[idx, -1] + self.noise[idx]
        return x, y


if __name__ == "__main__":
    dataloader = DataLoaderEuropean("data/european_one_dimensional.npy")
    print(dataloader[1])
