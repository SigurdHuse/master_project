from torch.utils.data import Dataset
import numpy as np
import torch


class DataLoaderEuropean(Dataset):

    def __init__(self, filename: str):
        super(DataLoaderEuropean, self).__init__()
        self.filename = filename

        self.X = torch.from_numpy(np.load(filename))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx, :-1]
        y = self.X[idx, -1]
        return x, y


if __name__ == "__main__":
    dataloader = DataLoaderEuropean("data/european_one_dimensional.npy")
    print(dataloader[1])
