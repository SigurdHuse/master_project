import numpy as np
import torch


class DataloaderEuropean1D:
    def __init__(self, time_range, S_range, K, r, sigma):
        self.time_range = time_range
        self.S_range = S_range
        self.r = r
        self.sigma = sigma
        self.K = K

    def option_function(self, X):
        return np.fmax(X - self.K, 0)

    def get_pde_data(self, n):
        X = np.concatenate([np.random.uniform(*self.time_range, (n, 1)),
                            np.random.uniform(*self.S_range, (n, 1))], axis=1)
        y = np.zeros((n, 1))  # price
        return X, y

    def get_pde_data_tensor(self, N_sample, mul=4):
        X1, y1 = self.get_pde_data(mul*N_sample)
        X1 = torch.from_numpy(X1).float().requires_grad_()
        y1 = torch.from_numpy(y1).float()
        return X1, y1

    def get_expiry_time_data(self, n, r):
        X = np.concatenate([np.ones((int(r*n), 1)),  # all at expiry time
                            np.random.uniform(*self.S_range, (int(r*n), 1))], axis=1)
        y = self.option_function(X[:, 1]).reshape(-1, 1)
        return X, y

    def get_expiry_time_tensor(self, N_sample, r=1):
        expiry_x, expiry_y = self.get_expiry_time_data(N_sample, r)
        expiry_x_tensor = torch.from_numpy(expiry_x).float()
        expiry_y_tensor = torch.from_numpy(expiry_y).float()
        return expiry_x_tensor, expiry_y_tensor

    def get_boundary_data(self, n, r1=1, r2=1):
        T = self.time_range[-1]
        lower_X = np.concatenate([np.random.uniform(*self.time_range, (int(n*r1), 1)),
                                  self.S_range[0] * np.ones((int(n*r1), 1))], axis=1)
        lower_y = np.zeros((int(n*r1), 1))

        upper_X = np.concatenate([np.random.uniform(*self.time_range, (int(r2*n), 1)),
                                  self.S_range[-1] * np.ones((int(r2*n), 1))], axis=1)
        upper_y = (self.S_range[-1] - self.K*np.exp(-self.r *
                                                    (T-upper_X[:, 0].reshape(-1)))).reshape(-1, 1)
        return lower_X, lower_y, upper_X, upper_y

    def get_boundary_data_tensor(self, N_sample, r1=1, r2=1):
        lower_x, lower_y, upper_x, upper_y = self.get_boundary_data(
            N_sample, r1, r2)
        lower_x_tensor = torch.from_numpy(lower_x).float()
        lower_y_tensor = torch.from_numpy(lower_y).float()
        upper_x_tensor = torch.from_numpy(upper_x).float()
        upper_y_tensor = torch.from_numpy(upper_y).float()
        return lower_x_tensor, lower_y_tensor, upper_x_tensor, upper_y_tensor
