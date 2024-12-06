import numpy as np
import torch
from scipy.stats import qmc
from tqdm import tqdm
from torch.distributions import Normal

# TODO write my own American 1D analytical solution


class DataGeneratorEuropean1D:
    def __init__(self, time_range, S_range, K: float, r: float, sigma: float, DEVICE, seed: int = 2024):
        self.time_range = time_range
        self.S_range = S_range
        self.r = r
        self.sigma = sigma
        self.K = K
        self.DEVICE = DEVICE
        self.sampler_2D = qmc.LatinHypercube(d=2, seed=seed)
        self.sampler_1D = qmc.LatinHypercube(d=1, seed=seed)

    def option_function(self, X):
        # Call option
        return np.fmax(X - self.K, 0)

    def get_pde_data(self, n):
        X = self.sampler_2D.random(n=n)
        # X = qmc.scale(sample, [self.time_range[0], self.S_range[0]], [
        #              self.time_range[1], self.S_range[1]])
        # X = np.concatenate([np.random.uniform(*self.time_range, (n, 1)),
        #                    np.random.uniform(*self.S_range, (n, 1)) + 1e-7], axis=1)
        y = np.zeros((n, 1))  # price
        return X, y

    def get_pde_data_tensor(self, N_sample, mul=4):
        X1, y1 = self.get_pde_data(mul*N_sample)
        X1 = torch.from_numpy(X1).float().requires_grad_()
        y1 = torch.from_numpy(y1).float()
        return X1.to(self.DEVICE), y1.to(self.DEVICE)

    def get_expiry_time_data(self, n, w=1):
        sample = self.sampler_1D.random(n=int(w*n))
        # sample = qmc.scale(sample, [self.S_range[0]], [self.S_range[1]])
        # X = np.concatenate([np.ones((int(r*n), self.time_range[1])),  # all at expiry time
        #                    np.random.uniform(*self.S_range, (int(r*n), 1)) + 1e-7], axis=1)
        X = np.concatenate([np.ones((int(w*n), self.time_range[1])),  # all at expiry time
                            sample], axis=1)
        y = self.option_function(X[:, 1]).reshape(-1, 1)

        return X, y

    def get_expiry_time_tensor(self, N_sample, w=1):
        expiry_x, expiry_y = self.get_expiry_time_data(N_sample, w)
        expiry_x_tensor = torch.from_numpy(expiry_x).float()
        expiry_y_tensor = torch.from_numpy(expiry_y).float()
        return expiry_x_tensor.to(self.DEVICE), expiry_y_tensor.to(self.DEVICE)

    def get_boundary_data(self, n, w1=1, w2=1):
        T = self.time_range[-1]
        lower_sample = self.sampler_1D.random(n=int(n*w1))
        # lower_sample = qmc.scale(
        #    lower_sample, [self.time_range[0]], [self.time_range[1]])
        # lower_X = np.concatenate([np.random.uniform(*self.time_range, (int(n*r1), 1)),
        #                          self.S_range[0] * np.ones((int(n*r1), 1))], axis=1)
        lower_X = np.concatenate([lower_sample,
                                  self.S_range[0] * np.ones((int(w1*n), 1))], axis=1)
        lower_y = np.zeros((int(w1*n), 1))

        upper_sample = self.sampler_1D.random(n=int(w2*n))
        # upper_sample = qmc.scale(
        #    upper_sample, [self.time_range[0]], [self.time_range[1]])

        # upper_X = np.concatenate([np.random.uniform(*self.time_range, (int(r2*n), 1)),
        #                          self.S_range[-1] * np.ones((int(r2*n), 1))], axis=1)
        upper_X = np.concatenate([upper_sample,
                                  self.S_range[-1] * np.ones((int(w2*n), 1))], axis=1)
        upper_y = (self.S_range[-1] - self.K*np.exp(-self.r *
                                                    (T-upper_X[:, 0].reshape(-1)))).reshape(-1, 1)

        return lower_X, lower_y, upper_X, upper_y

    def get_boundary_data_tensor(self, N_sample, w1=1, w2=1):
        lower_x, lower_y, upper_x, upper_y = self.get_boundary_data(
            N_sample, w1, w2)
        lower_x_tensor = torch.from_numpy(lower_x).float()
        lower_y_tensor = torch.from_numpy(lower_y).float()
        upper_x_tensor = torch.from_numpy(upper_x).float()
        upper_y_tensor = torch.from_numpy(upper_y).float()
        return lower_x_tensor.to(self.DEVICE), lower_y_tensor.to(self.DEVICE), upper_x_tensor.to(self.DEVICE), upper_y_tensor.to(self.DEVICE)

    def get_analytical_solution(self, S, t):
        T = self.time_range[-1]
        t2m = T-t  # Time to maturity
        d1 = (torch.log(S / self.K) + (self.r + 0.5 * self.sigma**2)
              * t2m) / (self.sigma * torch.sqrt(t2m))

        d2 = d1 - self.sigma * torch.sqrt(t2m)

        # Normal cumulative distribution function (CDF)
        standard_normal = Normal(0, 1)

        Nd1 = standard_normal.cdf(d1)
        Nd2 = standard_normal.cdf(d2)

        # Calculate the option price
        F = S * Nd1 - self.K * Nd2 * torch.exp(-self.r * t2m)
        return F

    def normalize(self, X):
        min_values = torch.tensor(
            [self.time_range[0], self.S_range[0]]).to(self.DEVICE)
        max_values = torch.tensor(
            [self.time_range[1], self.S_range[1]]).to(self.DEVICE)
        return (X - min_values) / (max_values - min_values)


class DataGeneratorEuropeanMultiDimensional(DataGeneratorEuropean1D):
    def __init__(self, time_range: list, S_range: np.array, K: float, r: float, sigma: float, DEVICE: torch.device):
        self.time_range = time_range
        self.S_range = S_range
        self.r = r
        self.sigma = sigma
        self.K = K
        self.DEVICE = DEVICE

        self.S_range_mean = np.exp(np.mean(np.log(S_range[:, 1])))

    def option_function(self, X):
        # Geometric mean basket option
        log_X = np.log(X)
        geometric_mean = np.exp(np.mean(log_X, axis=1))
        # zero_tensor = torch.full(geometric_mean.size(), 0).to(self.DEVICE)
        return np.fmax(geometric_mean - self.K, 0)

    def get_pde_data(self, n):
        X = np.random.uniform(*self.time_range, (n, 1))

        for i in range(len(self.S_range)):
            X = np.concatenate(
                [X, np.random.uniform(*self.S_range[i], (n, 1)) + 1e-7], axis=1)
        y = np.zeros((n, 1))  # price
        return X, y

    def get_expiry_time_data(self, n, r):
        X = np.ones((int(r*n), 1))

        for i in range(len(self.S_range)):
            X = np.concatenate(
                [X, np.random.uniform(*self.S_range[i], (int(r*n), 1)) + 1e-7], axis=1)

        # X = torch.from_numpy(X).float().to(self.DEVICE)
        y = self.option_function(X[:, 1:]).reshape(-1, 1)
        return X, y

    def get_boundary_data(self, n, r1=1, r2=1):
        T = self.time_range[-1]
        lower_X = np.random.uniform(*self.time_range, (int(n*r1), 1))

        for i in range(len(self.S_range)):
            lower_X = np.concatenate(
                [lower_X, self.S_range[i][0] * np.ones((int(n*r1), 1))], axis=1)

        lower_y = np.zeros((int(n*r1), 1))

        upper_X = np.random.uniform(*self.time_range, (int(r2*n), 1))

        for i in range(len(self.S_range)):
            upper_X = np.concatenate(
                [upper_X, self.S_range[i][-1] * np.ones((int(r2*n), 1))], axis=1)

        upper_y = (self.S_range_mean - self.K*np.exp(-self.r *
                                                     (T-upper_X[:, 0].reshape(-1)))).reshape(-1, 1)
        return lower_X, lower_y, upper_X, upper_y

    def get_analytical_solution(self, S, t):
        sigma_bar = np.sqrt(np.sum(np.sum(self.sigma))) / len(self.sigma)
        sigma_diag = np.sum(np.diag(self.sigma)**2)/(2 * len(self.sigma))
        log_S = torch.log(S)
        geometric_mean = torch.exp(torch.mean(log_S, dim=0))
        T = self.time_range[-1]

        F_t = geometric_mean * \
            torch.exp((self.r - sigma_diag)*(T - t) +
                      sigma_bar**2 * (T - t) / 2)

        t2m = T-t  # Time to maturity
        d1 = (torch.log(F_t / self.K) + (self.r + 0.5 * sigma_bar**2)
              * t2m) / (sigma_bar * torch.sqrt(t2m))

        d2 = d1 - sigma_bar * torch.sqrt(t2m)

        # Normal cumulative distribution function (CDF)
        def N0(value): return 0.5 * (1 + torch.erf(value / (2**0.5)))
        Nd1 = N0(d1)
        Nd2 = N0(d2)

        # Calculate the option price
        F = F_t * Nd1 - self.K * Nd2 * torch.exp(-self.r * t2m)
        return F

    def normalize(self, X):
        min_values = torch.tensor(
            [self.time_range[0]] + [t[0] for t in self.S_range]).to(self.DEVICE)
        max_values = torch.tensor(
            [self.time_range[1]] + [t[1] for t in self.S_range]).to(self.DEVICE)
        return (X - min_values) / (max_values - min_values)


class DataGeneratorAmerican1D(DataGeneratorEuropean1D):
    def option_function(self, X):
        # Put option
        return np.fmax(self.K - X, 0)

    def get_boundary_data(self, n, w1=1, w2=1):
        T = self.time_range[-1]

        lower_sample = self.sampler_1D.random(n=int(n*w1))
        # lower_sample = qmc.scale(
        #    lower_sample, [self.time_range[0]], [self.time_range[1]])
        lower_X = np.concatenate([lower_sample,
                                  self.S_range[0] * np.ones((int(w1*n), 1))], axis=1)

        lower_y = self.K * np.ones((int(n*w1), 1))

        upper_sample = self.sampler_1D.random(n=int(w2*n))
        # upper_sample = qmc.scale(
        #    upper_sample, [self.time_range[0]], [self.time_range[1]])

        upper_X = np.concatenate([upper_sample,
                                  self.S_range[-1] * np.ones((int(w2*n), 1))], axis=1)
        upper_y = np.zeros((int(w2*n), 1))
        return lower_X, lower_y, upper_X, upper_y

    def _compute_analytical_solution(self, S, t, n=250):
        T = self.time_range[-1]-t
        delta_t = T / n
        u = np.exp(self.sigma * np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp(self.r * delta_t) - d) / (u - d)
        # Initialize option values at maturity
        option_values = np.maximum(
            self.K - S * u**np.arange(n+1) * d**(n-np.arange(n+1)), np.zeros(n+1))
        # Backward induction
        for i in range(n-1, -1, -1):
            option_values = np.maximum((self.K - S * u**np.arange(i+1) * d**(i-np.arange(i+1))),
                                       np.exp(-self.r * delta_t) * (p * option_values[:i+1] + (1 - p) * option_values[1:i+2]))
        return option_values[0]

    def get_analytical_solution(self, S, t, n=250):
        res = np.array([self._compute_analytical_solution(
            S[i], t[i], n=n) for i in tqdm(range(len(S)))])
        return res
