import numpy as np
import torch
from scipy.stats import qmc
from tqdm import tqdm
from torch.distributions import Normal
from scipy.stats import norm

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
        X = qmc.scale(X, [self.time_range[0], self.S_range[0]], [
                      self.time_range[1], self.S_range[1]])
        y = np.zeros((n, 1))  # price
        return X, y

    def get_pde_data_tensor(self, N_sample, mul=4):
        X1, y1 = self.get_pde_data(mul*N_sample)
        X1 = torch.from_numpy(X1).float().requires_grad_()
        y1 = torch.from_numpy(y1).float()
        return X1.to(self.DEVICE), y1.to(self.DEVICE)

    def get_expiry_time_data(self, n, w=1):
        sample = self.sampler_1D.random(n=int(w*n))
        sample = qmc.scale(sample, [self.S_range[0]], [self.S_range[1]])

        X = np.concatenate([np.ones((int(w*n), 1))*self.time_range[1],  # all at expiry time
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
        lower_sample = qmc.scale(
            lower_sample, [self.time_range[0]], [self.time_range[1]])

        lower_X = np.concatenate([lower_sample,
                                  self.S_range[0] * np.ones((int(w1*n), 1))], axis=1)
        lower_y = np.zeros((int(w1*n), 1))

        upper_sample = self.sampler_1D.random(n=int(w2*n))
        upper_sample = qmc.scale(
            upper_sample, [self.time_range[0]], [self.time_range[1]])

        upper_X = np.concatenate([upper_sample,
                                  self.S_range[-1] * np.ones((int(w2*n), 1))], axis=1)
        upper_y = self.S_range[-1] - self.K * \
            np.exp(-self.r * (T-upper_X[:, 0].reshape(-1))).reshape(-1, 1)

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
        tmp_S = S.cpu().detach().numpy().flatten()
        tmp_t = t.cpu().detach().numpy()

        T = self.time_range[-1]
        t2m = T-tmp_t  # Time to maturity
        t2m = t2m.flatten()

        d1 = (np.log(tmp_S / self.K) + (self.r + 0.5 * self.sigma**2)
              * t2m) / (self.sigma * np.sqrt(t2m))

        d2 = d1 - self.sigma * np.sqrt(t2m)

        # Normal cumulative distribution function (CDF)
        normal = norm(loc=0, scale=1)

        Nd1 = normal.cdf(d1)
        Nd2 = normal.cdf(d2)

        # Calculate the option price
        F = tmp_S * Nd1 - self.K * Nd2 * np.exp(-self.r * t2m)
        return F

    def normalize(self, X):
        min_values = torch.tensor(
            [self.time_range[0], self.S_range[0]]).to(self.DEVICE)
        max_values = torch.tensor(
            [self.time_range[1], self.S_range[1]]).to(self.DEVICE)
        return (X - min_values) / (max_values - min_values)


class DataGeneratorEuropeanMultiDimensional(DataGeneratorEuropean1D):
    def __init__(self, time_range: list, S_range: np.array, K: float, r: float, sigma: np.array, DEVICE: torch.device, seed=2024):
        self.time_range = time_range
        self.S_range = S_range
        self.N = len(S_range)
        self.r = r
        self.sigma = sigma
        self.sigma_torch = torch.tensor(sigma).to(DEVICE)
        # self.cov = torch.tensor(sigma@sigma.T).to(DEVICE)
        self.K = K
        self.DEVICE = DEVICE
        self.S_range_mean = np.exp(np.mean(np.log(S_range[:, 1])))

        self.scaler_min = [self.time_range[0]] + [S[0] for S in self.S_range]
        self.scaler_max = [self.time_range[1]] + [S[1] for S in self.S_range]

        self.min_values = torch.tensor(
            [self.time_range[0]] + [S[0] for S in self.S_range]).to(self.DEVICE)
        self.max_values = torch.tensor(
            [self.time_range[1]] + [S[1] for S in self.S_range]).to(self.DEVICE)

        """ self.sampler_multi = qmc.Halton(d=self.N + 1, seed=seed)
        self.sampler_no_time = qmc.Halton(d=self.N, seed=seed)
        self.sampler_1D = qmc.Halton(d=1, seed=seed) """
        self.sampler_multi = qmc.LatinHypercube(d=self.N + 1, seed=seed)
        self.sampler_no_time = qmc.LatinHypercube(d=self.N, seed=seed)
        self.sampler_1D = qmc.LatinHypercube(d=1, seed=seed)

    def option_function(self, X):
        # Geometric mean basket option
        log_X = np.log(X)
        geometric_mean = np.exp(np.mean(log_X, axis=1))
        # zero_tensor = torch.full(geometric_mean.size(), 0).to(self.DEVICE)
        return np.fmax(geometric_mean - self.K, 0)

    def get_pde_data(self, n):
        X = self.sampler_multi.random(n=n)
        X = qmc.scale(X, self.scaler_min, self.scaler_max)
        y = np.zeros((n, 1))
        return X, y

    def get_expiry_time_data(self, n, w=1):
        sample = self.sampler_no_time.random(n=int(w*n))
        sample = qmc.scale(sample, self.scaler_min[1:], self.scaler_max[1:])

        X = np.concatenate([np.ones((int(w*n), 1))*self.time_range[1],  # all at expiry time
                            sample], axis=1)
        y = self.option_function(X[:, 1:]).reshape(-1, 1)
        return X, y

    def get_boundary_data(self, n, w1=1, w2=1):
        T = self.time_range[-1]
        """ lower_X = self.sampler_multi.random(n=int(n*w1))
        lower_X = qmc.scale(lower_X, self.scaler_min, self.scaler_max)

        idx = np.random.randint(1, self.N+1, int(n*w1))
        for i, cur in enumerate(idx):
            lower_X[i, cur] = 0.0 """

        lower_sample = self.sampler_1D.random(n=int(n*w1))
        lower_X = qmc.scale(
            lower_sample, [self.time_range[0]], [self.time_range[1]])

        for i in range(self.N):
            lower_X = np.concatenate(
                [lower_X, self.S_range[i][0] * np.ones((int(n*w1), 1))], axis=1)

        lower_y = np.zeros((int(w1*n), 1))

        upper_sample = self.sampler_1D.random(n=int(w2*n))
        upper_X = qmc.scale(
            upper_sample, [self.time_range[0]], [self.time_range[1]])

        for i in range(len(self.S_range)):
            upper_X = np.concatenate(
                [upper_X, self.S_range[i][-1] * np.ones((int(w2*n), 1))], axis=1)

        upper_y = self.S_range_mean - self.K * \
            np.exp(-self.r * (T-upper_X[:, 0].reshape(-1))).reshape(-1, 1)

        return lower_X, lower_y, upper_X, upper_y

    def get_analytical_solution(self, S, t):
        tmp_S = S.cpu().detach().numpy()
        tmp_t = t.cpu().detach().numpy()

        G = np.exp(np.mean(np.log(tmp_S), axis=1)).flatten()

        T = self.time_range[-1]
        t2m = T-tmp_t  # Time to maturity
        t2m = t2m.flatten()

        sigma_eff_sq = 0

        for j in range(self.N):
            tmp = 0
            for i in range(self.N):
                tmp += self.sigma[i, j]
            sigma_eff_sq += tmp**2

        sigma_eff_sq /= self.N**2
        sigma_eff = np.sqrt(sigma_eff_sq)
        # print(sigma_eff_sq)
        # print(G, self.K, self.r, sigma_eff_sq, t2m)
        d1 = (np.log(G / self.K) + (self.r + 0.5 * sigma_eff_sq)
              * t2m) / (sigma_eff * np.sqrt(t2m))
        d2 = d1 - sigma_eff * np.sqrt(t2m)
        # print(d1.shape, d2.shape)
        # Normal cumulative distribution function (CDF)

        # Standard normal cumulative distribution function.
        normal = norm(loc=0, scale=1)
        Phi_d1 = normal.cdf(d1)
        Phi_d2 = normal.cdf(d2)

        # Compute the analytical option price.
        price = G * Phi_d1 - self.K * np.exp(-self.r * t2m) * Phi_d2
        return price

    def normalize(self, X: torch.tensor):
        res = (X - self.min_values) / (self.max_values - self.min_values)
        return res


class DataGeneratorAmerican1D(DataGeneratorEuropean1D):
    def option_function(self, X):
        # Put option
        return np.fmax(self.K - X, 0)

    def get_boundary_data(self, n, w1=1, w2=1):
        lower_sample = self.sampler_1D.random(n=int(n*w1))
        lower_sample = qmc.scale(
            lower_sample, [self.time_range[0]], [self.time_range[1]])
        lower_X = np.concatenate([lower_sample,
                                  self.S_range[0] * np.ones((int(w1*n), 1))], axis=1)

        lower_y = self.K * np.ones((int(n*w1), 1))  # * \
        # np.exp(- self.r * (T - lower_X[:, 0].reshape(-1))).reshape(-1, 1)

        upper_sample = self.sampler_1D.random(n=int(w2*n))
        upper_sample = qmc.scale(
            upper_sample, [self.time_range[0]], [self.time_range[1]])

        upper_X = np.concatenate([upper_sample,
                                  self.S_range[-1] * np.ones((int(w2*n), 1))], axis=1)
        upper_y = np.zeros((int(w2*n), 1))
        return lower_X, lower_y, upper_X, upper_y

    def _compute_analytical_solution(self, S, t, M=1024):
        T = self.time_range[-1]-t
        delta_t = T / M
        u = np.exp(self.sigma * np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp(self.r * delta_t) - d) / (u - d)
        # Initialize option values at maturity
        option_values = np.maximum(
            self.K - S * u**np.arange(M+1) * d**(M-np.arange(M+1)), np.zeros(M+1))
        # Backward induction
        for i in range(M-1, -1, -1):
            option_values = np.maximum((self.K - S * u**np.arange(i+1) * d**(i-np.arange(i+1))),
                                       np.exp(-self.r * delta_t) * (p * option_values[:i+1] + (1 - p) * option_values[1:i+2]))
        return option_values[0]

    def get_analytical_solution(self, S, t, M=1024):
        res = np.array([self._compute_analytical_solution(
            S[i], t[i], M=M) for i in tqdm(range(len(S)), miniters=1_000, maxinterval=1_000)])
        return res
