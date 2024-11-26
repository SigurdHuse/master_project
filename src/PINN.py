import torch.nn as nn
import torch
from scipy.stats import norm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


class PINNforwards(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_function=nn.Tanh):
        super(PINNforwards, self).__init__()
        self.activation = activation_function

        self.input_layer = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            self.activation())

        layers = []

        for _ in range(N_LAYERS):
            layers.append(nn.Linear(N_HIDDEN, N_HIDDEN))
            layers.append(self.activation())

        self.hidden_layers = nn.Sequential(*layers)

        self.output_layer = nn.Linear(N_HIDDEN, N_OUTPUT)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def estimate_greeks_call(self, X_scaled: torch.tensor, X: torch.tensor, sigma: float, T: float):
        F = self.forward(X_scaled)

        grads = torch.autograd.grad(F, X, grad_outputs=torch.ones(F.shape).to(
            DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
        dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

        grads2nd = torch.autograd.grad(dVdS, X, grad_outputs=torch.ones(
            dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
        d2VdS2 = grads2nd[:, 1].view(-1, 1)

        S = X[:, 1].view(-1, 1)
        t = X[:, 0].view(-1, 1)

        delta = dVdS
        gamma = d2VdS2
        theta = dVdt
        nu = sigma * (T - t) * S**2 * gamma
        rho = -(T - t) * (F - S * delta)

        return delta, gamma, theta, nu, rho


class PINNbackwards(PINNforwards):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_function=nn.Tanh, initial_sigma=0.1):
        super(PINNbackwards, self).__init__(N_INPUT, N_OUTPUT, N_HIDDEN,
                                            N_LAYERS, activation_function)
        self.sigma = nn.Parameter(torch.tensor(
            initial_sigma, requires_grad=True))

    def foward_with_european_1D(self, X_scaled: torch.tensor, X: torch.tensor, r: float):
        y1_hat = super().forward(X_scaled)

        grads = torch.autograd.grad(y1_hat, X, grad_outputs=torch.ones(y1_hat.shape).to(
            DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
        dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

        grads2nd = torch.autograd.grad(dVdS, X, grad_outputs=torch.ones(
            dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
        d2VdS2 = grads2nd[:, 1].view(-1, 1)

        S1 = X[:, 1].view(-1, 1)

        dt_term = dVdt
        sigma_term = 0.5*(self.sigma * self.sigma) * (S1 * S1) * d2VdS2 / 100
        ds_term = r * S1 * dVdS
        y1_hat_term = r * y1_hat

        bs_pde = dt_term + ds_term + sigma_term - y1_hat_term

        # print(nn.MSELoss()(sigma_term, torch.zeros_like(sigma_term)))
        # print(ds_term + dt_term + y1_hat_term)
        # print("Sigma term", d2VdS2)
        return y1_hat, bs_pde
