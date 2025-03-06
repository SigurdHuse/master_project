import torch.nn as nn
import torch
from scipy.stats import norm
import rff

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


class PINNforwards(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_function=nn.Tanh,
                 use_fourier_transform: bool = True, sigma_FF: float = 1.0, encoded_size: int = 128, custom_arc: list[int] = None):
        super(PINNforwards, self).__init__()

        print("Training model with: ")
        if custom_arc == None:
            print(f"LAYERS : {N_LAYERS}")
            print(f"NODES : {N_HIDDEN}")

        else:
            print(f"ARCITECTHURE : {custom_arc}")

        print(f"USING FOURIER : {use_fourier_transform}")
        if use_fourier_transform:
            print(f"FOURIER SIGMA : {sigma_FF}")
            print(f"ENCODED SIZE  : {encoded_size} \n")
        else:
            print("")

        self.use_fourier_transform = use_fourier_transform
        self.activation = activation_function

        if use_fourier_transform:
            self.encoder = rff.layers.GaussianEncoding(
                sigma=sigma_FF, input_size=N_INPUT, encoded_size=encoded_size).to(DEVICE)
            N_INPUT = encoded_size * 2

        layers = []

        if custom_arc is None:
            self.input_layer = nn.Sequential(
                nn.Linear(N_INPUT, N_HIDDEN),
                self.activation())

            for _ in range(N_LAYERS):
                layers.append(nn.Linear(N_HIDDEN, N_HIDDEN))
                layers.append(self.activation())

            self.output_layer = nn.Linear(N_HIDDEN, N_OUTPUT)
        else:
            self.input_layer = nn.Sequential(
                nn.Linear(N_INPUT, custom_arc[0]),
                self.activation())

            for i in range(1, len(custom_arc)):
                layers.append(nn.Linear(custom_arc[i-1], custom_arc[i]))
                layers.append(self.activation())

            # print(layers)
            self.output_layer = nn.Linear(custom_arc[-1], N_OUTPUT)

        self.hidden_layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        if self.use_fourier_transform:
            x = self.encoder(x)

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


""" class PINNbackwards(PINNforwards):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_function=nn.Tanh, initial_sigma=0.1):
        super(PINNbackwards, self).__init__(N_INPUT, N_OUTPUT, N_HIDDEN,
                                            N_LAYERS, activation_function)
        self.sigma = nn.Parameter(torch.tensor(
            initial_sigma, requires_grad=True))

    def foward_with_european_1D(self, X_scaled: torch.tensor, X: torch.tensor, r: float):
        y1_hat = super().forward(X_scaled)
        prediction = super().forward(X_scaled)

        grads = torch.autograd.grad(y1_hat, X, grad_outputs=torch.ones(y1_hat.shape).to(
            DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
        dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

        grads2nd = torch.autograd.grad(dVdS, X, grad_outputs=torch.ones(
            dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
        d2VdS2 = grads2nd[:, 1].view(-1, 1)

        S1 = X[:, 1].view(-1, 1)

        dt_term = dVdt
        sigma_term = 0.5*(self.sigma * self.sigma) * (S1 * S1) * d2VdS2
        ds_term = r * S1 * dVdS
        y1_hat_term = r * y1_hat

        bs_pde = dt_term + ds_term + sigma_term - y1_hat_term

        # print(nn.MSELoss()(sigma_term, torch.zeros_like(sigma_term)))
        # print(ds_term + dt_term + y1_hat_term)
        # print("Sigma term", d2VdS2)
        return prediction, bs_pde
 """
