import torch.nn as nn
import torch
import rff
from typing import Tuple


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


class PINNforwards(nn.Module):
    """Class for feed-forward neural network"""

    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int, activation_function=nn.Tanh,
                 use_fourier_transform: bool = True, sigma_FF: float = 5.0, encoded_size: int = 128, custom_arc: list[int] = None) -> None:
        """Constructor

        Args:
            N_INPUT (int):   Dimension of input.
            N_OUTPUT (int):  Dimension of output.
            N_HIDDEN (int):  Number of nodes in hidden layers.
            N_LAYERS (int):  Number of hidden layers (network gets one more than specified, due to input layer).   
            activation_function (nn.function, optional): Activation function for all layers. Defaults to nn.Tanh.
            use_fourier_transform (bool, optional):      Bool indicating if network should Fourier transform the input. Defaults to True.
            sigma_FF (float, optional):                  Variance of Fourier embedding. Defaults to 1.0.
            encoded_size (int, optional):                Fourier emebedding size. Defaults to 128.
            custom_arc (list[int], optional):            List containing number of nodes in each layer. Defaults to None, if not None it owerwrites model arcitechture.
        """

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

        # Constructing layers
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

            self.output_layer = nn.Linear(custom_arc[-1], N_OUTPUT)

        self.hidden_layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Glorot initializes weights"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Performs a forward pass of the network

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output of forward pass
        """
        if self.use_fourier_transform:
            x = self.encoder(x)

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def estimate_greeks_call(self,
                             X_scaled: torch.tensor,
                             X: torch.tensor,
                             sigma: float,
                             T: float) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """Estimates the Greeks for a European call option

        Args:
            X_scaled (torch.tensor):    Scaled input, consisting of asset prices and time
            X (torch.tensor):           Non-scaled input
            sigma (float):              Volatility in model
            T (float):                  Final time in market

        Returns:
            torch.tensor: Approximated Greeks in the input points.
        """

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
