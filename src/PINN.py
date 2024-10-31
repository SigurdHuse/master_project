import torch.nn as nn
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


class PINNforwards(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_function=nn.Tanh()):
        super(PINNforwards, self).__init__()

        self.N_HIDDEN = N_HIDDEN
        self.activation = activation_function

        self.input_layer = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            self.activation)

        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                self.activation
            ) for _ in range(N_LAYERS)])

        self.output_layer = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class PINNbackwards(PINNforwards):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_function=nn.Tanh(), initial_sigma=0.1):
        super(PINNbackwards, self).__init__(N_INPUT, N_OUTPUT, N_HIDDEN,
                                            N_LAYERS, activation_function)
        self.sigma = nn.Parameter(torch.tensor(initial_sigma))

    def foward_with_european_1D(self, x: torch.tensor, r: float):
        y1_hat = PINNforwards.forward(self, x)

        grads = torch.autograd.grad(y1_hat, x, grad_outputs=torch.ones(y1_hat.shape).to(
            DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
        dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

        grads2nd = torch.autograd.grad(dVdS, x, grad_outputs=torch.ones(
            dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
        d2VdS2 = grads2nd[:, 1].view(-1, 1)

        S1 = x[:, 1].view(-1, 1)

        dt_term = dVdt + 0.5
        sigma_term = (self.sigma * self.sigma) * (S1 * S1) * d2VdS2
        ds_term = r * S1 * dVdS
        y1_hat_term = r * y1_hat

        bs_pde = dt_term + ds_term + sigma_term - y1_hat_term

        # print(nn.MSELoss()(sigma_term, torch.zeros_like(sigma_term)))
        # print(ds_term + dt_term + y1_hat_term)
        # print("Sigma term", d2VdS2)
        return y1_hat, bs_pde
