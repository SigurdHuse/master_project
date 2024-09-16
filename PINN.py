import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_function=nn.ReLU()):
        super().__init__()

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

        self.output_layer = nn.Sequential(
            nn.Linear(N_HIDDEN, N_OUTPUT),
            self.activation)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
