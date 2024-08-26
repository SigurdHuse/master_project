from PINN import PINN
from dataloader import Dataloader
import torch

import logging
from datetime import datetime

N_EPOCH = 80000
LEARNING_RATE = 1e-5
HIDDEN_LAYER = 4
HIDDEN_WIDTH = 128
N_sample = 8000
DEVICE = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)

K = 40
r = 0.05
sigma = 0.2
T = 1
S_range = [0, 160]
t_range = [0, T]

euro_call_data = Dataloader(t_range, S_range, K, r, sigma)

model = PINN(2, 1, HIDDEN_WIDTH, HIDDEN_LAYER).to(DEVICE)
