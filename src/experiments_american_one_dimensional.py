from train import black_scholes_american_1D, DEVICE
from data_generator import DataGeneratorAmerican1D

from train import create_validation_data, try_multiple_activation_functions, try_different_learning_rates

import torch
import torch.nn as nn
import numpy as np


torch.set_default_device(DEVICE)
torch.manual_seed(2025)
np.random.seed(2025)

config = {}

config["w_expiry"] = 1
config["w_lower"] = 1
config["w_upper"] = 1

config["weight_decay"] = 0
config["gamma"] = 0.9
config["scheduler_step"] = 5000

config["N_sample"] = 512
config["lambda_pde"] = 1
config["lambda_boundary"] = 1
config["lambda_expiry"] = 1
config["update_lambda"] = 500
config["alpha_lambda"] = 0.9

config["K"] = 40
config["t_range"] = [0, 1]
config["S_range"] = [0, 200]
config["sigma"] = 0.5
config["r"] = 0.04

config["learning_rate"] = 1e-3
config["save_model"] = False
config["save_loss"] = False
config["N_INPUT"] = 2

dataloader = DataGeneratorAmerican1D(
    time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE)

validation_data = create_validation_data(
    dataloader=dataloader, N_validation=5_000, config=config)

test_data = create_validation_data(
    dataloader=dataloader, N_validation=20_000, config=config)

""" tmp_X = test_data["X1_validation"].cpu().detach().numpy()
test = dataloader.get_analytical_solution(tmp_X[:, 1], tmp_X[:, 0], n=1024)
np.save("data/test_data_american_1D", test) """


""" torch.manual_seed(2026)
np.random.seed(2026)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_american_1D,
                             filename2="important_results/american_1D/learning_rates_epochs.txt",
                             filename1="important_results/american_1D/learning_rates.txt",
                             learning_rates=[5e-3, 1e-3, 5e-4], batch_sizes=[64, 128, 256, 512],
                             validation_data=validation_data, test_data=test_data,  analytical_solution_filename="data/test_data_american_1D.npy", epochs=350_000)
 """
torch.manual_seed(2026)
np.random.seed(2026)
try_multiple_activation_functions(config=config, dataloader=dataloader, PDE=black_scholes_american_1D,
                                  filename1="important_results/american_1D/activation_1D_american.txt",
                                  filename2="important_results/american_1D/activation_epochs_1D_american.txt",
                                  activation_functions=[
                                      nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh],
                                  layers=[2, 4, 6, 8], validation_data=validation_data, test_data=test_data, analytical_solution_filename="data/test_data_american_1D.npy", epochs=400_000)
