from train import black_scholes_1D, black_scholes_american_1D, black_scholes_multi_dimensional, DEVICE
from train import try_multiple_activation_functions, try_different_learning_rates, try_different_architectures, train_multiple_times, create_validation_data
from dataloader import DataloaderEuropean1D

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
config["N_sample"] = 1_000
config["pde_learning_rate"] = 80
config["K"] = 40
config["t_range"] = [0, 1]
config["S_range"] = [0, 200]
config["sigma"] = 0.25
config["r"] = 0.04
config["learning_rate"] = 2e-5
config["save_model"] = False
config["save_loss"] = False
config["N_INPUT"] = 2

dataloader = DataloaderEuropean1D(
    time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE)

validation_data = create_validation_data(
    dataloader=dataloader, N_validation=5_000, config=config)

test_data = create_validation_data(
    dataloader=dataloader, N_validation=20_000, config=config)


""" torch.manual_seed(2025)
np.random.seed(2025)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/MSE_lr_1d_european.txt", filename2="important_results/epoch_lr_1D_european.txt",
                             learning_rates=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5], batch_sizes=[100, 1_000, 5_000, 10_000],
                             validation_data=validation_data, test_data=test_data) """

""" torch.manual_seed(2025)
np.random.seed(2025)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/MSE_lr_1d_european_fine.txt", filename2="important_results/epoch_lr_1D_european_fine.txt",
                             learning_rates=[7e-5, 2e-5, 1e-5, 5e-6], batch_sizes=[5_00, 1_000, 2_000, 3_000, 4_000, 5_000],
                             validation_data=validation_data, test_data=test_data) """

""" torch.manual_seed(2025)
np.random.seed(2025)
try_multiple_activation_functions(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                                  filename1="important_results/activation_1D_european.txt",
                                  filename2="important_results/activation_epochs_1D_european.txt",
                                  activation_functions=[
                                      nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()],
                                  layers=[1, 2, 4, 8, 12], validation_data=validation_data, test_data=test_data) """

""" torch.manual_seed(2025)
np.random.seed(2025)
layers = [1, 2, 4, 6, 8]
nodes = [50, 100, 200, 400, 600, 800]
try_different_architectures(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                            filename1="important_results/arc_1D_european.txt", filename2="important_results/arc_epochs_1D_european.txt",
                            layers=layers, nodes=nodes, validation_data=validation_data, test_data=test_data) """


for lr, filename in zip([2e-5, 1e-4, 1e-3, 1e-2], ["lr_1", "lr_2", "lr_3", "lr_4"]):
    train_multiple_times(seeds=list(range(2024, 2034 + 1)), layers=4, nodes=500, PDE=black_scholes_1D, filename=filename,
                         nr_of_epochs=25_000, dataloader=dataloader, config=config, validation_data=validation_data)

config["learning_rate"] = 2e-5
