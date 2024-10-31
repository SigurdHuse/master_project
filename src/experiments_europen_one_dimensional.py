from train import black_scholes_1D, black_scholes_american_1D, black_scholes_multi_dimensional, DEVICE
from train import try_multiple_activation_functions, try_different_learning_rates, try_different_architectures, train_multiple_times, create_validation_data
from train import train
from data_generator import DataGeneratorEuropean1D
from PINN import PINNforwards

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
config["scheduler_step"] = 2000

config["N_sample"] = 2_00
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

dataloader = DataGeneratorEuropean1D(
    time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE)

validation_data = create_validation_data(
    dataloader=dataloader, N_validation=5_000, config=config)

test_data = create_validation_data(
    dataloader=dataloader, N_validation=20_000, config=config)

""" model = PINNforwards(2, 1, 128, 2, nn.Tanh())
train(model, 10_000, config["learning_rate"], dataloader,
      config, "test", black_scholes_1D, validation_data) """

torch.manual_seed(2025)
np.random.seed(2025)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/MSE_lr_1d_european.txt", filename2="important_results/epoch_lr_1D_european.txt",
                             learning_rates=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5], batch_sizes=[50, 100, 150, 200, 250],
                             validation_data=validation_data, test_data=test_data)

torch.manual_seed(2025)
np.random.seed(2025)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/MSE_lr_1d_european_fine.txt", filename2="important_results/epoch_lr_1D_european_fine.txt",
                             learning_rates=[3e-3, 2e-3, 1e-3, 9e-4, 8e-4], batch_sizes=[80, 1_00, 110, 120, 130, 140],
                             validation_data=validation_data, test_data=test_data)

torch.manual_seed(2025)
np.random.seed(2025)
try_multiple_activation_functions(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                                  filename1="important_results/activation_1D_european.txt",
                                  filename2="important_results/activation_epochs_1D_european.txt",
                                  activation_functions=[
                                      nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()],
                                  layers=[1, 2, 4, 6, 8], validation_data=validation_data, test_data=test_data)

torch.manual_seed(2025)
np.random.seed(2025)
layers = [1, 2, 3, 4, 5, 6]
nodes = [32, 64, 128, 256, 512, 1024]
try_different_architectures(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                            filename1="important_results/arc_1D_european.txt", filename2="important_results/arc_epochs_1D_european.txt",
                            layers=layers, nodes=nodes, validation_data=validation_data, test_data=test_data)


""" for lr, filename in zip([2e-5, 1e-4, 1e-3, 1e-2], ["lr_1", "lr_2", "lr_3", "lr_4"]):
    config["learning_rate"] = lr
    train_multiple_times(seeds=list(range(2024, 2034 + 1)), layers=4, nodes=500, PDE=black_scholes_1D, filename=filename,
                         nr_of_epochs=25_000, dataloader=dataloader, config=config, validation_data=validation_data)

config["learning_rate"] = 2e-5 """
