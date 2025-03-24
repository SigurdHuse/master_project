from train import black_scholes_multi_dimensional, create_validation_data

from training_functions import try_multiple_activation_functions, try_different_learning_rates, try_different_architectures
from training_functions import train_multiple_times, try_sigma_fourier_and_embedding_size, try_multiple_dimensions
from data_generator import DataGeneratorEuropeanMultiDimensional
from PINN import PINNforwards

import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


config = {}

config["american_option"] = False
config["N_INPUT"] = 6
config["use_fourier_transform"] = False
config["sigma_fourier"] = 5.0
config["fourier_encoded_size"] = 512

config["w_expiry"] = 1
config["w_lower"] = 1
config["w_upper"] = 1

config["weight_decay"] = 0
config["gamma"] = 0.99
# config["scheduler_step"] = 1_000

config["N_sample"] = 512
config["lambda_pde"] = 1
config["lambda_boundary"] = 1
config["lambda_expiry"] = 1
# config["update_lambda"] = 500
# config["alpha_lambda"] = 0.9

config["K"] = 1
config["t_range"] = [0, 1]

""" config["S_range"] = np.array([[0, 6_00]])

config["sigma"] = np.array([[1.0]]) """

config["S_range"] = np.array([[0, 25],
                              [0, 25],
                              [0, 20],
                              [0, 20],
                              [0, 15]])

config["sigma"] = np.array([[0.1, 0.2, 0.1, 0.3, 0.1],
                            [0.15, 0.1, 0.4, 0.1, 0.1],
                            [0.1, 0.1, 0.3, 0.1, 0.25],
                            [0.3, 0.1, 0.5, 0.1, 0.1],
                            [0.1, 0.7, 0.1, 0.1, 0.6]])

config["sigma_torch"] = torch.tensor(config["sigma"]).to(DEVICE)
# config["cov"] = config["sigma"]@config["sigma"].T
# config["cov_torch"] = torch.tensor(config["cov"]).to(DEVICE)
config["r"] = 0.04

config["learning_rate"] = 1e-3
config["save_model"] = False
config["save_loss"] = False
config["epochs_before_validation"] = 30

# config["epochs_before_validation"]
config["epochs_before_validation_loss_saved"] = 600
config["epochs_before_loss_saved"] = 600

if __name__ == "__main__":
    torch.manual_seed(2024)
    np.random.seed(2024)
    dataloader = DataGeneratorEuropeanMultiDimensional(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2024)

    validation_data = create_validation_data(
        dataloader=dataloader, N_validation=1024, config=config)

    test_data = create_validation_data(
        dataloader=dataloader, N_validation=20_000, config=config)

    """ torch.manual_seed(2025)
    np.random.seed(2025)
    # config["save_model"] = True
    # config["save_loss"] = True
    dataloader = DataGeneratorEuropeanMultiDimensional(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_multi_dimensional,
                                 filename1="important_results/european_multi/RMSE_test.txt", filename2="important_results/european_multi/epoch_test.txt",
                                 learning_rates=[1e-3], batch_sizes=[128, 256, 512, 1024],
                                 validation_data=validation_data, test_data=test_data, epochs=800_000, custom_arc=[256, 256, 256, 256, 256]) """
    # config["save_model"] = False

    """ torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropeanMultiDimensional(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_multi_dimensional,
                                 filename1="important_results/european_multi/RMSE_lr_first.txt", filename2="important_results/european_multi/epoch_lr_first.txt",
                                 learning_rates=[1e-3, 5e-3, 5e-4], batch_sizes=[256, 512, 1024],
                                 validation_data=validation_data, test_data=test_data, epochs=800_000) """

    """ torch.manual_seed(2025)
    np.random.seed(2025)
    config["save_model"] = True
    config["epochs_before_validation_loss_saved"] = 600
    config["epochs_before_loss_saved"] = 600
    config["save_loss"] = True
    dataloader = DataGeneratorEuropeanMultiDimensional(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_multi_dimensional,
                         filename="multi_dim", nr_of_epochs=800_000, dataloader=dataloader, config=config,
                         validation_data=validation_data, test_data=test_data)
    config["save_model"] = False
    config["epochs_before_validation_loss_saved"] = 600
    config["epochs_before_loss_saved"] = 600
    config["save_loss"] = False """

    """ try_multiple_dimensions(dimensions=list(range(1, 6 + 1)), config=config, PDE=black_scholes_multi_dimensional, filename1="important_results/european_multi/RMSE_dim_1.txt",
                            filename2="important_results/european_multi/epoch_dim_1.txt") """

    try_multiple_dimensions(dimensions=list(range(7, 10 + 1)), config=config, PDE=black_scholes_multi_dimensional, filename1="important_results/european_multi/RMSE_dim_2.txt",
                            filename2="important_results/european_multi/epoch_dim_2.txt")
