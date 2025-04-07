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


# Indicates we are working with an American option
config["american_option"] = False
# Number of inputs to model (t,S_1,S_2,S_3,S_4,S_5)
config["N_INPUT"] = 6

# Indicates if model should Fourier embed input
config["use_fourier_transform"] = False
# Variance of Fourier emebedding
config["sigma_fourier"] = 0.0
# Embedding size of Fourier encoding
config["fourier_encoded_size"] = 0


# Weighting of number of points to sample from expiry
config["w_expiry"] = 1
# Weighting of number of points to sample from lower part of domain, S = 0
config["w_lower"] = 1
# Weighting of number of points to sample from upper part of domain, S = S^max
config["w_upper"] = 1


# Constant to multiply L2-regularization with
config["weight_decay"] = 0
# Decay rate of learning rate
config["gamma"] = 0.99

# Number of points N to sample, remember N_u = 4N, N_b = 2N, N_h = N
config["N_sample"] = 512
# Weighting of PDE residual loss
config["lambda_pde"] = 1
# Weighting of boundary loss
config["lambda_boundary"] = 1
# Weighting of expiry loss
config["lambda_expiry"] = 1

# Strik price
config["K"] = 1
# Time range
config["t_range"] = [0, 1]
# Asset price ranges
config["S_range"] = np.array([[0, 25],
                              [0, 25],
                              [0, 20],
                              [0, 20],
                              [0, 15]])
# Volatility matrix
config["sigma"] = np.array([[0.05, 0.05, 0.1, 0.15, 0.1],
                            [0.01, 0.1, 0.05, 0.05, 0.01],
                            [0.01, 0.05, 0.1, 0.1, 0.15],
                            [0.05, 0.1, 0.05, 0.15, 0.1],
                            [0.1, 0.1, 0.01, 0.05, 0.05]])
# Volatility matrix as a tensor
config["sigma_torch"] = torch.tensor(config["sigma"]).to(DEVICE)
# Risk-free interest rate
config["r"] = 0.04


# Initial learning rate
config["learning_rate"] = 1e-3
# Indicates if we should save model at the end of training
config["save_model"] = False
# Indicates if we should save loss at the end of training
config["save_loss"] = False
# Number of epochs before validation loss is computed
config["epochs_before_validation"] = 30

# Number of epochs before validation loss is saved
config["epochs_before_validation_loss_saved"] = 600
# Number of epochs before training loss is saved
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

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropeanMultiDimensional(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_different_learning_rates(config=config,
                                 dataloader=dataloader,
                                 PDE=black_scholes_multi_dimensional,
                                 filename1="important_results/european_multi/RMSE_lr_first.txt",
                                 filename2="important_results/european_multi/epoch_lr_first.txt",
                                 learning_rates=[1e-3, 5e-3, 5e-4],
                                 batch_sizes=[256, 512, 1024],
                                 validation_data=validation_data,
                                 test_data=test_data,
                                 epochs=800_000)

    torch.manual_seed(2025)
    np.random.seed(2025)
    config["save_model"] = True
    config["epochs_before_validation_loss_saved"] = 600
    config["epochs_before_loss_saved"] = 600
    config["save_loss"] = True
    dataloader = DataGeneratorEuropeanMultiDimensional(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    train_multiple_times(seeds=list(range(1, 10 + 1)),
                         layers=4,
                         nodes=128,
                         PDE=black_scholes_multi_dimensional,
                         filename="multi_dim",
                         nr_of_epochs=800_000,
                         dataloader=dataloader,
                         config=config,
                         validation_data=validation_data,
                         test_data=test_data)

    config["save_model"] = False
    config["epochs_before_validation_loss_saved"] = 600
    config["epochs_before_loss_saved"] = 600
    config["save_loss"] = False
    torch.manual_seed(2025)
    np.random.seed(2025)
    try_multiple_dimensions(dimensions=list(range(1, 13 + 1)),
                            config=config,
                            PDE=black_scholes_multi_dimensional,
                            filename1=f"important_results/european_multi/RMSE_dim.txt",
                            filename2=f"important_results/european_multi/epoch_dim.txt")
