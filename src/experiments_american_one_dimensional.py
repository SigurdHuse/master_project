from train import black_scholes_american_1D
from data_generator import DataGeneratorAmerican1D
from train import create_validation_data

from training_functions import experiment_with_binomial_model, try_multiple_activation_functions, try_different_learning_rates, train_multiple_times, try_different_lambdas

import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


config = {}

# Indicates we are working with an American option
config["american_option"] = True
# Number of inputs to model (t,S_1)
config["N_INPUT"] = 2
# Indicates if model should Fourier embed input
config["use_fourier_transform"] = True
# Variance of Fourier emebedding
config["sigma_fourier"] = 5.0
# Embedding size of Fourier encoding
config["fourier_encoded_size"] = 128

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
# Weighting of free exercise boundary loss
config["lambda_exercise"] = 1

# Strike price
config["K"] = 20
# Time range
config["t_range"] = [0, 1]
# Asset price range
config["S_range"] = [0, 400]
# Volatility
config["sigma"] = 0.5
# Risk-free rate
config["r"] = 0.04

# Initial learning rate
config["learning_rate"] = 5e-4
# Indicates if we should save model at the end of training
config["save_model"] = False
# Indicates if we should save loss at the end of training
config["save_loss"] = False
# Number of epochs before validation loss is computed
config["epochs_before_validation"] = 30

# Number of epochs before validation loss is saved
config["epochs_before_validation_loss_saved"] = config["epochs_before_validation"]
# Number of epochs before training loss is saved
config["epochs_before_loss_saved"] = 1


if __name__ == "__main__":
    torch.manual_seed(2024)
    np.random.seed(2024)
    dataloader = DataGeneratorAmerican1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2024)

    validation_data = create_validation_data(
        dataloader=dataloader, N_validation=5_000, config=config)

    test_data = create_validation_data(
        dataloader=dataloader, N_validation=20_000, config=config)

    experiment_with_binomial_model(M_values=[32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048], dataloader=dataloader, test_data=test_data,
                                   filename1="important_results/american_1D/RMSE_binomial.txt", filename2="important_results/american_1D/timings_binomial.txt")

    tmp_X = test_data["X1_validation"].cpu().detach().numpy()
    test = dataloader.get_analytical_solution(tmp_X[:, 1], tmp_X[:, 0], M=2048)
    np.save("data/test_data_american_1D", test)

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorAmerican1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_different_learning_rates(config=config,
                                 dataloader=dataloader,
                                 PDE=black_scholes_american_1D,
                                 filename2="important_results/american_1D/epochs_lr.txt",
                                 filename1="important_results/american_1D/RMSE_lr.txt",
                                 learning_rates=[5e-3, 1e-3, 5e-4],
                                 batch_sizes=[256, 512, 1024],
                                 validation_data=validation_data,
                                 test_data=test_data,
                                 analytical_solution_filename="data/test_data_american_1D.npy",
                                 epochs=600_000)

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorAmerican1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_multiple_activation_functions(config=config, dataloader=dataloader, PDE=black_scholes_american_1D,
                                      filename1="important_results/american_1D/RMSE_activation.txt",
                                      filename2="important_results/american_1D/epochs_activation.txt",
                                      activation_functions=[
                                          nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh],
                                      layers=[1, 2, 4, 8],
                                      validation_data=validation_data,
                                      test_data=test_data,
                                      analytical_solution_filename="data/test_data_american_1D.npy",
                                      epochs=600_000)

    config["save_model"] = True
    config["epochs_before_validation_loss_saved"] = 600
    config["epochs_before_loss_saved"] = 600
    config["save_loss"] = True
    dataloader = DataGeneratorAmerican1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    train_multiple_times(seeds=list(range(1, 10 + 1)),
                         layers=4,
                         nodes=128,
                         PDE=black_scholes_american_1D,
                         filename="american_multiple",
                         nr_of_epochs=600_000,
                         dataloader=dataloader,
                         config=config,
                         validation_data=validation_data,
                         test_data=test_data,
                         analytical_solution_filename="data/test_data_american_1D.npy")
