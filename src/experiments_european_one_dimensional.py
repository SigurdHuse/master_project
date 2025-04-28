from train import black_scholes_1D, create_validation_data, train
from training_functions import try_multiple_activation_functions, try_different_learning_rates, try_different_architectures
from training_functions import train_multiple_times, try_sigma_fourier_and_embedding_size
from training_functions import computing_the_greeks, trying_weight_decay, try_different_lambdas, compute_test_loss
from data_generator import DataGeneratorEuropean1D
from PINN import PINNforwards

import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


config = {}

# Indicates we are working with an American option
config["american_option"] = False
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
config["N_sample"] = 1024
# Weighting of PDE residual loss
config["lambda_pde"] = 1
# Weighting of boundary loss
config["lambda_boundary"] = 1
# Weighting of expiry loss
config["lambda_expiry"] = 1

# Strik price
config["K"] = 40
# Time range
config["t_range"] = [0, 1]
# Asset price range
config["S_range"] = [0, 400]
# Volatility
config["sigma"] = 0.5
# Risk-free rate
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
config["epochs_before_validation_loss_saved"] = config["epochs_before_validation"]
# Number of epochs before training loss is saved
config["epochs_before_loss_saved"] = 1

if __name__ == "__main__":
    torch.manual_seed(2024)
    np.random.seed(2024)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2024)

    validation_data = create_validation_data(
        dataloader=dataloader, N_validation=5_000, config=config)

    test_data = create_validation_data(
        dataloader=dataloader, N_validation=20_000, config=config)

    """ torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_different_learning_rates(config=config,
                                 dataloader=dataloader,
                                 PDE=black_scholes_1D,
                                 filename1="important_results/european_1D/RMSE_lr_first.txt",
                                 filename2="important_results/european_1D/epoch_lr_first.txt",
                                 learning_rates=[5e-3, 1e-3, 5e-4],
                                 batch_sizes=[256, 512, 1024, 2048, 4096],
                                 validation_data=validation_data,
                                 test_data=test_data,
                                 epochs=600_000)

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_sigma_fourier_and_embedding_size(config=config,
                                         dataloader=dataloader,
                                         PDE=black_scholes_1D,
                                         filename1="important_results/european_1D/RMSE_fourier.txt",
                                         filename2="important_results/european_1D/epoch_fourier.txt",
                                         sigma_fourier=[1.0, 5.0, 10.0],
                                         embedding_size=[
                                             32, 64, 128, 256, 512],
                                         validation_data=validation_data,
                                         test_data=test_data,
                                         epochs=600_000)

    config["epochs_before_validation_loss_saved"] = 600
    config["epochs_before_loss_saved"] = 600
    config["use_fourier_transform"] = False
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    train_multiple_times(seeds=list(range(1, 10 + 1)),
                         layers=0,
                         nodes=0,
                         PDE=black_scholes_1D,
                         filename="no_fourier",
                         nr_of_epochs=600_000,
                         dataloader=dataloader,
                         config=config,
                         validation_data=validation_data,
                         test_data=test_data,
                         custom_arc=[256, 128, 128, 128, 128, 128])
    config["use_fourier_transform"] = True

    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    train_multiple_times(seeds=list(range(1, 10 + 1)),
                         layers=4,
                         nodes=128,
                         PDE=black_scholes_1D,
                         filename="with_fourier",
                         nr_of_epochs=600_000,
                         dataloader=dataloader,
                         config=config,
                         validation_data=validation_data,
                         test_data=test_data)
    config["epochs_before_validation_loss_saved"] = config["epochs_before_validation"]
    config["epochs_before_loss_saved"] = 1

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_multiple_activation_functions(config=config,
                                      dataloader=dataloader,
                                      PDE=black_scholes_1D,
                                      filename1="important_results/european_1D/RMSE_activation.txt",
                                      filename2="important_results/european_1D/epoch_activation.txt",
                                      activation_functions=[
                                          nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh],
                                      layers=[1, 2, 4, 8],
                                      validation_data=validation_data,
                                      test_data=test_data,
                                      epochs=600_000)

    torch.manual_seed(2025)
    np.random.seed(2025)
    layers = [1, 2, 4, 8]
    nodes = [32, 64, 128, 256]
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    try_different_architectures(config=config,
                                dataloader=dataloader,
                                PDE=black_scholes_1D,
                                filename1="important_results/european_1D/RMSE_arc.txt",
                                filename2="important_results/european_1D/epoch_arcs.txt",
                                layers=layers,
                                nodes=nodes,
                                validation_data=validation_data,
                                test_data=test_data,
                                epochs=600_000)

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    trying_weight_decay(config=config,
                        dataloader=dataloader,
                        PDE=black_scholes_1D,
                        test_data=test_data,
                        filename1="important_results/european_1D/RMSE_wd.txt",
                        filename2="important_results/european_1D/epoch_wd.txt",
                        weight_decays=[1e-2, 1e-3, 1e-4,
                                       1e-5, 1e-6, 1e-7, 0.0],
                        validation_data=validation_data,
                        epochs=600_000)

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    config["save_model"] = True
    computing_the_greeks(config=config,
                         dataloader=dataloader,
                         PDE=black_scholes_1D,
                         filename="important_results/european_1D/greeks.txt",
                         validation_data=validation_data,
                         test_data=test_data,
                         epochs=600_000)
    config["save_model"] = False

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    config["scheduler_step"] = 1_000_000
    try_different_learning_rates(config=config,
                                 dataloader=dataloader,
                                 PDE=black_scholes_1D,
                                 filename1="important_results/european_1D/RMSE_lr_scheduler.txt",
                                 filename2="important_results/european_1D/epoch_lr_scheduler.txt",
                                 learning_rates=[5e-3, 1e-3, 5e-4],
                                 batch_sizes=[256, 512, 1024],
                                 validation_data=validation_data,
                                 test_data=test_data,
                                 epochs=600_000)
    config["scheduler_step"] = 1_000 """

    torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    config["save_model"] = True
    config["epochs_before_validation_loss_saved"] = 1200
    config["epochs_before_loss_saved"] = 1200
    config["save_loss"] = True
    model = PINNforwards(2, 1, 256, 4, use_fourier_transform=True,
                         sigma_FF=5.0, encoded_size=128)
    best_epoch = train(model, 3_000_000, config["learning_rate"], dataloader,
                       config, "large_model", black_scholes_1D, validation_data)
    config["save_model"] = False
    config["save_loss"] = False
    config["epochs_before_validation_loss_saved"] = config["epochs_before_validation"]
    config["epochs_before_loss_saved"] = 1
    RMSE = compute_test_loss(model=model, test_data=test_data, dataloader=dataloader,
                             analytical_solution_filename=None)

    with open("important_results/european_1D/large_model.txt", 'w') as outfile:
        outfile.write(f"Best epoch : {best_epoch}\n")
        outfile.write(f"RMSE : {RMSE}")

    """ torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE, seed=2025)
    config["save_model"] = True
    config["epochs_before_validation_loss_saved"] = 90
    config["epochs_before_loss_saved"] = 90
    config["save_loss"] = True
    model = PINNforwards(2, 1, 64, 2, use_fourier_transform=True,
                         sigma_FF=5.0, encoded_size=128)
    best_epoch = train(model, 300_000, config["learning_rate"], dataloader,
                       config, "small_model", black_scholes_1D, validation_data)
    config["save_model"] = False
    config["save_loss"] = False
    config["epochs_before_validation_loss_saved"] = config["epochs_before_validation"]
    config["epochs_before_loss_saved"] = 1
    RMSE = compute_test_loss(model=model, test_data=test_data, dataloader=dataloader,
                             analytical_solution_filename=None)

    with open("important_results/european_1D/small_model.txt", 'w') as outfile:
        outfile.write(f"Best epoch : {best_epoch}\n")
        outfile.write(f"RMSE : {RMSE}") """
