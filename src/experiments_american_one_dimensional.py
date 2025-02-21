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

# config["american_option"] = True
config["N_INPUT"] = 2
config["use_fourier_transform"] = True
config["sigma_fourier"] = 5.0
config["fourier_encoded_size"] = 128

config["w_expiry"] = 1
config["w_lower"] = 1
config["w_upper"] = 1

config["weight_decay"] = 0
config["gamma"] = 0.99
config["scheduler_step"] = 1_000

config["N_sample"] = 512
config["lambda_pde"] = 1
config["lambda_boundary"] = 1
config["lambda_expiry"] = 1
# config["lambda_exercise"] = 1
# config["update_lambda"] = 500
# config["alpha_lambda"] = 0.9

config["K"] = 40
config["t_range"] = [0, 1]
config["S_range"] = [0, 400]
config["sigma"] = 0.5
config["r"] = 0.04

config["learning_rate"] = 1e-3
config["save_model"] = False
config["save_loss"] = False
config["epochs_before_validation"] = 30

config["epochs_before_validation_loss_saved"] = config["epochs_before_validation"]
config["epochs_before_loss_saved"] = 1


if __name__ == "__main__":
    torch.manual_seed(2026)
    np.random.seed(2026)
    dataloader = DataGeneratorAmerican1D(
        time_range=config["t_range"], S_range=config["S_range"], K=config["K"], r=config["r"], sigma=config["sigma"], DEVICE=DEVICE)

    validation_data = create_validation_data(
        dataloader=dataloader, N_validation=5_000, config=config)

    test_data = create_validation_data(
        dataloader=dataloader, N_validation=20_000, config=config)

    """ experiment_with_binomial_model(M_values=[32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048], dataloader=dataloader, test_data=test_data,
                                filename1="important_results/american_1D/RMSE_binomial.txt", filename2="important_results/american_1D/timings_binomial.txt") """

    """ tmp_X = test_data["X1_validation"].cpu().detach().numpy()
    test = dataloader.get_analytical_solution(tmp_X[:, 1], tmp_X[:, 0], M=2048)
    np.save("data/test_data_american_1D", test)

    make_3D_american_plot("plots/american_3D.png", tmp_X) """

    """ torch.manual_seed(2026)
    np.random.seed(2026)
    try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_american_1D,
                                 filename2="important_results/american_1D/epochs_test.txt",
                                 filename1="important_results/american_1D/RMSE_test.txt",
                                 learning_rates=[1e-3], batch_sizes=[1024],
                                 validation_data=validation_data, test_data=test_data,  analytical_solution_filename="data/test_data_american_1D.npy", epochs=200_000) """

    """ torch.manual_seed(2026)
    np.random.seed(2026)
    try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_american_1D,
                                 filename2="important_results/american_1D/epochs_lr.txt",
                                 filename1="important_results/american_1D/RMSE_lr.txt",
                                 learning_rates=[1e-3, 5e-3, 5e-4], batch_sizes=[256, 512, 1024],
                                 validation_data=validation_data, test_data=test_data,  analytical_solution_filename="data/test_data_american_1D.npy", epochs=600_000) """

    torch.manual_seed(2026)
    np.random.seed(2026)
    try_multiple_activation_functions(config=config, dataloader=dataloader, PDE=black_scholes_american_1D,
                                      filename1="important_results/american_1D/RMSE_activation.txt",
                                      filename2="important_results/american_1D/epochs_activation.txt",
                                      activation_functions=[
                                          nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh],
                                      layers=[1, 2, 4, 8], validation_data=validation_data, test_data=test_data, analytical_solution_filename="data/test_data_american_1D.npy", epochs=600_000)

    config["save_model"] = True
    config["epochs_before_validation_loss_saved"] = 600
    config["epochs_before_loss_saved"] = 600
    config["save_loss"] = True
    train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_american_1D, filename="american_multiple",
                         nr_of_epochs=600_000, dataloader=dataloader, config=config, validation_data=validation_data, test_data=test_data, analytical_solution_filename="data/test_data_american_1D.npy")

    """ torch.manual_seed(2025)
    np.random.seed(2025)
    try_different_lambdas(config=config, dataloader=dataloader, PDE=black_scholes_american_1D,
                          filename1="important_results/american_1D/RMSE_lambda.txt", filename2="important_results/american_1D/epoch_lambda.txt",
                          lambdas=[[1, 1, 1, 1e-2], [1, 1, 1, 1e-1], [1, 1, 1,
                                                                      1], [1, 1, 1, 10], [1, 1, 1, 50], [1, 1, 1, 100]],
                          validation_data=validation_data, test_data=test_data, epochs=600_000, analytical_solution_filename="data/test_data_american_1D.npy") """
