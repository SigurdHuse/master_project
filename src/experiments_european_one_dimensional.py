from train import black_scholes_1D, DEVICE
from train import try_multiple_activation_functions, try_different_learning_rates, try_different_architectures
from train import train_multiple_times, create_validation_data
from train import train, computing_the_greeks, trying_weight_decay
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
config["scheduler_step"] = 5_000

config["N_sample"] = 256
config["lambda_pde"] = 1
config["lambda_boundary"] = 1
config["lambda_expiry"] = 1
config["update_lambda"] = 1_000_000
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


""" torch.manual_seed(2025)
np.random.seed(2025)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/european_1D/MSE_lr_1d_european_test.txt", filename2="important_results/european_1D/epoch_lr_1D_european_test.txt",
                             learning_rates=[7e-3, 5e-3, 2e-3, 1e-3], batch_sizes=[64, 128],
                             validation_data=validation_data, test_data=test_data, epochs=600_000) """

""" model = PINNforwards(2, 1, 128, 2, nn.Tanh())
train(model, 10_000, config["learning_rate"], dataloader,
      config, "test", black_scholes_1D, validation_data) """


""" torch.manual_seed(2025)
np.random.seed(2025)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/european_1D/MSE_lr_no_lambda.txt", filename2="important_results/european_1D/epoch_lr_no_lambda.txt",
                             learning_rates=[5e-3, 1e-3, 5e-4], batch_sizes=[16, 32, 64, 128, 256],
                             validation_data=validation_data, test_data=test_data, epochs=400_000) """


torch.manual_seed(2025)
np.random.seed(2025)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/european_1D/MSE_lr_first.txt", filename2="important_results/european_1D/epoch_lr_first.txt",
                             learning_rates=[1e-2,  5e-3, 1e-3, 5e-4], batch_sizes=[32, 64, 128, 256],
                             validation_data=validation_data, test_data=test_data, epochs=600_000)


""" torch.manual_seed(2025)
np.random.seed(2025)
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/european_1D/MSE_lr_1d_european_fine.txt", filename2="important_results/european_1D/epoch_lr_1D_european_fine.txt",
                             learning_rates=[2e-3, 1e-3, 9e-4], batch_sizes=[64, 96, 128, 192, 256],
                             validation_data=validation_data, test_data=test_data, epochs=400_000) """


""" torch.manual_seed(2025)
np.random.seed(2025)
try_multiple_activation_functions(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                                  filename1="important_results/european_1D/activation_1D_european.txt",
                                  filename2="important_results/european_1D/activation_epochs_1D_european.txt",
                                  activation_functions=[
                                      nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh],
                                  layers=[1, 2, 4, 8], validation_data=validation_data, test_data=test_data, epochs=400_000) """


""" torch.manual_seed(2025)
np.random.seed(2025)
layers = [1, 2, 4, 6, 8]
nodes = [32, 64, 128, 256, 512, 1024]
try_different_architectures(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                            filename1="important_results/european_1D/arc_1D_european.txt", filename2="important_results/european_1D/arc_epochs_1D_european.txt",
                            layers=layers, nodes=nodes, validation_data=validation_data, test_data=test_data, epochs=350_000) """


""" trying_weight_decay(config=config, dataloader=dataloader, PDE=black_scholes_1D, test_data=test_data,
                    filename1="important_results/european_1D/wd_MSE.txt", filename2="important_results/european_1D/wd_epochs.txt",
                    weight_decays=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0], validation_data=validation_data,
                    epochs=400_000) """

""" train_multiple_times(seeds=list(range(1, 10 + 1)), layers=5, nodes=256, PDE=black_scholes_1D, filename="test",
                        nr_of_epochs=350_000, dataloader=dataloader, config=config, validation_data=validation_data) """

""" torch.manual_seed(2025)
np.random.seed(2025)
config["save_model"] = True
computing_the_greeks(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                     filename="important_results/european_1d/greeks.txt", validation_data=validation_data, test_data=test_data, epochs=400_000)
config["save_model"] = False """


# Try large PDE lambda
""" torch.manual_seed(2025)
np.random.seed(2025)
config["update_lambda"] = 1_000_000
try_different_learning_rates(config=config, dataloader=dataloader, PDE=black_scholes_1D,
                             filename1="important_results/european_1D/MSE_lr_1d_european_no_lambda.txt", filename2="important_results/european_1D/epoch_lr_1D_european_no_lambda.txt",
                             learning_rates=[1e-2, 5e-3, 1e-3, 5e-4], batch_sizes=[32, 64, 128, 256, 512],
                             validation_data=validation_data, test_data=test_data, epochs=400_000)
config["update_lambda"] = 500 """

""" train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=256, PDE=black_scholes_1D, filename="with_lambda",
                     nr_of_epochs=400_000, dataloader=dataloader, config=config, validation_data=validation_data) """


""" torch.manual_seed(2025)
np.random.seed(2025)
config["update_lambda"] = 1_000_000
config["N_sample"] = 64
train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=256, PDE=black_scholes_1D, filename="no_lambda_64",
                     nr_of_epochs=400_000, dataloader=dataloader, config=config, validation_data=validation_data)
config["N_sample"] = 256
config["update_lambda"] = 500 """
