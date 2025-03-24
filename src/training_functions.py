from data_generator import DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D
import torch
import torch.nn as nn
import numpy as np
from PINN import PINNforwards
from train import train
import copy
import os
from torch.distributions import Normal
import time
import datetime as dt
from train import create_validation_data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


def MSE_numpy(target: np.array, prediction: np.array):
    return np.square(np.subtract(target.flatten(), prediction.flatten())).mean()


def RMSE_numpy(target: np.array, prediction: np.array):
    return np.sqrt(MSE_numpy(target, prediction))


def standard_normal_pdf(x):
    return (1 / torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-x.pow(2) / 2)


def compute_test_loss(model, test_data: dict, dataloader, analytical_solution_filename: str = None):
    X1_test = test_data["X1_validation"]
    X1_test_scaled = test_data["X1_validation_scaled"]

    expiry_x_tensor_test = test_data["expiry_x_tensor_validation_scaled"]
    expiry_y_tensor_test = test_data["expiry_y_tensor_validation"]

    lower_x_tensor_test = test_data["lower_x_tensor_validation_scaled"]
    lower_y_tensor_test = test_data["lower_y_tensor_validation"]

    upper_x_tensor_test = test_data["upper_x_tensor_validation_scaled"]
    upper_y_tensor_test = test_data["upper_y_tensor_validation"]

    if analytical_solution_filename is None:
        analytical_solution = dataloader.get_analytical_solution(
            X1_test[:, 1:], X1_test[:, 0])  # .cpu().detach().numpy()
    else:
        analytical_solution = np.load(analytical_solution_filename)

    analytical_solution = analytical_solution.reshape(
        analytical_solution.shape[0], -1)

    MSE = nn.MSELoss()
    RMSE = 0
    total_number_of_elements = 0

    with torch.no_grad():
        predicted_pde = model(X1_test_scaled).cpu().detach().numpy()

        predicted_expiry = model(expiry_x_tensor_test)

        predicted_lower = model(lower_x_tensor_test)

        predicted_upper = model(upper_x_tensor_test)

    RMSE = np.square(np.subtract(
        analytical_solution, predicted_pde)).mean() * analytical_solution.size
    total_number_of_elements += analytical_solution.size

    RMSE += MSE(expiry_y_tensor_test,
                predicted_expiry).item() * torch.numel(predicted_expiry)
    total_number_of_elements += torch.numel(predicted_expiry)

    RMSE += MSE(lower_y_tensor_test,
                predicted_lower).item()*torch.numel(predicted_lower)
    total_number_of_elements += torch.numel(predicted_lower)

    RMSE += MSE(upper_y_tensor_test,
                predicted_upper).item() * torch.numel(predicted_upper)
    total_number_of_elements += torch.numel(predicted_upper)

    RMSE /= total_number_of_elements
    RMSE = np.sqrt(RMSE)
    return RMSE


def try_multiple_activation_functions(config: dict, dataloader, PDE, filename1: str, filename2: str, activation_functions: list, layers: list, validation_data: dict, test_data, analytical_solution_filename: str = None, epochs=600_000):
    mse_data = np.zeros((len(activation_functions), len(layers)))
    used_epochs = np.zeros((len(activation_functions), len(layers)))

    MSE = nn.MSELoss()
    for i, activation_function in enumerate(activation_functions):
        for j, layer in enumerate(layers):
            model = PINNforwards(N_INPUT=config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=128, N_LAYERS=layer, activation_function=activation_function,
                                 use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
            model.to(DEVICE)
            model.train(True)
            epoch = train(model, epochs, config["learning_rate"], dataloader,
                          config, "", PDE, validation_data)

            model.train(False)
            model.eval()

            mse_data[i, j] = compute_test_loss(
                model=model, test_data=test_data, dataloader=dataloader, analytical_solution_filename=analytical_solution_filename)

            used_epochs[i, j] = epoch

            print(f"Trial {j + i*len(layers) + 1} / {len(layers)
                  * len(activation_functions)} Done")
            print("RMSE", mse_data[i, j])
    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, used_epochs)


def try_different_learning_rates(config: dict, dataloader, PDE, filename1: str, filename2: str, learning_rates: list, batch_sizes: list, validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 600_000, custom_arc: list[int] = None):
    cur_config = copy.deepcopy(config)

    epoch_data = np.zeros((len(learning_rates), len(batch_sizes)))
    mse_data = np.zeros((len(learning_rates), len(batch_sizes)))

    if custom_arc is None:
        model = PINNforwards(N_INPUT=config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=128,
                             N_LAYERS=4, use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
    else:
        model = PINNforwards(N_INPUT=config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=None, N_LAYERS=None,
                             use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"],
                             encoded_size=config["fourier_encoded_size"], custom_arc=custom_arc)

    start_model = copy.deepcopy(model.state_dict())

    for i, learning_rate in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            cur_config["N_sample"] = batch_size
            model.load_state_dict(start_model)

            model.train(True)
            best_epoch = train(model, epochs, learning_rate, dataloader, cur_config, f"learning_{
                               learning_rate}_{batch_size}", PDE, validation_data)

            model.train(False)
            model.eval()

            mse_data[i, j] = compute_test_loss(
                model=model, test_data=test_data, dataloader=dataloader, analytical_solution_filename=analytical_solution_filename)

            epoch_data[i, j] = best_epoch

            print(f"Trial {j + i*len(batch_sizes) +
                  1} / {len(batch_sizes)*len(learning_rates)} Done")
            print("RMSE", mse_data[i, j])
    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, epoch_data)


def try_different_architectures(config: dict, dataloader, PDE, filename1: str, filename2: str, layers: list, nodes: list, validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 600_000):
    epoch_data = np.zeros((len(layers), len(nodes)))
    mse_data = np.zeros((len(layers), len(nodes)))

    MSE = nn.MSELoss()

    for i, layer in enumerate(layers):
        for j, node in enumerate(nodes):
            model = PINNforwards(N_INPUT=config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=node,
                                 N_LAYERS=layer, use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
            model.train(True)
            best_epoch = train(
                model, epochs, config["learning_rate"], dataloader, config, "", PDE, validation_data)

            model.train(False)
            model.eval()

            mse_data[i, j] = compute_test_loss(
                model=model, test_data=test_data, dataloader=dataloader, analytical_solution_filename=analytical_solution_filename)

            epoch_data[i, j] = best_epoch

            print(f"Trial {j + i*len(nodes) +
                  1} / {len(layers)*len(nodes)} Done")
            print("RMSE", mse_data[i, j])

    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, epoch_data)


def train_multiple_times(seeds: list[int], layers: int, nodes: int, PDE, filename: str,
                         nr_of_epochs: int, dataloader, config: dict, validation_data: dict, test_data: dict, analytical_solution_filename: str = None, custom_arc: list[int] = None):
    cur_config = config.copy()
    cur_config["save_loss"] = True
    cur_config["save_model"] = False

    types_of_loss = ["total_loss", "loss_boundary",
                     "loss_pde", "loss_expiry", "loss_lower", "loss_upper"]
    if config["american_option"]:
        types_of_loss = types_of_loss + ["loss_free_boundary"]

    results_train = np.zeros(
        (len(seeds), nr_of_epochs // config["epochs_before_loss_saved"], len(types_of_loss)))
    results_val = np.zeros(
        (len(seeds), nr_of_epochs // config["epochs_before_validation_loss_saved"], len(types_of_loss)))

    mse_data = np.zeros(len(seeds))

    best_test_MSE = float("inf")
    best_model = None

    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PINNforwards(N_INPUT=cur_config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=nodes, N_LAYERS=layers,
                             use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"],
                             encoded_size=config["fourier_encoded_size"], custom_arc=custom_arc)

        _ = train(model, nr_of_epochs, config["learning_rate"], dataloader, cur_config, str(
            seed), PDE, validation_data)

        cur_train = np.load(f"results/loss_{seed}.npy")
        cur_val = np.load(f"results/validation_{seed}.npy")

        results_train[i] = cur_train
        results_val[i] = cur_val

        model.train(False)
        model.eval()

        mse_data[i] = compute_test_loss(
            model=model, test_data=test_data, dataloader=dataloader, analytical_solution_filename=analytical_solution_filename)

        os.remove(f"results/loss_{seed}.npy")
        os.remove(f"results/validation_{seed}.npy")

        if mse_data[i] < best_test_MSE:
            best_test_MSE = mse_data[i]
            best_model = copy.deepcopy(model.state_dict())

        print(f"Run {i + 1} / {len(seeds)} done")
        print(f"RMSE : {mse_data[i]:.2e} \n")

    if config["save_model"]:
        torch.save(best_model, f"models/" + filename + ".pth")

    np.save("results/average_loss_" +
            filename, np.vstack([results_train.mean(axis=0), results_train.std(axis=0)]))

    np.save("results/average_validation_" +
            filename, np.vstack([results_val.mean(axis=0), results_val.std(axis=0)]))
    np.savetxt("results/rmse_data_" + filename + ".txt", mse_data)


def trying_weight_decay(config: dict, dataloader, PDE, filename1: str, filename2: str, weight_decays: list, validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 250_000):
    cur_config = copy.deepcopy(config)

    epoch_data = np.zeros((1, len(weight_decays)))
    mse_data = np.zeros((1, len(weight_decays)))

    model = PINNforwards(N_INPUT=config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=128,
                         N_LAYERS=4, use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
    start_model = copy.deepcopy(model.state_dict())

    for i, weight_decay in enumerate(weight_decays):
        cur_config["weight_decay"] = weight_decay
        model.load_state_dict(start_model)
        model.train(True)
        best_epoch = train(model, epochs, config["learning_rate"], dataloader, cur_config, f"wd_{
                           weight_decay}", PDE, validation_data)

        model.train(False)
        model.eval()

        mse_data[0, i] = compute_test_loss(
            model=model, test_data=test_data, dataloader=dataloader, analytical_solution_filename=analytical_solution_filename)

        epoch_data[0, i] = best_epoch

        print(f"Trial {i} /  {len(weight_decays)} Done")
        print("RMSE", mse_data[0, i])
    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, epoch_data)


def try_different_lambdas(config: dict, dataloader, PDE, filename1: str, filename2: str, lambdas: list[list[float]], validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 600_000):
    cur_config = copy.deepcopy(config)

    epoch_data = np.zeros(len(lambdas))
    mse_data = np.zeros(len(lambdas))

    model = PINNforwards(N_INPUT=config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=128,
                         N_LAYERS=4, use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
    start_model = copy.deepcopy(model.state_dict())

    for i, cur in enumerate(lambdas):

        cur_config["lambda_pde"] = cur[0]
        cur_config["lambda_boundary"] = cur[1]
        cur_config["lambda_expiry"] = cur[2]
        # cur_config["lambda_exercise"] = cur[3]

        model.load_state_dict(start_model)
        model.train(True)
        best_epoch = train(
            model, epochs, cur_config["learning_rate"], dataloader, cur_config, "", PDE, validation_data)

        model.train(False)
        model.eval()

        mse_data[i] = compute_test_loss(
            model=model, test_data=test_data, dataloader=dataloader, analytical_solution_filename=analytical_solution_filename)

        epoch_data[i] = best_epoch

        print(f"Trial {i} Done")
        print("RMSE", mse_data[i])
    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, epoch_data)


def try_sigma_fourier_and_embedding_size(config: dict, dataloader, PDE, filename1: str, filename2: str, sigma_fourier: list, embedding_size: list, validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 200_000):
    cur_config = copy.deepcopy(config)

    epoch_data = np.zeros((len(sigma_fourier), len(embedding_size)))
    mse_data = np.zeros((len(sigma_fourier), len(embedding_size)))

    for i, sg_fourier in enumerate(sigma_fourier):
        for j, em_size in enumerate(embedding_size):
            cur_config["sigma_fourier"] = sg_fourier
            cur_config["fourier_encoded_size"] = em_size

            model = PINNforwards(N_INPUT=config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=128,
                                 N_LAYERS=4, use_fourier_transform=config["use_fourier_transform"], sigma_FF=sg_fourier, encoded_size=em_size)
            model.train(True)
            best_epoch = train(
                model, epochs, config["learning_rate"], dataloader, cur_config, "", PDE, validation_data)

            model.train(False)
            model.eval()

            mse_data[i, j] = compute_test_loss(
                model=model, test_data=test_data, dataloader=dataloader, analytical_solution_filename=analytical_solution_filename)

            epoch_data[i, j] = best_epoch

            print(f"Trial {j + i*len(embedding_size) +
                  1} / {len(embedding_size)*len(sigma_fourier)} Done")
            print("RMSE", mse_data[i, j])
    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, epoch_data)


def computing_the_greeks(config: dict, dataloader, PDE, filename: str, validation_data: dict, test_data: dict, epochs: int = 250_000):
    X1_test = test_data["X1_validation"]
    X1_test_scaled = test_data["X1_validation_scaled"]

    S = X1_test[:, 1].view(-1, 1)
    t = X1_test[:, 0].view(-1, 1)

    r = config["r"]
    sigma = config["sigma"]
    K = config["K"]
    T = config["t_range"][-1]

    t2m = T-t

    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2)
          * t2m) / (sigma * torch.sqrt(t2m))

    d2 = d1 - sigma * torch.sqrt(t2m)

    standard_normal = Normal(0, 1)

    analytical_delta = standard_normal.cdf(d1).cpu().detach().numpy()
    analytical_gamma = (standard_normal_pdf(
        d1) / (S * sigma * t2m)).cpu().detach().numpy()
    analytical_rho = (t2m * K * torch.exp(-r * t2m) *
                      standard_normal.cdf(d2)).cpu().detach().numpy()

    analytical_theta = (-S * sigma / (2 * torch.sqrt(t2m)) * standard_normal_pdf(
        d1) - r * K * torch.exp(- r * t2m) * standard_normal.cdf(d2)).cpu().detach().numpy()

    analytical_nu = (S * torch.sqrt(t2m) *
                     standard_normal_pdf(d1)).cpu().detach().numpy()

    model = PINNforwards(N_INPUT=config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=128,
                         N_LAYERS=4, use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
    # model.load_state_dict(torch.load("models/greeks.pth", weights_only=True))
    model.train(True)
    best_epoch = train(
        model, epochs, config["learning_rate"], dataloader, config, "greeks", PDE, validation_data)
    model.train(False)
    model.eval()
    # print(analytical_delta.shape, analytical_gamma.shape, analytical_rho.shape, analytical_theta.shape, analytical_nu.shape)
    delta, gamma, theta, nu, rho = model.estimate_greeks_call(
        X1_test_scaled, X1_test, sigma, T)

    delta = delta.cpu().detach().numpy()
    gamma = gamma.cpu().detach().numpy()
    theta = theta.cpu().detach().numpy()
    nu = nu.cpu().detach().numpy()
    rho = rho.cpu().detach().numpy()

    """ plt.plot(rho.flatten())
    plt.savefig("plots/pred_rho.png")
    plt.clf()
    plt.plot(analytical_rho.flatten())
    plt.savefig("plots/rho.png") """
    # print(np.max(analytical_delta), np.max(analytical_gamma), np.max(analytical_theta), np.max(analytical_nu), np.max(analytical_rho))
    # print(np.min(analytical_delta), np.min(analytical_gamma), np.min(analytical_theta), np.min(analytical_nu), np.min(analytical_rho))

    with open(filename, 'w') as outfile:
        outfile.write(f"EPOCHS : {best_epoch}\n")
        outfile.write(f"DELTA  : {RMSE_numpy(
            analytical_delta, delta):.2e} \n")
        outfile.write(f"GAMMA  : {RMSE_numpy(
            analytical_gamma, gamma):.2e} \n")
        outfile.write(f"THETA  : {RMSE_numpy(
            analytical_theta, theta):.2e} \n")
        outfile.write(
            f"NU     : {RMSE_numpy(analytical_nu, nu):.2e} \n")
        outfile.write(f"RHO    : {RMSE_numpy(analytical_rho, rho):.2e}")


def experiment_with_binomial_model(M_values: list[int], dataloader, test_data, filename1: str, filename2: str):
    tmp_X = test_data["X1_validation"].cpu().detach().numpy()
    prev = np.zeros(tmp_X.shape[0])

    timings = np.zeros(len(M_values))
    error = np.zeros(len(M_values))

    for i, M in enumerate(M_values):
        start_time = time.time()
        cur = dataloader.get_analytical_solution(
            tmp_X[:, 1], tmp_X[:, 0], M=M)
        end_time = time.time()

        timings[i] = end_time - start_time
        error[i] = RMSE_numpy(cur, prev)
        prev = copy.deepcopy(cur)

    np.savetxt(filename1, error)
    np.savetxt(filename2, timings)


def try_multiple_dimensions(dimensions: list, config, PDE, filename1: str, filename2: str):
    cur_config = copy.deepcopy(config)

    epoch_data = np.zeros((1, len(dimensions)))
    rmse_data = np.zeros((1, len(dimensions)))

    for i, d in enumerate(dimensions):
        cur_S_range = np.array([[0, 20] for i in range(d)])
        cur_sigma = np.full((d, d), 0.15)

        cur_config["sigma"] = cur_sigma
        cur_config["sigma_torch"] = torch.tensor(
            cur_config["sigma"]).to(DEVICE)
        cur_config["S_range"] = cur_S_range
        cur_config["N_INPUT"] = d + 1
        # print(cur_S_range)
        # print(cur_sigma)

        dataloader_val = DataGeneratorEuropeanMultiDimensional(
            time_range=config["t_range"], S_range=cur_S_range, K=cur_config["K"], r=cur_config["r"], sigma=cur_sigma, DEVICE=DEVICE, seed=2024)
        validation_data = create_validation_data(
            dataloader=dataloader_val, N_validation=1024, config=cur_config)

        test_data = create_validation_data(
            dataloader=dataloader_val, N_validation=20_000, config=cur_config)

        torch.manual_seed(2025)
        np.random.seed(2025)
        dataloader = DataGeneratorEuropeanMultiDimensional(
            time_range=config["t_range"], S_range=cur_S_range, K=cur_config["K"], r=cur_config["r"], sigma=cur_sigma, DEVICE=DEVICE, seed=2025)

        model = PINNforwards(N_INPUT=d + 1, N_OUTPUT=1, N_HIDDEN=128,
                             N_LAYERS=4, use_fourier_transform=cur_config["use_fourier_transform"], sigma_FF=cur_config["sigma_fourier"], encoded_size=cur_config["fourier_encoded_size"])

        model.train(True)
        best_epoch = train(
            model, 800_000, config["learning_rate"], dataloader, cur_config, f"d_{d}", PDE, validation_data)

        model.train(False)
        model.eval()

        rmse_data[0, i] = compute_test_loss(
            model=model, test_data=test_data, dataloader=dataloader, analytical_solution_filename=None)

        epoch_data[0, i] = best_epoch

        print(f"Trial {i + 1} /  {len(dimensions)} Done")
        print("RMSE", rmse_data[0, i])
    np.savetxt(filename1, rmse_data)
    np.savetxt(filename2, epoch_data)
