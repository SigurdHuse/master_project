from PINN import PINNforwards
from data_generator import DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy

import os
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


def black_scholes_multi_dimensional(y_hat, X1, config):
    sigma = config["sigma"]
    r = config["r"]

    t = X1[:, 0].unsqueeze(1)  # Extract the first column as time t
    # Extract the remaining columns as asset prices S
    S = X1[:, 1:]

    # Compute first derivatives
    grads = torch.autograd.grad(outputs=y_hat, inputs=X1,
                                grad_outputs=torch.ones_like(y_hat), create_graph=True)
    # First derivative w.r.t. each asset price (ignore time dimension)
    dV_dS = grads[0][:, 1:]
    dV_dt = grads[0][:, 0].unsqueeze(1)  # First derivative w.r.t. time

    # Compute second derivatives and cross terms
    d2V_dS2 = torch.stack([torch.autograd.grad(dV_dS[:, i], X1, grad_outputs=torch.ones_like(dV_dS[:, i]), create_graph=True)[0][:, i+1]
                           # Diagonal second derivatives w.r.t. each asset price
                           for i in range(S.shape[1])], dim=1)

    # Mixed second derivatives using correlation matrix
    cross_term = torch.zeros_like(y_hat)
    for i in range(S.shape[1]):
        for j in range(i + 1, S.shape[1]):
            d2V_dSij = torch.autograd.grad(dV_dS[:, i], X1, grad_outputs=torch.ones_like(
                dV_dS[:, i]), create_graph=True)[0][:, j+1]

            cross_term += (sigma[i, j] * S[:, i] *
                           S[:, j] * d2V_dSij).view(-1, 1)

    diffusion_term = 0.5 * torch.sum(torch.stack(
        [sigma[i, i] * S[:, i]**2 * d2V_dS2[:, i] for i in range(S.shape[1])], dim=1), dim=1) + cross_term

    drift_term = r * torch.sum(S * dV_dS, dim=1)
    bs_pde = dV_dt + diffusion_term + drift_term - r * y_hat.squeeze()

    return bs_pde


def black_scholes_1D(y1_hat, X1, config):
    sigma = config["sigma"]
    r = config["r"]

    grads = torch.autograd.grad(y1_hat, X1, grad_outputs=torch.ones(y1_hat.shape).to(
        DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
    dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

    grads2nd = torch.autograd.grad(dVdS, X1, grad_outputs=torch.ones(
        dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
    d2VdS2 = grads2nd[:, 1].view(-1, 1)

    S1 = X1[:, 1].view(-1, 1)
    bs_pde = dVdt + (0.5 * ((sigma**2) * (S1 ** 2)) * d2VdS2) + \
        (r * S1 * dVdS) - (r * y1_hat)

    return bs_pde


def black_scholes_american_1D(y1_hat, X1, config):
    sigma = config["sigma"]
    r = config["r"]
    K = config["K"]

    grads = torch.autograd.grad(y1_hat, X1, grad_outputs=torch.ones(y1_hat.shape).to(
        DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
    dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
    grads2nd = torch.autograd.grad(dVdS, X1, grad_outputs=torch.ones(
        dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
    d2VdS2 = grads2nd[:, 1].view(-1, 1)
    S1 = X1[:, 1].view(-1, 1)
    bs_pde = dVdt + (0.5 * ((sigma * S1) ** 2) * d2VdS2) + \
        (r * S1 * dVdS) - (r * y1_hat)

    # free region: option exercise immediately
    yint = torch.max(K - S1, torch.zeros_like(S1))
    free_pde = yint - y1_hat

    combined_pde = bs_pde*free_pde
    # print(type(combined_pde[1]))
    return combined_pde


def create_validation_data(dataloader:
                           DataGeneratorEuropean1D, N_validation: int, config: dict) -> dict:
    w_expiry = config["w_expiry"]
    w_lower = config["w_lower"]
    w_upper = config["w_upper"]

    validation_data = {}

    expiry_x_tensor_validation, expiry_y_tensor_validation = dataloader.get_expiry_time_tensor(
        N_validation, w_expiry)
    expiry_x_tensor_validation_scaled = dataloader.normalize(
        expiry_x_tensor_validation)
    # config["encoder"](expiry_x_tensor_validation)

    validation_data["expiry_x_tensor_validation"] = expiry_x_tensor_validation.to(
        DEVICE)
    validation_data["expiry_x_tensor_validation_scaled"] = expiry_x_tensor_validation_scaled.to(
        DEVICE)
    validation_data["expiry_y_tensor_validation"] = expiry_y_tensor_validation.to(
        DEVICE)

    lower_x_tensor_validation, lower_y_tensor_validation, upper_x_tensor_validation, upper_y_tensor_validation = dataloader.get_boundary_data_tensor(
        N_validation, w_lower, w_upper)
    lower_x_tensor_validation_scaled = dataloader.normalize(
        lower_x_tensor_validation)

    # config["encoder"](
    #    lower_x_tensor_validation)
    upper_x_tensor_validation_scaled = dataloader.normalize(
        upper_x_tensor_validation)

    # config["encoder"](    upper_x_tensor_validation)

    validation_data["lower_x_tensor_validation"] = lower_x_tensor_validation.to(
        DEVICE)
    validation_data["lower_x_tensor_validation_scaled"] = lower_x_tensor_validation_scaled.to(
        DEVICE)
    validation_data["lower_y_tensor_validation"] = lower_y_tensor_validation.to(
        DEVICE)

    validation_data["upper_x_tensor_validation"] = upper_x_tensor_validation.to(
        DEVICE)
    validation_data["upper_x_tensor_validation_scaled"] = upper_x_tensor_validation_scaled.to(
        DEVICE)
    validation_data["upper_y_tensor_validation"] = upper_y_tensor_validation.to(
        DEVICE)

    X1_validation, y1_validation = dataloader.get_pde_data_tensor(
        N_validation)
    X1_validation_scaled = dataloader.normalize(X1_validation)
    # config["encoder"](X1_validation)

    validation_data["X1_validation"] = X1_validation

    validation_data["X1_validation_scaled"] = X1_validation_scaled

    validation_data["y1_validation"] = y1_validation

    return validation_data


def train_one_epoch(model, dataloader, loss_function, optimizer, config, loss_history, PDE):
    model.train()

    w_expiry = config["w_expiry"]
    w_lower = config["w_lower"]
    w_upper = config["w_upper"]
    N_sample = config["N_sample"]

    # Expiry time data
    expiry_x_tensor, expiry_y_tensor = dataloader.get_expiry_time_tensor(
        N_sample, w_expiry)

    # config["encoder"](expiry_x_tensor)
    expiry_x_tensor = dataloader.normalize(expiry_x_tensor)

    expiry_y_pred = model(expiry_x_tensor)
    mse_expiry = loss_function(expiry_y_tensor, expiry_y_pred)

    # Get boundary data
    lower_x_tensor, lower_y_tensor, upper_x_tensor, upper_y_tensor = dataloader.get_boundary_data_tensor(
        N_sample, w_lower, w_upper)

    # config["encoder"](lower_x_tensor)
    lower_x_tensor = dataloader.normalize(lower_x_tensor)

    # config["encoder"](upper_x_tensor)
    upper_x_tensor = dataloader.normalize(upper_x_tensor)

    lower_y_pred = model(lower_x_tensor)
    mse_lower = loss_function(lower_y_tensor, lower_y_pred)

    upper_y_pred = model(upper_x_tensor)
    mse_upper = loss_function(upper_y_tensor, upper_y_pred)

    # Loss for boundary conditions
    # We have to divide by 2 to get the MSE of the boundary condition as we both upper and lower
    loss_boundary = config["lambda_expiry"] * mse_expiry + \
        config["lambda_boundary"] * (mse_lower + mse_upper)/2

    # Compute the "Black-Scholes loss"
    X1, y1 = dataloader.get_pde_data_tensor(N_sample)

    X1_scaled = dataloader.normalize(X1)  # config["encoder"](X1)

    y1_hat = model(X1_scaled)

    bs_pde = PDE(y1_hat, X1, config)

    loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
    # Backpropagate joint loss
    loss = loss_boundary + config["lambda_pde"] * loss_pde

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.to("cpu").detach()
    loss_boundary = loss_boundary.to("cpu").detach()
    loss_pde = loss_pde.to("cpu").detach()
    mse_expiry = mse_expiry.to("cpu").detach()
    mse_lower = mse_lower.to("cpu").detach()
    mse_upper = mse_upper.to("cpu").detach()

    if config["epoch"] % config["epochs_before_loss_saved"] == 0:
        loss_history["total_loss"].append(
            mse_expiry.item() + mse_lower.item() + mse_upper.item() + loss_pde.item())
        loss_history["loss_boundary"].append(
            mse_lower.item() + mse_upper.item())
        loss_history["loss_pde"].append(loss_pde.item())
        loss_history["loss_expiry"].append(mse_expiry.item())
        loss_history["loss_lower"].append(mse_lower.item())
        loss_history["loss_upper"].append(mse_upper.item())

    """ if config["epoch"] % config["update_lambda"] == 0:
        # We have to divide by 2 to get the MSE of the boundary condition
        mse_boundary = (mse_lower.item() + mse_upper.item())/2

        mse_sum = mse_boundary + mse_expiry.item() + loss_pde.item()

        new_lambda_boundary = mse_boundary / max(mse_sum, 1e-6)
        new_lambda_expiry = mse_expiry.item() / max(mse_sum, 1e-6)
        new_lambda_pde = loss_pde.item() / max(mse_sum, 1e-6)

        alpha = config["alpha_lambda"]
        config["lambda_boundary"] = alpha * \
            config["lambda_boundary"] + (1 - alpha) * new_lambda_boundary

        config["lambda_expiry"] = alpha * \
            config["lambda_expiry"] + (1 - alpha) * new_lambda_expiry

        config["lambda_pde"] = alpha * \
            config["lambda_pde"] + (1 - alpha) * new_lambda_pde

        for lambda_name in ["lambda_pde", "lambda_boundary", "lambda_expiry"]:
            loss_history[lambda_name].append(config[lambda_name])

        # print(config["lambda_pde"], config["lambda_boundary"],
        #      config["lambda_expiry"]) """


def train(model, nr_of_epochs: int, learning_rate: float, dataloader, config: dict, filename: str, PDE, validation_data: dict = {}, final_learning_rate=1e-5):
    epochs_before_validation = config["epochs_before_validation"]
    # config["lambda_pde"] = 1
    # config["lambda_boundary"] = 1
    # config["lambda_expiry"] = 1
    n = np.log(final_learning_rate / learning_rate) / np.log(config["gamma"])

    scheduler_step = int(nr_of_epochs // n)

    # Make sure we do not modulo w.r.t 0
    if scheduler_step == 0:
        scheduler_step = nr_of_epochs

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])

    scheduler = ExponentialLR(optimizer, config["gamma"])

    types_of_loss = ["total_loss", "loss_boundary",
                     "loss_pde", "loss_expiry", "loss_lower", "loss_upper"]
    loss_history = {i: [] for i in types_of_loss}
    loss_history_validation = {i: [] for i in types_of_loss}

    """ for lambda_name in ["lambda_pde", "lambda_boundary", "lambda_expiry"]:
        loss_history[lambda_name] = [config[lambda_name]] """

    loss_function = nn.MSELoss()
    best_validation = float("inf")
    best_validation_epoch = 0
    best_model = None

    expiry_x_tensor_validation = validation_data["expiry_x_tensor_validation_scaled"]
    expiry_y_tensor_validation = validation_data["expiry_y_tensor_validation"]

    lower_x_tensor_validation = validation_data["lower_x_tensor_validation_scaled"]
    lower_y_tensor_validation = validation_data["lower_y_tensor_validation"]

    upper_x_tensor_validation = validation_data["upper_x_tensor_validation_scaled"]
    upper_y_tensor_validation = validation_data["upper_y_tensor_validation"]

    X1_validation = validation_data["X1_validation"]
    X1_validation_scaled = validation_data["X1_validation_scaled"]
    y1_validation = validation_data["y1_validation"]

    for epoch in tqdm(range(1, nr_of_epochs + 1), miniters=10_000, maxinterval=10_000):
        config["epoch"] = epoch

        model.train(True)
        train_one_epoch(
            model, dataloader, loss_function, optimizer, config, loss_history, PDE)

        if epoch % scheduler_step == 0:
            scheduler.step()
            # print(f"learning rate : {optimizer.param_groups[0]['lr']}")

        model.train(False)
        model.eval()

        if epoch % epochs_before_validation == 0:
            with torch.no_grad():
                expiry_y_pred = model(expiry_x_tensor_validation)
                mse_expiry = loss_function(
                    expiry_y_tensor_validation, expiry_y_pred)

                lower_y_pred = model(lower_x_tensor_validation)
                mse_lower = loss_function(
                    lower_y_tensor_validation, lower_y_pred)

                upper_y_pred = model(upper_x_tensor_validation)
                mse_upper = loss_function(
                    upper_y_tensor_validation, upper_y_pred)

                # We have to divide by 2 to get the MSE of the boundary condition
                loss_boundary = mse_expiry + (mse_lower + mse_upper)/2

            y1_hat = model(X1_validation_scaled)
            bs_pde = PDE(y1_hat, X1_validation, config)

            loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))

            loss = loss_boundary + loss_pde

            # Make sure the validation prediction does not affect the training
            optimizer.zero_grad()

            loss = loss.to("cpu").detach()
            loss_boundary = loss_boundary.to("cpu").detach()
            loss_pde = loss_pde.to("cpu").detach()
            mse_expiry = mse_expiry.to("cpu").detach()
            mse_lower = mse_lower.to("cpu").detach()
            mse_upper = mse_upper.to("cpu").detach()

            bs_pde = bs_pde.to("cpu").detach()

            if epoch % config["epochs_before_validation_loss_saved"] == 0:
                loss_history_validation["total_loss"].append(loss.item())
                loss_history_validation["loss_boundary"].append(
                    loss_boundary.item())
                loss_history_validation["loss_pde"].append(loss_pde.item())
                loss_history_validation["loss_expiry"].append(
                    mse_expiry.item())
                loss_history_validation["loss_lower"].append(mse_lower.item())
                loss_history_validation["loss_upper"].append(mse_upper.item())

            if loss.item() < best_validation:
                best_validation = loss.item()
                # print("best validation", best_validation)
                best_validation_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

    validation_array = np.zeros(
        (nr_of_epochs // config["epochs_before_validation_loss_saved"], len(types_of_loss)))
    loss_array = np.zeros(
        (nr_of_epochs // config["epochs_before_loss_saved"], len(types_of_loss)))
    # lambda_values = np.zeros((nr_of_epochs // config["update_lambda"] + 1, 3))

    for i, name in enumerate(types_of_loss):
        validation_array[:, i] = loss_history_validation[name]
        loss_array[:, i] = loss_history[name]

    """ for i, lambda_name in enumerate(["lambda_pde", "lambda_boundary", "lambda_expiry"]):
        lambda_values[:, i] = loss_history[lambda_name] """

    if config["save_loss"]:
        np.save("results/loss_" + filename, loss_array)
        np.save("results/validation_" +
                filename, validation_array)
        # np.save("results/lambda_values_" + filename, lambda_values)

    if config["save_model"]:
        torch.save(
            best_model, f"models/" + filename + ".pth")

    # Load best model based on validation data
    model.load_state_dict(best_model)

    return best_validation_epoch

    """ if config["save_model"]:
        torch.save(best_model, f"models/" + filename + ".pth")

    np.save("results/average_loss_" +
            filename, np.vstack([results_train.mean(axis=0), results_train.std(axis=0)]))

    np.save("results/average_validation_" +
            filename, np.vstack([results_val.mean(axis=0), results_val.std(axis=0)]))
    np.savetxt("results/rmse_data_" + filename + ".txt", mse_data) """


if __name__ == "__main__":
    torch.manual_seed(2024)
    np.random.seed(2024)

    config = {}

    # S_range = [0, 100]
    # t_range = [0, 1]
    # S_range = np.array([[0, 100] for i in range(5)])
    # sigma = np.ones((5, 5))

    # dataloader = DataloaderEuropean1D(t_range, S_range, K, r, sigma, DEVICE)

    config["w_expiry"] = 1
    config["w_lower"] = 1
    config["w_upper"] = 1
    config["N_sample"] = 1_000
    config["pde_learning_rate"] = 60
    config["K"] = 40
    config["t_range"] = [0, 1]
    config["S_range"] = [0, 100]
    # config["BVP1_PENALTY"] = 8
    config["sigma"] = 0.25
    config["r"] = 0.04
    config["learning_rate"] = 1e-4
    config["save_model"] = True
    config["save_loss"] = True

    """ try_different_learning_rates(
        config, DataloaderAmerican1D, black_scholes_american_1D, "important_results/mse_data_learning_rates_american.txt", "important_results/epoch_data_learning_rates_american.txt") """

    config["S_range"] = np.array([[0, 100], [0, 10], [0, 20], [0, 200]])
    config["sigma"] = np.array([[1, 0.1, 0.1, 0.1], [0.1, 1, 0.1, 0.1],
                                [0.1, 0.1, 1, 0.1], [0.1, 0.1, 0.1, 1]])

    # try_multiple_activation_function_european_1D(
    #    config, DataloaderEuropeanMultiDimensional, black_scholes_multi_dimensional, "important_results/mse_data_activation_multi.txt")

    """ model = PINNforwards(2, 1, 300, 5)
    dataloader = DataloaderAmerican1D([0, 1], [0, 100], 40, 0.04, 0.25, DEVICE)
    validation_data = create_validation_data(dataloader, 2_000, config)
    train(model, 10_000, 1e-4, dataloader, config, "america_test",
          black_scholes_american_1D, validation_data) """

    """ try_different_learning_rates(
        config, DataloaderEuropean1D, black_scholes_1D, "important_results/mse_data_learning_rates.txt", "important_results/epoch_data_learning_rates.txt")
    try_multiple_activation_function_european_1D(
        config, DataloaderEuropean1D, black_scholes_1D, "important_results/different_activation.txt") """

    """ model = PINNforwards(2, 1, 500, 2).to(DEVICE)
    validation_data = create_validation_data(dataloader, 1_000, config)
    train(model, 10_000, 1e-4, dataloader, config,
          "test_analytical", black_scholes_1D, validation_data) """

    """ for nr_of_layers in [1, 2, 3, 4, 5, 6, 7]:
        for width in [10, 20, 40, 80, 160, 320, 560]:
            filename = f"multi_{nr_of_layers}_{width}"
            model = PINNforwards(6, 1, width, nr_of_layers).to(DEVICE)
            train(model, 30_000, LEARNING_RATE, dataloader, config,
                  filename, black_scholes_multi_dimensional, validation_data) """

    """ for N in [100, 500, 1_000, 2_000, 4_000, 8_000, 16_000]:
        config["N_sample"] = N
        filename = f"test_N_{N}"
        model = PINNforwards(2, 1, HIDDEN_WIDTH, HIDDEN_LAYER).to(DEVICE)
        train(model, 30_000, LEARNING_RATE, dataloader,
            config, filename, black_scholes_1D, validation_data) """

    """ X1_validation = validation_data["X1_validation"]
    analytical_solution = dataloader.get_analytical_solution(
        X1_validation[:, 1], X1_validation[:, 0]).cpu().detach().numpy()

    np.savetxt("results/analytical_" + filename +
               ".txt", analytical_solution)
    np.savetxt("results/validation_data.txt",
               X1_validation.cpu().detach().numpy()) """
