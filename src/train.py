from PINN import PINNforwards
from data_generator import DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
from torch.distributions import Normal
import os
import matplotlib.pyplot as plt

seed = 2024
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

    combined_pde = bs_pde*free_pde,
    # print(type(combined_pde[1]))
    return combined_pde[0]


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
    upper_x_tensor_validation_scaled = dataloader.normalize(
        upper_x_tensor_validation)

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

    validation_data["X1_validation"] = X1_validation

    validation_data["X1_validation_scaled"] = X1_validation_scaled

    validation_data["y1_validation"] = y1_validation

    return validation_data


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
            X1_test[:, 1], X1_test[:, 0]).cpu().detach().numpy()
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


def MASE(target: np.array, prediction: np.array):
    MAE = np.abs(target.flatten() - prediction.flatten()).mean()
    AMD = np.abs(target.flatten() - target.flatten().mean()).mean()

    return 100 * MAE / AMD


def RMSE_numpy(target: np.array, prediction: np.array):
    return np.sqrt(np.square(np.subtract(target.flatten(), prediction.flatten())).mean())


def relative_L2_error(target: np.array, prediction: np.array):
    difference = np.linalg.norm(target - prediction, ord=2)
    return difference / np.linalg.norm(target, ord=2)


def standard_normal_pdf(x):
    return (1 / torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-x.pow(2) / 2)


def train_one_epoch(model, dataloader, loss_function, optimizer, config, loss_history, PDE):
    model.train()

    w_expiry = config["w_expiry"]
    w_lower = config["w_lower"]
    w_upper = config["w_upper"]
    N_sample = config["N_sample"]

    # Expiry time data
    expiry_x_tensor, expiry_y_tensor = dataloader.get_expiry_time_tensor(
        N_sample, w_expiry)

    expiry_x_tensor = dataloader.normalize(expiry_x_tensor)

    expiry_y_pred = model(expiry_x_tensor)
    mse_expiry = loss_function(expiry_y_tensor, expiry_y_pred)

    # Get boundary data
    lower_x_tensor, lower_y_tensor, upper_x_tensor, upper_y_tensor = dataloader.get_boundary_data_tensor(
        N_sample, w_lower, w_upper)

    lower_x_tensor = dataloader.normalize(lower_x_tensor)
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

    X1_scaled = dataloader.normalize(X1)

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


def try_multiple_activation_functions(config: dict, dataloader, PDE, filename1: str, filename2: str, activation_functions: list, layers: list, validation_data: dict, test_data, analytical_solution_filename: str = None, epochs=150_000):
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
            X1_test[:, 1], X1_test[:, 0]).cpu().detach().numpy()
    else:
        analytical_solution = np.load(analytical_solution_filename)

    analytical_solution = analytical_solution.reshape(
        analytical_solution.shape[0], -1)
    mse_data = np.zeros((len(activation_functions), len(layers)))
    used_epochs = np.zeros((len(activation_functions), len(layers)))

    MSE = nn.MSELoss()
    for i, activation_function in enumerate(activation_functions):
        for j, layer in enumerate(layers):
            model = PINNforwards(
                config["N_INPUT"], 1, 128, layer, activation_function)
            model.to(DEVICE)
            model.train(True)
            epoch = train(model, epochs, config["learning_rate"], dataloader,
                          config, "", PDE, validation_data)

            model.train(False)
            model.eval()
            with torch.no_grad():
                predicted_pde = model(X1_test_scaled).cpu().detach().numpy()
                predicted_expiry = model(expiry_x_tensor_test)
                predicted_lower = model(lower_x_tensor_test)
                predicted_upper = model(upper_x_tensor_test)

            total_number_of_elements = 0
            mse_data[i, j] = np.square(np.subtract(
                analytical_solution, predicted_pde)).mean() * analytical_solution.size
            total_number_of_elements += analytical_solution.size

            mse_data[i, j] += MSE(expiry_y_tensor_test,
                                  predicted_expiry).item() * torch.numel(predicted_expiry)
            total_number_of_elements += torch.numel(predicted_expiry)

            mse_data[i, j] += MSE(lower_y_tensor_test,
                                  predicted_lower).item()*torch.numel(predicted_lower)
            total_number_of_elements += torch.numel(predicted_lower)

            mse_data[i, j] += MSE(upper_y_tensor_test,
                                  predicted_upper).item() * torch.numel(predicted_upper)
            total_number_of_elements += torch.numel(predicted_upper)

            mse_data[i, j] /= total_number_of_elements
            mse_data[i, j] = np.sqrt(mse_data[i, j])

            used_epochs[i, j] = epoch

            print(f"Trial {j + i*len(layers) + 1} / {len(layers)
                  * len(activation_functions)} Done")
            print("RMSE", mse_data[i, j])
    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, used_epochs)


def try_different_learning_rates(config: dict, dataloader, PDE, filename1: str, filename2: str, learning_rates: list, batch_sizes: list, validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 200_000):
    cur_config = copy.deepcopy(config)

    epoch_data = np.zeros((len(learning_rates), len(batch_sizes)))
    mse_data = np.zeros((len(learning_rates), len(batch_sizes)))

    model = PINNforwards(config["N_INPUT"], 1, 128, 4)
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


def try_different_architectures(config: dict, dataloader, PDE, filename1: str, filename2: str, layers: list, nodes: list, validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 250_000):
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
            X1_test[:, 1], X1_test[:, 0]).cpu().detach().numpy()
    else:
        analytical_solution = np.load(analytical_solution_filename)

    analytical_solution = analytical_solution.reshape(
        analytical_solution.shape[0], -1)

    epoch_data = np.zeros((len(layers), len(nodes)))
    mse_data = np.zeros((len(layers), len(nodes)))

    MSE = nn.MSELoss()

    for i, layer in enumerate(layers):
        for j, node in enumerate(nodes):
            model = PINNforwards(config["N_INPUT"], 1, node, layer)
            model.train(True)
            best_epoch = train(
                model, epochs, config["learning_rate"], dataloader, config, "", PDE, validation_data)

            model.train(False)
            model.eval()

            with torch.no_grad():
                predicted_pde = model(X1_test_scaled).cpu().detach().numpy()
                predicted_expiry = model(expiry_x_tensor_test)
                predicted_lower = model(lower_x_tensor_test)
                predicted_upper = model(upper_x_tensor_test)

            total_number_of_elements = 0
            mse_data[i, j] = np.square(np.subtract(
                analytical_solution, predicted_pde)).mean() * analytical_solution.size
            total_number_of_elements += analytical_solution.size

            mse_data[i, j] += MSE(expiry_y_tensor_test,
                                  predicted_expiry).item() * torch.numel(predicted_expiry)
            total_number_of_elements += torch.numel(predicted_expiry)

            mse_data[i, j] += MSE(lower_y_tensor_test,
                                  predicted_lower).item()*torch.numel(predicted_lower)
            total_number_of_elements += torch.numel(predicted_lower)

            mse_data[i, j] += MSE(upper_y_tensor_test,
                                  predicted_upper).item() * torch.numel(predicted_upper)
            total_number_of_elements += torch.numel(predicted_upper)

            mse_data[i, j] /= total_number_of_elements
            mse_data[i, j] = np.sqrt(mse_data[i, j])

            epoch_data[i, j] = best_epoch

            print(f"Trial {j + i*len(nodes) +
                  1} / {len(layers)*len(nodes)} Done")
            print("RMSE", mse_data[i, j])

    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, epoch_data)


def train_multiple_times(seeds: list[int], layers: int, nodes: int, PDE, filename: str,
                         nr_of_epochs: int, dataloader, config: dict, validation_data: dict, test_data: dict, analytical_solution_filename: str = None):
    cur_config = config.copy()
    cur_config["save_loss"] = True

    types_of_loss = ["total_loss", "loss_boundary",
                     "loss_pde", "loss_expiry", "loss_lower", "loss_upper"]

    results_train = np.zeros((len(seeds), nr_of_epochs, len(types_of_loss)))
    results_val = np.zeros(
        (len(seeds), nr_of_epochs // config["epochs_before_validation"], len(types_of_loss)))

    mse_data = np.zeros(len(seeds))

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
            X1_test[:, 1], X1_test[:, 0]).cpu().detach().numpy()
    else:
        analytical_solution = np.load(analytical_solution_filename)

    analytical_solution = analytical_solution.reshape(
        analytical_solution.shape[0], -1)

    MSE = nn.MSELoss()
    best_test_MSE = float("inf")
    best_model = None

    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PINNforwards(cur_config["N_INPUT"], 1, nodes, layers)
        train(model, nr_of_epochs, config["learning_rate"], dataloader, cur_config, str(
            seed), PDE, validation_data)

        cur_train = np.load(f"results/loss_{seed}.npy")
        cur_val = np.load(f"results/validation_{seed}.npy")

        results_train[i] = cur_train
        results_val[i] = cur_val

        model.train(False)
        model.eval()

        with torch.no_grad():
            predicted_pde = model(X1_test_scaled).cpu().detach().numpy()
            predicted_expiry = model(expiry_x_tensor_test)
            predicted_lower = model(lower_x_tensor_test)
            predicted_upper = model(upper_x_tensor_test)

        total_number_of_elements = 0
        mse_data[i] = np.square(np.subtract(
            analytical_solution, predicted_pde)).mean() * analytical_solution.size
        total_number_of_elements += analytical_solution.size

        mse_data[i] += MSE(expiry_y_tensor_test,
                           predicted_expiry).item() * torch.numel(predicted_expiry)
        total_number_of_elements += torch.numel(predicted_expiry)

        mse_data[i] += MSE(lower_y_tensor_test,
                           predicted_lower).item()*torch.numel(predicted_lower)
        total_number_of_elements += torch.numel(predicted_lower)

        mse_data[i] += MSE(upper_y_tensor_test,
                           predicted_upper).item() * torch.numel(predicted_upper)
        total_number_of_elements += torch.numel(predicted_upper)

        mse_data[i] /= total_number_of_elements
        mse_data[i] = np.sqrt(mse_data[i])

        os.remove(f"results/loss_{seed}.npy")
        os.remove(f"results/validation_{seed}.npy")

        if mse_data[i] < best_test_MSE:
            best_test_MSE = mse_data[i]
            best_model = copy.deepcopy(model.state_dict())
        print(f"Run {i} / {len(seeds)} done \n")

    if config["save_model"]:
        torch.save(best_model, f"models/" + filename + ".pth")

    np.save("results/average_loss_" +
            filename, np.vstack([results_train.mean(axis=0), results_train.std(axis=0)]))

    np.save("results/average_validation_" +
            filename, np.vstack([results_val.mean(axis=0), results_val.std(axis=0)]))
    np.savetxt("results/mse_data_" + filename + ".txt", mse_data)


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

    model = PINNforwards(config["N_INPUT"], 1, 128, 4)
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


def trying_weight_decay(config: dict, dataloader, PDE, filename1: str, filename2: str, weight_decays: list, validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 250_000):
    cur_config = copy.deepcopy(config)

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
            X1_test[:, 1], X1_test[:, 0]).cpu().detach().numpy()
    else:
        analytical_solution = np.load(analytical_solution_filename)

    analytical_solution = analytical_solution.reshape(
        analytical_solution.shape[0], -1)

    epoch_data = np.zeros((1, len(weight_decays)))
    mse_data = np.zeros((1, len(weight_decays)))

    model = PINNforwards(config["N_INPUT"], 1, 128, 4)
    start_model = copy.deepcopy(model.state_dict())

    MSE = nn.MSELoss()
    for i, weight_decay in enumerate(weight_decays):
        cur_config["weight_decay"] = weight_decay
        model.load_state_dict(start_model)
        model.train(True)
        best_epoch = train(model, epochs, config["learning_rate"], dataloader, cur_config, f"wd_{
                           weight_decay}", PDE, validation_data)

        model.train(False)
        model.eval()

        with torch.no_grad():
            predicted_pde = model(X1_test_scaled).cpu().detach().numpy()
            predicted_expiry = model(expiry_x_tensor_test)
            predicted_lower = model(lower_x_tensor_test)
            predicted_upper = model(upper_x_tensor_test)

        total_number_of_elements = 0
        mse_data[0, i] = np.square(np.subtract(
            analytical_solution, predicted_pde)).mean() * analytical_solution.size
        total_number_of_elements += analytical_solution.size

        mse_data[0, i] += MSE(expiry_y_tensor_test,
                              predicted_expiry).item() * torch.numel(predicted_expiry)
        total_number_of_elements += torch.numel(predicted_expiry)

        mse_data[0, i] += MSE(lower_y_tensor_test,
                              predicted_lower).item()*torch.numel(predicted_lower)
        total_number_of_elements += torch.numel(predicted_lower)

        mse_data[0, i] += MSE(upper_y_tensor_test,
                              predicted_upper).item() * torch.numel(predicted_upper)
        total_number_of_elements += torch.numel(predicted_upper)

        mse_data[0, i] /= total_number_of_elements
        mse_data[0, i] = np.sqrt(mse_data[0, i])

        epoch_data[0, i] = best_epoch

        print(f"Trial {i} /  {len(weight_decays)} Done")
        print("RMSE", mse_data[0, i])
    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, epoch_data)


def try_different_lambdas(config: dict, dataloader, PDE, filename1: str, filename2: str, lambdas: list[list[float]], validation_data: dict, test_data: dict, analytical_solution_filename: str = None, epochs: int = 600_000):
    cur_config = copy.deepcopy(config)

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
            X1_test[:, 1], X1_test[:, 0]).cpu().detach().numpy()
    else:
        analytical_solution = np.load(analytical_solution_filename)

    analytical_solution = analytical_solution.reshape(
        analytical_solution.shape[0], -1)

    epoch_data = np.zeros(len(lambdas))
    mse_data = np.zeros(len(lambdas))

    model = PINNforwards(config["N_INPUT"], 1, 128, 4)
    start_model = copy.deepcopy(model.state_dict())

    MSE = nn.MSELoss()
    for i, cur in enumerate(lambdas):

        config["lambda_pde"] = cur[0]
        config["lambda_boundary"] = cur[1]
        config["lambda_expiry"] = cur[2]

        model.load_state_dict(start_model)
        model.train(True)
        best_epoch = train(
            model, epochs, cur_config["learning_rate"], dataloader, cur_config, "", PDE, validation_data)

        model.train(False)
        model.eval()

        with torch.no_grad():
            predicted_pde = model(X1_test_scaled).cpu().detach().numpy()
            predicted_expiry = model(expiry_x_tensor_test)
            predicted_lower = model(lower_x_tensor_test)
            predicted_upper = model(upper_x_tensor_test)

        total_number_of_elements = 0
        mse_data[i] = np.square(np.subtract(
            analytical_solution, predicted_pde)).mean() * analytical_solution.size
        total_number_of_elements += analytical_solution.size

        mse_data[i] += MSE(expiry_y_tensor_test,
                           predicted_expiry).item() * torch.numel(predicted_expiry)
        total_number_of_elements += torch.numel(predicted_expiry)

        mse_data[i] += MSE(lower_y_tensor_test,
                           predicted_lower).item()*torch.numel(predicted_lower)
        total_number_of_elements += torch.numel(predicted_lower)

        mse_data[i] += MSE(upper_y_tensor_test,
                           predicted_upper).item() * torch.numel(predicted_upper)
        total_number_of_elements += torch.numel(predicted_upper)

        mse_data[i] /= total_number_of_elements
        mse_data[i] = np.sqrt(mse_data[i])

        epoch_data[i] = best_epoch

        print(f"Trial {i} Done")
        print("RMSE", mse_data[i])
    np.savetxt(filename1, mse_data)
    np.savetxt(filename2, epoch_data)


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
