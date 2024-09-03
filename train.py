from PINN import PINN
from dataloader import DataloaderEuropean1D
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

seed = 2024
DEVICE = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


# TODO find out what happens with expiry_x_tensor (what is r_ivp? r_ivp is weight connected to ivp compared to boundary data)
# TODO find what BVP1-penalty is

def create_validation_data(dataloader: DataloaderEuropean1D, N_validation: int, config: dict) -> dict:
    w_expiry = config["w_expiry"]
    w_lower = config["w_lower"]
    w_upper = config["w_upper"]

    validation_data = {}

    expiry_x_tensor_validation, expiry_y_tensor_validation = dataloader.get_expiry_time_tensor(
        N_validation, w_expiry)
    validation_data["expiry_x_tensor_validation"] = expiry_x_tensor_validation.to(
        DEVICE)
    validation_data["expiry_y_tensor_validation"] = expiry_y_tensor_validation.to(
        DEVICE)

    lower_x_tensor_validation, lower_y_tensor_validation, upper_x_tensor_validation, upper_y_tensor_validation = dataloader.get_boundary_data_tensor(
        N_validation, w_lower, w_upper)

    validation_data["lower_x_tensor_validation"] = lower_x_tensor_validation.to(
        DEVICE),
    validation_data["lower_y_tensor_validation"] = lower_y_tensor_validation.to(
        DEVICE)

    validation_data["upper_x_tensor_validation"] = upper_x_tensor_validation.to(
        DEVICE),
    validation_data["upper_y_tensor_validation"] = upper_y_tensor_validation.to(
        DEVICE)

    X1_validation, y1_validation = dataloader.get_pde_data(N_validation)
    validation_data["X1_validation"] = torch.from_numpy(
        X1_validation).float().requires_grad_().to(DEVICE)

    validation_data["y1_validation"] = torch.from_numpy(
        y1_validation).float().to(DEVICE)

    return validation_data


def train_one_epoch_european_1D(model, dataloader, loss_function, optimizer, config, loss_history):
    model.train()

    w_expiry = config["w_expiry"]
    w_lower = config["w_lower"]
    w_upper = config["w_upper"]
    N_sample = config["N_sample"]
    pde_learning_rate = config["pde_learning_rate"]
    BVP1_PENALTY = config["BVP1_PENALTY"]
    sigma = config["sigma"]
    r = config["r"]
    # Expiry time data
    expiry_x_tensor, expiry_y_tensor = dataloader.get_expiry_time_tensor(
        N_sample, w_expiry)
    expiry_x_tensor = expiry_x_tensor.to(DEVICE)
    expiry_y_tensor = expiry_y_tensor.to(DEVICE)

    expiry_y_pred = model(expiry_x_tensor)
    mse_expiry = loss_function(expiry_y_tensor, expiry_y_pred)

    # Get boundary data
    lower_x_tensor, lower_y_tensor, upper_x_tensor, upper_y_tensor = dataloader.get_boundary_data_tensor(
        N_sample, w_lower, w_upper)
    lower_x_tensor, lower_y_tensor = lower_x_tensor.to(
        DEVICE), lower_y_tensor.to(DEVICE)

    upper_x_tensor, upper_y_tensor = upper_x_tensor.to(
        DEVICE), upper_y_tensor.to(DEVICE)

    lower_y_pred = model(lower_x_tensor)
    mse_lower = loss_function(lower_y_tensor, lower_y_pred)

    upper_y_pred = model(upper_x_tensor)
    mse_upper = loss_function(upper_y_tensor, upper_y_pred)

    # Loss for boundary conditions
    loss_boundary = mse_expiry + BVP1_PENALTY*mse_lower + mse_upper

    # Compute the "Black-Scholes loss"
    X1, y1 = dataloader.get_pde_data(N_sample)
    X1 = torch.from_numpy(X1).float().requires_grad_().to(DEVICE)
    y1 = torch.from_numpy(y1).float().to(DEVICE)
    y1_hat = model(X1)
    grads = torch.autograd.grad(y1_hat, X1, grad_outputs=torch.ones(y1_hat.shape).to(
        DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
    dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
    grads2nd = torch.autograd.grad(dVdS, X1, grad_outputs=torch.ones(
        dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
    d2VdS2 = grads2nd[:, 1].view(-1, 1)
    S1 = X1[:, 1].view(-1, 1)
    bs_pde = dVdt + (0.5 * ((sigma * S1) ** 2) * d2VdS2) + \
        (r * S1 * dVdS) - (r * y1_hat)
    loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))

    # Backpropagate joint loss
    loss = loss_boundary + pde_learning_rate * loss_pde

    loss_history["total_loss"].append(loss.item())
    loss_history["loss_boundary"].append(loss_boundary.item())
    loss_history["loss_pde"].append(loss_pde.item())
    loss_history["loss_expiry"].append(mse_expiry.item())
    loss_history["loss_lower"].append(mse_lower.item())
    loss_history["loss_upper"].append(mse_upper.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(model, nr_of_epochs: int, learning_rate: float, dataloader, config: dict, filename: str, validation_data: dict = {}):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    types_of_loss = ["total_loss", "loss_boundary",
                     "loss_pde", "loss_expiry", "loss_lower", "loss_upper"]
    loss_history = {i: [] for i in types_of_loss}
    loss_history_validation = {i: [] for i in types_of_loss}

    BVP1_PENALTY = config["BVP1_PENALTY"]
    pde_learning_rate = config["pde_learning_rate"]
    sigma = config["sigma"]
    r = config["r"]
    loss_function = nn.MSELoss()
    best_validation = float("inf")
    best_validation_epoch = 0
    best_model = None

    expiry_x_tensor_validation = validation_data["expiry_x_tensor_validation"]
    expiry_y_tensor_validation = validation_data["expiry_y_tensor_validation"]

    lower_x_tensor_validation,  = validation_data["lower_x_tensor_validation"]
    lower_y_tensor_validation = validation_data["lower_y_tensor_validation"]

    upper_x_tensor_validation,  = validation_data["upper_x_tensor_validation"]
    upper_y_tensor_validation = validation_data["upper_y_tensor_validation"]

    X1_validation = validation_data["X1_validation"]
    y1_validation = validation_data["y1_validation"]

    for epoch in tqdm(range(nr_of_epochs)):
        model.train(True)
        train_one_epoch_european_1D(
            model, dataloader, loss_function, optimizer, config, loss_history)
        model.train(False)
        model.eval()

        expiry_y_pred = model(expiry_x_tensor_validation)
        mse_expiry = loss_function(
            expiry_y_tensor_validation, expiry_y_pred)

        lower_y_pred = model(lower_x_tensor_validation)
        mse_lower = loss_function(lower_y_tensor_validation, lower_y_pred)

        upper_y_pred = model(upper_x_tensor_validation)
        mse_upper = loss_function(upper_y_tensor_validation, upper_y_pred)

        loss_boundary = mse_expiry + BVP1_PENALTY*mse_lower + mse_upper

        y1_hat = model(X1_validation)
        grads = torch.autograd.grad(y1_hat, X1_validation, grad_outputs=torch.ones(y1_hat.shape).to(
            DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
        dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
        grads2nd = torch.autograd.grad(dVdS, X1_validation, grad_outputs=torch.ones(
            dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
        d2VdS2 = grads2nd[:, 1].view(-1, 1)
        S1 = X1_validation[:, 1].view(-1, 1)
        bs_pde = dVdt + (0.5 * ((sigma * S1) ** 2) * d2VdS2) + \
            (r * S1 * dVdS) - (r * y1_hat)
        loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))

        loss = loss_boundary + pde_learning_rate * loss_pde
        loss_history_validation["total_loss"].append(loss.item())
        loss_history_validation["loss_boundary"].append(
            loss_boundary.item())
        loss_history_validation["loss_pde"].append(loss_pde.item())
        loss_history_validation["loss_expiry"].append(mse_expiry.item())
        loss_history_validation["loss_lower"].append(mse_lower.item())
        loss_history_validation["loss_upper"].append(mse_upper.item())

        if loss.item() < best_validation:
            best_validation = loss.item()
            best_validation_epoch = epoch + 1
            best_model = model.state_dict()

    validation_array = np.zeros((nr_of_epochs, len(types_of_loss)))
    loss_array = np.zeros((nr_of_epochs, len(types_of_loss)))

    for i, name in enumerate(types_of_loss):
        validation_array[:, i] = loss_history_validation[name]
        loss_array[:, i] = loss_history[name]

    np.savetxt("results/loss_" + filename + ".txt", loss_array)
    np.savetxt("results/validation_" + filename + ".txt", validation_array)
    torch.save(
        best_model, f"models/epoch_{best_validation_epoch}_" + filename + ".pth")


if __name__ == "__main__":
    torch.manual_seed(2024)

    config = {}
    LEARNING_RATE = 1e-5
    HIDDEN_LAYER = 6
    HIDDEN_WIDTH = 256

    K = 40
    r = 0.05
    sigma = 0.2
    T = 1
    S_range = [0, 160]
    t_range = [0, T]
    dataloader = DataloaderEuropean1D(t_range, S_range, K, r, sigma)
    model = PINN(2, 1, HIDDEN_WIDTH, HIDDEN_LAYER).to(DEVICE)

    config["w_expiry"] = 1
    config["w_lower"] = 1
    config["w_upper"] = 1
    config["N_sample"] = 8000
    config["pde_learning_rate"] = 52
    config["BVP1_PENALTY"] = 8
    config["sigma"] = 0.2
    config["r"] = 0.05

    validation_data = create_validation_data(dataloader, 20_000, config)

    train(model, 1_000, LEARNING_RATE, dataloader,
          config, "test", validation_data)
