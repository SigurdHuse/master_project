from PINN import PINNforwards
from data_generator import DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
from typing import Callable, Union


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


def black_scholes_multi_dimensional(y_hat: torch.tensor, X1: torch.tensor, config: dict) -> torch.tensor:
    """Computes the PDE residual for the multi-dimensional Black-Scholes PDE

    Args:
        y_hat (torch.tensor): Predicted option prices
        X1 (torch.tensor):    Input points to network
        config (dict):        Dictionary with hyperparameters

    Returns:
        torch.tensor:         Numerically approximated PDE residual in input points
    """

    sigma = config["sigma_torch"]
    r = config["r"]

    S = X1[:, 1:]   # shape: [N, n]
    N, n = S.shape

    # A tensor of ones for use as grad_outputs.
    ones = torch.ones_like(y_hat)

    # Compute gradients of y_hat with respect to all components of X1.
    # grad_all has shape [N, 1+n]: first column for t, remaining for S.
    grad_all = torch.autograd.grad(
        y_hat, X1, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    # Extract time derivative
    V_t = grad_all[:, :1]  # shape: [N, 1]

    # Extract first derivatives with respect to asset prices
    V_S = grad_all[:, 1:]  # shape: [N, n]

    # Compute second derivatives with respect to asset prices.
    # We want V_SS[:, i,j] = \partial^2 V/(\partial S_i  \partial S_j) for i,j=1,...,n.
    V_SS = torch.zeros(N, n, n, device=X1.device)
    for i in range(n):
        # Compute gradient of V_S[:, i] with respect to X1.
        grad_i = torch.autograd.grad(
            V_S[:, i:i+1], X1, grad_outputs=torch.ones_like(V_S[:, i:i+1]),
            create_graph=True, retain_graph=True
        )[0]
        # The asset derivatives are in the columns 1: (ignoring time).
        V_SS[:, i, :] = grad_i[:, 1:]

    # First-order term: sum_i r * S_i * (∂V/∂S_i)
    term1 = r * torch.sum(S * V_S, dim=1, keepdim=True)

    term2 = 0
    for i in range(n):
        for j in range(n):
            sigma_eff = 0
            for k in range(n):
                sigma_eff += sigma[i, k] * sigma[j, k]

            tmp = S[:, i]*S[:, j]
            tmp *= V_SS[:, i, j]
            tmp *= sigma_eff
            term2 += tmp

    term2 *= 0.5
    term2 = term2.view(-1, 1)

    # PDE residual: V_t + (first-order term) + (second-order term) - r * V.
    residual = V_t + term1 + term2 - r * y_hat
    return residual


def black_scholes_1D(y1_hat: torch.tensor, X1: torch.tensor, config: dict) -> torch.tensor:
    """Computes the PDE residual for the Black-Scholes PDE

    Args:
        y_hat (torch.tensor): Predicted option prices
        X1 (torch.tensor):    Input points to network
        config (dict):        Dictionary with hyperparameters

    Returns:
        torch.tensor:         Numerically approximated PDE residual in input points
    """

    sigma = config["sigma"]
    r = config["r"]

    grads = torch.autograd.grad(y1_hat, X1, grad_outputs=torch.ones(y1_hat.shape).to(
        DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
    dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

    grads2nd = torch.autograd.grad(dVdS, X1, grad_outputs=torch.ones(
        dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
    d2VdS2 = grads2nd[:, 1].view(-1, 1)

    S1 = X1[:, 1].view(-1, 1)
    residual = dVdt + (0.5 * ((sigma**2) * (S1 ** 2)) * d2VdS2) + \
        (r * S1 * dVdS) - (r * y1_hat)
    return residual


def black_scholes_american_1D(y1_hat: torch.tensor, X1: torch.tensor, config: dict) -> torch.tensor:
    """Computes the PDE residual for the American put Black-Scholes PDE

    Args:
        y_hat (torch.tensor): Predicted option prices
        X1 (torch.tensor):    Input points to network
        config (dict):        Dictionary with hyperparameters

    Returns:
        torch.tensor:         Numerically approximated PDE residual in input points
    """

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

    # Put option payoff function
    yint = torch.max(K - S1, torch.zeros_like(S1))
    free_pde = y1_hat - yint

    residual = bs_pde * free_pde

    # Price should always be bigger than the immidieate payoff
    free_boundary = torch.min(free_pde, torch.zeros_like(y1_hat))
    return residual, free_boundary


def create_validation_data(dataloader:  Union[DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D],
                           N_validation: int,
                           config: dict) -> dict:
    """Creates a dictionary containing validation data

    Args:
        dataloader Union[DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D]:   Dataloader used to generate points
        N_validation (int):                     Number of points to sample, recall that this is scaled different for different regions.
        config (dict):                          Dictionary with hyperparameters

    Returns:
        dict: Dictionary with sampled scaled points and targets for inner domain and boundary,
              expiry_x_tensor_validation - Point from expiry at t = T.
              expiry_y_tensor_validation - Analytical solution at expiry.
              lower_x_tensor_validation  - Point from S = S_min.
              lower_y_tensor_validation  - Analytical solution at S= S_min.
              upper_x_tensor_validation  - Point from S = S_max.
              upper_y_tensor_validation  - Analytical solution at S= S_max.
              X1_validation              - Points from the inner domain.
              y1_validation              - Just a tensor with zeros, as the PDE residual should equal zero.
    """

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


def train_one_epoch(model: PINNforwards,
                    dataloader: Union[DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D],
                    loss_function: Callable[[torch.tensor, torch.tensor], torch.tensor],
                    optimizer: torch.optim.Optimizer,
                    config: dict,
                    loss_history: dict,
                    PDE: Callable[[torch.tensor, torch.tensor], torch.tensor]) -> None:
    """Performs one epoch of training

    Args:
        model (PINNforwards):               Model currently being trained.
        dataloader (Union[DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D]): Dataloader used to sample points.
        loss_function (Callable[[torch.tensor, torch.tensor], torch.tensor]): Loss function from torch.nn 
        optimizer (torch.optim.Optimizer):  The optimizer for updating the model's parameters.
        config (dict): _description_        Dictionary with hyperparameters.
        loss_history (dict):                Dictionary used to store the loss history.
        PDE (Callable):                     Function which computes the PDE residual for the inner domain points.
    """
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

    X1_scaled = dataloader.normalize(X1)  # config["encoder"](X1)

    y1_hat = model(X1_scaled)

    if config["american_option"]:
        bs_pde, free_boundary = PDE(y1_hat, X1, config)

        loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
        loss_free_boundary = loss_function(
            free_boundary, torch.zeros_like(free_boundary))
        loss = loss_boundary + \
            config["lambda_pde"] * loss_pde + \
            config["lambda_pde"] * loss_free_boundary
    else:
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

    if config["american_option"]:
        loss_free_boundary = loss_free_boundary.to("cpu").detach()

    if config["epoch"] % config["epochs_before_loss_saved"] == 0:
        loss_history["total_loss"].append(
            mse_expiry.item() + mse_lower.item() + mse_upper.item() + loss_pde.item())
        loss_history["loss_boundary"].append(
            mse_lower.item() + mse_upper.item())
        loss_history["loss_pde"].append(loss_pde.item())
        loss_history["loss_expiry"].append(mse_expiry.item())
        loss_history["loss_lower"].append(mse_lower.item())
        loss_history["loss_upper"].append(mse_upper.item())

        if config["american_option"]:
            loss_history["loss_free_boundary"].append(
                loss_free_boundary.item())


def train(model: PINNforwards,
          nr_of_epochs: int,
          learning_rate: float,
          dataloader: Union[DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D],
          config: dict,
          filename: str,
          PDE: Callable[[torch.tensor, torch.tensor], torch.tensor],
          validation_data: dict = {},
          final_learning_rate: float = 1e-5) -> int:
    """Main training function

    Args:
        model (PINNforwards):   Model currently being trained.
        nr_of_epochs (int):     Number of epochs to train model for.
        learning_rate (float):  Initial learning rate to use in optimizer.
        dataloader (Union[DataGeneratorEuropean1D, DataGeneratorEuropeanMultiDimensional, DataGeneratorAmerican1D]): Dataloader used to generate training data.
        config (dict):          Dictionary with hyperparameters
        filename (str):         Filename to store loss as
        PDE (Callable[[torch.tensor, torch.tensor], torch.tensor]): Function which computes the PDE residual for the inner domain points.
        validation_data (dict, optional): Dictionary containing validation data. Defaults to {}.
        final_learning_rate (float, optional): Final learning rate. Defaults to 1e-5.

    Returns:
        int: Best validation epoch
    """

    epochs_before_validation = config["epochs_before_validation"]
    n = np.log(final_learning_rate / learning_rate) / np.log(config["gamma"])

    scheduler_step = int(nr_of_epochs // n)

    # Make sure we do not modulo w.r.t 0
    if scheduler_step <= 0:
        scheduler_step = nr_of_epochs

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])

    scheduler = ExponentialLR(optimizer, config["gamma"])

    types_of_loss = ["total_loss", "loss_boundary",
                     "loss_pde", "loss_expiry", "loss_lower", "loss_upper"]
    if config["american_option"]:
        types_of_loss = types_of_loss + ["loss_free_boundary"]

    loss_history = {i: [] for i in types_of_loss}
    loss_history_validation = {i: [] for i in types_of_loss}

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
                loss_boundary = config["lambda_expiry"] * \
                    mse_expiry + config["lambda_boundary"] * \
                    (mse_lower + mse_upper)/2

            y1_hat = model(X1_validation_scaled)

            if config["american_option"]:
                bs_pde, free_boundary = PDE(y1_hat, X1_validation, config)

                loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
                loss_free_boundary = loss_function(
                    free_boundary, torch.zeros_like(free_boundary))

                loss_pde = config["lambda_pde"]*loss_pde + \
                    config["lambda_pde"]*loss_free_boundary

                loss = loss_boundary + loss_pde
            else:
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

            if config["american_option"]:
                loss_free_boundary = loss_free_boundary.to("cpu").detach()

            if epoch % config["epochs_before_validation_loss_saved"] == 0:
                loss_history_validation["total_loss"].append(loss.item())
                loss_history_validation["loss_boundary"].append(
                    loss_boundary.item())
                loss_history_validation["loss_pde"].append(loss_pde.item())
                loss_history_validation["loss_expiry"].append(
                    mse_expiry.item())
                loss_history_validation["loss_lower"].append(mse_lower.item())
                loss_history_validation["loss_upper"].append(mse_upper.item())

                if config["american_option"]:
                    loss_history_validation["loss_free_boundary"].append(
                        loss_free_boundary.item())

            if loss.item() < best_validation:
                best_validation = loss.item()
                best_validation_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

    validation_array = np.zeros(
        (nr_of_epochs // config["epochs_before_validation_loss_saved"], len(types_of_loss)))
    loss_array = np.zeros(
        (nr_of_epochs // config["epochs_before_loss_saved"], len(types_of_loss)))

    for i, name in enumerate(types_of_loss):
        validation_array[:, i] = loss_history_validation[name]
        loss_array[:, i] = loss_history[name]

    if config["save_loss"]:
        np.save("results/loss_" + filename, loss_array)
        np.save("results/validation_" +
                filename, validation_array)

    if config["save_model"]:
        torch.save(
            best_model, f"models/" + filename + ".pth")

    # Load best model based on validation data
    model.load_state_dict(best_model)

    return best_validation_epoch
