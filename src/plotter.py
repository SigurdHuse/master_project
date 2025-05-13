from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from torch.distributions import Normal
from train import DEVICE, create_validation_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import torch
import numpy as np
from PINN import PINNforwards
from data_generator import DataGeneratorAmerican1D, DataGeneratorEuropean1D
from matplotlib.colors import LogNorm, SymLogNorm
import experiments_european_one_dimensional as one_euro
import experiments_american_one_dimensional as one_american
from training_functions import compute_test_loss
import matplotlib.ticker as ticker
from typing import Callable, Union

torch.set_default_device(DEVICE)


mpl.rcParams["figure.titlesize"] = 22
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["legend.fontsize"] = "large"
mpl.rcParams["xtick.labelsize"] = 15
mpl.rcParams["ytick.labelsize"] = 15
mpl.rcParams["figure.dpi"] = 1_300


def get_analytical_solution(S: torch.tensor, t: torch.tensor, t_range: list[float], sigma: float, r: float, K: float) -> torch.tensor:
    """Computes the analytical solution of a European call option

    Args:
        S (torch.tensor):       Asset prices
        t (torch.tensor):       Times
        t_range (list[float]):  Time range
        sigma (float):          Volatility
        r (float):              Risk-free rate
        K (float):              Strike price

    Returns:
        torch.tensor: Computed analytical price
    """

    T = t_range[-1]
    t2m = T-t  # Time to maturity
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2)
          * t2m) / (sigma * torch.sqrt(t2m))

    d2 = d1 - sigma * torch.sqrt(t2m)

    # Normal cumulative distribution function (CDF)
    standard_normal = Normal(0, 1)

    Nd1 = standard_normal.cdf(d1)
    Nd2 = standard_normal.cdf(d2)

    # Calculate the option price
    F = S * Nd1 - K * Nd2 * torch.exp(-r * t2m)
    return F


def make_training_plot(filename: str) -> None:
    """Makes training plot comparing using Fourier to not using Fourier for a 1D European call"""

    X_loss_fourier = np.load("results/average_loss_with_fourier.npy")
    X_loss = np.load("results/average_loss_no_fourier.npy")

    n_loss = X_loss_fourier.shape[0]

    X_validation_fourier = np.load(
        "results/average_validation_with_fourier.npy")
    X_validation = np.load("results/average_validation_no_fourier.npy")

    n_val = X_validation_fourier.shape[0]

    skip = 100
    plot_every = 1
    x_loss = np.arange(600*skip, n_loss//2 * 600, 600*plot_every)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(x_loss, X_loss_fourier[skip:n_loss//2:plot_every, 0],
               label="Fourier", color="midnightblue")
    ax[0].plot(x_loss, X_loss[skip:n_loss//2:plot_every, 0],
               label="No Fourier", color="red")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[0].set_title("Training loss")
    ax[0].grid()
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plot_every = 1
    x_val = np.arange(600*skip, n_val//2 * 600, 600*plot_every)

    ax[1].plot(x_val, X_validation_fourier[skip:n_val//2:plot_every,
               0], label="Fourier", color="midnightblue")
    ax[1].plot(x_val, X_validation[skip:n_val//2:plot_every, 0],
               label="No Fourier", color="red")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Epoch")
    # as[1].legend()
    ax[1].set_title("Validation loss")
    ax[1].grid()
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    fig.tight_layout()
    plt.savefig(filename)


def standard_normal_pdf(x: torch.tensor) -> torch.tensor:
    """Computes normal distribution pdf for mean = 0 and variance=1

    Args:
        x (torch.tensor): Points to compute pdf for.

    Returns:
        torch.tensor: Computed pdf
    """
    return (1 / torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-x.pow(2) / 2)


def plots_greeks(n: int, time: float, path_to_weights: str, name_of_plot: str) -> None:
    """Plots predicted Greeks versus analytical

    Args:
        n (int):                Number of points to plot for
        time (float):           Time to plot Greeks at
        path_to_weights (str):  Path to weights used in model to predict Greeks.
        name_of_plot (str):     Filename to save plot as
    """

    S_range = [0, 400]
    t_range = [0, 1]
    S = np.linspace(*S_range, n)
    T = np.full(n, time)

    X = torch.tensor(np.column_stack((T, S)),
                     dtype=torch.float, requires_grad=True)
    min_values = torch.tensor(
        [t_range[0], S_range[0]]).to(DEVICE)
    max_values = torch.tensor(
        [t_range[1], S_range[1]]).to(DEVICE)

    X_scaled = (X - min_values) / (max_values - min_values)

    model = PINNforwards(
        2, 1, 128, 4, use_fourier_transform=True, sigma_FF=5.0, encoded_size=128)
    model.load_state_dict(torch.load(path_to_weights, weights_only=True))
    model = model.to(DEVICE)

    r = 0.04
    sigma = 0.5
    K = 40

    t2m = 1.0-T
    t2m = torch.from_numpy(t2m).to(DEVICE)
    S_torch = torch.from_numpy(S).to(DEVICE)

    d1 = (torch.log(S_torch / K) + (r + 0.5 * sigma**2)
          * t2m) / (sigma * torch.sqrt(t2m))

    d2 = d1 - sigma * torch.sqrt(t2m)

    standard_normal = Normal(0, 1)

    analytical_delta = standard_normal.cdf(d1).cpu().detach().numpy()
    analytical_gamma = (standard_normal_pdf(
        d1) / (S_torch * sigma * t2m)).cpu().detach().numpy()
    analytical_rho = (t2m * K * torch.exp(-r * t2m) *
                      standard_normal.cdf(d2)).cpu().detach().numpy()

    analytical_theta = (-S_torch * sigma / (2 * torch.sqrt(t2m)) * standard_normal_pdf(
        d1) - r * K * torch.exp(- r * t2m) * standard_normal.cdf(d2)).cpu().detach().numpy()

    analytical_nu = (S_torch * torch.sqrt(t2m) *
                     standard_normal_pdf(d1)).cpu().detach().numpy()

    delta, gamma, theta, nu, rho = model.estimate_greeks_call(
        X_scaled, X, 0.5, t_range[1])

    Nd1 = standard_normal.cdf(d1)
    Nd2 = standard_normal.cdf(d2)

    # Calculate the option price
    analytical_price = (S_torch * Nd1 - K * Nd2 *
                        torch.exp(-r * t2m)).to("cpu").detach().numpy().flatten()

    price = model(X_scaled)

    delta = delta.to("cpu").detach().numpy().flatten()
    gamma = gamma.to("cpu").detach().numpy().flatten()
    theta = theta.to("cpu").detach().numpy().flatten()
    nu = nu.to("cpu").detach().numpy().flatten()
    rho = rho.to("cpu").detach().numpy().flatten()
    price = price.to("cpu").detach().numpy().flatten()

    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    linestyle = (0, (4, 4))
    linewidth = 3
    # Add line plots to the first row
    ax[0, 0].plot(S, delta, color="midnightblue", linewidth=linewidth)
    ax[0, 0].plot(S, analytical_delta, color="red",
                  linestyle=linestyle, linewidth=linewidth)
    ax[0, 0].set_title(r"$\Delta$")

    ax[0, 1].plot(S, gamma, color="midnightblue",
                  label="Predicted", linewidth=linewidth)
    ax[0, 1].plot(S, analytical_gamma, color="red",
                  linestyle=linestyle, label="Analytical", linewidth=linewidth)
    ax[0, 1].set_title(r"$\Gamma$")
    ax[0, 1].legend()

    ax[0, 2].plot(S, theta, color="midnightblue", linewidth=linewidth)
    ax[0, 2].plot(S, analytical_theta, color="red",
                  linestyle=linestyle, linewidth=linewidth)
    ax[0, 2].set_title(r"$\Theta$")

    ax[1, 0].plot(S, nu, color="midnightblue", linewidth=linewidth)
    ax[1, 0].plot(S, analytical_nu,  color="red",
                  linestyle=linestyle, linewidth=linewidth)
    ax[1, 0].set_title(r"$\nu$")
    ax[1, 0].set_xlabel("Asset price")

    ax[1, 1].plot(S, rho, color="midnightblue", linewidth=linewidth)
    ax[1, 1].plot(S, analytical_rho, color="red",
                  linestyle=linestyle, linewidth=linewidth)
    ax[1, 1].set_title(r"$\rho$")
    ax[1, 1].set_xlabel("Asset price")

    ax[1, 2].plot(S, price, color="midnightblue", linewidth=linewidth)
    ax[1, 2].plot(S, analytical_price, color="red",
                  linestyle=linestyle, linewidth=linewidth)
    ax[1, 2].set_title(r"Pricing function")
    ax[1, 2].set_xlabel("Asset price")

    for i in range(2):
        for j in range(3):
            ax[i, j].grid()

    fig.tight_layout()
    plt.savefig(name_of_plot)


def binomial_plot(filename: str) -> None:
    """Plots RMSE using Binomial and run time

    Args:
        filename (str): Filename to save plot as
    """
    rmse = np.loadtxt("important_results/american_1D/RMSE_binomial.txt")
    timings = np.loadtxt("important_results/american_1D/timings_binomial.txt")

    M_values = np.array([32, 64, 128, 256, 384, 512, 768,
                        1024, 1280, 1536, 1792, 2048])

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))

    ax[0].plot(M_values[1:], rmse[1:], label="RMSE", color="midnightblue")
    ax[0].plot(M_values[1:], 1 / M_values[1:] * rmse[1] * M_values[1]*1.15,
               label=r"$O(M^{-1})$", color="red")
    ax[0].set_yscale('log')
    ax[0].set_xscale("log")
    ax[0].set_ylabel("RMSE")
    ax[0].set_xlabel("M")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(M_values, timings, label="Runtime", color="midnightblue")
    ax[1].plot(M_values, M_values**2 / M_values[0]**2 * timings[0]*1.15,
               label=r"$O(M^2)$", color="red")
    ax[1].set_yscale('log')
    ax[1].set_xscale("log")
    ax[1].set_ylabel("Runtime")
    ax[1].set_xlabel("M")
    ax[1].legend()
    ax[1].grid()

    fig.tight_layout()
    plt.savefig(filename)


def plot_different_loss(name_of_model: str, filename: str, x_values: np.array, values_to_skip: int = 100, skip_every: int = 5, use_average: bool = False, american: bool = False) -> None:
    """Plots training, validation and different loss over epochs

    Args:
        name_of_model (str):            Name of data to plot
        filename (str):                 Filename to save plot as
        x_values (np.array):            Epochs values to plot on x-axis.
        values_to_skip (int, optional): Values to skip plotting at the start. Defaults to 100.
        skip_every (int, optional):     We plot only the points which occur skip_every. Defaults to 5.
        use_average (bool, optional):   Indicates if we are plotting average loss. Defaults to False.
        american (bool, optional):      Indicates of the loss if for a American option. Defaults to False.
    """

    if use_average:
        X_loss = np.load(f"results/average_loss_{name_of_model}.npy")
        X_loss = X_loss[: X_loss.shape[0] // 2]
        X_validation = np.load(
            f"results/average_validation_{name_of_model}.npy")
        X_validation = X_validation[: X_validation.shape[0] // 2]
    else:
        X_loss = np.load(f"results/loss_{name_of_model}.npy")
        X_validation = np.load(f"results/validation_{name_of_model}.npy")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(x_values[values_to_skip::skip_every], X_loss[values_to_skip::skip_every, 0],
               label="Training loss", color="midnightblue")
    ax[0].plot(x_values[values_to_skip::skip_every], X_validation[values_to_skip::skip_every, 0],
               linestyle=(0, (4, 4)), label="Validation loss", color="red")
    ax[0].set_yscale("log")
    # ax[0].set_xscale("log")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[0].grid()
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax[1].plot(x_values[values_to_skip::5],
               X_loss[values_to_skip::skip_every, 2], label="PDE")
    ax[1].plot(x_values[values_to_skip::5],
               X_loss[values_to_skip::skip_every, 3], label="Expiry")
    ax[1].plot(x_values[values_to_skip::skip_every],
               X_loss[values_to_skip::skip_every, 4], label="Lower")
    ax[1].plot(x_values[values_to_skip::skip_every],
               X_loss[values_to_skip::skip_every, 5], label="Upper")

    if american:
        ax[1].plot(x_values[values_to_skip::skip_every],
                   X_loss[values_to_skip::skip_every, 6], label="Free boundary")

    ax[1].grid()
    ax[1].set_xlabel("Epoch")
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.plot(X_loss[100::5, 0], label = "Loss")
    leg = ax[1].legend()
    if american:
        leg.set_loc("best")
        leg.set_bbox_to_anchor((0.6, 0.6))

    ax[1].set_yscale("log")
    fig.tight_layout()
    plt.savefig(filename)


def plot_heat_map_of_predicted_versus_analytical(model: PINNforwards,
                                                 test_data: dict[torch.tensor],
                                                 dataloader: Union[DataGeneratorEuropean1D, DataGeneratorAmerican1D],
                                                 filename: str,
                                                 analytical_solution_filename: str = None,
                                                 model2: PINNforwards = None) -> None:
    """Plots heatmap of hetmap versus anlytical for a given model.

    Args:
        model (PINNforwards): Model we plot heatmap for
        test_data (dict[torch.tensor]): Dictionary containing test data, which we use for plotting
        dataloader (Union[DataGeneratorEuropean1D, DataGeneratorAmerican1D]): Dataloader used to compute analytical solution
        filename (str): Filename to save plot as
        analytical_solution_filename (str, optional):  If analyzing American option, providing this will load the data. Defaults to None.
        model2 (PINNforwards, optional): Model to compare model against if specified. Defaults to None.
    """

    X1_test = test_data["X1_validation"]
    X1_test_scaled = test_data["X1_validation_scaled"]

    expiry_x_tensor_test = test_data["expiry_x_tensor_validation"]
    expiry_x_tensor_test_scaled = test_data["expiry_x_tensor_validation_scaled"]
    expiry_y_tensor_test = test_data["expiry_y_tensor_validation"]

    lower_x_tensor_test = test_data["lower_x_tensor_validation"]
    lower_x_tensor_test_scaled = test_data["lower_x_tensor_validation_scaled"]
    lower_y_tensor_test = test_data["lower_y_tensor_validation"]

    upper_x_tensor_test = test_data["upper_x_tensor_validation"]
    upper_x_tensor_test_scaled = test_data["upper_x_tensor_validation_scaled"]
    upper_y_tensor_test = test_data["upper_y_tensor_validation"]

    if analytical_solution_filename is None:
        analytical_solution = dataloader.get_analytical_solution(
            X1_test[:, 1], X1_test[:, 0])  # .cpu().detach().numpy()
    else:
        analytical_solution = np.load(analytical_solution_filename)

    analytical_solution = analytical_solution.reshape(
        analytical_solution.shape[0], -1)

    with torch.no_grad():
        predicted_pde = model(X1_test_scaled).cpu().detach().numpy()
        predicted_expiry = model(
            expiry_x_tensor_test_scaled).cpu().detach().numpy()
        predicted_lower = model(
            lower_x_tensor_test_scaled).cpu().detach().numpy()
        predicted_upper = model(
            upper_x_tensor_test_scaled).cpu().detach().numpy()

    if model2 is not None:
        with torch.no_grad():
            predicted_pde2 = model2(X1_test_scaled).cpu().detach().numpy()
            predicted_expiry2 = model2(
                expiry_x_tensor_test_scaled).cpu().detach().numpy()
            predicted_lower2 = model2(
                lower_x_tensor_test_scaled).cpu().detach().numpy()
            predicted_upper2 = model2(
                upper_x_tensor_test_scaled).cpu().detach().numpy()

    X1_test = X1_test.cpu().detach().numpy()
    expiry_x_tensor_test = expiry_x_tensor_test.cpu().detach().numpy()
    lower_x_tensor_test = lower_x_tensor_test.cpu().detach().numpy()
    upper_x_tensor_test = upper_x_tensor_test.cpu().detach().numpy()

    expiry_y_tensor_test = expiry_y_tensor_test.cpu().detach().numpy()
    lower_y_tensor_test = lower_y_tensor_test.cpu().detach().numpy()
    upper_y_tensor_test = upper_y_tensor_test.cpu().detach().numpy()

    all_points = np.vstack(
        [X1_test, expiry_x_tensor_test, lower_x_tensor_test, upper_x_tensor_test])
    all_preds = np.vstack(
        [predicted_pde, predicted_expiry, predicted_lower, predicted_upper])
    all_y = np.vstack([analytical_solution, expiry_y_tensor_test,
                      lower_y_tensor_test, upper_y_tensor_test])

    if model2 is not None:
        all_preds_2 = np.vstack(
            [predicted_pde2, predicted_expiry2, predicted_lower2, predicted_upper2])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        scatter1 = ax[0].scatter(
            all_points[:, 0], all_points[:, 1], c=np.abs(all_y - all_preds_2).ravel(), norm=LogNorm(),  cmap='plasma')
        cbar1 = plt.colorbar(scatter1, ax=ax[0])
        ax[0].set_title("Small model")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Asset price")

        scatter2 = ax[1].scatter(all_points[:, 0], all_points[:, 1], c=np.fmax(np.abs(
            all_y - all_preds), 1e-8).ravel(), cmap='plasma', norm=LogNorm())
        cbar2 = plt.colorbar(scatter2, ax=ax[1])
        ax[1].set_title("Large model")
        ax[1].set_xlabel("Time")
        fig.tight_layout()

    else:
        scatter2 = plt.scatter(all_points[:, 0], all_points[:, 1], c=np.abs(
            all_y - all_preds), cmap='plasma', norm=LogNorm())
        cbar2 = plt.colorbar(scatter2)
        plt.xlabel("Time")
        plt.ylabel("Asset price")

        plt.tight_layout()
    plt.savefig(filename, format="jpg", dpi=180,
                pil_kwargs={"quality": 100}, bbox_inches='tight')


def plot_loss_and_sigma_backwards_problem(lambda_pdes: list, filename: str, pre: str = "scale_") -> None:
    """Plot loss and sigma for inverse problem

    Args:
        lambda_pdes (list): List with different scalings used for PDE loss.
        filename (str):     Filename to save plot as.
    """

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(0, 15_000, 1)

    conv = {"0": "0", "1": r"$10^{-1}$", "2": r"$10^{-2}$", "3": r"$10^{-3}$",
            "4": r"$10^{-4}$", "one": r"$1.0$", "10": r"$10^{1}$"}

    for scale in lambda_pdes:
        X_loss = np.load(f"results_backwards/average_loss_{pre}{scale}.npy")
        X_loss = X_loss[:X_loss.shape[0] // 2, :]

        X_sigma = np.load(f"results_backwards/average_sigma_{pre}{scale}.npy")
        X_sigma_std = X_sigma[X_sigma.shape[0]//2:, :]
        X_sigma = X_sigma[: X_sigma.shape[0]//2, :]

        if pre == "scale_":
            label = r"$\lambda_{PDE}$ =" + f"{scale}"
        else:
            label = r"$\lambda_{PDE}$ =" + conv[scale]
        # print(type(label))
        ax[0].plot(x, X_loss[:, 1] + X_loss[:, 0], label=label)
        ax[1].plot(x, X_sigma.ravel())
        # ax[1].plot(x, X_sigma_std.ravel(), label = r"$\lambda_{PDE}$ =" +f"{scale}")
    if pre == "scale_":
        ax[1].plot(x, 0.5*np.ones(15_000), label=r"True $\sigma$")
    ax[1].legend()
    ax[1].grid()

    ax[0].legend()
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Average loss")
    ax[0].set_xlabel("Epoch")

    ax[0].grid()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(r"Average $\sigma$")

    fig.tight_layout()

    plt.savefig(filename)


def plot_all_sigma_for_two(filename: str) -> None:
    """Plot all sigma for two different lambda_PDE for inverse problem.

    Args:
        filename (str): Filename to save plot as.
    """

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    X1 = np.load("results_backwards/sigma_scale_0.001.npy")
    X2 = np.load("results_backwards/sigma_scale_1000.npy")

    for i in range(X1.shape[0]):
        ax[0].plot(X1[i])
        ax[1].plot(X2[i])

    ax[0].set_title(r"$\lambda_{PDE} = 10^{-3}$")
    ax[0].grid()
    ax[0].set_ylabel(r"$\sigma$")
    ax[0].set_xlabel("Epoch")
    ax[1].grid()
    ax[1].set_title(r"$\lambda_{PDE} = 1000$")
    ax[1].set_ylabel(r"$\sigma$")
    ax[1].set_xlabel("Epoch")

    fig.tight_layout()
    plt.savefig(filename)


def plot_log_log_dimensions(filename: str) -> None:
    """Plot log-log plot of run-time and test RMSE as a function of dimension for a geometric mean option.

    Args:
        filename (str): Filename to save plot as.
    """
    dims = np.array(list(range(2, 14 + 1)))
    rmse = np.loadtxt("important_results/european_multi/RMSE_dim.txt")
    timings = np.loadtxt("important_results/european_multi/timings_dim.txt")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(dims, timings, label="Training time", color="midnightblue")
    ax[0].plot(dims, (dims)**3 * timings[-1] /
               dims[-1]**3 * 2.5, label=r"$O(D^3)$", color="red")
    ax[0].legend()
    ax[0].set_xlabel("Dimension")
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[0].xaxis.set_minor_formatter(ticker.NullFormatter())
    ax[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # ax[0].set_xlim(2, 13)
    ax[0].grid()
    ax[0].set_ylabel("Training time")

    ax[1].plot(dims, rmse, label="Test RMSE", color="midnightblue")
    # ax[1].plot(dims, (dims + 1)**4 - (dims + 0.999999)**4, label=r"$O(N^4)$")
    ax[1].set_xlabel("Dimension")
    ax[1].legend()
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-4, 1e-1)
    ax[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # ax[1].set_xlim(1, 15)
    # ax[1].set_xscale("log")
    ax[1].grid()

    fig.tight_layout()
    plt.savefig(filename)


def plot_heat_map_of_predicted_versus_analytical_inverse(model: PINNforwards,
                                                         test_data: str,
                                                         filename: str,
                                                         model2: PINNforwards = None) -> None:
    """Plots heatmap of hetmap versus anlytical for a given model.

    Args:
        model (PINNforwards): Model we plot heatmap for
        test_data (dict[torch.tensor]): Filename of test data
        dataloader (Union[DataGeneratorEuropean1D, DataGeneratorAmerican1D]): Dataloader used to compute analytical solution
        filename (str): Filename to save plot as
        analytical_solution_filename (str, optional):  If analyzing American option, providing this will load the data. Defaults to None.
        model2 (PINNforwards, optional): Model to compare model against if specified. Defaults to None.
    """
    min_values = torch.tensor([0.0, 90.34]).to(DEVICE)
    max_values = torch.tensor([2.42, 293.20]).to(DEVICE)

    data = np.load(test_data)
    X1_test = torch.from_numpy(data[:, :2]).to(DEVICE)
    X1_test_scaled = (X1_test - min_values) / (max_values - min_values)

    analytical_solution = data[:, 2].reshape(
        data.shape[0], -1)

    with torch.no_grad():
        predicted_pde = model(X1_test_scaled).cpu().detach().numpy()

    if model2 is not None:
        with torch.no_grad():
            predicted_pde2 = model2(X1_test_scaled).cpu().detach().numpy()

    X1_test = X1_test.cpu().detach().numpy()

    all_points = X1_test
    all_preds = predicted_pde

    all_y = analytical_solution

    if model2 is not None:
        all_preds_2 = predicted_pde2
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        scatter1 = ax[0].scatter(
            all_points[:, 0], all_points[:, 1], c=np.abs(all_y - all_preds_2).ravel(), norm=LogNorm(),  cmap='plasma')
        cbar1 = plt.colorbar(scatter1, ax=ax[0])
        ax[0].set_title(r"$\lambda_{PDE} = 0.0$")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Asset price")

        scatter2 = ax[1].scatter(all_points[:, 0], all_points[:, 1], c=np.fmax(np.abs(
            all_y - all_preds), 1e-8).ravel(), cmap='plasma', norm=LogNorm())
        cbar2 = plt.colorbar(scatter2, ax=ax[1])
        ax[1].set_title(r"$\lambda_{PDE} = 10^{-3}$")
        ax[1].set_xlabel("Time")
        fig.tight_layout()

    else:
        scatter2 = plt.scatter(all_points[:, 0], all_points[:, 1], c=np.abs(
            all_y - all_preds), cmap='plasma', norm=LogNorm())
        cbar2 = plt.colorbar(scatter2)
        plt.xlabel("Time")
        plt.ylabel("Asset price")
        plt.title("Absolute error")

        plt.tight_layout()
    plt.savefig(filename, format="jpg", dpi=180,
                pil_kwargs={"quality": 100}, bbox_inches='tight')


if __name__ == "__main__":
    make_training_plot("plots/fourier_loss.pdf")
    plt.clf()

    binomial_plot("plots/binomial.pdf")
    plt.clf()

    plots_greeks(10_000, 0.5, "models/greeks.pth",
                 "plots/one_dim_european_greeks.pdf")
    plt.clf()

    plot_different_loss("large_model", "plots/large_model_loss.pdf",
                        x_values=np.arange(1, 3_000_000 + 1, 1200))
    plt.clf()
    plot_different_loss("small_model", "plots/small_model_loss.pdf",
                        x_values=np.arange(1, 300_000 + 1, 90)[1:])
    plt.clf()

    plot_different_loss("american_multiple", "plots/american_loss.pdf", values_to_skip=100,
                        x_values=np.arange(1, 600_000 + 1, 600), use_average=True, american=True)
    plt.clf()

    torch.manual_seed(2024)
    np.random.seed(2024)
    dataloader = DataGeneratorEuropean1D(
        time_range=[0, 1], S_range=[0, 400], K=40, r=0.04, sigma=0.5, DEVICE=DEVICE, seed=2024)

    validation_data = create_validation_data(
        dataloader=dataloader, N_validation=5_000, config=one_euro.config)

    test_data = create_validation_data(
        dataloader=dataloader, N_validation=20_000, config=one_euro.config)
    large_model = PINNforwards(2, 1, 256, 4, use_fourier_transform=True,
                               sigma_FF=5.0, encoded_size=128)
    large_model.load_state_dict(torch.load(
        "models/large_model.pth", weights_only=True))

    small_model = PINNforwards(2, 1, 64, 2, use_fourier_transform=True,
                               sigma_FF=5.0, encoded_size=128)
    small_model.load_state_dict(torch.load(
        "models/small_model.pth", weights_only=True))

    plot_heat_map_of_predicted_versus_analytical(
        large_model, test_data, dataloader, "plots/european_scatter.jpg", model2=small_model)
    plt.clf()

    torch.manual_seed(2024)
    np.random.seed(2024)
    dataloader_american = DataGeneratorAmerican1D(
        time_range=one_american.config["t_range"], S_range=one_american.config["S_range"],
        K=one_american.config["K"], r=one_american.config["r"], sigma=one_american.config["sigma"], DEVICE=DEVICE, seed=2024)

    validation_data = create_validation_data(
        dataloader=dataloader_american, N_validation=5_000, config=one_american.config)

    test_data = create_validation_data(
        dataloader=dataloader_american, N_validation=20_000, config=one_american.config)

    american_model = PINNforwards(
        2, 1, 128, 4, use_fourier_transform=True, sigma_FF=5.0, encoded_size=128)
    american_model.load_state_dict(torch.load(
        "models/american_multiple.pth", weights_only=True))

    plot_heat_map_of_predicted_versus_analytical(
        american_model, test_data, dataloader_american, "plots/american_model.jpg", analytical_solution_filename="data/test_data_american_1D.npy")

    plot_different_loss("multi_dim", "plots/multi_dim.pdf",
                        x_values=np.arange(1, 800_000 + 1, 600), use_average=True)
    plt.clf()

    plot_loss_and_sigma_backwards_problem(
        lambda_pdes=["1e-05", "0.0001", "0.001", "0.0", "10", "1000"], filename="plots/loss_vs_sigma.pdf")
    plt.clf()

    plot_all_sigma_for_two(filename="plots/sigmas.pdf")
    plt.clf()
    plot_log_log_dimensions(filename="plots/dimensions.pdf")
    plt.clf()

    plot_loss_and_sigma_backwards_problem(
        lambda_pdes=["0", "2", "3", "4", "one"], filename="plots/loss_vs_sigma_apple.pdf", pre="apple_data_")
    plt.clf()

    model_apple_3 = PINNforwards(N_INPUT=2, N_OUTPUT=1, N_HIDDEN=128, N_LAYERS=4,
                                 use_fourier_transform=True, sigma_FF=5.0, encoded_size=128)
    model_apple_3.load_state_dict(torch.load(
        "models/apple_data_3.pth", weights_only=True))

    model_apple_0 = PINNforwards(N_INPUT=2, N_OUTPUT=1, N_HIDDEN=128, N_LAYERS=4,
                                 use_fourier_transform=True, sigma_FF=5.0, encoded_size=128)
    model_apple_0.load_state_dict(torch.load(
        "models/apple_data_0.pth", weights_only=True))

    plot_heat_map_of_predicted_versus_analytical_inverse(
        model=model_apple_3, test_data="data/apple_data_test.npy", filename="plots/apple_model.jpg", model2=model_apple_0)
    plt.clf()
