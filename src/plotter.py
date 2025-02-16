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
from matplotlib.colors import LogNorm
import experiments_european_one_dimensional as one_euro

torch.set_default_device(DEVICE)


mpl.rcParams["figure.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 15
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["figure.dpi"] = 500


def get_analytical_solution(S, t, t_range, sigma, r, K):
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


def make_training_plot(filename: str):
    X_loss_fourier = np.load("results/average_loss_with_fourier.npy")
    X_loss = np.load("results/average_loss_no_fourier.npy")

    n_loss = X_loss_fourier.shape[0]

    X_validation_fourier = np.load(
        "results/average_validation_with_fourier.npy")
    X_validation = np.load("results/average_validation_no_fourier.npy")

    n_val = X_validation_fourier.shape[0]

    skip = 0
    plot_every = 2_000
    x_loss = np.arange(skip, n_loss//2, plot_every)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(x_loss, X_loss_fourier[skip:n_loss//2:plot_every, 0],
               label="Fourier", color="midnightblue")
    ax[0].plot(x_loss, X_loss[skip:n_loss//2:plot_every, 0],
               label="No Fourier", color="red")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epochs")
    ax[0].legend()
    ax[0].set_title("Training loss")
    ax[0].grid()

    plot_every = 100
    x_val = np.arange(skip, n_loss//2, 30*plot_every)

    ax[1].plot(x_val, X_validation_fourier[skip:n_val//2:plot_every,
               0], label="Fourier", color="midnightblue")
    ax[1].plot(x_val, X_validation[skip:n_val//2:plot_every, 0],
               label="No Fourier", color="red")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("epochs")
    # as[1].legend()
    ax[1].set_title("Validation loss")
    ax[1].grid()

    fig.tight_layout()
    plt.savefig(filename)


def visualize_one_dimensional(n: int, path_to_weights: str, name_of_plot: str):
    n = 50
    S_range = [0, 200]
    t_range = [0, 1]
    s = np.linspace(*S_range, n)
    t = np.linspace(*t_range, n)
    S, T = np.meshgrid(s, t)

    X = torch.tensor(np.column_stack(
        (T.flatten(), S.flatten())), dtype=torch.float)
    min_values = torch.tensor(
        [t_range[0], S_range[0]]).to(DEVICE)
    max_values = torch.tensor(
        [t_range[1], S_range[1]]).to(DEVICE)

    X_scaled = (X - min_values) / (max_values - min_values)

    model = PINNforwards(2, 1, 256, 4)
    model.load_state_dict(torch.load(path_to_weights, weights_only=True))
    model = model.to(DEVICE)

    Z_model = model(X_scaled).to("cpu").detach().numpy().reshape((n, n))
    Z_analytical = get_analytical_solution(X[:, 1], X[:, 0], t_range, 0.5, 0.04, 40).to(
        "cpu").detach().numpy().reshape((n, n))

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'surface'}, {'type': 'surface'},
                {'type': 'surface'}]],  # Specify 3D plots
        subplot_titles=("Predicted", "Analytical",
                        "Predicted subtracted Analytical"),
        # horizontal_spacing=0.5
    )

    # Add the first 3D surface plot
    fig.add_trace(
        go.Surface(z=Z_model, x=S, y=T, colorscale='Cividis',
                   showscale=True, colorbar=dict(title="Option price", x=0.65)),
        row=1, col=1
    )

    # Add the second 3D surface plot
    fig.add_trace(
        go.Surface(z=Z_analytical, x=S, y=T,
                   showscale=False, colorscale='Cividis'),
        row=1, col=2
    )

    fig.add_trace(
        go.Surface(z=Z_model - Z_analytical, x=S, y=T, colorscale='Viridis',
                   showscale=True, colorbar=dict(title="Difference", x=1.0)),
        row=1, col=3
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Stock price",
            yaxis_title="Time",
            zaxis_title="Option price",
            camera=dict(eye=dict(x=-1, y=2, z=2))  # Adjust view for both plots
        ),
        scene2=dict(  # Layout for the second plot
            xaxis_title="Stock price",
            yaxis_title="Time",
            zaxis_title="Option price",
            camera=dict(eye=dict(x=-1, y=2, z=2))  # Adjust view for both plots
        ),
        scene3=dict(  # Layout for the second plot
            xaxis_title="Stock price",
            yaxis_title="Time",
            zaxis_title="Option price",
            camera=dict(eye=dict(x=-1, y=2, z=2))  # Adjust view for both plots
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_annotations(font_size=20)
    # Show the plot
    fig.write_image(name_of_plot,
                    width=2200, height=800, scale=1)


def standard_normal_pdf(x):
    return (1 / torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-x.pow(2) / 2)


def plots_greeks(n: int, time: float, path_to_weights: str, name_of_plot: str):
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


def binomial_plot(filename):
    rmse = np.loadtxt("important_results/american_1D/RMSE_binomial.txt")
    timings = np.loadtxt("important_results/american_1D/timings_binomial.txt")

    M_values = np.array([32, 64, 128, 256, 384, 512, 768,
                        1024, 1280, 1536, 1792, 2048])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(M_values[1:], rmse[1:], label="RMSE", color="midnightblue")
    ax[0].plot(M_values[1:], 1 / M_values[1:],
               label=r"$O(n^{-1})$", color="red")
    ax[0].set_yscale('log')
    ax[0].set_xscale("log")
    ax[0].set_ylabel("RMSE")
    ax[0].set_xlabel("M")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(M_values, timings, label="Run time", color="midnightblue")
    ax[1].plot(M_values, M_values, label=r"$O(n)$", color="red")
    ax[1].set_yscale('log')
    ax[1].set_xscale("log")
    ax[1].set_ylabel("Run time [s]")
    ax[1].set_xlabel("M")
    ax[1].legend()
    ax[1].grid()

    fig.tight_layout()
    plt.savefig(filename)


def make_3D_american_plot(filename, X):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    y = np.load("data/test_data_american_1D.npy")
    # Create a 3D line plot
    ax.plot_trisurf(X[::1, 0], X[::1, 1], y[::1].ravel(),
                    cmap='viridis', edgecolor='none')

    # Optionally, you can also create a scatter plot if you want discrete points:
    # ax.scatter(x, y, z, color='r', marker='o', label='3D points')

    # Label the axes
    ax.set_xlabel('Time')
    ax.set_ylabel('Asset price')
    ax.set_zlabel('Price function')

    # Add a title and legend
    # ax.set_title('3D Plot Example')
    ax.view_init(elev=30, azim=30)
    ax.legend()

    fig.tight_layout()
    # Display the plot
    plt.savefig(filename)


def plot_different_loss(name_of_model: str, filename: str, x_values: np.array, values_to_skip=100, skip_every=5):
    X_loss = np.load(f"results/loss_{name_of_model}.npy")
    X_validation = np.load(f"results/validation_{name_of_model}.npy")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(x_values[values_to_skip::skip_every], X_loss[values_to_skip::skip_every, 0],
               label="Loss", color="midnightblue")
    ax[0].plot(x_values[values_to_skip::skip_every], X_validation[values_to_skip::skip_every, 0],
               linestyle=(0, (4, 4)), label="Validation", color="red")
    ax[0].set_yscale("log")
    # ax[0].set_xscale("log")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(x_values[values_to_skip::skip_every],
               X_loss[values_to_skip::skip_every, 4], label="loss_lower")
    ax[1].plot(x_values[values_to_skip::skip_every],
               X_loss[values_to_skip::skip_every, 5], label="loss_upper")
    ax[1].plot(x_values[values_to_skip::5],
               X_loss[values_to_skip::skip_every, 2], label="loss_pde")
    ax[1].plot(x_values[values_to_skip::5],
               X_loss[values_to_skip::skip_every, 3], label="loss_expiry")
    ax[1].grid()
    ax[1].set_xlabel("epoch")
    # plt.plot(X_loss[100::5, 0], label = "Loss")

    ax[1].legend()
    ax[1].set_yscale("log")
    plt.savefig(filename)


def plot_heat_map_of_predicted_versus_analytical(model, test_data: dict[torch.tensor], dataloader, filename):
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

    analytical_solution = dataloader.get_analytical_solution(
        X1_test[:, 1], X1_test[:, 0]).cpu().detach().numpy()

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

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    scatter1 = ax[0].scatter(all_points[:, 0], all_points[:, 1], c=all_preds)
    cbar1 = plt.colorbar(scatter1, ax=ax[0])
    ax[0].set_title("Predicted values")

    scatter2 = ax[1].scatter(all_points[:, 0], all_points[:, 1], c=np.abs(
        all_y - all_preds), cmap='plasma', norm=LogNorm())
    cbar2 = plt.colorbar(scatter2, ax=ax[1])
    ax[1].set_title("Absolute error")

    plt.tight_layout()
    plt.savefig(filename, format="jpg", dpi=120,
                pil_kwargs={"quality": 100}, bbox_inches='tight')


if __name__ == "__main__":
    """ visualize_one_dimensional(50, "models/greeks.pth",
                              "plots/one_dim_european.pdf") """
    # plt.clf()
    # make_training_plot("plots/fourier_loss.pdf")
    # plt.clf()
    """ binomial_plot("plots/binomial.pdf")
    plt.clf() """

    plots_greeks(10_000, 0.5, "models/greeks.pth",
                 "plots/one_dim_european_greeks.pdf")

    """ plot_different_loss("large_model", "plots/large_model_loss.pdf",
                        x_values=np.arange(1, 3_000_000 + 1, 1200))

    plot_different_loss("small_model", "plots/small_model_loss.pdf",
                        x_values=np.arange(1, 200_000 + 1, 90)[1:]) """
    """ torch.manual_seed(2025)
    np.random.seed(2025)
    dataloader = DataGeneratorEuropean1D(
        time_range=[0, 1], S_range=[0, 400], K=40, r=0.04, sigma=0.5, DEVICE=DEVICE)

    validation_data = create_validation_data(
        dataloader=dataloader, N_validation=5_000, config=one_euro.config)

    test_data = create_validation_data(
        dataloader=dataloader, N_validation=20_000, config=one_euro.config)
    large_model = PINNforwards(2, 1, 128, 8, use_fourier_transform=True,
                               sigma_FF=5.0, encoded_size=128)
    large_model.load_state_dict(torch.load(
        "models/large_model.pth", weights_only=True))

    small_model = PINNforwards(2, 1, 64, 2, use_fourier_transform=True,
                               sigma_FF=5.0, encoded_size=128)
    small_model.load_state_dict(torch.load(
        "models/small_model.pth", weights_only=True))

    plot_heat_map_of_predicted_versus_analytical(
        large_model, test_data, dataloader, "plots/large_model.jpg")
    plot_heat_map_of_predicted_versus_analytical(
        small_model, test_data, dataloader, "plots/small_model.jpg") """
