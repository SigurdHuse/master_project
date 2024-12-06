from torch.distributions import Normal
from train import DEVICE
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import torch
import numpy as np
from PINN import PINNforwards
torch.set_default_device(DEVICE)


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


def make_training_plot(filename: str, for_validation: bool = False):
    types_of_loss = ["total_loss", "loss_boundary",
                     "loss_pde", "loss_expiry", "loss_lower", "loss_upper"]
    if for_validation:
        X_loss = np.loadtxt("results/validation_" + filename + ".txt")
    else:
        X_loss = np.loadtxt("results/loss_" + filename + ".txt")

    df_loss = pd.DataFrame(columns=types_of_loss, data=X_loss)
    df_loss["epoch"] = np.arange(1, len(df_loss) + 1)
    df_melted = df_loss.melt(
        id_vars="epoch", value_vars=types_of_loss, var_name='Category', value_name="Loss")

    y_name = "Validation loss " if for_validation else "Loss"
    fig = px.line(df_melted, x="epoch", y="Loss", color="Category", log_y=True)
    fig.show()


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


def plots_greeks(n: int, time: float, path_to_weights: str, name_of_plot: str):
    n = 1_000
    S_range = [0, 200]
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

    model = PINNforwards(2, 1, 256, 4)
    model.load_state_dict(torch.load(path_to_weights, weights_only=True))
    model = model.to(DEVICE)

    delta, gamma, theta, nu, rho = model.estimate_greeks_call(
        X_scaled, X, 0.5, t_range[1])

    delta = delta.to("cpu").detach().numpy().flatten()
    gamma = gamma.to("cpu").detach().numpy().flatten()
    theta = theta.to("cpu").detach().numpy().flatten()
    nu = nu.to("cpu").detach().numpy().flatten()
    rho = rho.to("cpu").detach().numpy().flatten()

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],  # Row 1: 3 columns
            [{'type': 'xy'}, {'type': 'xy'}, None]            # Row 2: 2 columns
        ],
        subplot_titles=(r"$\Delta$", r"$\Gamma$",
                        r"$\Theta$", r"$\nu$", r"$\rho$"),
        horizontal_spacing=0.1,  # Adjust horizontal spacing
        vertical_spacing=0.2     # Adjust vertical spacing
    )

    # Add line plots to the first row
    fig.add_trace(go.Scatter(x=S, y=delta, mode='lines',
                  line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=S, y=gamma, mode='lines',
                  line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=S, y=theta, mode='lines',
                  line=dict(width=2)), row=1, col=3)

    # Add line plots to the second row
    fig.add_trace(go.Scatter(x=S, y=nu, mode='lines',
                  line=dict(width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=S, y=rho, mode='lines',
                  line=dict(width=2)), row=2, col=2)

    # Update layout
    fig.update_layout(
        title=f"Different Greeks at t = {time}",
        height=800,  # Adjust height to fit all subplots
        width=1000,  # Adjust width for better aspect ratio
        font=dict(size=20),  # Adjust font size
        margin=dict(l=20, r=20, t=50, b=20),  # Tight margins
        showlegend=False
    )
    fig.update_annotations(font_size=20)
    fig.update_xaxes(title_text="Stock Price", row=1, col=1)
    fig.update_xaxes(title_text="Stock Price", row=1, col=2)
    fig.update_xaxes(title_text="Stock Price", row=1, col=3)
    fig.update_xaxes(title_text="Stock Price", row=2, col=1)
    fig.update_xaxes(title_text="Stock Price", row=2, col=2)
    # Show the plot
    fig.write_image(name_of_plot, width=2200, height=800, scale=1)


if __name__ == "__main__":
    visualize_one_dimensional(50, "models/greeks.pth",
                              "plots/one_dim_european.pdf")
    plots_greeks(10_000, 0.5, "models/greeks.pth",
                 "plots/one_dim_european_greeks.pdf")
