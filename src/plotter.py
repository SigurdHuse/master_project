from train import DEVICE
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
import numpy as np
torch.set_default_device(DEVICE)


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
