from data_generator import DataGeneratorEuropean1D
import numpy as np
import torch
import pandas as pd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


def generate_1D_european(nr_of_points: int, config: dict, filename: str) -> None:
    """Generate data for a 1D European call option

    Args:
        nr_of_points (int): Number of point to generate
        config (dict):      Dictionary with hyperparameters
        filename (str):     Filename to save point as
    """
    time_range = config["time_range"]
    S_range = config["S_range"]
    K = config["K"]
    r = config["r"]
    sigma = config["sigma"]
    seed = config["seed"]

    dataloader = DataGeneratorEuropean1D(
        time_range=time_range, S_range=S_range, K=K, r=r, sigma=sigma, DEVICE=DEVICE, seed=seed)

    X_pde, y_pde = dataloader.get_pde_data_tensor(N_sample=nr_of_points, mul=1)
    solution = dataloader.get_analytical_solution(S=X_pde[:, 1], t=X_pde[:, 0])

    # X_pde_scaled = dataloader.normalize(X_pde)

    X_pde = X_pde.cpu().detach().numpy()
    # X_pde_scaled = X_pde_scaled.cpu().detach().numpy()
    solution = solution
    solution = solution.reshape((nr_of_points, 1))

    X_pde = np.concatenate([X_pde, solution], axis=1).astype(np.float32)

    np.save("data/" + filename, X_pde)


def extract_apple_data(nr_of_validation: int, nr_of_training: int, nr_of_test: int):
    """Extract apple data from /data/apple_data.csv, and split into training, test and validation

    Args:
        nr_of_validation (int): Number of validation points to extract.
        nr_of_training (int):   Number of training points to extract.
        nr_of_test (int):       Number of validation points to extract.
    """

    df = pd.read_csv("data/apple_data.csv", delimiter=",")
    values = df.values[:, :3]
    values = values[~np.isnan(values).any(axis=1)]
    values = values.astype(np.float32)
    N = values.shape[0]

    assert nr_of_validation + nr_of_training + \
        nr_of_test == N, f"You need to use all {N} points"

    # Scramble data to avoid bias
    idx = np.random.permutation(N)
    values = values[idx]

    print(f"Max stock price: {np.max(values[:, 1]):.2f}")
    print(f"Min stock price: {np.min(values[:, 1]):.2f}")

    print(f"Max time: {np.max(values[:, 0]):.2f}")
    print(f"Min time: {np.min(values[:, 0]):.2f}")

    n1 = nr_of_validation
    n2 = n1 + nr_of_training
    n3 = n2 + nr_of_test

    val_X = values[:n1]
    train_X = values[n1:n2]
    test_X = values[n2:n3]

    np.save("data/apple_data_test.npy", test_X)
    np.save("data/apple_data_val.npy", val_X)
    np.save("data/apple_data_train.npy", train_X)


if __name__ == "__main__":
    config = {
        "K": 40,
        "time_range": [0, 1],
        "S_range": [0, 400],
        "sigma":  0.5,
        "r": 0.04,
    }
    """ config["seed"] = 1
    generate_1D_european(1280, config, "european_one_dimensional_train.npy")
    config["seed"] = 2
    generate_1D_european(5_00, config, "european_one_dimensional_val.npy")
    config["seed"] = 3
    generate_1D_european(4_000, config, "european_one_dimensional_test.npy")

    np.random.seed(1000)
    extract_apple_data(nr_of_validation=691,
                       nr_of_test=3_457, nr_of_training=9_680) """
    for sig in [1, 2, 3, 4]:
        config["sigma"] = sig
        config["seed"] = 1
        generate_1D_european(
            1280, config, f"european_one_dimensional_train_{sig}.npy")
        config["seed"] = 2
        generate_1D_european(
            5_00, config, f"european_one_dimensional_val_{sig}.npy")
        config["seed"] = 3
        generate_1D_european(
            4_000, config, f"european_one_dimensional_test_{sig}.npy")
