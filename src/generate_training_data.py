from data_generator import DataGeneratorEuropean1D
import numpy as np
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


def generate_1D_european(nr_of_points: int, config: dict, filename: str):
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
    solution = solution.cpu().detach().numpy()
    solution = solution.reshape((nr_of_points, 1))

    X_pde = np.concatenate([X_pde, solution], axis=1)

    np.save("data/" + filename, X_pde)


if __name__ == "__main__":
    config = {
        "K": 40,
        "time_range": [0, 1],
        "S_range": [0, 400],
        "sigma":  0.5,
        "r": 0.04,
    }
    config["seed"] = 1
    generate_1D_european(1280, config, "european_one_dimensional_train.npy")
    config["seed"] = 2
    generate_1D_european(5_00, config, "european_one_dimensional_val.npy")
    config["seed"] = 3
    generate_1D_european(4_000, config, "european_one_dimensional_test.npy")
