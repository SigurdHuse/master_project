from torch.utils.data import DataLoader
from dataloader import DataLoaderEuropean
from PINN import PINNbackwards
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


def train(model: PINNbackwards, nr_of_epochs: int, config: dict):
    config["lambda_pde"] = 1
    config["lambda_target"] = 1

    dataset = DataLoaderEuropean(config["train_filename"], config)

    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    dataset_val = DataLoaderEuropean(config["val_filename"], config)

    dataloader_val = DataLoader(
        dataset_val, batch_size=128, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    scheduler = ExponentialLR(optimizer, config["gamma"])

    loss_function = nn.MSELoss()
    best_validation = float("inf")
    best_validation_epoch = 0
    best_model = None

    time_range, S_range = config["t_range"], config["S_range"]
    min_values = torch.tensor(
        [time_range[0], S_range[0]]).to(DEVICE)
    max_values = torch.tensor(
        [time_range[1], S_range[1]]).to(DEVICE)

    for epoch in tqdm(range(1, nr_of_epochs + 1)):
        model.train(True)
        total_loss = 0
        total_loss_pde = 0
        total_loss_target = 0

        for batch_idx, (x, y) in enumerate(dataloader, 1):

            if epoch == 1_000:
                optimizer = torch.optim.Adam(
                    [model.sigma], lr=config["learning_rate"], weight_decay=config["weight_decay"])

                scheduler = ExponentialLR(optimizer, config["gamma"])

            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.requires_grad_(True)
            x_scaled = (x - min_values) / (max_values - min_values)
            y = y.unsqueeze(1)
            y_hat, bs_pde = model.foward_with_european_1D(
                x_scaled, x, config["r"])

            loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
            loss_target = loss_function(y_hat, y)

            if 1_000 <= epoch:
                loss = loss_pde
            else:
                loss = loss_target
            # config["lambda_target"]*loss_target

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.to("cpu").detach()
            loss_pde = loss_pde.to("cpu").detach()
            loss_target = loss_target.to("cpu").detach()

            total_loss += loss.item()
            total_loss_pde += loss_pde.item()
            total_loss_target += loss_target.item()

            """ if epoch % config["update_lambda"] == 0:
                l2_pde += torch.linalg.vector_norm(loss_pde).item()**2
                l2_target += torch.linalg.vector_norm(loss_target).item()**2 """

        print(total_loss_pde / len(dataloader),
              total_loss_target / len(dataloader))
        print(total_loss, model.sigma.item())

        if epoch % config["scheduler_step"] == 0:
            scheduler.step()

        model.train(False)
        model.eval()

        """ if epoch % config["update_lambda"] == 0:
            l2_pde = np.sqrt(l2_pde)
            l2_target = np.sqrt(l2_target)

            l2_sum = l2_pde + l2_target

            new_lambda_target = l2_sum / max(l2_target, 1e-6)
            new_lambda_pde = l2_sum / max(l2_pde, 1e-6)

            alpha = config["alpha_lambda"]
            config["lambda_target"] = min(alpha *
                                          config["lambda_target"] + (1 - alpha) * new_lambda_target, 1_000)

            config["lambda_pde"] = min(alpha *
                                       config["lambda_pde"] + (1 - alpha) * new_lambda_pde, 20)

            print("Lambda pde", config["lambda_pde"])
            print("Target lambda", config["lambda_target"]) """

        """ if epoch % config["epochs_before_validation"] == 0:
            loss_val = []
            for batch_idx, (x, y) in enumerate(dataloader_val):
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.requires_grad_(True)
                x_scaled = (x - min_values) / (max_values - min_values)
                y = y.unsqueeze(1)
                y_hat, bs_pde = model.foward_with_european_1D(
                    x_scaled, x, config["r"])

                loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
                loss_target = loss_function(y_hat, y)

                loss = loss_pde + loss_target
                loss_val.append(loss.item())
            cur_loss_val = sum(loss_val) / len(loss_val)
            # Make sure the validation prediction does not affect the training
            optimizer.zero_grad()

            if cur_loss_val < best_validation:
                best_validation_epoch = epoch
                best_validation = cur_loss_val
                best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model) """

    return best_validation_epoch


def try_multiple_activation_functions(config: dict, filename1: str, filename2: str, filename3: str, activation_functions: list, layers: list):
    dataset_test = DataLoaderEuropean(config["test_filename"])

    dataloader_test = DataLoader(
        dataset_test, batch_size=32, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    errors = np.zeros((len(activation_functions), len(layers)))
    sigma_error = np.zeros((len(activation_functions), len(layers)))
    epochs = np.zeros((len(activation_functions), len(layers)))

    for i, activation_function in enumerate(activation_functions):
        for j, layer in enumerate(layers):
            model = PINNbackwards(
                config["N_INPUT"], 1, 256, layer, activation_function)
            model.train(True)
            epoch = train(model, 10_000, config)

            model.train(False)
            model.eval()
            for batch_idx, (x, y) in enumerate(dataloader_test):
                with torch.no_grad():
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    x = x.requires_grad_(True)
                    y = y.unsqueeze(1)
                    y_hat = model.forward(x)

                    errors[i, j] += torch.sum(
                        (y - y_hat)**2).item()
                    epochs[i, j] = epoch

            errors[i, j] /= len(dataset_test)
            sigma_error[i, j] = config["true_sigma"] - model.sigma
    np.savetxt(filename1, errors)
    np.savetxt(filename2, epochs)
    np.savetxt(filename3, sigma_error)


if __name__ == "__main__":
    config = {"train_filename": "data/european_one_dimensional_train.npy",
              "val_filename": "data/european_one_dimensional_val.npy",
              "test_filename": "data/european_one_dimensional_test.npy",
              "learning_rate": 1e-3,
              "weight_decay": 0.0,
              "gamma": 0.9,
              "r": 0.04,
              "epochs_before_validation": 10,
              "scheduler_step": 100,
              "pde_loss_weight": 1,
              "N_INPUT": 2,
              "true_sigma": 0.5,
              "S_range": [0, 200],
              "t_range": [0, 1],
              "DEVICE": DEVICE}

    model = PINNbackwards(2, 1, 256, 4, initial_sigma=1.0)

    model.to(DEVICE)
    print(model.sigma)
    train(model, 2_000, config)
    print(model.sigma)
    """ try_multiple_activation_functions(
        config, "important_results_backwards/MSE_activation.txt",
        "important_results_backwards/epochs_activation.txt",
        "important_results_backwards/sigma_activation.txt",
        [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()],
        [1, 2, 3, 4, 5, 6]) """
