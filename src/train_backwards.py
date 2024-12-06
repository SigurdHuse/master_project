from torch.utils.data import DataLoader
from dataloader import DataLoaderEuropean
from PINN import PINNforwards
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt


def black_scholes_1D(y1_hat, X1, sigma, r):
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


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


def train(model: PINNforwards, nr_of_epochs: int, config: dict, PDE):
    print("Target Scale", config["target_scale"])
    dataset = DataLoaderEuropean(config["train_filename"], config)

    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=True, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    dataset_val = DataLoaderEuropean(config["val_filename"], config)

    dataloader_val = DataLoader(
        dataset_val, batch_size=128, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    sigma = torch.nn.Parameter(torch.tensor(
        [2.0], requires_grad=True)).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters())+[sigma], lr=config["learning_rate"], weight_decay=config["weight_decay"])

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

    sigmas = []
    for epoch in tqdm(range(1, nr_of_epochs + 1), miniters=1_00, maxinterval=1_00):
        model.train(True)
        total_loss = 0
        total_loss_pde = 0
        total_loss_target = 0

        for batch_idx, (x, y) in enumerate(dataloader, 1):
            optimizer.zero_grad()

            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.requires_grad_(True)
            x_scaled = (x - min_values) / (max_values - min_values)
            y = y.unsqueeze(1)

            # Call model.forward only once
            y_hat = model.forward(x_scaled)
            prediction = model.forward(x_scaled)

            # Compute the PDE residual
            bs_pde = PDE(y_hat, x, sigma, config["r"])

            loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
            loss_target = loss_function(prediction, y)

            loss = loss_pde + config["target_scale"] * loss_target

            loss.backward()
            optimizer.step()

            loss = loss.to("cpu").detach()
            loss_pde = loss_pde.to("cpu").detach()
            loss_target = loss_target.to("cpu").detach()

            total_loss += loss.item()
            total_loss_pde += loss_pde.item()
            total_loss_target += loss_target.item()

        if epoch % config["epochs_before_validation"] == 0:
            loss_val = []
            for batch_idx, (x, y) in enumerate(dataloader_val):
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.requires_grad_(True)
                x_scaled = (x - min_values) / (max_values - min_values)
                y = y.unsqueeze(1)

                y_hat = model.forward(x_scaled)
                prediction = model.forward(x_scaled)

                # Compute the PDE residual
                bs_pde = PDE(y_hat, x, sigma, config["r"])

                loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
                loss_target = loss_function(prediction, y)

                loss = loss_pde + loss_target
                loss_val.append(loss.item())
            cur_loss_val = sum(loss_val) / len(loss_val)
            # Make sure the validation prediction does not affect the training
            optimizer.zero_grad()

            if cur_loss_val < best_validation:
                best_validation_epoch = epoch
                best_validation = cur_loss_val
                best_model = copy.deepcopy(model.state_dict())

        # print(total_loss_pde / len(dataloader),
        #      total_loss_target / len(dataloader))
        # print(total_loss_pde + total_loss_target, sigma.item())
        sigmas.append(sigma.item())
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
    # Load best model based on validation data
    model.load_state_dict(best_model)
    # plt.plot(sigmas)
    # plt.savefig("plots/test.png")
    return np.array(sigmas), best_validation_epoch


def try_multiple_activation_functions(config: dict, filename1: str, filename2: str,  activation_functions: list, layers: list):
    epochs = 2_000

    dataset_test = DataLoaderEuropean(config["test_filename"], config)

    dataloader_test = DataLoader(
        dataset_test, batch_size=32, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    errors = np.zeros((len(activation_functions), len(layers)))
    all_sigmas = np.zeros((len(activation_functions) * len(layers), epochs))

    time_range, S_range = config["t_range"], config["S_range"]
    min_values = torch.tensor(
        [time_range[0], S_range[0]]).to(DEVICE)
    max_values = torch.tensor(
        [time_range[1], S_range[1]]).to(DEVICE)

    for i, activation_function in enumerate(activation_functions):
        for j, layer in enumerate(layers):
            model = PINNforwards(
                config["N_INPUT"], 1, 256, layer, activation_function)
            model.train(True)
            sigmas = train(model, epochs, config, black_scholes_1D)

            model.train(False)
            model.eval()
            for batch_idx, (x, y) in enumerate(dataloader_test):
                with torch.no_grad():
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    x_scaled = (x - min_values) / (max_values - min_values)
                    y = y.unsqueeze(1)
                    y_hat = model.forward(x_scaled)

                    errors[i, j] += torch.sum(
                        (y - y_hat)**2).item()

            errors[i, j] /= len(dataset_test)

            all_sigmas[i * len(layers) + j, :] = sigmas
            print(f"Trial {j + i*len(layers) +
                  1} / {len(layers)*len(activation_functions)} Done")
    np.savetxt(filename1, errors)
    np.save(filename2, all_sigmas)


def try_learning_rate_and_target_scale(config: dict, filename1: str, filename2: str, filename3,  learning_rates: list, target_scales: list):
    epochs = 2_000

    tmp_config = copy.deepcopy(config)
    dataset_test = DataLoaderEuropean(config["test_filename"], tmp_config)

    dataloader_test = DataLoader(
        dataset_test, batch_size=256, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    errors = np.zeros((len(learning_rates), len(target_scales)))
    best_epochs = np.zeros((len(learning_rates), len(target_scales)))
    all_sigmas = np.zeros((len(learning_rates) * len(target_scales), epochs))

    time_range, S_range = config["t_range"], config["S_range"]
    min_values = torch.tensor(
        [time_range[0], S_range[0]]).to(DEVICE)
    max_values = torch.tensor(
        [time_range[1], S_range[1]]).to(DEVICE)

    model = PINNforwards(config["N_INPUT"], 1, 256, 4)
    model = model.to(DEVICE)
    start_model = copy.deepcopy(model.state_dict())

    for i, learning_rate in enumerate(learning_rates):
        for j, target_scale in enumerate(target_scales):
            tmp_config["target_scale"] = target_scale
            tmp_config["learning_rate"] = learning_rate

            model.load_state_dict(start_model)
            model.train(True)
            sigmas, epoch = train(model, epochs, tmp_config, black_scholes_1D)

            model.train(False)
            model.eval()
            for batch_idx, (x, y) in enumerate(dataloader_test):
                with torch.no_grad():
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    x_scaled = (x - min_values) / (max_values - min_values)
                    y = y.unsqueeze(1)
                    y_hat = model.forward(x_scaled)

                    errors[i, j] += torch.sum(
                        (y - y_hat)**2).item()

            print(sigmas[epoch])
            errors[i, j] /= len(dataset_test)
            best_epochs[i, j] = epoch
            all_sigmas[i * len(target_scales) + j, :] = sigmas

            print(f"Trial {j + i*len(target_scales) +
                  1} / {len(target_scales)*len(learning_rates)} Done")
    np.savetxt(filename1, errors)
    np.save(filename2, all_sigmas)
    np.savetxt(filename3, best_epochs)


if __name__ == "__main__":
    config = {"train_filename": "data/european_one_dimensional_train.npy",
              "val_filename": "data/european_one_dimensional_val.npy",
              "test_filename": "data/european_one_dimensional_test.npy",
              "learning_rate": 1e-3,
              "weight_decay": 1e-5,
              "gamma": 0.9,
              "r": 0.04,
              "epochs_before_validation": 10,
              "scheduler_step": 2000,
              "pde_loss_weight": 1,
              "N_INPUT": 2,
              "true_sigma": 0.5,
              "S_range": [0, 200],
              "t_range": [0, 1],
              "DEVICE": DEVICE,
              "target_scale": 100}

    # model = PINNforwards(2, 1, 256, 4)

    # model.to(DEVICE)

    # train(model, 2_000, config, black_scholes_1D)
    """ torch.manual_seed(2025)
    np.random.seed(2025)
    try_learning_rate_and_target_scale(config, "important_results_backwards/MSE_learning.txt",
                                       "important_results_backwards/sigmas_learning.npy", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4], [10, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]) """

    torch.manual_seed(2025)
    np.random.seed(2025)
    try_learning_rate_and_target_scale(config, "important_results_backwards/MSE_learning_test.txt",
                                       "important_results_backwards/sigmas_learning_test.npy", "important_results_backwards/epochs_learning_test.txt", [3e-3, 1e-3, 5e-4], [1_000, 5_000, 10_000, 20_000])

    """ torch.manual_seed(2025)
    np.random.seed(2025)
    try_multiple_activation_functions(
        config, "important_results_backwards/MSE_activation.txt",
        "important_results_backwards/sigmas_activation.npy",
        [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh],
        [1, 2, 3, 4, 5, 6]) """
