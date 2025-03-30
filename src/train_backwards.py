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
from data_generator import DataGeneratorEuropean1D
import os


def black_scholes_1D_backwards(y1_hat, X1, sigma, r):
    grads = torch.autograd.grad(y1_hat, X1, grad_outputs=torch.ones(y1_hat.shape).to(
        DEVICE), retain_graph=True, create_graph=True, only_inputs=True)[0]
    dVdt, dVdS = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

    grads2nd = torch.autograd.grad(dVdS, X1, grad_outputs=torch.ones(
        dVdS.shape).to(DEVICE), create_graph=True, only_inputs=True)[0]
    d2VdS2 = grads2nd[:, 1].view(-1, 1)

    S1 = X1[:, 1].view(-1, 1)
    bs_pde = dVdt + 0.5 * sigma**2 * S1 ** 2 * d2VdS2 + \
        r * S1 * dVdS - r * y1_hat

    return bs_pde


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


def train(model: PINNforwards, nr_of_epochs: int, config: dict, PDE, pde_dataloader, final_learning_rate: float = 1e-5, filename: str = ""):
    # print("PDE Scale", config["pde_scale"])
    n = np.log(final_learning_rate /
               config["learning_rate"]) / np.log(config["gamma"])
    scheduler_step = int(nr_of_epochs // n)
    types_of_loss = ["loss_pde", "loss_target"]

    loss_history = {i: [] for i in types_of_loss}
    loss_history_validation = {i: [] for i in types_of_loss}
    # Make sure we do not modulo w.r.t 0
    if scheduler_step <= 0:
        scheduler_step = nr_of_epochs
    # print(scheduler_step)

    dataset = DataLoaderEuropean(
        config["train_filename"], config["training_noise"])

    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                            shuffle=True, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))
    number_of_batches = len(dataset) // config["batch_size"]
    if len(dataset) % config["batch_size"] != 0:
        number_of_batches += 1

    dataset_val = DataLoaderEuropean(config["val_filename"])

    dataloader_val = DataLoader(
        dataset_val, batch_size=len(dataset_val), num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

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
        total_loss_pde = 0
        total_loss_target = 0

        for batch_idx, (x, y) in enumerate(dataloader, 1):
            optimizer.zero_grad()

            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.requires_grad_(True)
            x_scaled = (x - min_values) / (max_values - min_values)
            y = y.unsqueeze(1)

            # print(torch.max(x_scaled))
            y_hat = model(x_scaled)
            loss_target = loss_function(y_hat, y)

            # loss = config["pde_scale"] * loss_pde + loss_target
            loss = loss_target

            if config["use_target_points_for_PDE"]:
                bs_pde = PDE(y_hat, x, sigma, config["r"])
                loss_pde = config["pde_scale"] * \
                    loss_function(bs_pde, torch.zeros_like(bs_pde))

                loss = loss + loss_pde
            else:
                if epoch > config["PDE_epochs"] and config["PDE_batch"] > 0:
                    X1, y1 = pde_dataloader.get_pde_data_tensor(
                        config["PDE_batch"], mul=1)

                    X1_scaled = pde_dataloader.normalize(X1)
                    y_pde = model(X1_scaled)
                    bs_pde = PDE(y_pde, X1, sigma, config["r"])
                    loss_pde = config["pde_scale"] * \
                        loss_function(bs_pde, torch.zeros_like(bs_pde))

                    loss = loss + loss_pde

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_target = loss_target.to("cpu").detach()
            total_loss_target += loss_target.item() * x.shape[0]

            if epoch > config["PDE_epochs"] and config["PDE_batch"] > 0:
                loss_pde = loss_pde.to("cpu").detach()
                total_loss_pde += loss_pde.item() * config["PDE_batch"]

        if config["PDE_batch"] > 0:
            total_loss_pde /= number_of_batches * config["PDE_batch"]
        total_loss_target /= len(dataset)

        loss_history["loss_pde"].append(total_loss_pde)
        loss_history["loss_target"].append(total_loss_target)

        # print(total_loss_pde, total_loss_target)
        if epoch % scheduler_step == 0:
            scheduler.step()

        model.train(False)
        model.eval()
        if epoch % config["epochs_before_validation"] == 0:
            # loss_val = []
            total_val_loss_pde = 0
            total_val_loss_target = 0

            for batch_idx, (x, y) in enumerate(dataloader_val):
                x, y = x.to(DEVICE), y.to(DEVICE)

                x = x.requires_grad_(True)
                x_scaled = (x - min_values) / (max_values - min_values)
                y = y.unsqueeze(1)

                y_hat = model(x_scaled)

                # prediction = model.forward(x_scaled)
                loss_target = loss_function(y_hat, y)

                if config["PDE_batch"] > 0:
                    # Compute the PDE residual
                    bs_pde = PDE(y_hat, x, sigma, config["r"])

                    loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))

                    loss_pde = loss_pde.to("cpu").detach()
                    total_val_loss_pde += loss_pde.item() * x.shape[0]

                loss_target = loss_target.to("cpu").detach()
                total_val_loss_target += loss_target.item() * x.shape[0]
                # loss_val.append(loss.item())

            total_val_loss_target /= len(dataset_val)
            total_val_loss_pde /= len(dataset_val)

            loss_history_validation["loss_target"].append(
                total_val_loss_target)
            loss_history_validation["loss_pde"].append(
                total_val_loss_target)

            loss_val = total_loss_target + \
                total_val_loss_pde * config["pde_scale"]

            # Make sure the validation prediction does not affect the training
            optimizer.zero_grad()
            if loss_val < best_validation:
                best_validation_epoch = epoch
                best_validation = loss_val
                best_model = copy.deepcopy(model.state_dict())

        sigmas.append(sigma.item())

    # Load best model based on validation data
    model.load_state_dict(best_model)

    validation_array = np.zeros(
        (nr_of_epochs // config["epochs_before_validation"], len(types_of_loss)))
    loss_array = np.zeros(
        (nr_of_epochs, len(types_of_loss)))
    # lambda_values = np.zeros((nr_of_epochs // config["update_lambda"] + 1, 3))

    for i, name in enumerate(types_of_loss):
        validation_array[:, i] = loss_history_validation[name]
        loss_array[:, i] = loss_history[name]

    if config["save_loss"]:
        np.save("results_backwards/loss_" + filename, loss_array)
        np.save("results_backwards/validation_" +
                filename, validation_array)

    return np.array(sigmas), best_validation_epoch


def try_multiple_activation_functions(config: dict, filename1: str, filename2: str,  activation_functions: list, layers: list, epochs=15_000):

    dataset_test = DataLoaderEuropean(config["test_filename"])

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
                config["N_INPUT"], 1, 128, layer, activation_function, use_fourier_transform=config["use_fourier_transform"],
                sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
            model.train(True)
            sigmas = train(model, epochs, config, black_scholes_1D_backwards)

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
            print(
                f"Trial {j + i*len(layers) + 1} / {len(layers)*len(activation_functions)} Done")
    np.savetxt(filename1, errors)
    np.save(filename2, all_sigmas)


def try_learning_rate_and_target_scale(config: dict, filename1: str, filename2: str, filename3,  learning_rates: list, target_scales: list, pde_dataloader, epochs=15_000):
    tmp_config = copy.deepcopy(config)
    dataset_test = DataLoaderEuropean(config["test_filename"])

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

    model = PINNforwards(config["N_INPUT"], 1, 128, 4, use_fourier_transform=config["use_fourier_transform"],
                         sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
    model = model.to(DEVICE)
    start_model = copy.deepcopy(model.state_dict())

    for i, learning_rate in enumerate(learning_rates):
        for j, target_scale in enumerate(target_scales):
            tmp_config["pde_scale"] = target_scale
            tmp_config["learning_rate"] = learning_rate

            model.load_state_dict(start_model)
            model.train(True)
            sigmas, epoch = train(model, epochs, tmp_config,
                                  black_scholes_1D_backwards, pde_dataloader)

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

            print("Sigma", sigmas[epoch - 1])
            errors[i, j] /= len(dataset_test)
            errors[i, j] = np.sqrt(errors[i, j])
            print(errors[i, j])
            best_epochs[i, j] = epoch
            all_sigmas[i * len(target_scales) + j, :] = sigmas

            print(f"Trial {j + i*len(target_scales) +
                  1} / {len(target_scales)*len(learning_rates)} Done")
    np.savetxt(filename1, errors)
    np.save(filename2, all_sigmas)
    np.savetxt(filename3, best_epochs)


def try_batch_sizes(config: dict, filename1: str, filename2: str, filename3,  batch_PDE: list, batch_target: list, dataloader, epochs=15_000):
    tmp_config = copy.deepcopy(config)
    dataset_test = DataLoaderEuropean(config["test_filename"])

    dataloader_test = DataLoader(
        dataset_test, batch_size=256, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    errors = np.zeros((len(batch_PDE), len(batch_target)))
    best_epochs = np.zeros((len(batch_PDE), len(batch_target)))
    all_sigmas = np.zeros((len(batch_PDE) * len(batch_target), epochs))

    time_range, S_range = config["t_range"], config["S_range"]
    min_values = torch.tensor(
        [time_range[0], S_range[0]]).to(DEVICE)
    max_values = torch.tensor(
        [time_range[1], S_range[1]]).to(DEVICE)

    model = PINNforwards(config["N_INPUT"], 1, 128, 4, use_fourier_transform=config["use_fourier_transform"],
                         sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
    model = model.to(DEVICE)
    start_model = copy.deepcopy(model.state_dict())

    for i, pde_batch in enumerate(batch_PDE):
        for j, target_batch in enumerate(batch_target):

            tmp_config["batch_size"] = target_batch
            tmp_config["PDE_batch"] = pde_batch

            model.load_state_dict(start_model)
            model.train(True)
            sigmas, epoch = train(model=model, nr_of_epochs=epochs, config=tmp_config,
                                  PDE=black_scholes_1D_backwards, pde_dataloader=dataloader)

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

            print("Sigma", sigmas[epoch - 1])
            errors[i, j] /= len(dataset_test)
            errors[i, j] = np.sqrt(errors[i, j])
            print("RMSE", errors[i, j])
            best_epochs[i, j] = epoch
            all_sigmas[i * len(batch_target) + j, :] = sigmas

            print(f"Trial {j + i*len(batch_target) +
                  1} / {len(batch_target)*len(batch_PDE)} Done")

    np.savetxt(filename1, errors)
    np.save(filename2, all_sigmas)
    np.savetxt(filename3, best_epochs)


def train_multiple_times(seeds: list[int], layers: int, nodes: int, PDE, filename: str,
                         nr_of_epochs: int, pde_dataloader, config: dict, custom_arc: list[int] = None, save_all_sigma=False):
    cur_config = config.copy()
    cur_config["save_loss"] = True
    cur_config["save_model"] = False

    types_of_loss = ["loss_pde", "loss_target"]

    dataset_test = DataLoaderEuropean(config["test_filename"])

    dataloader_test = DataLoader(
        dataset_test, batch_size=256, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    results_train = np.zeros(
        (len(seeds), nr_of_epochs, len(types_of_loss)))
    results_val = np.zeros(
        (len(seeds), nr_of_epochs // config["epochs_before_validation"], len(types_of_loss)))

    rmse_data = np.zeros(len(seeds))
    sigmas_results = np.zeros((len(seeds), nr_of_epochs))

    best_test_MSE = float("inf")
    best_model = None
    time_range, S_range = config["t_range"], config["S_range"]
    min_values = torch.tensor(
        [time_range[0], S_range[0]]).to(DEVICE)
    max_values = torch.tensor(
        [time_range[1], S_range[1]]).to(DEVICE)

    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PINNforwards(N_INPUT=cur_config["N_INPUT"], N_OUTPUT=1, N_HIDDEN=nodes, N_LAYERS=layers,
                             use_fourier_transform=config["use_fourier_transform"], sigma_FF=config["sigma_fourier"],
                             encoded_size=config["fourier_encoded_size"], custom_arc=custom_arc)
        model = model.to(DEVICE)

        sigmas, _ = train(model=model, nr_of_epochs=nr_of_epochs, PDE=PDE, pde_dataloader=pde_dataloader, config=cur_config, filename=str(
            seed))
        sigmas_results[i] = sigmas

        cur_train = np.load(f"results_backwards/loss_{seed}.npy")
        cur_val = np.load(f"results_backwards/validation_{seed}.npy")

        results_train[i] = cur_train
        results_val[i] = cur_val

        model.train(False)
        model.eval()

        for batch_idx, (x, y) in enumerate(dataloader_test):
            with torch.no_grad():
                x, y = x.to(DEVICE), y.to(DEVICE)
                x_scaled = (x - min_values) / (max_values - min_values)
                y = y.unsqueeze(1)
                y_hat = model.forward(x_scaled)

                rmse_data[i] += torch.sum(
                    (y - y_hat)**2).item()

        rmse_data[i] /= len(dataset_test)
        rmse_data[i] = np.sqrt(rmse_data[i])

        os.remove(f"results_backwards/loss_{seed}.npy")
        os.remove(f"results_backwards/validation_{seed}.npy")

        if rmse_data[i] < best_test_MSE:
            best_test_MSE = rmse_data[i]
            best_model = copy.deepcopy(model.state_dict())

        print(f"Run {i + 1} / {len(seeds)} done")
        print(f"RMSE : {rmse_data[i]:.2e} \n")

    if config["save_model"]:
        torch.save(best_model, f"models/" + filename + ".pth")

    np.save("results_backwards/average_loss_" +
            filename, np.vstack([results_train.mean(axis=0), results_train.std(axis=0)]))

    np.save("results_backwards/average_validation_" +
            filename, np.vstack([results_val.mean(axis=0), results_val.std(axis=0)]))

    # print(np.vstack([sigmas_results.mean(axis=0),
    #      sigmas_results.std(axis=0)]).shape)
    np.save("results_backwards/average_sigma_" + filename,
            np.vstack([sigmas_results.mean(axis=0), sigmas_results.std(axis=0)]))

    np.savetxt("results_backwards/rmse_data_" + filename + ".txt", rmse_data)

    if save_all_sigma is True:
        np.save("results_backwards/sigma_" + filename, sigmas_results)


def try_adding_noise(config: dict, filename1: str, filename2: str, filename3,  traing_noise, dataloader, epochs=15_000):
    tmp_config = copy.deepcopy(config)
    dataset_test = DataLoaderEuropean(config["test_filename"])

    dataloader_test = DataLoader(
        dataset_test, batch_size=256, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    errors = np.zeros(len(traing_noise))
    best_epochs = np.zeros(len(traing_noise))
    all_sigmas = np.zeros((len(traing_noise), epochs))

    time_range, S_range = config["t_range"], config["S_range"]
    min_values = torch.tensor(
        [time_range[0], S_range[0]]).to(DEVICE)
    max_values = torch.tensor(
        [time_range[1], S_range[1]]).to(DEVICE)

    model = PINNforwards(config["N_INPUT"], 1, 128, 4, use_fourier_transform=config["use_fourier_transform"],
                         sigma_FF=config["sigma_fourier"], encoded_size=config["fourier_encoded_size"])
    model = model.to(DEVICE)
    start_model = copy.deepcopy(model.state_dict())

    for i, cur_noise in enumerate(traing_noise):
        tmp_config["training_noise"] = cur_noise

        model.load_state_dict(start_model)
        model.train(True)
        sigmas, epoch = train(model=model, nr_of_epochs=epochs, config=tmp_config,
                              PDE=black_scholes_1D_backwards, pde_dataloader=dataloader)

        model.train(False)
        model.eval()
        for batch_idx, (x, y) in enumerate(dataloader_test):
            with torch.no_grad():
                x, y = x.to(DEVICE), y.to(DEVICE)
                x_scaled = (x - min_values) / (max_values - min_values)
                y = y.unsqueeze(1)
                y_hat = model.forward(x_scaled)

                errors[i] += torch.sum(
                    (y - y_hat)**2).item()

        print("Sigma", sigmas[epoch - 1])
        errors[i] /= len(dataset_test)
        errors[i] = np.sqrt(errors[i])
        print("RMSE", errors[i])
        best_epochs[i] = epoch
        all_sigmas[i, :] = sigmas

        print(f"Trial {i + 1} / {len(traing_noise)} Done")

    np.savetxt(filename1, errors)
    np.save(filename2, all_sigmas)
    np.savetxt(filename3, best_epochs)


if __name__ == "__main__":
    config = {"train_filename": "data/european_one_dimensional_train.npy",
              "val_filename": "data/european_one_dimensional_val.npy",
              "test_filename": "data/european_one_dimensional_test.npy",
              "learning_rate": 1e-3,
              "weight_decay": 0.0,
              "gamma": 0.99,
              "r": 0.04,
              "epochs_before_validation": 10,
              "N_INPUT": 2,
              "S_range": [0, 400],
              "t_range": [0, 1],
              "DEVICE": DEVICE,
              "pde_scale": 1e-4,
              "use_fourier_transform": True,
              "sigma_fourier": 5.0,
              "fourier_encoded_size": 128,
              "batch_size": 64,
              "PDE_batch": 1024,
              "PDE_epochs": 1_000,
              "save_model": False,
              "save_loss": False,
              "use_target_points_for_PDE": False,
              "training_noise": 0.0}

    # model.to(DEVICE)

    """ torch.manual_seed(1000)
    np.random.seed(1000)
    pde_dataloader = DataGeneratorEuropean1D(
        config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
    try_batch_sizes(config, "important_results_backwards/RMSE_batch_fine.txt",
                    "important_results_backwards/sigmas_batch_fine.npy", "important_results_backwards/epochs_batch_fine.txt",
                    batch_PDE=[0, 128, 256, 512, 1024], batch_target=[32, 64, 96, 128],
                    dataloader=pde_dataloader, epochs=15_000) """

    """ torch.manual_seed(1000)
    np.random.seed(1000)
    pde_dataloader = DataGeneratorEuropean1D(
        config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
    config["use_target_points_for_PDE"] = True
    try_batch_sizes(config, "important_results_backwards/RMSE_same.txt",
                    "important_results_backwards/sigmas_same.npy", "important_results_backwards/epochs_same.txt",
                    batch_PDE=[0], batch_target=[16, 32, 64, 96, 128],
                    dataloader=pde_dataloader, epochs=15_000)
    config["use_target_points_for_PDE"] = False """

    # config["gamma"] = 1
    """ torch.manual_seed(1000)
    np.random.seed(1000)
    pde_dataloader = DataGeneratorEuropean1D(
        config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
    try_adding_noise(config, "important_results_backwards/RMSE_noise.txt",
                     "important_results_backwards/sigmas_noise.npy", "important_results_backwards/epochs_noise.txt",
                     [1e-3, 1e-2, 1e-1, 0.0, 1.0, 5.0, 10.0, 20.0], pde_dataloader, 15_000) """

    """ for pde_scale in [1e-6]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [1e-5]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [1e-4]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config, save_all_sigma=True) """

    """ for pde_scale in [1e-3]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [1e-2]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [1e-1]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [0.0]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [10]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [100]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [1000]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config, save_all_sigma=True) """

    """ for pde_scale in [5_000]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    """ for pde_scale in [10_000]:
        config["pde_scale"] = pde_scale
        pde_dataloader = DataGeneratorEuropean1D(
            config["t_range"], config["S_range"], K=None, r=config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
        train_multiple_times(seeds=list(range(1, 10 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                             filename=f"scale_{pde_scale}", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=config) """

    apple_config = copy.deepcopy(config)
    apple_config["train_filename"] = "data/apple_data_train.npy"
    apple_config["val_filename"] = "data/apple_data_val.npy"
    apple_config["test_filename"] = "data/apple_data_test.npy"
    apple_config["S_range"] = [90.34, 258.46]
    apple_config["r"] = 2.20
    apple_config["t_range"] = [0.0, 2.42]
    apple_config["save_model"] = True
    apple_config["use_fourier_transform"] = True

    """ apple_config["pde_scale"] = 1e-4
    torch.manual_seed(1000)
    np.random.seed(1000)
    pde_dataloader = DataGeneratorEuropean1D(
        apple_config["t_range"], apple_config["S_range"], K=None, r=apple_config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
    train_multiple_times(seeds=list(range(1, 1 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                         filename=f"apple_data_4_no_fourier", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=apple_config) """

    """ apple_config["pde_scale"] = 0.0
    torch.manual_seed(1000)
    np.random.seed(1000)
    pde_dataloader = DataGeneratorEuropean1D(
        apple_config["t_range"], apple_config["S_range"], K=None, r=apple_config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
    train_multiple_times(seeds=list(range(1, 1 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                         filename=f"apple_data_0_no_fourier", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=apple_config) """

    """ apple_config["pde_scale"] = 1e-3
    torch.manual_seed(1000)
    np.random.seed(1000)
    pde_dataloader = DataGeneratorEuropean1D(
        apple_config["t_range"], apple_config["S_range"], K=None, r=apple_config["r"], sigma=None, DEVICE=DEVICE, seed=1000)
    train_multiple_times(seeds=list(range(1, 1 + 1)), layers=4, nodes=128, PDE=black_scholes_1D_backwards,
                         filename=f"apple_data_lambda_3_no_fourier", nr_of_epochs=15_000, pde_dataloader=pde_dataloader, config=apple_config) """
