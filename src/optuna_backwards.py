from torch.optim.lr_scheduler import ExponentialLR
from PINN import PINNbackwards
import optuna
import numpy as np
import torch
import torch.optim as optim
from dataloader import DataLoaderEuropean
from torch.utils.data import DataLoader
from optuna.trial import TrialState


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


config = {"train_filename": "data/european_one_dimensional_train.npy",
          "val_filename": "data/european_one_dimensional_val.npy",
          "test_filename": "data/european_one_dimensional_test.npy",
          "r": 0.04,
          "epochs_before_validation": 5,
          "scheduler_step": 100,
          "N_INPUT": 2,
          "true_sigma": 0.5}

def define_model(trial):
    print(f"Model loaded!")
    n_layers = trial.suggest_int("n_layers", 2, 6)
    n_nodes = trial.suggest_int("n_nodes", 128, 768)
    #initial_sigma = trial.suggest_float("initial_sigma", 1.0, 3.0)

    model = PINNbackwards(2, 1, n_nodes, n_layers, initial_sigma=3.0)

    return model


def objective(trial):
    model = define_model(trial).to(DEVICE)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 48)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-4)
    gamma_lr = 0.9 # trial.suggest_float("gamma_lr", 0.8, 0.99)
    pde_loss_weight = trial.suggest_float("pde_loss_weight", 10.0, 250.0)

    dataset = DataLoaderEuropean(config["train_filename"])

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    dataset_val = DataLoaderEuropean(config["val_filename"])

    dataloader_val = DataLoader(
        dataset_val, batch_size=128, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]) # , , "SGD"
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ExponentialLR(optimizer, gamma_lr)
    
    loss_function = torch.nn.MSELoss()

    best_validation = float("inf")
    validation_sigma_error = 100

    for epoch in range(1, 1_200 + 1):
        model.train(True)

        for batch_idx, (x, y) in enumerate(dataloader, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.requires_grad_(True)
            y = y.unsqueeze(1)
            y_hat, bs_pde = model.foward_with_european_1D(x, config["r"])

            loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
            loss_target = loss_function(y_hat, y)

            loss = pde_loss_weight*loss_pde + loss_target

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % config["scheduler_step"] == 0:
            scheduler.step()

        model.train(False)
        #model.eval()

        if epoch % config["epochs_before_validation"] == 0:
            total_loss_val = 0
            for batch_idx, (x, y) in enumerate(dataloader_val):
                with torch.no_grad():
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    x = x.requires_grad_(True)
                    y = y.unsqueeze(1)
                y_hat, bs_pde = model.foward_with_european_1D(
                    x, config["r"])

                loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
                loss_target = loss_function(y_hat, y)

                loss = pde_loss_weight*loss_pde + loss_target
                total_loss_val += loss

            if total_loss_val < best_validation:
                best_validation = total_loss_val

                validation_sigma_error = np.abs(config["true_sigma"] - model.sigma.item())
            #print("sigma:", model.sigma.item(), sigma_error)
            trial.report(validation_sigma_error, step = epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return validation_sigma_error


if __name__ == "__main__":
    SEED = 2024
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    study = optuna.create_study(direction="minimize", storage="sqlite:///db.sqlite3", study_name = "backwards_problem_sigma_48_finer_with_val_correct")
    study.optimize(objective, n_trials=None, timeout=3600*48)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
