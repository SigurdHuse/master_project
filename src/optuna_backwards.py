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
          "learning_rate": 1e-3,
          "weight_decay": 0,
          "gamma": 0.9,
          "r": 0.04,
          "epochs_before_validation": 10,
          "scheduler_step": 20,
          "N_INPUT": 2,
          "true_sigma": 0.5}
NR_OF_MODELS = 0

def define_model(trial):
    print(f"Model loaded!")
    n_layers = trial.suggest_int("n_layers", 1, 4)
    n_nodes = trial.suggest_int("n_nodes", 50, 600, step=50)

    model = PINNbackwards(2, 1, n_nodes, n_layers)

    return model


def objective(trial):
    model = define_model(trial).to(DEVICE)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 10, 120, step = 10)
    gamma_lr = trial.suggest_float("gamma_lr", 0.8, 0.95)

    dataset = DataLoaderEuropean(config["train_filename"])

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    dataset_val = DataLoaderEuropean(config["val_filename"])

    dataloader_val = DataLoader(
        dataset_val, batch_size=128, num_workers=10, pin_memory=True, generator=torch.Generator(device='cuda'))

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    scheduler = ExponentialLR(optimizer, gamma_lr)

    loss_function = torch.nn.MSELoss()
    best_validation = float("inf")

    for epoch in range(1, 1_000 + 1):
        model.train(True)

        for batch_idx, (x, y) in enumerate(dataloader, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.requires_grad_(True)
            y = y.unsqueeze(1)
            y_hat, bs_pde = model.foward_with_european_1D(x, config["r"])

            loss_pde = loss_function(bs_pde, torch.zeros_like(bs_pde))
            loss_target = loss_function(y_hat, y)

            loss = loss_pde + loss_target

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % config["scheduler_step"] == 0:
            scheduler.step()

        model.train(False)
        model.eval()

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

            loss = loss_pde + loss_target
            total_loss_val += loss

        trial.report(total_loss_val, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return total_loss_val


if __name__ == "__main__":
    SEED = 2024
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    study = optuna.create_study(direction="minimize", storage="sqlite:///db.sqlite3", study_name = "backwards_problem")
    study.optimize(objective, n_trials=100, timeout=600)

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
