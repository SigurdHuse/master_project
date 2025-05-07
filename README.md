# Master thesis
This repository contains code for the master thesis: 

**Physics-informed neural networks for option pricing and volatility calibration using the Black–Scholes PDE**
 
by Sigurd Kjelsbøl Huse 

Supervised by Håkon Andreas Hoel. 

## Abstract

This thesis investigates the use of physics-informed neural networks (PINNs) for option pricing and volatility calibration using the Black-Scholes partial differential equation (PDE). The forward problem is addressed by solving the PDE for a European call option, an American put option, and a multidimensional geometric mean call option. The inverse problem focuses on estimating a constant volatility from European call option prices, using both synthetic and real market data. The experiments are motivated by the potential of PINNs to provide a data driven meshfree modeling approach that incorporates both observational data and prior knowledge through the inclusion of the PDE residual in the loss function.

In the forward setting, using PINNs, we achieve root mean square errors (RMSE) of $2.10\cdot10^{-3}$ for the European call, $4.28 \cdot 10^{-2}$ for the American put, and $1.57 \cdot 10^{-2}$ for the multidimensional geometric mean call. By evaluating computation times across increasing dimensionality, we find that PINNs appear to overcome the curse of dimensionality that limits traditional numerical methods. For the inverse problem, the volatility is inferred by minimizing the error in the predicted option price. On synthetic data, the PINNs achieves an RMSE of $4.37 \cdot 10^{-2}$, outperforming a standard feedforward neural network (FFNN), which produces $1.44 \cdot 10^1$. For real market data, PINNs achieves an RMSE of $8.44 \cdot 10^{-1}$, only marginally better than the FFNN result of $8.94 \cdot 10^{-1}$. However, in both cases, the PINN fails to recover the true constant volatility, indicating that, while it can fit the option prices accurately, it does not learn the underlying hidden PDE parameter.

## Installation 
Code was ran with python 3.12.4, install dependencies with:

```bash
> pip install -r requirements.txt
```

## Generate data and plots from thesis
Run commands:

```bash
> python3 src/experiments_european_one_dimensional.py
> python3 src/experiments_american_one_dimensional.py
> python3 src/experiments_european_multi_dimensional.py
> python3 src/train_backwards.py
> python3 src/plotter.py
```


## Hardware used
All experiments were run on:

Machine learning infrastructure (ML Nodes), University Centre for Information Technology, University Of Oslo, Norway.
equipped with:

- NVIDIA RTX 2080 Ti GPUs

- AMD EPYC 7282 16-Core CPUs

## Repository Structure
```text
├── data                           # Training data for inverse problem \
├── important_results              # Results from forward problem experiments \
│   ├── american_1D                # American put option results \
│   ├── european_1D                # European call option results \
│   └── european_multi             # Multidimensional geometric mean results\
├── important_results_backwards    # Results from the inverse problem \
├── models                         # Saved PINN models used for plotting \
├── plots                          # Generated figures \
├── raw_apple_data                 # Raw Apple (AAPL) American call option dataset \
├── results                        # Training outputs for forward problems \
├── results_backwards              # Training outputs for inverse problems \
├── src                            # Source code \
│   ├── data_generator.py          # Point sampling for forward/inverse problems \
│   ├── dataloader.py              # Dataset loader for the inverse problem \
│   ├── experiments_american_one_dimensional.py   # 1D American put experiments\
│   ├── experiments_european_multi_dimensional.py # 1D European call experiments\
│   ├── experiments_european_one_dimensional.py   # Multidimensional geometric mean call option experiments\
│   ├── generate_training_data.py  # Creates datasets for the inverse problem\
│   ├── PINN.py                    # Class to represent a physics-informed neural network \
│   ├── plotter.py                 # Creates plots displayed in thesis \
│   ├── train_backwards.py         # Training functions and experiments for the inverse problem\
│   ├── training_functions.py      # Helper functions for training \
│   └── train.py                   # Training functions for the forward problem \
└── Various_plots_and_apple_data.ipynb # Notebook with some plots and extraction of Apple data\
```

## How to cite
```bibtex
@mastersthesis{huse2025, 
  author       = {Sigurd Kjelsbøl Huse},
  title        = {Physics-informed neural networks for option pricing and volatility calibration using the Black–Scholes PDE},
  school       = {University of Oslo},
  month        = {May},
  year         = {2025}, 
  type         = {Master’s thesis}, 
  address      = {Oslo, Norway}, 
  supervisor   = {Håkon Andreas Hoel} 
}
```
