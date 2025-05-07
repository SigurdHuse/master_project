# Master thesis
This repository contains code for the master thesis: 

**Physics-informed neural networks for option pricing and volatility calibration using the Black–Scholes PDE**
 
by Sigurd Kjelsbøl Huse 

Supervised by Håkon Andreas Hoel. 


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
  year         = {2025}, 
  type         = {Master’s thesis}, 
  address      = {Oslo, Norway}, 
  supervisor   = {Håkon Andreas Hoel} 
}
```
