# Master thesis
This repository contains code for the master thesis: 

''Physics-informed neural networks for option pricing and volatility calibration using the Black–Scholes PDE'', 

written by Sigurd Kjelsbøl Huse and supervised by  Håkon Andreas Hoel. 


## Installation 
Code was ran with python 3.12.4, to install packages run command:

```Terminal
> pip install -r requirements.txt
```

## Generate data and plots from thesis
Run commands:

```Terminal
> python3 src/experiments_european_one_dimensional.py
> python3 src/experiments_american_one_dimensional.py
> python3 src/experiments_european_multi_dimensional.py
> python3 src/train_backwards.py
> python3 src/plotter.py
```


## Hardware used
All code was run on:

Machine learning infrastructure (ML Nodes), University Centre for Information Technology, University Of Oslo, Norway.

Using RTX2080Ti graphic cards and AMD EPYC 7282 16-Core processors.

## Structure of repository
```text
├── data                           # Data used to train models for the inverse problem\
├── important_results              # Folder with results from the forward problem \
│   ├── american_1D                # American put option results \
│   ├── european_1D                # European call option results \
│   └── european_multi             # Multidimensional geometric mean results\
├── important_results_backwards    # Folder with results from the inverse problem \
├── models                         # Folder with trained models used for plotting \
├── plots                          # Folder with generated plots \
├── raw_apple_data                 # Apple stock American call options (raw data) \
├── results                        # Folder with results from training for the forward problem \
├── results_backwards              # Folder with results from training for the inverse problem \
├── src                            # Folder with code \
│   ├── data_generator.py          # Used to sample data points \
│   ├── dataloader.py              # Loads the dataset for the inverse problem \
│   ├── experiments_american_one_dimensional.py   # American put option experiments\
│   ├── experiments_european_multi_dimensional.py # European call option experiments\
│   ├── experiments_european_one_dimensional.py   # Multidimensional geometric mean call option experiments\
│   ├── generate_training_data.py  # Creates datasets for the inverse problem\
│   ├── PINN.py                    # Data class to represent physics-informed neural network \
│   ├── plotter.py                 # Creates plots displayed in thesis \
│   ├── train_backwards.py         # Training functions and experiments for the inverse problem\
│   ├── training_functions.py      # Helper functions for training \
│   └── train.py                   # Training functions for the forward problem \
└── Various_plots_and_apple_data.ipynb # Notebook with some plots and extraction of Apple data\
```
