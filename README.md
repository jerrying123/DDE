## Data-Driven Encoding

This repository contains the source code for the paper "Data-Driven Encoding: A New Numerical Method for Computation of the Koopman Operator" (https://arxiv.org/abs/2301.06542). 

The pendulum folder corresponds to a physical system represented below:

![pendulum](images/pendulum.png)

In the pendulum folder, there are three subfolders, each pertaining to different data distributions mentioned in the paper. These folders are:
- `gaussian`: This folder contains the data for the truncated Gaussian distribution.
- `uniform`: This folder contains the data for the uniform distribution.
- `trajectories`: This folder contains the data for the trajectories.

The winchbot folder corresponds to the physical system represented below:

![winchbot](images/winch.png)

In the data distribution folders and in the `winchbot` folder, there are a few common scripts. These scripts include:

- `data_DE.py`: This script runs the data-driven encoding algorithm on the set of data located in the `data` folder of each system.
- `calc_EDMD.py`: This script runs the extended dynamic mode decomposition algorithm. 
- `gen_error_space.py`: This script generates the error plots over the state space dynamic range of each method. This is not in the `winchbot` folder because the state space is of too high dimension to plot easily.


