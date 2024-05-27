# FedMAP: Federated Maximum A Posteriori

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

FedMAP (Federated Maximum A Posteriori) is a novel federated learning (FL) algorithm that incorporates a global prior distribution over the local model parameters, enabling personalized FL. This repository contains the implementation of the FedMAP algorithm.

## Overview

The FedMAP algorithm consists of three main steps:

1. **Initialization**: A client is randomly selected, and its model parameters are used to initialize the global model and the local model parameters for all clients.

2. **Local Optimization**: Each client optimizes their model parameters by minimizing the negative log-likelihood of the posterior distribution, which includes a prior term that penalizes deviations from the global model parameters.

3. **Global Aggregation**: The server aggregates the optimized local model parameters from all clients using a weighted average, where the weights are the weighting factors computed during the Local Optimization step. The updated global model is then broadcast to all clients for the next round of Local Optimization.

## Repository Structure

- 📂 FedMAP
   - 📜 fedmap.py              # Contains the implementation of the FedMAP algorithm
   - 📜 utils.py               # Utility functions for data preprocessing, model initialization, and other helper functions
   - 📂 models                 # Directory containing the model architectures used in the experiments
   - 📂 datasets               # Directory containing the datasets used in the experiments
   - 📂 experiments            # Directory containing scripts for running experiments and reproducing results
   - 📜 README.md              # The file you're currently reading
   - 📜 .gitignore             # Specifies intentionally untracked files to ignore

## Installation

To run the experiments:
`python simulation.py`

