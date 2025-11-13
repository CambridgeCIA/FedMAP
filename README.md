# FedMAP: Federated Maximum A Posteriori

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

FedMAP (Federated Maximum A Posteriori) is a novel federated learning (FL) algorithm that incorporates a global prior distribution over local model parameters using Input Convex Neural Networks (ICNNs), enabling personalized federated learning. This repository contains the complete implementation of the FedMAP algorithm with support for multiple healthcare datasets and tasks.

## Overview

The FedMAP algorithm consists of three main steps:

1. **Initialisation**: A client is randomly selected, and its model parameters are used to initialise the global model and the local model parameters for all clients.

2. **Local Optimisation**: Each client optimises their model parameters by minimizing the negative log-likelihood of the posterior distribution, which includes an ICNN-based prior term that penalizes deviations from the global model parameters.

3. **Global Aggregation**: The server aggregates the optimized local model parameters from all clients using a weighted average, where the weights are the contribution scores computed during local optimisation. The updated global model and ICNN modules are then broadcast to all clients for the next round.

## Key Features

- **ICNN-based Prior**: Adaptive prior distribution using Input Convex Neural Networks
- **Multiple Task Support**: Classification, binary classification, and survival analysis (Cox Proportional Hazards)
- **Healthcare Applications**: Pre-configured for INTERVAL, eICU, CPRD, and synthetic datasets
- **Three-Tier Deployment**: Support for training, fine-tuning, and inference workflows
- **Flexible Architecture**: Modular design supporting various neural network architectures

## Repository Structure

```
FedMAP/
├── 📂 src/
│   ├── 📂 client/           # Client-side federated learning logic
│   │   └── client_app.py    # Client training and evaluation handlers
│   ├── 📂 server/           # Server-side aggregation and orchestration
│   │   └── server_app.py    # Server initialisation and FedMAP strategy
│   ├── 📂 strategies/       # Federated learning aggregation strategies
│   │   └── fedmap.py        # FedMAP strategy with ICNN training
│   ├── 📂 models/           # Neural network architectures
│   │   ├── iron_classifier.py      # Dense classifier for INTERVAL
│   │   ├── multimodal_ffn.py       # Multi-modal network for eICU
│   │   ├── example_classifier.py   # Simple MLP for examples
│   │   └── __init__.py
│   ├── 📂 loss_modules/     # Loss functions and priors
│   │   └── map.py           # FedMAP loss with ICNN prior
│   ├── 📂 tasks/            # Task-specific implementations
│   │   ├── interval.py      # Iron deficiency classification (INTERVAL)
│   │   ├── eicu.py          # ICU mortality prediction (eICU)
│   │   ├── cprd.py          # CVD risk prediction (CPRD)
│   │   └── example.py       # Synthetic data example
│   ├── 📂 tiers/            # Multi-tier deployment scripts
│   │   ├── tier2_finetune.py  # Fine-tuning on new clients
│   │   └── tier3_infer.py     # Inference on unseen clients
│   ├── 📂 utils/            # Utility functions
│   │   └── train_helper.py  # ICNN implementation and training utilities
│   └── 📂 checkpoints/      # Model checkpoints (created during training)
├── 📂 config/               # Configuration files
│   └── 📂 task/
│       ├── interval.toml    # INTERVAL task configuration
│       ├── eicu.toml        # eICU task configuration
│       ├── cprd.toml        # CPRD task configuration
│       └── example.toml     # Example task configuration
├── 📂 datasets/             # Data directory (not included in repo)
│   ├── interval/
│   ├── eicu/
│   ├── cprd/
│   └── example/
├── 📂 scripts/              # Execution scripts
│   ├── run_t1.sh           # Run Tier 1 (training)
│   ├── run_t2.sh           # Run Tier 2 (fine-tuning)
│   └── run_t3.sh           # Run Tier 3 (inference)
├── 📂 results/              # Output metrics (created during execution)
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Docker Compose setup
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Python dependencies
└── README.md               
```

## Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional but recommended)
- NVIDIA Container Toolkit (for GPU support)

### Option 1: Docker Installation (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FedMAP
   ```

2. **Build and start the Docker container**
   ```bash
   docker compose up --build
   ```
   
   This will:
   - Build a Docker image based on PyTorch 2.3.0 with CUDA 12.1
   - Install Python 3.10 and all required dependencies
   - Mount the current directory to `/app` in the container
   - Enable GPU support (if available)

3. **Access the container**
   
   Open a new terminal and run:
   ```bash
   docker exec -it fedmap_container bash
   ```

### Option 2: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FedMAP
   ```

2. **Create a virtual environment (Python 3.10)**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```
   
   Or if using requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following main packages:
- `flwr[simulation]>=1.8.0, <2.0` - Flower federated learning framework
- `torch>=2.3.0` - PyTorch deep learning framework
- `hydra-core==1.3.0` - Configuration management
- `numpy==2.2.6` - Numerical computing
- `pandas==2.3.0` - Data manipulation
- `scikit-learn==1.7.0` - Machine learning utilities
- `pycox==0.3.0` - Survival analysis (for CPRD task)
- `torchtuples==0.2.2` - PyTorch utilities

## Dataset Setup

Before running experiments, you need to prepare your datasets. Place your data in the `datasets/` directory with the following structure:

### INTERVAL Dataset (Iron Deficiency Classification)
```
datasets/interval/
├── INTERVAL_irondef_site_1_train.csv
├── INTERVAL_irondef_site_1_val.csv
├── INTERVAL_irondef_site_2_train.csv
├── INTERVAL_irondef_site_2_val.csv
├── INTERVAL_irondef_site_3_train.csv
└── INTERVAL_irondef_site_3_val.csv
```

### eICU Dataset (ICU Mortality Prediction)
```
datasets/eicu/group_A/
├── {hospital_id}/
│   ├── mortality_train.csv
│   ├── mortality_val.csv
│   ├── medications_train.csv
│   ├── medications_val.csv
│   ├── diagnosis_train.csv
│   ├── diagnosis_val.csv
│   ├── physio_train.csv
│   └── physio_val.csv
```

### CPRD Dataset (CVD Risk Prediction)
```
datasets/cprd/
└── risk_factors_all.csv
```

### Example Dataset (Synthetic Data)
```
datasets/example/
├── partition_0_train.csv
├── partition_0_test.csv
├── partition_1_train.csv
├── partition_1_test.csv
└── ...
```

## Usage

### Tier 1: Federated Training

Train the global model with FedMAP across multiple clients:

**Using Flower CLI:**
```bash
flwr run . --run-config="./config/task/interval.toml"
```

**Using shell script:**
```bash
bash scripts/run_t1.sh
```

**Configuration options** (in `config/task/[task_name].toml`):
- `task-name`: Task identifier (interval, eicu, cprd, example)
- `model`: Model architecture to use
- `num-server-rounds`: Number of federated learning rounds
- `local-epochs`: Number of local training epochs per round
- `fraction-evaluate`: Fraction of clients to use for evaluation
- `learning-rate`: Learning rate for local optimisation
- `batch-size`: Batch size for training

### Tier 2: Fine-tuning on New Clients

Fine-tune the trained global model on new clients with ICNN prior:

```bash
bash scripts/run_t2.sh
```

This will:
- Load the trained global model from `src/checkpoints/global_model_{task_name}.pt`
- Load the trained ICNN modules from `src/checkpoints/icnn_modules.pt`
- Fine-tune on clients
- Save metrics to `results/{task_name}_metrics_test.csv`

### Tier 3: Inference on Unseen Clients

Evaluate the global model on completely new clients without fine-tuning:

```bash
bash scripts/run_t3.sh
```

This will:
- Load the trained global model
- Evaluate on clients 
- Save metrics to `results/{task_name}_metrics_test.csv`

## Algorithm Details

### FedMAP Objective Function

The FedMAP algorithm (Tier 1) involves each client $k$ minimising a local objective function. This objective is based on Maximum a Posteriori (MAP) estimation and combines the local data loss with a global, learnable ICNN-based prior (see Eq 7 in the paper):

$$
\theta_{k}^{*} = \underset{\theta \in \Theta}{\arg\min} \left( \frac{1}{N_{k}}\mathcal{L}(\theta; Z_{k}) + \mathcal{R}(\theta; \mu, \psi) \right)
$$


Where:

-   $\mathcal{L}(\theta; Z_{k})$: The local data loss (negative log-likelihood) on the local dataset $Z_k$.
-   $\mathcal{R}(\theta; \mu, \psi)$: The ICNN-based prior term, which regularises the local parameters $\theta$.
-   $\theta$: Local model parameters.
-   $\mu$: Global model parameters.
-   $\psi$: ICNN (prior) parameters.
-   $N_k$: The number of data points at site $k$.

### Contribution Calculation

Each client's contribution weight $\omega_k$ for the global aggregation step is calculated based on the posterior probability, which is proportional to the likelihood multiplied by the prior (see Eq 8 in the paper):

$$
\omega_{k} = \mathbb{P}(Z_{k} | \theta_{k}) \times \exp(-\mathcal{R}(\theta_{k}; \mu, \psi))
$$

This quantifies how well the local model fits its data (the likelihood $\mathbb{P}(Z_{k} | \theta_{k})$) while adhering to the global prior (the $\exp(-\mathcal{R}(\...))$ term).

### ICNN Prior

The ICNN (Input Convex Neural Network) provides a learned, convex prior. As defined in Equation 6 of the paper, the regulariser $\mathcal{R}$ is:

$$
\mathcal{R}(\theta; \mu, \psi) = f_{\psi}(\theta, \mu) + \alpha\|\theta - \mu\|^{2} + \epsilon(\|\theta\|^{2} + \|\mu\|^{2})
$$

Where $f_{\psi}$ is the ICNN itself, and $\alpha$ and $\epsilon$ are hyperparameters that ensure strong convexity.

## Tested Tasks

### 1. INTERVAL (Iron Deficiency Classification)
- **Model**: DenseClassifier (2-layer dense network)
- **Task**: Binary classification of iron deficiency
- **Features**: 18 features (16 haematology + age + sex)


### 2. eICU (ICU Mortality Prediction)
- **Model**: MultimodalFFN (multi-modal feedforward network)
- **Task**: Binary classification of ICU mortality
- **Modalities**: Medications (1411), Diagnosis (686), Physiology (7)

### 3. CPRD (CVD Risk Prediction)
- **Model**: CoxPH (Cox Proportional Hazards)
- **Task**: Survival analysis for cardiovascular disease risk
- **Features**: 8 risk factors (age, sex, SBP, cholesterol, etc.)


### 4. Default Example (Synthetic Data)
- **Model**: MLP (3-layer multilayer perceptron)
- **Task**: Binary classification
- **Features**: 31 synthetic features


## Outputs

### Checkpoints
Training saves the following checkpoints:
- `src/checkpoints/global_model_{task_name}.pt` - Final global model
- `src/checkpoints/icnn_modules.pt` - Trained ICNN prior modules


### Metrics
Performance metrics are saved to CSV files:
- `results/{task_name}_metrics_test.csv` - Test/validation metrics per round
- `results/{task_name}_metrics_train.csv` - Training metrics (if logged)

Metrics include:
- Loss, Accuracy, Balanced Accuracy
- ROC AUC, AUPRC (for classification tasks)
- Concordance Index, Integrated Brier Score (for survival tasks)
- Confusion matrix (TP, TN, FP, FN)

## Customization

### Adding a New Task

1. **Create a task class** in `src/tasks/your_task.py`:
   ```python
   class YourTask:
       def __init__(self, cid, config, device):
           # Initialize task
           pass
       
       def set_models(self, global_model, cnnet_modules):
           # Set up models
           pass
       
       def train(self, patience=3, batch_size=32):
           # Training logic
           return best_state_dict, contribution
       
       def validate(self, batch_size=32):
           # Validation logic
           return avg_loss, metrics
   ```

2. **Add configuration** in `config/task/your_task.toml`:
   ```toml
   task-name = "your_task"
   model = "YourModel"
   num-server-rounds = 10
   local-epochs = 5
   ```

3. **Update imports** in `src/client/client_app.py` and `src/server/server_app.py`

### Adding a New Model

1. Create your model in `src/models/your_model.py`:
   ```python
   import torch.nn as nn
   
   class YourModel(nn.Module):
       def __init__(self, input_dim, output_dim):
           super().__init__()
           # Define layers
       
       def forward(self, x):
           # Forward pass
           return output
   ```

2. Import it in `src/models/__init__.py`

3. Add initialisation logic in `src/server/server_app.py`

## GPU Support

The Docker setup automatically enables GPU support if available. To verify GPU access:

```bash
docker exec -it fedmap_container python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

To adjust GPU resources, modify `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # or specify number
          capabilities: [gpu]
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) errors**
   - Reduce batch size in configuration
   - Reduce model size or hidden dimensions
   - Increase shared memory: `shm_size: 100gb` in docker-compose.yml

2. **Dataset not found errors**
   - Verify dataset paths in task files
   - Ensure data is properly mounted (check docker-compose.yml volumes)
   - Check file permissions

3. **Client evaluation failures**
   - Check for empty validation sets
   - Verify class balance in datasets
   - Ensure proper data preprocessing

## Citation

If you use FedMAP in your research, please cite:

```bibtex
@misc{zhang2025fedmappersonalisedfederatedlearning,
  title={FedMAP: Personalised Federated Learning for Real Large-Scale Healthcare Systems},
  author={Fan Zhang and Daniel Kreuter and Carlos Esteve-Yagüe and Sören Dittmer and Javier Fernandez-Marques and Samantha Ip and BloodCounts! Consortium and Norbert C. J. de Wit and Angela Wood and James HF Rudd and Nicholas Lane and Nicholas S Gleadall and Carola-Bibiane Schönlieb and Michael Roberts},
  year={2025},
  eprint={2405.19000},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2405.19000}
}

```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## Acknowledgments

- Built on the [Flower](https://flower.dev/) federated learning framework
- Survival analysis powered by [pycox](https://github.com/havakv/pycox)