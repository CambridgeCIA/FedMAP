"""pytorchexample: A Flower / PyTorch FedMAP app for the iron task."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.strategies import FedMAP
from src.models import DenseClassifier, MultimodalFFN, MLP
from src.utils.train_helper import InputConvexNN
import torchtuples as tt

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    num_rounds: int = context.run_config["num-server-rounds"]
    local_epochs: int = context.run_config["local-epochs"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    task_name: str = context.run_config["task-name"]
    learning_rate: float = context.run_config.get("learning-rate", 0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Server using device: {device}")

   
    def model_init_fn():
        if task_name == "interval":
            return DenseClassifier().to(device)
        elif task_name == "eicu":
            return MultimodalFFN(input_dim_drugs=1411, input_dim_dx=686, input_dim_physio=7).to(device)
        elif task_name == "cprd":
            in_features = 8 
            num_nodes = [64, 32, 16]  
            out_features = 1 
            batch_norm = True
            dropout = 0.1
            return tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout).to(device)
        else:
            return MLP(31).to(device)
    
    global_model = model_init_fn()
    icnn_dict = {}
    param_name_mapping = {} 
    
    for name, param in global_model.named_parameters():
        param_size = param.numel()
        sanitized_name = name.replace('.', '__')
        icnn_dict[sanitized_name] = InputConvexNN(param_size=param_size).to(device)
        param_name_mapping[name] = sanitized_name
        
    cnnet_modules = nn.ModuleDict(icnn_dict)
        
    strategy = FedMAP(fraction_evaluate=fraction_evaluate, icnn_modules=cnnet_modules)
    arrays = ArrayRecord(global_model.state_dict())


    print(f"Starting FedMAP strategy for {num_rounds} rounds...")
    config = ConfigRecord({"lr": learning_rate, "local_epochs": local_epochs, "task_name": task_name})
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=config,
        evaluate_config=config,
        num_rounds=num_rounds
    )

   
    print("\nSaving final model to disk...")
    final_model = model_init_fn()
    final_model.to(device)
    state_dict = result.arrays.to_torch_state_dict()
    final_model.load_state_dict(state_dict)
    torch.save(state_dict, f"src/checkpoints/global_model_{task_name}.pt")
    print(f"Training complete. Final model saved as 'global_model_{task_name}.pt'")