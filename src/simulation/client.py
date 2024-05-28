import os
from collections import OrderedDict
from typing import Dict, Tuple, List

import torch
import flwr as fl
from torch.utils.data import DataLoader
from flwr.common import Metrics
from flwr.common.typing import Scalar

from utils import train, test
from dataset import load_data_synthetic
from models.map_cnn import MLP

def write_to_log(message, log_file="./results/logfile.log"):
    """Write a message to a log file."""
    with open(log_file, "a") as f:
        f.write(message + "\n")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, cid=None):
        self.trainset = trainset
        self.valset = valset
        self.cid = cid
        
        shape = self.trainset.dataset[0][0].shape[0]
        self.model = MLP(shape)
        self.gamma = MLP(shape)
        self.previous_model = MLP(shape) 
        self.variance = 15
    
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) 
        self.gamma.to(self.device) 
        self.previous_model.to(self.device) 
        
    def get_parameters(self, config):
        """Get model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            
    def fit(self, parameters, config):
        """Fit the model with given parameters and configuration."""
        if config['fit_strategy'] == 'fedomni':
            return self.fit_fedomni(parameters, config)
        elif config['fit_strategy'] == 'fedavg':
            return self.fit_fedavg(parameters, config)
    
    def fit_fedomni(self, parameters, config):
        """Fit the model using FedOmni strategy."""
        model_path = f'./models/model_{str(self.cid)}.pth'
        
        if config['server_round'] == 1:
            set_params(self.gamma, parameters)
            set_params(self.model, parameters)
        else:
            set_params(self.gamma, parameters)
            set_params(self.model, parameters)

            self.previous_model.load_state_dict(torch.load(model_path))
            self.model.replace_linear_layer(self.previous_model)
            
        batch, epochs = config["batch_size"], config["epochs"]

        # Train the model
        train(self.model, self.trainset, epochs=epochs, device=self.device, cid=self.cid, gamma=list(self.gamma.parameters()), variance=self.variance)
        torch.save(self.model.state_dict(), model_path)
        loss, accuracy = test(self.model, self.valset, device=self.device, cid=self.cid)
        
        write_to_log(f"Client {self.cid} accuracy: {accuracy}")

        sample_size = len(self.trainset.dataset)
        return self.get_parameters(config={}), sample_size, {}
    
    def fit_fedavg(self, parameters, config):
        """Fit the model using FedAvg strategy."""
        set_params(self.model, parameters)
        batch, epochs = config["batch_size"], config["epochs"]
        
        train(self.model, self.trainset, epochs=epochs, device=self.device, cid=self.cid)
        return self.get_parameters(config={}), len(self.trainset.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model with given parameters and configuration."""
        loss, accuracy = 0, 0 
        return float(loss), len(self.trainset.dataset), {"accuracy": float(accuracy)}

def get_client_fn():
    """Return a function to construct a client."""
    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        trainset, valset = load_data_synthetic(f'./data/synthetic_local/partitions_{cid}.csv', LABEL)
        
        print(f"Client {cid} has {len(trainset.dataset)} training examples and {len(valset.dataset)}")
        return FlowerClient(trainset, valset, cid=cid).to_client()
    return client_fn

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
