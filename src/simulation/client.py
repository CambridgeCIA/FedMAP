import os
from collections import OrderedDict
from typing import List

import torch
import flwr as fl
from  ..utils.logger import log_results
from ..data_loader import load_data_synthetic, load_image_data
from ..models import MLP, CNNModel
from datasets import generate_data_from_config
from ..utils.train_helper import train, test


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, cid=None, model=None, variance=None):
        self.trainset = trainset
        self.valset = valset
        self.cid = cid        
        shape = self.trainset.dataset[0][0].shape[0]

        self.model = model(shape)
        self.gamma = model(shape)
        self.previous_model = model(shape)
        self.variance = variance
      
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
        if config['fit_strategy'] == 'fedmap':
            return self.fit_fedmap(parameters, config)
        elif config['fit_strategy'] == 'fedavg' or config['fit_strategy'] == 'fedprox':
            return self.fit_fedavg(parameters, config)
        elif config['fit_strategy'] == 'fedbn':
            return self.fit_fedbn(parameters, config)   
    
    def fit_fedmap(self, parameters, config):
        """Fit the model using FedOmni strategy."""
        model_path = f'./src/model_temp/model_{str(self.cid)}.pth'
        
        if config['server_round'] == 1:
            set_params(self.gamma, parameters)
            set_params(self.model, parameters)
        else:
            set_params(self.gamma, parameters)
            self.model.load_state_dict(torch.load(model_path))
            
        epochs = config["epochs"]

        # Train the model
        contribution = train(self.model, self.trainset, epochs=epochs, device=self.device, cid=self.cid, strategy_name='fedmap', gamma=self.gamma, variance=self.variance)
        torch.save(self.model.state_dict(), model_path)
        sample_size = len(self.trainset.dataset)
        return self.get_parameters(config={}), sample_size, {"weight": contribution}
    
    def fit_fedavg(self, parameters, config):
        """Fit the model using FedAvg strategy."""    
        set_params(self.model, parameters)
        
        self.train_fedavg(config)
        return self.get_parameters(config={}), len(self.trainset.dataset), {}     
    
    def fit_fedbn(self, parameters, config):
        model_path = f'./src/model_temp/model_{str(self.cid)}.pth'
        
        if config['server_round'] == 1:
            set_params(self.model, parameters)
        else:
            self.model.load_state_dict(torch.load(model_path))
            set_params_partial(self.model, parameters)
        
        self.train_fedavg(config)
        torch.save(self.model.state_dict(), model_path)
        return self.get_parameters(config={}), len(self.trainset.dataset), {}
    
    def train_fedavg(self, config):
        epochs = config["epochs"]
        train(self.model, self.trainset, epochs=epochs, device=self.device, cid=self.cid, strategy_name='fedavg', isMultiClass=config["is_multi_class"])
        
    def evaluate(self, parameters, config):
        if config['fit_strategy'] == 'fedmap':
            return self.evaluate_fedmap(parameters, config)
        elif config['fit_strategy'] == 'fedbn':
            return self.evaluate_fedbn(parameters, config)
        else:
            return self.evaluate_fedavg(parameters, config)
        
        
    def evaluate_fedavg(self, parameters, config):
        set_params(self.model, parameters)
        loss, accuracy = test(self.model, self.valset, device=self.device, cid=self.cid, isMultiClass=config["is_multi_class"])

        log_results(f"FedAVG Client {self.cid} accuracy: {accuracy}")
        return float(loss), len(self.trainset.dataset), {"accuracy": float(accuracy)}

    def evaluate_fedmap(self, parameters, config):
        model_path = f'./src/model_temp/model_{str(self.cid)}.pth'     
        self.model.load_state_dict(torch.load(model_path))
        loss, accuracy = test(self.model, self.valset, device=self.device, cid=self.cid, isMultiClass=config["is_multi_class"])
        log_results(f"FedMAP Client {self.cid} accuracy: {accuracy}")

        return float(loss), len(self.valset.dataset), {"accuracy": float(accuracy)}

    def evaluate_fedbn(self, parameters, config):
        model_path = f'./src/model_temp/model_{str(self.cid)}.pth'        
        self.model.load_state_dict(torch.load(model_path))
        set_params_partial(self.model, parameters)       
        loss, accuracy = test(self.model, self.valset, device=self.device, cid=self.cid, isMultiClass=config["is_multi_class"])

        log_results(f"FedBN Client {self.cid} accuracy: {accuracy}")
        return float(loss), len(self.valset.dataset), {"accuracy": float(accuracy)}
    

def get_client_fn(cfg):
    """Return a function to construct a client."""
    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        
        if cfg.datasets.name == 'synthetic':
            datasets_dir = f'./datasets/synthetic/partitions_{cid}.csv'
            if not os.path.exists(datasets_dir):
                generate_data_from_config(cfg.datasets)
        
            trainset, valset = load_data_synthetic(datasets_dir, cfg.datasets.label)

        elif cfg.datasets.name == 'office_31':          
            trainset, valset = load_image_data(cid)
        
        model = MLP if cfg.datasets.model == 'MLP' else CNNModel 
        variance = cfg.envs.variance if cfg.envs.name == 'fedmap' else None
        
        print(f"Client {cid} has {len(trainset.dataset)} training examples and {len(valset.dataset)}")
        return FlowerClient(trainset, valset, cid=cid, model=model, variance=variance).to_client()
    return client_fn

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    new_state_dict = dict()
    
    # Iterate over the model's state_dict and the provided parameters
    for (name, param), new_param in zip(model.state_dict().items(), params):
        # Convert the NumPy array to a PyTorch tensor
        new_param = torch.from_numpy(new_param)
        
        # Ensure the shapes match
        assert param.shape == new_param.shape, f"Shape mismatch for parameter {name}: {param.shape} != {new_param.shape}"
        
        # Update the new state_dict with the new parameter
        new_state_dict[name] = new_param
    
    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)
    
def set_params_partial(model: torch.nn.Module, params: List[torch.Tensor]):
    """Set model weights from a list of NumPy ndarrays, excluding batch normalization layers."""
    # Create a zip object of model's state dictionary keys and the provided parameters
    params_dict = zip(model.state_dict().keys(), params)

    # Filter out batch normalization parameters and convert the rest to Torch Tensors
    state_dict = OrderedDict()
    for key, value in params_dict:
     
        # Check if the key corresponds to a batch normalization layer in the FedResNet50 model
        if 'bn' not in key:
            state_dict[key] = torch.tensor(value)

    # Load the non-batch normalization parameters into the model
    model.load_state_dict(state_dict, strict=False)
