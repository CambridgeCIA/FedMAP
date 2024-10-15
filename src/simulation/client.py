import os
from collections import OrderedDict
from typing import List, Dict

import torch
import flwr as fl
from ..utils.logger import log_results

from datasets import load_individual_dataset, load_image_data, generate_data_from_config
from ..models import MLP, CNNModel
from ..utils.train_helper import train, test


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, input_shape: int):
        """Factory method to create a model instance."""
        if model_type == 'MLP':
            return MLP(input_shape)
        elif model_type == 'CNN':
            return CNNModel(31)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class FitStrategy:
    """Base class for fit strategies."""

    def fit(self, client: 'FlowerClient', parameters, config):
        raise NotImplementedError("Fit method must be implemented.")


class FedMapStrategy(FitStrategy):
    def fit(self, client: 'FlowerClient', parameters, config):
        model_path = f'./src/model_temp/model_{str(client.cid)}.pth'
        
        if config['server_round'] == 1:
            set_params(client.gamma, parameters)
            set_params(client.model, parameters)
        else:
            set_params(client.gamma, parameters)
            client.model.load_state_dict(torch.load(model_path))

        epochs = config["epochs"]
        contribution = train(client.model, client.trainset, epochs=epochs, device=client.device, 
                             cid=client.cid, strategy_name='fedmap', isMultiClass=config["is_multi_class"], gamma=client.gamma, 
                             variance=client.variance)

       
        torch.save(client.model.state_dict(), model_path)
        return client.get_parameters(config={}), len(client.trainset.dataset), {"weight": contribution}


class FedAvgStrategy(FitStrategy):
    def fit(self, client: 'FlowerClient', parameters, config):
        set_params(client.model, parameters)
        client.train(config)
        return client.get_parameters(config={}), len(client.trainset.dataset), {}


class FedProxStrategy(FitStrategy):
    def fit(self, client: 'FlowerClient', parameters, config):
        set_params(client.model, parameters)
        client.train_fedprox(config)
        return client.get_parameters(config={}), len(client.trainset.dataset), {}


class FedBNStrategy(FitStrategy):
    def fit(self, client: 'FlowerClient', parameters, config):
        model_path = f'./src/model_temp/model_{str(client.cid)}.pth'
        if config['server_round'] == 1:
            set_params(client.model, parameters)
        else:
            client.model.load_state_dict(torch.load(model_path))
            set_params_partial(client.model, parameters)

        client.train(config)
        torch.save(client.model.state_dict(), model_path)
        return client.get_parameters(config={}), len(client.trainset.dataset), {}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, cid=None, model_type='MLP', variance=None):
        self.trainset = trainset
        self.valset = valset
        self.cid = cid        
        shape = self.trainset.dataset[0][0].shape[0]

        self.model = ModelFactory.create_model(model_type, shape)
        self.gamma = ModelFactory.create_model(model_type, shape)
        self.previous_model = ModelFactory.create_model(model_type, shape)
        self.variance = variance

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) 
        self.gamma.to(self.device) 
        self.previous_model.to(self.device)

        # Strategy mapping
        self.strategy_map = {
            'fedmap': FedMapStrategy(),
            'fedavg': FedAvgStrategy(),
            'fedprox': FedProxStrategy(),
            'fedbn': FedBNStrategy()
        }

    def get_parameters(self, config):
        """Get model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Fit the model with given parameters and configuration."""
        fit_strategy = self.strategy_map.get(config['fit_strategy'])
        if fit_strategy:
            return fit_strategy.fit(self, parameters, config)
        else:
            raise ValueError(f"Unknown fit strategy: {config['fit_strategy']}")

    def evaluate(self, parameters, config):
        if config['fit_strategy'] in {'fedbn'}:
            return self.evaluate_fedbn(parameters, config)
        elif config['fit_strategy'] in {'fedmap'}:
            return self.evaluate_fedmap(parameters, config)
        else:
            return self.evaluate_fedavg(parameters, config)

    def evaluate_fedavg(self, parameters, config):
        set_params(self.model, parameters)
        loss, accuracy = test(self.model, self.valset, device=self.device, cid=self.cid, 
                              isMultiClass=config["is_multi_class"])
        log_results(f"{config['fit_strategy']} Client {self.cid} accuracy: {accuracy}")
        return float(loss), len(self.trainset.dataset), {"accuracy": float(accuracy)}

    def evaluate_fedbn(self, parameters, config):
        model_path = f'./src/model_temp/model_{str(self.cid)}.pth'        
        self.model.load_state_dict(torch.load(model_path))
        set_params_partial(self.model, parameters)       
        loss, accuracy = test(self.model, self.valset, device=self.device, cid=self.cid, 
                              isMultiClass=config["is_multi_class"])
        log_results(f"FedBN Client {self.cid} accuracy: {accuracy}")
        return float(loss), len(self.valset.dataset), {"accuracy": float(accuracy)}
    
    def evaluate_fedmap(self, parameters, config):
        model_path = f'./src/model_temp/model_{str(self.cid)}.pth'
        self.model.load_state_dict(torch.load(model_path))
       
        loss, accuracy = test(self.model, self.valset, device=self.device, cid=self.cid, 
                              isMultiClass=config["is_multi_class"])
        
        log_results(f"FedMAP Client {self.cid} accuracy: {accuracy}")
        return float(loss), len(self.trainset.dataset), {"accuracy": float(accuracy)}

    def train(self, config):
        epochs = config["epochs"]
        train(self.model, self.trainset, epochs=epochs, device=self.device, 
              cid=self.cid, strategy_name='fedavg', isMultiClass=config["is_multi_class"])

    def train_fedprox(self, config):
        epochs = config["epochs"]
        train(self.model, self.trainset, epochs=epochs, device=self.device, 
              cid=self.cid, strategy_name='fedprox', isMultiClass=config["is_multi_class"], 
              proximal_mu=config["proximal_mu"])


def get_client_fn(cfg):
    """Return a function to construct a client."""
    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        if cfg.datasets.name == 'synthetic':
            datasets_dir = f'./datasets/synthetic/partitions_0.csv'
            if not os.path.exists(datasets_dir):
                generate_data_from_config(cfg.datasets)

            trainset, valset = load_individual_dataset(cid)

        elif cfg.datasets.name == 'office_31':          
            trainset, valset = load_image_data(cid)

        model_type = cfg.datasets.model

        variance = cfg.envs.variance if cfg.envs.name == 'fedmap' else None

        print(f"Client {cid} has {len(trainset.dataset)} training examples and {len(valset.dataset)}")
        return FlowerClient(trainset, valset, cid=cid, model_type=model_type, variance=variance).to_client()
    return client_fn


def set_params(model: torch.nn.Module, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    new_state_dict = {}
    for (name, param), new_param in zip(model.state_dict().items(), params):
        new_param = torch.from_numpy(new_param)
        assert param.shape == new_param.shape, f"Shape mismatch for parameter {name}: {param.shape} != {new_param.shape}"
        new_state_dict[name] = new_param
    model.load_state_dict(new_state_dict)


def set_params_partial(model: torch.nn.Module, params: List[torch.Tensor]):
    """Set model weights from a list of NumPy ndarrays, excluding batch normalization layers."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict()
    for key, value in params_dict:
        if 'bn' not in key:
            state_dict[key] = torch.tensor(value)
    model.load_state_dict(state_dict, strict=False)
