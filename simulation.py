import os
import flwr as fl
from omegaconf import DictConfig
from typing import Dict
import hydra
from flwr.common.typing import Scalar
from src.simulation.client import get_client_fn
from src.strategies import FedMAP
from datasets import synthetic_data_gen
from src.simulation import individual

def fit_config(server_round: int, epochs, fit_strategy, weights_contribution='sample_size', is_multi_class=False) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    return {
        "epochs": epochs,
        "batch_size": 64,
        "server_round": server_round,
        "fit_strategy": fit_strategy,
        "weights_contribution": weights_contribution,
        "is_multi_class": is_multi_class
    }
    
def evaluate_config(fit_strategy, is_multi_class=False):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds, one to three, then increase to ten local
    evaluation steps.
    """
    return {
        "fit_strategy": fit_strategy,
        "is_multi_class": is_multi_class,
    }

@hydra.main(config_path='./config', config_name='config', version_base=None)
def main(cfg: DictConfig):
    
    for i in range(cfg.number_clients):
        try:
            os.remove(f"./models/model_{i}.pth")
        except FileNotFoundError:
            pass    

    is_multi_class = True
    if cfg.datasets.name == 'synthetic':
        synthetic_data_gen.generate_data_from_config(cfg)
        is_multi_class = False
        

    # Configure strategies
    strategy_fedavg = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=cfg.number_clients,
        min_evaluate_clients=cfg.number_clients,
        min_available_clients=cfg.number_clients,
        on_fit_config_fn=lambda server_round: fit_config(server_round, cfg.epochs, cfg.envs.name, False, is_multi_class),
        on_evaluate_config_fn=lambda server_round: evaluate_config(cfg.envs.name, is_multi_class)
    )
    
    strategy_fedbn = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=cfg.number_clients,
        min_evaluate_clients=cfg.number_clients,
        min_available_clients=cfg.number_clients,
        on_fit_config_fn=lambda server_round: fit_config(server_round, cfg.epochs, cfg.envs.name, False, is_multi_class),
        on_evaluate_config_fn=lambda server_round: evaluate_config(cfg.envs.name, is_multi_class)
    )
    
    strategy_fedprox = fl.server.strategy.FedProx(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=cfg.number_clients,
        min_evaluate_clients=cfg.number_clients,
        min_available_clients=cfg.number_clients,
        on_fit_config_fn=lambda server_round: fit_config(server_round, cfg.epochs, cfg.envs.name, False, is_multi_class),
        proximal_mu=5.0,
        on_evaluate_config_fn=lambda server_round: evaluate_config(cfg.envs.name, is_multi_class)
    )
    
    strategy_fedmap = FedMAP(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=cfg.number_clients,
        min_evaluate_clients=cfg.number_clients,
        min_available_clients=cfg.number_clients,
        on_fit_config_fn=lambda server_round: fit_config(server_round, cfg.epochs, cfg.envs.name, weights_contribution='contribution', is_multi_class=is_multi_class),
        on_evaluate_config_fn=lambda server_round: evaluate_config(cfg.envs.name, is_multi_class)
    )
    
    strategies = {
        "fedavg": strategy_fedavg,
        "fedmap": strategy_fedmap,
        "fedprox":  strategy_fedprox,
        "fedbn":  strategy_fedbn,
    }
    
    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": cfg.num_cpus,
        "num_gpus": cfg.num_gpus,
    }

    if cfg.envs.name == "individual":
        individual.train_val((cfg.datasets.name == 'synthetic'), cfg.envs.epochs, num_classes=cfg.datasets.num_classes, num_clients=cfg.number_clients)
    else:
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=get_client_fn(cfg),
            num_clients=cfg.number_clients,
            client_resources=client_resources,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategies[cfg.envs.name]
        )

if __name__ == "__main__":
    main()
