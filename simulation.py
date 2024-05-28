import os
import flwr as fl
from omegaconf import DictConfig
import hydra

from client import get_client_fn, set_params, weighted_average
from strategies import FedMAP

def fit_config(server_round: int, epochs, fit_strategy, weights_contribution='sample_size') -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    return {
        "epochs": epochs,
        "batch_size": 64,
        "server_round": server_round,
        "fit_strategy": fit_strategy,
        "weights_contribution": weights_contribution
    }

@hydra.main(config_path='./conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    
    for i in range(cfg.number_clients):
        try:
            os.remove(f"./models/model_{i}.pth")
        except FileNotFoundError:
            pass    
    # Load validation data
    val = ''
    
    # Configure strategies
    strategy_fedavg = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=cfg.number_client,
        min_evaluate_clients=cfg.number_client,
        min_available_clients=5,
        on_fit_config_fn=lambda server_round: fit_config(server_round, cfg.epochs, cfg.envs),
    )
    
    strategy_fedbn = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=cfg.number_client,
        min_evaluate_clients=cfg.number_client,
        min_available_clients=5,
        on_fit_config_fn=lambda server_round: fit_config(server_round, cfg.epochs, cfg.envs)
    )
    
    strategy_fedprox = fl.server.strategy.FedProx(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=cfg.number_client,
        min_evaluate_clients=cfg.number_client,
        on_fit_config_fn=lambda server_round: fit_config(server_round, cfg.epochs, cfg.envs),
        proximal_mu=2.0
    )
    
    strategy_fedomni = FedMAP(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=cfg.number_clients,
        min_evaluate_clients=cfg.number_clients,
        min_available_clients=cfg.number_clients,
        on_fit_config_fn=lambda server_round: fit_config(server_round, cfg.epochs, cfg.envs)
    )
    
    strategies = {
        "fedavg": strategy_fedavg,
        "fedomni": strategy_fedomni,
        "fedprox":  strategy_fedprox,
        "fedbn":  strategy_fedbn,
    }
    
    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(),
        num_clients=cfg.number_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategies[cfg.envs],
    )

if __name__ == "__main__":
    main()
