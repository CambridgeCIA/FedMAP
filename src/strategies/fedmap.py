from typing import Callable, Union
import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Scalar
from flwr.server.strategy.aggregate import aggregate
from functools import reduce
import numpy as np
import torch

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.logger import log

class FedMAP(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # if self.inplace:
        #     # Does in-place weighted average of results
        #     aggregated_ndarrays = aggregate_inplace(results)
        else:

            # Convert results
            weights_results = [
                # (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics['weight'])
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
     
            # aggregated_ndarrays, variance = self.aggregate(weights_results)
            aggregated_ndarrays = aggregate(weights_results)
            # variance = 8.0
            # aggregated_ndarrays.append(np.array(variance))
                        
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

      
        return parameters_aggregated, metrics_aggregated
    
    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        total_weights = sum([weight for _, _, weight in results])
        
        weighted_weights = [
            [layer * weight for layer in parameters] for parameters, _, weight in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / total_weights
            for layer_updates in zip(*weighted_weights)
        ]
        variance = 7.0
     
        return weights_prime, variance
    
    def aggregate_with_weights(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        variance = self.update_varirance(results, weights_prime)
        return weights_prime, variance

    
    def update_varirance(self, model_parameters, weights_prime):
        """
        Update the sigma value based on the current model parameters, their expected values,
        and the sizes of the datasets for each task.

        Parameters:
        - model_parameters (list of torch.Tensor): Current model parameters for each task.
        - weights_prime (torch.Tensor): The global parameter for the model parameters.
        - dataset_sizes (list of int): The sizes of the datasets for each task.
        
        Returns:
        - sigma (torch.Tensor): Updated sigma value.
        """
        total_squared_diff = 0.0
        total_size = 0
        n = sum(param.size for param in weights_prime)
        
        for params, m_i in model_parameters:
            result_tensors = [torch.tensor(p) - torch.tensor(wp) for p, wp in zip(params, weights_prime)]
            squared_diff = sum(torch.sum(tensor ** 2) for tensor in result_tensors)
            total_squared_diff += m_i * squared_diff
            total_size += m_i

        sigma = torch.sqrt(total_squared_diff / (n * total_size))
        return sigma.item()