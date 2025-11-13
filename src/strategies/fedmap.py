from collections.abc import Iterable
from logging import INFO, WARNING, ERROR
from typing import Callable, Optional, Tuple
import os

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
    Array,
)
from flwr.server import Grid

from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    sample_nodes,
    validate_message_reply_consistency,
)

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class FedMAP(Strategy):
    """Federated MAP strategy with ICNN prior.

    Implementation based on ICNN-based prior for personalized aggregation.

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation.
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    weighted_by_key : str (default: "omega")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages (the FedMAP contribution).
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    icnn_key : str (default: "icnn")
        Key used to store the ICNN ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    icnn_modules : Optional[nn.ModuleDict] (default: None)
        ModuleDict containing ICNN modules with parameter names as keys.
    train_metrics_aggr_fn : Optional[callable] (default: None)
        Function to aggregate training metrics.
    evaluate_metrics_aggr_fn : Optional[callable] (default: None)
        Function to aggregate evaluation metrics.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "omega",
        arrayrecord_key: str = "arrays",
        icnn_key: str = "icnn",
        configrecord_key: str = "config",
        icnn_modules: Optional[nn.ModuleDict] = None,
        train_metrics_aggr_fn: Optional[
            Callable[[list[RecordDict], str], MetricRecord]
        ] = None,
        evaluate_metrics_aggr_fn: Optional[
            Callable[[list[RecordDict], str], MetricRecord]
        ] = None,
    ) -> None:
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.icnn_key = icnn_key
        
        # Store ICNN modules
        self.icnn_modules = icnn_modules
        
        # Convert to ArrayRecord for sending to clients
        if icnn_modules is not None:
            self.icnn_arrays = self._modules_to_arrayrecord(icnn_modules)
        else:
            self.icnn_arrays = None
            
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.weighted_by_key = weighted_by_key
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key
        self.train_metrics_aggr_fn = train_metrics_aggr_fn or aggregate_metricrecords
        self.evaluate_metrics_aggr_fn = (
            evaluate_metrics_aggr_fn or aggregate_metricrecords
        )
        
 
        self.icnn_lr = 1e-3            
        self.icnn_steps = 2             
        self.icnn_device = torch.device("cpu")
   
        if self.fraction_evaluate == 0.0:
            self.min_evaluate_nodes = 0
            log(
                WARNING,
                "fraction_evaluate is set to 0.0. "
                "Federated evaluation will be skipped.",
            )
        if self.fraction_train == 0.0:
            self.min_train_nodes = 0
            log(
                WARNING,
                "fraction_train is set to 0.0. Federated training will be skipped.",
            )

    def _modules_to_arrayrecord(self, modules: nn.ModuleDict) -> ArrayRecord:
        """Convert ICNN ModuleDict to ArrayRecord for transmission."""
        state_dict = modules.state_dict()
        arrays = {}
        for key, tensor in state_dict.items():
            # Convert to numpy and let Array handle it
            numpy_array = tensor.cpu().detach().numpy()
            arrays[key] = Array(numpy_array)
        
        return ArrayRecord(arrays)

    def summary(self) -> None:
        """Log summary configuration of the strategy."""

        log(INFO, "Minimum available nodes: %d", self.min_available_nodes)

    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: str
    ) -> Iterable[Message]:
        """Construct N Messages carrying the same RecordDict payload."""
        messages = []
        for node_id in node_ids:
            message = Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        if self.fraction_train == 0.0:
            return []
            

        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        config["server-round"] = server_round
 
        # Construct record with model parameters, config, and ICNN modules
        record = RecordDict(
            {
                self.arrayrecord_key: arrays,
                self.configrecord_key: config,
                self.icnn_key: self.icnn_arrays,
            }
        )

        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def _check_and_log_replies(
        self, replies: Iterable[Message], is_train: bool, validate: bool = True
    ) -> tuple[list[Message], list[Message]]:
        """Check replies for errors and log them."""
        if not replies:
            return [], []

        # Filter messages that carry content
        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(
            INFO,
            "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate",
            len(valid_replies),
            len(error_replies),
        )
   
        for msg in error_replies:
            log(
                INFO,
                "\t> Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error.reason,
            )

        if validate and valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=is_train,
            )

        return valid_replies, error_replies
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Extract client parameters and weights for ICNN training
            client_params_list = []
            weights_list = []
            
            for reply in reply_contents:
                if self.arrayrecord_key in reply:
                    client_arrays = reply[self.arrayrecord_key]
                    client_params_list.append(client_arrays)
                    
                    # Get omega weight for this client
                    if "metrics" in reply and self.weighted_by_key in reply["metrics"]:
                        weight = reply["metrics"][self.weighted_by_key]
                        weights_list.append(weight)
                    else:
                        weights_list.append(1.0)  # Default weight

            # Aggregate model parameters using weighted average
            arrays = aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
            )

            # Train ICNN modules with client parameters
            if arrays is not None and len(client_params_list) > 0:
                self._train_icnn(client_params_list, arrays, weights_list)

            # Aggregate metrics
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
            
        return arrays, metrics

    def _train_icnn(
        self, 
        client_params_list: list[ArrayRecord], 
        global_params: ArrayRecord,
        weights: list[float]
    ) -> None:
        """
        Train ICNN modules using client and global parameters.
        ICNN modules are stored in a ModuleDict with sanitized keys (dots replaced with '__').
        """
        if self.icnn_modules is None:
            log(WARNING, "ICNN modules not initialized, skipping ICNN training")
            return
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        log(INFO, f"Training ICNN with {len(client_params_list)} clients, weights: {normalized_weights}")
        
        # Set ICNN modules to train mode
        for cnnet in self.icnn_modules.values():
            cnnet.train()
        
        # Create mapping from original parameter names to sanitized ICNN keys
        # The ICNN keys have dots replaced with '__'
        param_to_icnn_key = {}
        for param_name in global_params.keys():
            sanitized_name = param_name.replace('.', '__')
            if sanitized_name in self.icnn_modules:
                param_to_icnn_key[param_name] = sanitized_name
        
        if len(param_to_icnn_key) == 0:
            log(ERROR, "No matching parameters found between ICNN modules and global_params!")
            log(ERROR, f"Global params keys (sample): {list(global_params.keys())[:3]}")
            log(ERROR, f"ICNN keys (sample): {list(self.icnn_modules.keys())[:3]}")
            return
        
        log(INFO, f"Processing {len(param_to_icnn_key)} parameters with ICNN modules")
        
        # Verify parameter sizes match ICNN expectations
        size_mismatch = False
        for param_name, icnn_key in param_to_icnn_key.items():
            array = global_params[param_name].numpy()
            param_size = array.size
            icnn_module = self.icnn_modules[icnn_key]
            expected_size = icnn_module.net[0].in_features // 2
            
            if param_size != expected_size:
                log(
                    ERROR, 
                    f"Parameter '{param_name}': size={param_size}, ICNN expects={expected_size}"
                )
                size_mismatch = True
        
        if size_mismatch:
            log(ERROR, "Parameter size mismatches detected! Cannot train ICNN.")
            return
        
        # Training loop for ICNN
        for step in range(self.icnn_steps):
            # Zero gradients
            for cnnet in self.icnn_modules.values():
                for param in cnnet.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            
            total_loss = 0.0
            
            # Iterate over each parameter and its corresponding ICNN
            for param_name, icnn_key in param_to_icnn_key.items():
                cnnet = self.icnn_modules[icnn_key]
                
                # Get global parameter (gamma)
                gamma_array = global_params[param_name].numpy()
                gamma_flat = torch.from_numpy(gamma_array).float().flatten().unsqueeze(0).to(self.icnn_device)
                
                # Compute weighted loss over all clients
                for client_idx, client_params in enumerate(client_params_list):
                    if param_name not in client_params:
                        log(WARNING, f"Client {client_idx} missing parameter '{param_name}', skipping")
                        continue
                    
                    weight = normalized_weights[client_idx]
                    
                    # Get client parameter (theta)
                    theta_array = client_params[param_name].numpy()
                    theta_flat = torch.from_numpy(theta_array).float().flatten().unsqueeze(0).to(self.icnn_device)
                    
                    # Verify sizes match
                    if theta_flat.shape != gamma_flat.shape:
                        log(ERROR, f"Client {client_idx} param '{param_name}' shape mismatch")
                        continue
                    
                    # Compute ICNN output
                    output = cnnet(theta_flat, gamma_flat)
                    
                    # Weighted loss contribution
                    loss = weight * output.sum()
                    total_loss += loss
            
            if total_loss > 0:
                total_loss.backward()
                
                # Gradient descent step
                with torch.no_grad():
                    for cnnet in self.icnn_modules.values():
                        for param in cnnet.parameters():
                            if param.grad is not None:
                                param.data -= self.icnn_lr * param.grad
            
                for cnnet in self.icnn_modules.values():
                    cnnet.enforce_convexity()
            
                log(INFO, f"ICNN step {step + 1}/{self.icnn_steps}, loss: {total_loss.item():.6f}")
            else:
                log(WARNING, f"ICNN step {step + 1}/{self.icnn_steps}: total_loss is 0")
        
        self.icnn_arrays = self._modules_to_arrayrecord(self.icnn_modules)
        self._save_icnn()
        log(INFO, "ICNN training complete")

    def _save_icnn(self) -> None:
        path = "src/checkpoints/icnn_modules.pt"
        if self.icnn_modules is None:
            log(WARNING, "No ICNN modules to save")
            return
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        try:
            torch.save(self.icnn_modules.state_dict(), path)
           
            self.icnn_arrays = self._modules_to_arrayrecord(self.icnn_modules)
            log(INFO, "Saved ICNN modules to %s", path)
        except Exception as exc:
            log(ERROR, "Failed to save ICNN modules to %s: %s", path, str(exc))
            
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return metrics