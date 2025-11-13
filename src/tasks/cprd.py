import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import math
from typing import Optional, Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import torchtuples as tt
from collections import OrderedDict

from src.utils.train_helper import InputConvexNN

GROUP_A = [670, 206, 272, 263, 229, 515, 351, 88, 677, 473, 194, 147, 111]

class CPRDCoxPHDataset(Dataset):
    """Dataset class for CPRD CVD risk prediction data"""
    def __init__(self, x_data, durations, events):
        """
        Args:
            x_data: Feature tensor or array
            durations: Time to event or censoring
            events: Event indicators (1=event, 0=censored)
        """
        if isinstance(x_data, np.ndarray):
            self.x = torch.FloatTensor(x_data)
        else:
            self.x = x_data.float()
        
        if isinstance(durations, np.ndarray):
            self.durations = torch.FloatTensor(durations)
        else:
            self.durations = durations.float()
        
        if isinstance(events, np.ndarray):
            self.events = torch.FloatTensor(events)
        else:
            self.events = events.float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], (self.durations[idx], self.events[idx])
    
class CoxPHFedMAP(CoxPH):
    """CoxPH model with FedMAP regularization using the existing FedMAPLoss"""
    def __init__(self, net, gamma_net, cnnet_modules, optimizer=None, 
                 device=None, lambda_prior=0.01):
        self.gamma_net = gamma_net
        self.cnnet_modules = cnnet_modules
        self.lambda_prior = lambda_prior
        
        # Initialize with standard CoxPH
        super().__init__(net, optimizer, device)
        
        # Import FedMAPLoss and ICNNPrior from the loss module
        from src.loss_modules.map import FedMAPLoss, ICNNPrior
        
        # Create prior
        prior = ICNNPrior(cnnet_modules)
        
        # Create CoxPH loss wrapper
        cox_loss_wrapper = CoxPHLossWrapper()
        
        # Create FedMAP loss (for training only, not metrics)
        self.fedmap_loss = FedMAPLoss(cox_loss_wrapper, prior, gamma_net)
        self.fedmap_loss.bind_model(self.net)
        
        # Keep standard cox_ph_loss for metrics
        from pycox.models.loss import cox_ph_loss
        self.loss = cox_ph_loss  # This is what torchtuples uses as the actual loss
    
    def compute_loss(self, log_h, target, reduction='mean'):
        """
        Override compute_loss to use FedMAP loss during training.
        
        Args:
            log_h: Log hazard predictions
            target: Tuple of (durations, events)
            reduction: Reduction method (ignored, kept for compatibility)
        """
        # Use FedMAP loss for training
        return self.fedmap_loss(log_h, target)
    
    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, 
            verbose=True, num_workers=0, shuffle=True, metrics=None, 
            val_data=None, val_batch_size=8224, **kwargs):
        # Don't pass any metrics - let torchtuples handle it automatically
        # It will use self.loss (cox_ph_loss) for the loss metric
        return super().fit(input, target, batch_size, epochs, callbacks, verbose,
                          num_workers, shuffle, metrics=None, val_data=val_data, 
                          val_batch_size=val_batch_size, **kwargs)

class CoxPHLossWrapper(nn.Module):
    """Wrapper to make cox_ph_loss compatible with FedMAPLoss interface"""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import cox_ph_loss
        self.cox_ph_loss_fn = cox_ph_loss
    
    def forward(self, log_h, targets):
        """
        Args:
            log_h: Log hazard predictions
            targets: Tuple of (durations, events)
        """
        durations, events = targets
        return self.cox_ph_loss_fn(log_h, durations, events)


class CPRD:    
    def __init__(self, cid, config, device):
        """
        Initialize CPRD client for federated learning.
        
        Args:
            cid: Client/Hospital ID
            config: Configuration dictionary containing training parameters
            device: torch device (cuda/cpu)
        """
        self.cid = cid
        self.config = config
        self.hospital_id = GROUP_A[cid]
        self.lr = config.get('lr', 0.001)
        self.local_epochs = config.get('local_epochs', 10)
        self.server_round = config.get('server_round', 0)
        self.device = device
        
        self.local_model = None
        self.global_model = None
        self.cnnet_modules = None
        
        self.risk_factors_path = "datasets/cprd/risk_factors_all.csv"
        self.landmark_age = config.get('landmark_age', 65)
        
        self.input_features = None
        self.num_hidden_nodes = config.get('num_hidden_nodes', [64, 32, 16])
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.x_mapper = None
        
        self.lambda_prior = config.get('lambda_prior', 0.01)
    
    def _get_coxph_net(self, in_features, num_nodes, batch_norm=True, dropout=0.1):
        """Create CoxPH network architecture"""
        out_features = 1
        return tt.practical.MLPVanilla(in_features, num_nodes, out_features, 
                                       batch_norm, dropout)
    
    def _load_cvd_data(self, pracid):
        df = pd.read_csv(self.risk_factors_path)
        df = df[df["cvd_before_landmark"] == 0]
        df = df[df["death_before_landmark"] == 0]
        df = df[df["end_time"] > 0]
        df = df[df["pracid"] == pracid].copy()
        return df
    
    def _prepare_hospital_data(self, train_split=0.8, seed=42):
        df_hospital = self._load_cvd_data(self.hospital_id)
        
        df_hospital["ten_year_cvd_event"] = df_hospital["cvd_ten_years_after_landmark"]
        df_hospital["duration"] = np.minimum(df_hospital["end_time"], 10.0)
        
        sex_encoder = LabelEncoder()
        df_hospital["sex"] = sex_encoder.fit_transform(df_hospital["sex"])
        
        df_train, df_val = train_test_split(
            df_hospital, 
            test_size=1-train_split, 
            random_state=seed,
            stratify=df_hospital["ten_year_cvd_event"]
        )
        
        if len(df_train) < 20 or len(df_val) < 5:
            raise ValueError(f"Hospital {self.hospital_id} has insufficient data")
        
        # Prepare test set
        df_test = df_val.copy()
        df_test["event"] = df_test["ten_year_cvd_event"]
        df_test = df_test[df_test["duration"] <= 10]
        
        if len(df_test) < 3:
            raise ValueError(f"Hospital {self.hospital_id} has insufficient test data")
        
        # Feature transformation using ColumnTransformer
        cols_standardize = ["sbp", "tchol", "hdl", "landmark_age"]
        cols_leave = ["sex", "smoking_status", "bp_med", "diabetes_diag"]
        
        # Create ColumnTransformer
        self.x_mapper = ColumnTransformer(
            transformers=[
                ('standardize', StandardScaler(), cols_standardize),
                ('passthrough', 'passthrough', cols_leave)
            ],
            remainder='drop'
        )
        
        # Transform features
        x_train = self.x_mapper.fit_transform(df_train).astype("float32")
        x_val = self.x_mapper.transform(df_val).astype("float32")
        x_test = self.x_mapper.transform(df_test).astype("float32")
        
        # Target preparation
        get_target = lambda df: (df["duration"].values, df["event"].values)
        y_train = get_target(df_train)
        y_val = get_target(df_val)
        durations_test, events_test = get_target(df_test)
        
        # Store data
        self.train_data = (
            torch.from_numpy(x_train).float(), 
            (torch.from_numpy(y_train[0]).float(), torch.from_numpy(y_train[1]).float())
        )
        
        self.val_data = (
            torch.from_numpy(x_val).float(), 
            (torch.from_numpy(y_val[0]).float(), torch.from_numpy(y_val[1]).float())
        ) if len(x_val) > 0 else None
        
        self.test_data = (x_test, durations_test, events_test)
        
        # Set input features dimension
        self.input_features = x_train.shape[1]
        
        print(f"Hospital {self.hospital_id} - Train: {len(x_train)}, "
              f"Val: {len(x_val)}, Test: {len(x_test)}")
        
        return self.input_features
    
    def set_models(self, global_model_state, cnnet_modules_state):
        """
        Set up local and global models with ICNN modules.
        
        Args:
            global_model_state: State dict of the global CoxPH model
            cnnet_modules_state: State dict of ICNN modules (OrderedDict) or nn.ModuleDict
        """
 
        if self.input_features is None:
            self._prepare_hospital_data()
        

        local_net = self._get_coxph_net(self.input_features, self.num_hidden_nodes)
        if global_model_state is not None:
            local_net.load_state_dict(global_model_state)
        

        global_net = self._get_coxph_net(self.input_features, self.num_hidden_nodes)
        if global_model_state is not None:
            global_net.load_state_dict(global_model_state)
        

        if cnnet_modules_state is not None:
            if isinstance(cnnet_modules_state, (dict, OrderedDict)):
                print(f"Reconstructing ICNN modules from state dict with {len(cnnet_modules_state)} parameters")
                
                icnn_dict = {}
                for name, param in local_net.named_parameters():
                    param_size = param.numel()
                    sanitized_name = name.replace('.', '__')
                    icnn_dict[sanitized_name] = InputConvexNN(
                        param_size=param_size,
                        hidden_dims=[64, 32],
                        alpha=0.001,
                        epsilon=1e-4
                    ).to(self.device)
                
                self.cnnet_modules = nn.ModuleDict(icnn_dict)
                

                self.cnnet_modules.load_state_dict(cnnet_modules_state)
            else:
                self.cnnet_modules = cnnet_modules_state
        else:
            self.cnnet_modules = None
        
        self.local_model = CoxPHFedMAP(
            local_net, global_net, self.cnnet_modules,
            tt.optim.Adam, self.device, self.lambda_prior
        )
        

        self.global_model = global_net
        self.local_model.optimizer.set_lr(self.lr)
    
    def train_loader(self, batch_size=128):
        if self.train_data is None:
            self._prepare_hospital_data()
        return self.train_data
    
    def val_loader(self, batch_size=128):
        if self.val_data is None:
            self._prepare_hospital_data()
        return self.val_data
    
    def train(self, patience=5, batch_size=128):
        """
        Train the CoxPH model on the client's dataset.
        
        Args:
            patience: Early stopping patience
            batch_size: Batch size for training
            
        Returns:
            best_state_dict: Best model state dict
            contribution: Client contribution score
        """
        if self.local_model is None:
            raise ValueError("Models not initialized. Call set_models() first.")
        
  
        x_train_tensor, y_train_tuple = self.train_data
        
        effective_batch_size = max(2, min(batch_size, len(x_train_tensor) // 4))
        if len(x_train_tensor) % effective_batch_size == 1:
            effective_batch_size = max(2, effective_batch_size - 1)
        
        print(f"Client {self.cid} (Hospital {self.hospital_id}) - "
              f"Training with batch size: {effective_batch_size}")
        
  
        callbacks = [tt.cb.EarlyStopping(patience=patience)]
        

        log = self.local_model.fit(
            x_train_tensor, y_train_tuple,
            effective_batch_size, self.local_epochs,
            callbacks=callbacks, verbose=False,
            val_data=self.val_data
        )
        

        log_df = log.to_pandas() if hasattr(log, 'to_pandas') else pd.DataFrame()
        final_train_loss = (log_df['train_loss'].iloc[-1] 
                           if not log_df.empty and 'train_loss' in log_df.columns 
                           else float('nan'))
        final_val_loss = (log_df['val_loss'].iloc[-1] 
                         if not log_df.empty and 'val_loss' in log_df.columns 
                         else float('nan'))
        
        print(f"Client {self.cid} - Train Loss: {final_train_loss:.4f}, "
              f"Val Loss: {final_val_loss:.4f}")
        

        contribution = self._calculate_contribution()
        

        best_state_dict = self.local_model.net.state_dict()
        
        return best_state_dict, contribution
    
    def validate(self, batch_size=128, tier=1):
        """
        Validate the model on test set.
        
        Args:
            batch_size: Batch size for evaluation
            
        Returns:
            avg_loss: Average test loss (not applicable for survival)
            metrics: Dictionary of validation metrics
        """
        if self.local_model is None:
            raise ValueError("Model not initialized. Call set_models() first.")
        
        self.local_model.net.eval()
        x_test, durations_test, events_test = self.test_data
        
        if len(x_test) == 0:
            return 0.0, {"concordance": 0.0, "integrated_brier_score": None}
        
        try:
            x_test_tensor = torch.from_numpy(x_test).float().to(self.device)
            

            _ = self.local_model.compute_baseline_hazards(
                x_test_tensor, (durations_test, events_test)
            )
            

            surv = self.local_model.predict_surv_df(x_test_tensor)

            ev = EvalSurv(surv, durations_test, events_test, censor_surv="km")
            concordance = float(ev.concordance_td())

            integrated_brier_score = None
            if durations_test.max() > durations_test.min():
                time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
                try:
                    integrated_brier_score = float(ev.integrated_brier_score(time_grid))
                except Exception as e:
                    print(f"  Brier score calculation failed: {e}")
            
            metrics = {
                "concordance": concordance,
                "integrated_brier_score": integrated_brier_score
            }
            
            print(f"Client {self.cid} - Concordance: {concordance:.4f}")
            
            return 0.0, metrics
            
        except Exception as e:
            print(f"Client {self.cid} - Evaluation error: {e}")
            return 0.0, {"concordance": 0.0, "integrated_brier_score": None}
    
    @torch.no_grad()
    def _calculate_contribution(self) -> float:
        """
        FedMAP client-side contribution computation for CoxPH.        
        Returns:
            contribution: Client contribution score
        """
        # Set networks to eval mode
        self.local_model.net.eval()
        self.global_model.eval()
        
        if self.cnnet_modules is not None:
            if isinstance(self.cnnet_modules, nn.ModuleDict):
                for cnnet in self.cnnet_modules.values():
                    cnnet.eval()
            else:
                for cnnet in self.cnnet_modules:
                    cnnet.eval()
        
        x_train_tensor, y_train_tuple = self.train_data
        
        # Calculate effective batch size
        effective_batch_size = max(2, min(128, len(x_train_tensor) // 4))
        
        # Create DataLoader for batch processing (like INTERVAL)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(x_train_tensor, y_train_tuple[0], y_train_tuple[1])
        likelihood_loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False)
        
        # Compute likelihood term (similar to INTERVAL's cross-entropy sum)
        total_neg_loglik = 0.0
        N = 0
        
        for x_batch, dur_batch, evt_batch in likelihood_loader:
            x_batch = x_batch.to(self.device)
            dur_batch = dur_batch.to(self.device)
            evt_batch = evt_batch.to(self.device)
            
            # Get log hazard predictions
            log_h = self.local_model.net(x_batch)
            
            # Compute CoxPH loss for this batch
            from pycox.models.loss import cox_ph_loss
            batch_loss = cox_ph_loss(log_h, dur_batch, evt_batch)
            
            # Accumulate negative log-likelihood (loss is negative log-likelihood)
            total_neg_loglik += batch_loss.item() * x_batch.size(0)
            N += x_batch.size(0)
        
        if N == 0:
            return 0.0
        
        # Convert to log-likelihood (negative of loss)
        sum_loglik = -total_neg_loglik
        mean_loglik = sum_loglik / N
        
        # Compute prior term (same as INTERVAL)
        prior_term = 0.0
        
        # Get cnnet dtype
        if isinstance(self.cnnet_modules, nn.ModuleDict):
            first_cnnet = next(iter(self.cnnet_modules.values()))
        else:
            first_cnnet = self.cnnet_modules[0]
        
        cnnet_dtype = (next(first_cnnet.parameters()).dtype 
                    if len(list(first_cnnet.parameters())) > 0 
                    else torch.float32)
        

        if isinstance(self.cnnet_modules, nn.ModuleDict):
            prior_term = self._compute_prior_module_dict(
                self.local_model.net, 
                self.global_model, 
                self.cnnet_modules, 
                self.device, 
                cnnet_dtype
            )
        else:
            prior_term = self._compute_prior_module_list(
                self.local_model.net,
                self.global_model,
                self.cnnet_modules,
                self.device,
                cnnet_dtype
            )
        

        log_contribution = mean_loglik - (prior_term / N)
        contribution = float(math.exp(log_contribution))
        
        print(f"  Client {self.cid} contribution: {contribution:.4f} "
            f"(mean_loglik: {mean_loglik:.4f}, prior_term: {prior_term:.4f})")
        
        return contribution
    
    def _compute_prior_module_list(
        self,
        net: nn.Module, 
        gamma: nn.Module, 
        cnnet_modules: nn.ModuleList,
        device: torch.device,
        cnnet_dtype: torch.dtype
    ) -> float:
        """Helper method for contribution calculation with ModuleList."""
        prior_term = 0.0
        
        for (lp, gp), cnnet in zip(zip(net.parameters(), gamma.parameters()), cnnet_modules):
            theta_flat = lp.detach().view(1, -1).to(device=device, dtype=cnnet_dtype)
            mu_flat = gp.detach().view(1, -1).to(device=device, dtype=cnnet_dtype)
            prior_term += cnnet(theta_flat, mu_flat).sum().item()
        
        return prior_term

    def _compute_prior_module_dict(
        self,
        net: nn.Module,
        gamma: nn.Module,
        cnnet_modules: nn.ModuleDict,
        device: torch.device,
        cnnet_dtype: torch.dtype
    ) -> float:
        """Helper method for contribution calculation with ModuleDict."""
        prior_term = 0.0
        
        net_params = {name.replace('.', '__'): param 
                    for name, param in net.named_parameters()}
        gamma_params = {name.replace('.', '__'): param 
                        for name, param in gamma.named_parameters()}
        
        for sanitized_name, cnnet in cnnet_modules.items():
            if sanitized_name not in net_params or sanitized_name not in gamma_params:
                continue
            
            local_param = net_params[sanitized_name]
            global_param = gamma_params[sanitized_name]
            
            theta_flat = local_param.detach().view(1, -1).to(device=device, dtype=cnnet_dtype)
            mu_flat = global_param.detach().view(1, -1).to(device=device, dtype=cnnet_dtype)
            
            prior_term += cnnet(theta_flat, mu_flat).sum().item()
        
        return prior_term
    def _record_performance(self, round_num, metrics, tier=1):
        """
        Log validation/test metrics to a CSV file.
        
        Args:
            round_num: Current federated learning round
            metrics: Dictionary of metrics to log
            filename: Path to CSV file
        """
        filename=f'../results/cprd_metrics_tier{tier}.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        headers = ['Client_ID', 'Hospital_ID', 'Round', 'Concordance', 
                   'Integrated_Brier_Score', 'Contribution']
        
        record = [
            int(self.cid),
            int(self.hospital_id),
            round_num,
            metrics.get('concordance', 0.0),
            metrics.get('integrated_brier_score', None),
            metrics.get('contribution', 0.0)
        ]
        
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(headers)
                csvwriter.writerow(record)
        else:
            with open(filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(record)


def initialize_cnnet_modules(model, hidden_dims=[64, 32], alpha=0.001, 
                            epsilon=1e-4, device='cpu'):
    """
    Initialize ICNN modules for FedMAP using the existing InputConvexNN class.
    
    Args:
        model: The neural network model
        hidden_dims: Hidden dimensions for ICNN
        alpha: Regularization parameter
        epsilon: Perturbation parameter
        device: Device to use
        
    Returns:
        ModuleList of ICNN modules
    """

    
    param_sizes = [p.numel() for p in model.parameters()]
    cnnet_modules = nn.ModuleList([
        InputConvexNN(param_size=size, hidden_dims=hidden_dims, 
                     alpha=alpha, epsilon=epsilon).to(device)
        for size in param_sizes
    ])
    return cnnet_modules