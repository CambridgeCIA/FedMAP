import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, 
    confusion_matrix, roc_curve, auc
)
from src.utils.train_helper import InputConvexNN
from src.loss_modules.map import FedMAPLoss, ICNNPrior
from typing import Optional
from collections import OrderedDict
import torch.nn.functional as F
import math
from src.models import MLP


class SyntheticDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features) if not isinstance(features, torch.Tensor) else features
        self.labels = torch.FloatTensor(labels) if not isinstance(labels, torch.Tensor) else labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class Example:
    def __init__(self, cid, config, device):
        """
        Initialize Exampleclient for federated learning.
        
        Args:
            cid: Client ID
            config: Configuration dictionary containing training parameters
            device: torch device (cuda/cpu)
        """
        self.cid = cid
        self.config = config
        self.lr = 0.0001
        self.local_epochs = config.get('local_epochs', 15)
        self.server_round = config.get('server-round', 0)
        self.device = device
        
        self.local_model = None
        self.global_model = None
        self.cnnet_modules = None
        self.prior = None
        
        self.data_path = './datasets/example'
        self.validation_ratio = 0.2
        
        self.input_dim = 31
        
    def _load_data_by_id(self, batch_size=64, mode="train"):
        """
        Load data for specific client ID.
        
        Args:
            batch_size: Batch size for data loaders
            
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        # Load data
        data_file = f'{self.data_path}/partition_{self.cid}_{mode}.csv'
        
        try:
            data = pd.read_csv(data_file)
        except FileNotFoundError:
            print("="*50)
            print(f"ERROR: Data file not found for client {self.cid} at: {data_file}")
            print("Please update the `data_path` in config")
            print("="*50)
            raise
        
      
        LABEL = 'ClassCategory_0'
        
        X = data.drop(columns=[LABEL])
        y = data[LABEL]
        

        if self.input_dim is None:
            self.input_dim = X.shape[1]
        print(f"Client {self.cid} - Input dimension: {self.input_dim}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.validation_ratio, 
            random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Client {self.cid} - Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
              f"Features: {self.input_dim}")
        
        return train_loader, val_loader
    
    def train_loader(self, batch_size=64):
        """Returns the training data loader."""
        train_loader, _ = self._load_data_by_id(batch_size, "train")
        return train_loader
    
    def val_loader(self, batch_size=64):
        """Returns the validation data loader."""
        _, val_loader = self._load_data_by_id(batch_size, "test")
        return val_loader
    
    def set_models(self, global_model, cnnet_modules):
        """
        Set up local and global models with ICNN modules.
        
        Args:
            global_model: State dict of the global model
            cnnet_modules: State dict of ICNN modules or nn.ModuleDict
        """

        self.local_model = MLP(self.input_dim).to(self.device)
        self.global_model = MLP(self.input_dim).to(self.device)
        
     
        if global_model is not None:
            self.global_model.load_state_dict(global_model)
            self.local_model.load_state_dict(global_model)
        
        if cnnet_modules is not None:
            if isinstance(cnnet_modules, (dict, OrderedDict)):
                icnn_dict = {}
                for name, param in self.local_model.named_parameters():
                    param_size = param.numel()
                    sanitized_name = name.replace('.', '__')
                    icnn_dict[sanitized_name] = InputConvexNN(param_size=param_size).to(self.device)
                
                self.cnnet_modules = nn.ModuleDict(icnn_dict)
                self.cnnet_modules.load_state_dict(cnnet_modules)
            else:
                self.cnnet_modules = cnnet_modules
            
            self.prior = ICNNPrior(self.cnnet_modules)
    
    def train(self, patience=3, batch_size=64):
        trainloader = self.train_loader(batch_size)
        valloader = self.val_loader(batch_size)
        

        criterion = FedMAPLoss(nn.BCELoss(), self.prior, self.global_model)
        criterion.bind_model(self.local_model)
        
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        best_val_loss = float('inf')
        best_state_dict = self.local_model.state_dict()
        epochs_without_improvement = 0
        
        # Training loop
        for epoch in range(self.local_epochs):
            self.local_model.train()
            train_loss = 0.0
            
            for batch_data, batch_label in trainloader:
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.local_model(batch_data).squeeze()
                loss = criterion(outputs, batch_label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_data.size(0)
            
            train_loss /= len(trainloader.dataset)
            
            # Validation
            self.local_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_label in valloader:
                    batch_data = batch_data.to(self.device)
                    batch_label = batch_label.to(self.device)
                    outputs = self.local_model(batch_data).squeeze()
                    loss = criterion(outputs, batch_label)
                    val_loss += loss.item() * batch_data.size(0)
            
            val_loss /= len(valloader.dataset)
            
            print(f"Client {self.cid} | Epoch [{epoch+1}/{self.local_epochs}]: "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = self.local_model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"Client {self.cid}: Early stopping triggered.")
                break
        
        contribution = self._calculate_contribution()
        return best_state_dict, contribution
    
    def validate(self, batch_size=64):
        """
        Validate the model on validation set.
        
        Args:
            batch_size: Batch size for validation
            
        Returns:
            avg_loss: Average validation loss
            metrics: Dictionary of validation metrics
        """
        testloader = self.val_loader(batch_size)
        
        self.local_model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0.0
        criterion = nn.BCELoss(reduction='sum')
        
        with torch.no_grad():
            for batch_data, batch_label in testloader:
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)
                outputs = self.local_model(batch_data).squeeze()
                
                loss = criterion(outputs, batch_label)
                total_loss += loss.item()
                
                predicted = (outputs >= 0.5).float()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_label.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        if not all_labels:
            print("Warning: Validation set is empty.")
            return 0.0, {}
        
        avg_loss = total_loss / len(testloader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        roc_auc = 0.0
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
        except ValueError:
            print("Warning: ROC AUC calculation failed.")
        
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        if cm.size == 1:
            if all_labels[0] == 0:
                cm = np.array([[cm[0][0], 0], [0, 0]])
            else:
                cm = np.array([[0, 0], [0, cm[0][0]]])
        elif cm.shape[0] == 1:
            if all_labels[0] == 0:
                cm = np.array([[cm[0][0], cm[0][1]], [0, 0]])
            else:
                cm = np.array([[0, 0], [cm[0][0], cm[0][1]]])
        
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "roc_auc": roc_auc,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
        try:
           self._record_performance(self.server_round, metrics)
        except Exception:
            pass
        return avg_loss, metrics
    
    def _record_performance(self, round_num, metrics):
        """
        Log validation/test metrics to a CSV file.
        
        Args:
            round_num: Current federated learning round
            metrics: Dictionary of metrics to log
        """
        filename = './results/example_metrics_test.csv'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        headers = ['Client_ID', 'Round', 'Loss', 'Accuracy', 'Balanced_Accuracy', 
                   'ROC_AUC', 'TN', 'FP', 'FN', 'TP']
        record = [
            int(self.cid),
            round_num,
            metrics.get('loss', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('balanced_accuracy', 0.0),
            metrics.get('roc_auc', 0.0),
            metrics.get('tn', 0),
            metrics.get('fp', 0),
            metrics.get('fn', 0),
            metrics.get('tp', 0)
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
    
    @torch.no_grad()
    def _calculate_contribution(self) -> float:
        """
        FedMAP client-side contribution computation.
        Uses the same algorithm as INTERVAL task.
        
        Returns:
            contribution: Client contribution score
        """
        self.local_model.eval()
        self.global_model.eval()
        
        if self.cnnet_modules is not None:
            if isinstance(self.cnnet_modules, nn.ModuleDict):
                for cnnet in self.cnnet_modules.values():
                    cnnet.eval()
            else:
                for cnnet in self.cnnet_modules:
                    cnnet.eval()
        
        train_loader_instance = self.train_loader(batch_size=64)
        dataset = train_loader_instance.dataset
        likelihood_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        # Compute likelihood term
        total_neg_loglik = 0.0
        N = 0
        
        for x, y in likelihood_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            logits = self.local_model(x).squeeze()
            # BCE loss is negative log-likelihood for binary classification
            total_neg_loglik += F.binary_cross_entropy(logits, y, reduction="sum").item()
            N += y.size(0)
        
        if N == 0:
            return 0.0
        
        sum_loglik = -total_neg_loglik
        mean_loglik = sum_loglik / N
        
        # Compute prior term
        prior_term = 0.0
        
        # Get cnnet dtype
        if isinstance(self.cnnet_modules, nn.ModuleDict):
            first_cnnet = next(iter(self.cnnet_modules.values()))
        else:
            first_cnnet = self.cnnet_modules[0]
        
        cnnet_dtype = (next(first_cnnet.parameters()).dtype 
                       if len(list(first_cnnet.parameters())) > 0 
                       else torch.float32)
        
        # Compute prior
        if isinstance(self.cnnet_modules, nn.ModuleDict):
            prior_term = self._compute_prior_module_dict(
                self.local_model, self.global_model, self.cnnet_modules,
                self.device, cnnet_dtype
            )
        else:
            prior_term = self._compute_prior_module_list(
                self.local_model, self.global_model, self.cnnet_modules,
                self.device, cnnet_dtype
            )
        
        # Calculate contribution (same formula as INTERVAL)
        log_contribution = mean_loglik - (prior_term / N)
        
        return float(math.exp(log_contribution))
    
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