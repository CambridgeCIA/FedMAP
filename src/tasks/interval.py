import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import math
import torch.nn as nn
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import auc, balanced_accuracy_score, roc_curve, accuracy_score, confusion_matrix
from src.models import DenseClassifier
from src.utils.train_helper import InputConvexNN
from src.loss_modules.map import FedMAPLoss, ICNNPrior 
from typing import Optional
import torch.nn.functional as F


hl_features = [
    "WBC_10_9_L", "RBC_10_12_L", "HGB_g_L", "HCT_PCT",
    "MCV_fL", "MCH_pg", "MCHC_g_dL", "PLT_10_9_L",
    "RDW_SD_fL", "NEUT_10_9_L", "LYMPH_10_9_L",
    "MONO_10_9_L", "EO_10_9_L", "BASO_10_9_L",
    "NRBC_10_9_L", "IG_10_9_L"
]


class INTERVALDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class Interval:
    def __init__(self, cid, config, device):
        self.cid = cid  
        self.config = config  
        self.lr = config['lr']
        self.local_epochs = config['local_epochs']
        self.server_round = config['server-round']
        self.device = device
        self.local_model = None
        self.global_model = None
        self.cnnet_modules = None
        self.prior = None
        
    def set_models(self, global_model, cnnet_modules):
        self.local_model = DenseClassifier().to(self.device)
        self.global_model = DenseClassifier().to(self.device)

        if cnnet_modules is not None:
            icnn_dict = {}
            for name, param in self.local_model.named_parameters():
                param_size = param.numel()
                sanitized_name = name.replace('.', '__') 
                icnn_dict[sanitized_name] = InputConvexNN(param_size=param_size).to(self.device)
            
            self.cnnet_modules = nn.ModuleDict(icnn_dict)
            self.cnnet_modules.load_state_dict(cnnet_modules)
            self.prior = ICNNPrior(self.cnnet_modules)
            
        self.global_model.load_state_dict(global_model)
        self.local_model.load_state_dict(global_model)
          
    def _load_data_by_id(self, batch_size=128):
        base_path = "datasets/interval" 

        files = [
            {"train": "INTERVAL_irondef_site_1_train.csv", "val": "INTERVAL_irondef_site_1_val.csv"},
            {"train": "INTERVAL_irondef_site_2_train.csv", "val": "INTERVAL_irondef_site_2_val.csv"},
            {"train": "INTERVAL_irondef_site_3_train.csv", "val": "INTERVAL_irondef_site_3_val.csv"},
        ]
        
        if self.cid >= len(files):
            raise ValueError(f"Client ID {self.cid} is out of bounds for file list.")

        train_file = f"{base_path}/{files[self.cid]['train']}"
        val_file = f"{base_path}/{files[self.cid]['val']}"
        
        try:
            train_df = pd.read_csv(train_file)
            val_df = pd.read_csv(val_file)
        except FileNotFoundError:
            print("="*50)
            print(f"ERROR: Data files not found for client {self.cid} at: {base_path}")
            print("Please update the `base_path` in pytorchexample/client_app.py")
            print("="*50)
            raise

        train_features = hl_features + ["Age", "Sex"]
        train_df['Sex'] = (train_df['Sex'] == 'M').astype(np.float32)
        val_df['Sex'] = (val_df['Sex'] == 'M').astype(np.float32)

        for col in train_features:
            if col != 'Sex':
                train_df[col] = train_df[col].astype(np.float32)
                val_df[col] = val_df[col].astype(np.float32)

        X_train = train_df[train_features].values
        y_train = train_df["ferritin_low"].values.astype(np.int64)
        X_val = val_df[train_features].values
        y_val = val_df["ferritin_low"].values.astype(np.int64)


        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / (std + 1e-8)
        X_val = (X_val - mean) / (std + 1e-8)

        train_dataset = INTERVALDataset(X_train, y_train)
        val_dataset = INTERVALDataset(X_val, y_val)

    
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[y_train]

        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    
    def train_loader(self, batch_size=128):
        train_loader, _ = self._load_data_by_id(batch_size)
        return train_loader

    def val_loader(self, batch_size=128):
        """Returns the validation data loader."""
        _, val_loader = self._load_data_by_id(batch_size)
        return val_loader

    def train(self, patience=3, batch_size=128):
        """
        Train the network on the client's dataset.
        """
        # Get data loaders
        trainloader = self.train_loader(batch_size)
        valloader = self.val_loader(batch_size)

        criterion = FedMAPLoss(nn.CrossEntropyLoss(), self.prior, self.global_model)
        criterion.bind_model(self.local_model)

        # Use learning rate from class config
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_state_dict = self.local_model.state_dict()
        epochs_without_improvement = 0

        # Use local epochs from class config
        for epoch in range(self.local_epochs):
            self.local_model.train()
            train_loss = 0.0
            for batch_data, batch_label in trainloader:
                batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.local_model(batch_data)
                loss = criterion(outputs, batch_label) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch_data.size(0)

            train_loss /= len(trainloader.dataset)

            self.local_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_label in valloader:
                    batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)
                    outputs = self.local_model(batch_data)
                    loss = criterion(outputs, batch_label) 
                    val_loss += loss.item() * batch_data.size(0)

            val_loss /= len(valloader.dataset)
            
            print(f"Client {self.cid} | Epoch [{epoch+1}/{self.local_epochs}]: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

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

    def validate(self, batch_size=128, tier=1):
        testloader = self.val_loader(batch_size)
        
        self.local_model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='sum')

        with torch.no_grad():
            for batch_data, batch_label in testloader:
                batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)
                outputs = self.local_model(batch_data)
                
                loss = criterion(outputs, batch_label)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_label.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

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
            print("Warning: ROC AUC calculation failed (e.g., only one class present).")

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
           self._record_performance(self.server_round, metrics, tier)
        except Exception:
            pass
        return avg_loss, metrics
    

    def _record_performance(self, round_num, metrics, tier=1):
        """
        Log validation/test metrics to a CSV file.
        Based on 'log_test_metrics_to_csv' function.
        """
        filename = f'./results/interval_metrics_test_tier{tier}.csv'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        headers = ['Client_ID', 'Round', 'Loss', 'Balanced_Accuracy', 'ROC_AUC', 'TN', 'FP', 'FN', 'TP']
        record = [
            int(self.cid) + 1, 
            round_num,
            metrics.get('loss', 0.0),
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
    def _calculate_contribution(
        self,
    ) -> float:
        """
        FedMAP client-side contribution computation.
        Based on 'calculate_contribution' function.
        """
        self.local_model.eval()
        self.global_model.eval()
        
        if isinstance(self.cnnet_modules, nn.ModuleDict):
            for cnnet in self.cnnet_modules.values():
                cnnet.eval()
        else:
            for cnnet in self.cnnet_modules:
                cnnet.eval()

        train_loader_instance = self.train_loader(batch_size=128)
        dataset = train_loader_instance.dataset
        bs = 128
        likelihood_loader = DataLoader(dataset, batch_size=bs, shuffle=False)

    
        total_neg_loglik, N = 0.0, 0
        for x, y in likelihood_loader:
            x, y = x.to(self.device), y.to(self.device)

            logits = self.local_model(x)
            total_neg_loglik += F.cross_entropy(logits, y, reduction="sum").item()


            N += y.size(0)
        
        if N == 0:
            return 0.0 

        sum_loglik = -total_neg_loglik
        mean_loglik = sum_loglik / N

        
        prior_term = 0.0
        
        
        if isinstance(self.cnnet_modules, nn.ModuleDict):
            first_cnnet = next(iter(self.cnnet_modules.values()))
        else:
            first_cnnet = self.cnnet_modules[0]
        
        cnnet_dtype = next(first_cnnet.parameters()).dtype if len(list(first_cnnet.parameters())) > 0 else torch.float32
        

        if isinstance(self.cnnet_modules, nn.ModuleDict):
            prior_term = self._compute_prior_module_dict(self.local_model, self.global_model, self.cnnet_modules, self.device, cnnet_dtype)
        else:
            prior_term = self._compute_prior_module_list(self.local_model, self.global_model, self.cnnet_modules, self.device, cnnet_dtype)

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
        """Helper method for contribution calculation."""
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
        """Helper method for contribution calculation."""
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


def log_train_metrics_to_csv(cid, epoch, train_loss, train_acc):
    """Log training metrics to a CSV file."""
    filename = 'metrics_train.csv'
    headers = ['Client_ID', 'Epoch', 'Train_Loss', 'Train_Accuracy']
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    record = [int(cid)+1, epoch, train_loss, train_acc]

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerow(record)
    else:
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(record)