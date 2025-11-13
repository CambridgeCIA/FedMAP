import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import math
import torch.nn as nn
import csv
from sklearn.metrics import auc, balanced_accuracy_score, roc_curve, accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from src.models import MultimodalFFN
from src.utils.train_helper import InputConvexNN
from src.loss_modules.map import FedMAPLoss, ICNNPrior 
from typing import Optional
import torch.nn.functional as F

SITES_A = [73, 79, 92, 110, 122, 140, 141, 142, 143, 144, 148, 154, 157, 165, 
           167, 171, 176, 181, 183, 184, 197, 199, 226, 227, 243, 248, 252, 264, 
           269, 271, 272, 277, 279, 280, 281, 283, 300, 338, 345, 353, 365, 416, 
           417, 419, 420, 421, 424, 444, 449, 452, 458]

def minmaxscale(column):
    max_, min_ = column.max(), column.min()
    if max_ == min_:
        return column * 0.0
    return (column - min_) / (max_ - min_)


class MultiModalDataset(Dataset):
    def __init__(self, drugs_df, dx_df, physio_df, mortality_df, patient_list):
        self.patient_ids = patient_list
        self.drugs = torch.FloatTensor(drugs_df.loc[patient_list].values)
        self.dx = torch.FloatTensor(dx_df.loc[patient_list].values)
        self.physio = torch.FloatTensor(physio_df.loc[patient_list].values)
        self.labels = torch.LongTensor(mortality_df.loc[patient_list]['expired'].astype(int).values)
        
    def __len__(self):
        return len(self.patient_ids)
        
    def __getitem__(self, idx):
        return (self.drugs[idx], self.dx[idx], self.physio[idx]), self.labels[idx]


class eICU:
    def __init__(self, cid, config, device):
        """
        Initialize eICU client for federated learning.
        
        Args:
            cid: Client/Hospital ID
            config: Configuration dictionary containing training parameters
            device: torch device (cuda/cpu)
        """
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
    
        self.input_dim_drugs = None
        self.input_dim_dx = None
        self.input_dim_physio = None
        
    def set_models(self, global_model, cnnet_modules):
        """
        Set up local and global models with ICNN modules.
        
        Args:
            global_model: State dict of the global model
            cnnet_modules: State dict of ICNN modules
            input_dims: Tuple of (dim_drugs, dim_dx, dim_physio)
        """
        self.input_dim_drugs, self.input_dim_dx, self.input_dim_physio = (1411, 686, 7)
        

        self.local_model = MultimodalFFN(
            input_dim_drugs=self.input_dim_drugs,
            input_dim_dx=self.input_dim_dx,
            input_dim_physio=self.input_dim_physio
        ).to(self.device)
        
        self.global_model = MultimodalFFN(
            input_dim_drugs=self.input_dim_drugs,
            input_dim_dx=self.input_dim_dx,
            input_dim_physio=self.input_dim_physio
        ).to(self.device)

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
  
    def _load_data_by_id(self, batch_size=32, mode='train', seed=42):
        """
        Load data for specific hospital ID.
        
        Args:
            batch_size: Batch size for data loaders
            seed: Random seed for reproducibility
            
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
            input_dims: Tuple of input dimensions
        """
        base_path = '/app/datasets/eicu/group_A'
        hospital_id = SITES_A[self.cid]
        
        try:
            mortality = pd.read_csv(
                f'{base_path}/{hospital_id}/mortality_{mode}.csv',
                usecols=['patientunitstayid', 'expired'],
                index_col='patientunitstayid'
            )
            drugs = pd.read_csv(
                f'{base_path}/{hospital_id}/medications_{mode}.csv',
                index_col='patientunitstayid'
            )
            dx = pd.read_csv(
                f'{base_path}/{hospital_id}/diagnosis_{mode}.csv',
                index_col='patientunitstayid'
            )
            physio = pd.read_csv(
                f'{base_path}/{hospital_id}/physio_{mode}.csv',
                index_col='patientunitstayid'
            )
        except FileNotFoundError:
            print("="*50)
            print(f"ERROR: Data files not found for hospital {hospital_id} at: {base_path}")
            print("Please update the `data_path` in config")
            print("="*50)
            raise

   
        drugs_scaled = drugs.apply(minmaxscale, axis=1)
        dx_scaled = dx.apply(minmaxscale, axis=1)
        physio_scaled = physio.apply(minmaxscale, axis=1)

     
        patient_ids = list(
            set(drugs_scaled.index) & set(dx_scaled.index) &
            set(physio_scaled.index) & set(mortality.index)
        )

        if not patient_ids:
            raise ValueError(f"No valid patient IDs for hospital {hospital_id} after intersection.")

 
        drugs_scaled = drugs_scaled.loc[patient_ids]
        dx_scaled = dx_scaled.loc[patient_ids]
        physio_scaled = physio_scaled.loc[patient_ids]
        mortality = mortality.loc[patient_ids]


        labels = mortality['expired'].astype(int).values
        n_positive = sum(labels)
        n_negative = len(labels) - n_positive
        print(f"Hospital {hospital_id} - Total samples: {len(labels)}, "
              f"Positive: {n_positive}, Negative: {n_negative}")

        if n_positive == 0 or len(labels) < batch_size:
            raise ValueError(f"Hospital {hospital_id} has no positive samples or too few samples ({len(labels)}).")

        
        train_ids, val_ids = train_test_split(
            patient_ids,
            test_size=0.3,
            stratify=labels,
            random_state=seed
        )

        if sum(mortality.loc[val_ids]['expired'].astype(int).values) == 0:
            raise ValueError(f"Validation set for hospital {hospital_id} has no positive samples.")

 
        train_dataset = MultiModalDataset(drugs_scaled, dx_scaled, physio_scaled, mortality, train_ids)
        val_dataset = MultiModalDataset(drugs_scaled, dx_scaled, physio_scaled, mortality, val_ids)

        train_labels = mortality.loc[train_ids]['expired'].astype(int).values
        class_counts = np.bincount(train_labels)
        class_weights = np.array([1.0 / count if count > 0 else 0.0 for count in class_counts])
        sample_weights = np.array([class_weights[label] for label in train_labels])
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


        effective_batch_size = min(batch_size, len(train_ids) // 2)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            sampler=train_sampler,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        return train_loader, val_loader
    
    def train_loader(self, batch_size=32):
        train_loader, _ = self._load_data_by_id(batch_size, mode='train')
        return train_loader

    def val_loader(self, batch_size=32):
        _, val_loader = self._load_data_by_id(batch_size, mode='val')
        return val_loader

    def train(self, patience=3, batch_size=32):
        """
        Train the network on the client's dataset.
        
        Args:
            patience: Early stopping patience
            batch_size: Batch size for training
            
        Returns:
            best_state_dict: Best model state dict
            contribution: Client contribution score
        """
        # Get data loaders
        trainloader, valloader = self._load_data_by_id(batch_size)

        criterion = FedMAPLoss(nn.CrossEntropyLoss(), self.prior, self.global_model)
        criterion.bind_model(self.local_model)

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_state_dict = self.local_model.state_dict()
        epochs_without_improvement = 0

       
        for epoch in range(self.local_epochs):
            self.local_model.train()
            train_loss = 0.0
            samples_processed = 0
            
            for (batch_drugs, batch_dx, batch_physio), batch_label in trainloader:
                if batch_drugs.size(0) <= 1:
                    continue
                    
                batch_drugs = batch_drugs.to(self.device)
                batch_dx = batch_dx.to(self.device)
                batch_physio = batch_physio.to(self.device)
                batch_label = batch_label.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.local_model(batch_drugs, batch_dx, batch_physio)
                loss = criterion(outputs, batch_label) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_drugs.size(0)
                samples_processed += batch_drugs.size(0)

            if samples_processed > 0:
                train_loss /= samples_processed
            else:
                train_loss = float('inf')

            # Validation
            self.local_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch_drugs, batch_dx, batch_physio), batch_label in valloader:
                    batch_drugs = batch_drugs.to(self.device)
                    batch_dx = batch_dx.to(self.device)
                    batch_physio = batch_physio.to(self.device)
                    batch_label = batch_label.to(self.device)
                    
                    outputs = self.local_model(batch_drugs, batch_dx, batch_physio)
                    loss = criterion(outputs, batch_label) 
                    val_loss += loss.item() * batch_drugs.size(0)

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
            
        # Calculate contribution
        contribution = self._calculate_contribution()
        return best_state_dict, contribution

    def validate(self, batch_size=32):
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
        criterion = nn.CrossEntropyLoss(reduction='sum')

        with torch.no_grad():
            for (batch_drugs, batch_dx, batch_physio), batch_label in testloader:
                batch_drugs = batch_drugs.to(self.device)
                batch_dx = batch_dx.to(self.device)
                batch_physio = batch_physio.to(self.device)
                batch_label = batch_label.to(self.device)
                
                outputs = self.local_model(batch_drugs, batch_dx, batch_physio)
                
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
        auprc = 0.0
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(all_labels, all_probs)
            auprc = average_precision_score(all_labels, all_probs)
        except ValueError:
            print("Warning: ROC AUC/AUPRC calculation failed (e.g., only one class present).")

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
            "auprc": auprc,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
        
        return avg_loss, metrics

    def _record_performance(self, round_num, metrics):
        """
        Log validation/test metrics to a CSV file.
        
        Args:
            round_num: Current federated learning round
            metrics: Dictionary of metrics to log
        """
        filename = './results/eicu_metrics_test.csv'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        headers = ['Client_ID', 'Round', 'Loss', 'Accuracy', 'Balanced_Accuracy', 
                   'ROC_AUC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']
        record = [
            int(self.cid), 
            round_num,
            metrics.get('loss', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('balanced_accuracy', 0.0),
            metrics.get('roc_auc', 0.0),
            metrics.get('auprc', 0.0),
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
        
        Returns:
            contribution: Client contribution score
        """
        self.local_model.eval()
        self.global_model.eval()
        
        if isinstance(self.cnnet_modules, nn.ModuleDict):
            for cnnet in self.cnnet_modules.values():
                cnnet.eval()
        else:
            for cnnet in self.cnnet_modules:
                cnnet.eval()

        train_loader_instance = self.train_loader(batch_size=32)
        dataset = train_loader_instance.dataset
        bs = 32
        likelihood_loader = DataLoader(dataset, batch_size=bs, shuffle=False)

        # Compute likelihood term
        total_neg_loglik, N = 0.0, 0
        for (x_drugs, x_dx, x_physio), y in likelihood_loader:
            x_drugs = x_drugs.to(self.device)
            x_dx = x_dx.to(self.device)
            x_physio = x_physio.to(self.device)
            y = y.to(self.device)

            logits = self.local_model(x_drugs, x_dx, x_physio)
            total_neg_loglik += F.cross_entropy(logits, y, reduction="sum").item()
            N += y.size(0)
        
        if N == 0:
            return 0.0 

        sum_loglik = -total_neg_loglik
        mean_loglik = sum_loglik / N

        # Compute prior term
        prior_term = 0.0
        
        if isinstance(self.cnnet_modules, nn.ModuleDict):
            first_cnnet = next(iter(self.cnnet_modules.values()))
        else:
            first_cnnet = self.cnnet_modules[0]
        
        cnnet_dtype = next(first_cnnet.parameters()).dtype if len(list(first_cnnet.parameters())) > 0 else torch.float32
        
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


def log_train_metrics_to_csv(cid, epoch, train_loss, train_acc):
    """Log training metrics to a CSV file."""
    filename = './results/eicu_metrics_train.csv'
    headers = ['Client_ID', 'Epoch', 'Train_Loss', 'Train_Accuracy']
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    record = [int(cid), epoch, train_loss, train_acc]

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerow(record)
    else:
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(record)