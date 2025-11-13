import math
import torch
import torch.nn as nn
import numpy as np
import os
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import auc, balanced_accuracy_score, roc_curve, accuracy_score, confusion_matrix
from src.loss_modules.map import FedMAPLoss

class InputConvexNN(nn.Module):
    """
    Input Convex Neural Network (ICNN) module.
    """
    def __init__(self, param_size, hidden_dims=[64, 32], alpha=0.01, epsilon=1e-4):
        super(InputConvexNN, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon  
        input_dim = param_size * 2
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0], bias=True))
        layers.append(nn.Softplus())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dims[-1], 1, bias=True))
        self.net = nn.Sequential(*layers)
        
        for layer in self.net:
            if isinstance(layer, nn.Linear) and layer != self.net[0]:
                layer.weight.data.clamp_(min=0)
    
    def forward(self, theta, gamma):
        """
        theta: The local model parameters.
        gamma: The global model parameters.
        """
        x = torch.cat((theta, gamma), dim=-1)
        base_term = self.net(x)
        quad_term = self.alpha / 2 * (theta - gamma).pow(2).sum(dim=-1, keepdim=True)
        perturbation = self.epsilon * (theta.pow(2).sum(dim=-1, keepdim=True) + gamma.pow(2).sum(dim=-1, keepdim=True))
        return base_term + quad_term + perturbation

    def enforce_convexity(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear) and layer != self.net[0]:
                layer.weight.data.clamp_(min=0)


def train(net, trainloader, valloader, gamma, prior, device, local_epochs, patience=3):
    criterion = FedMAPLoss(nn.CrossEntropyLoss(), prior, gamma)
    criterion.bind_model(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_loss = float('inf')
    best_state_dict = net.state_dict()
    epochs_without_improvement = 0

    for epoch in range(local_epochs):
        net.train()
        train_loss = 0.0
        for batch_data, batch_label in trainloader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            
            optimizer.zero_grad()
            outputs = net(batch_data)
            loss = criterion(outputs, batch_label) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_data.size(0)

        train_loss /= len(trainloader.dataset)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_label in valloader:
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)
                outputs = net(batch_data)
                loss = criterion(outputs, batch_label) 
                val_loss += loss.item() * batch_data.size(0)

        val_loss /= len(valloader.dataset)
        
        print(f"Epoch [{epoch+1}/{local_epochs}]: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = net.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return best_state_dict

def test(net, testloader, device):
    """
    Evaluate the model on the test set.
    Adapted from your script's `evaluate_model` function.
    
    Returns:
        float: The average loss.
        dict: A dictionary of performance metrics.
    """
    net.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for batch_data, batch_label in testloader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            outputs = net(batch_data)
            
        
            loss = criterion(outputs, batch_label)
            total_loss += loss.item()

     
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
    
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_label.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())


    avg_loss = total_loss / len(testloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        print("Warning: ROC AUC calculation failed.")

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    # Ensure cm is 2x2
    if cm.size == 1: # Only one class predicted and present
        if all_labels[0] == 0:
            cm = np.array([[cm[0][0], 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, cm[0][0]]])
    elif cm.shape[0] == 1: # only one class present
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
    
    return avg_loss, metrics


from typing import Optional
import torch.nn.functional as F


@torch.no_grad()
def calculate_contribution(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    gamma: torch.nn.Module,
    cnnet_modules, 
    device: torch.device,
    task_type: str = "classification",
    batch_size: Optional[int] = None,
    per_sample: bool = True
) -> float:
    """
    FedMAP client-side contribution computation.

    Returns:
        float: log_contribution (a scalar in log-space)
    """
    net.eval()
    gamma.eval()
    
    if isinstance(cnnet_modules, nn.ModuleDict):
        for cnnet in cnnet_modules.values():
            cnnet.eval()
    else:
        for cnnet in cnnet_modules:
            cnnet.eval()


    dataset = trainloader.dataset
    bs = batch_size or getattr(trainloader, "batch_size", 256) or 256
    likelihood_loader = DataLoader(dataset, batch_size=bs, shuffle=False)

   
    total_neg_loglik, N = 0.0, 0
    for x, y in likelihood_loader:
        x, y = x.to(device), y.to(device)

        if task_type == "classification":
            logits = net(x)
            total_neg_loglik += F.cross_entropy(logits, y, reduction="sum").item()

        elif task_type == "binary":
            logits = net(x)
            logits = logits.view(-1)
            y = y.float().view(-1)
            total_neg_loglik += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()

        else:
            raise NotImplementedError(f"Unsupported task_type: {task_type}")

        N += y.size(0)

    sum_loglik = -total_neg_loglik
    mean_loglik = sum_loglik / max(N, 1)

    prior_term = 0.0
    
    
    if isinstance(cnnet_modules, nn.ModuleDict):
        first_cnnet = next(iter(cnnet_modules.values()))
    else:
        first_cnnet = cnnet_modules[0]
    
    cnnet_dtype = next(first_cnnet.parameters()).dtype if len(list(first_cnnet.parameters())) > 0 else torch.float32
    

    if isinstance(cnnet_modules, nn.ModuleDict):
        prior_term = _compute_prior_module_dict(net, gamma, cnnet_modules, device, cnnet_dtype)
    else:
        prior_term = _compute_prior_module_list(net, gamma, cnnet_modules, device, cnnet_dtype)


    if per_sample:
        log_contribution = mean_loglik - (prior_term / max(N, 1))
    else:
        log_contribution = sum_loglik - prior_term

    return float(math.exp(log_contribution))


def _compute_prior_module_list(
    net: nn.Module, 
    gamma: nn.Module, 
    cnnet_modules: nn.ModuleList,
    device: torch.device,
    cnnet_dtype: torch.dtype
) -> float:

    prior_term = 0.0
    
    for (lp, gp), cnnet in zip(zip(net.parameters(), gamma.parameters()), cnnet_modules):
        theta_flat = lp.detach().view(1, -1).to(device=device, dtype=cnnet_dtype)
        mu_flat = gp.detach().view(1, -1).to(device=device, dtype=cnnet_dtype)
        prior_term += cnnet(theta_flat, mu_flat).sum().item()
    
    return prior_term


def _compute_prior_module_dict(
    net: nn.Module,
    gamma: nn.Module,
    cnnet_modules: nn.ModuleDict,
    device: torch.device,
    cnnet_dtype: torch.dtype
) -> float:
   
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
    
    # Ensure directory exists
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

def log_test_metrics_to_csv(cid, round_num, metrics):
    """Log validation/test metrics to a CSV file."""
    filename = './results/metrics_test.csv'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    headers = ['Client_ID', 'Round', 'Loss', 'Balanced_Accuracy', 'ROC_AUC', 'TN', 'FP', 'FN', 'TP']
    record = [
        int(cid) + 1, 
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
        with open(filename, 'a', newline='')as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(record)