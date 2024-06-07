from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
from ..loss_modules import MAPLoss
from sklearn.metrics import balanced_accuracy_score

def train(net, trainloader, epochs, device: str, cid: str, strategy_name='fedavg', isMultiClass=False, gamma=None, variance=None):
    """Train the model."""

    if strategy_name == 'fedmap':
        if isMultiClass:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.BCELoss()
            
        criterion = MAPLoss(loss_fn, gamma, variance)
    else:
        if isMultiClass:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
     
        
    optimizer = torch.optim.Adam(
        net.parameters(), lr=0.001
    )
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_data, batch_label in trainloader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            
            outputs = net(batch_data.to(device)).squeeze()
            loss = criterion(outputs, batch_label, net) if strategy_name == 'fedmap' else criterion(outputs, batch_label) 
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            # Calculate accuracy
            predicted = outputs.round()
            correct_predictions += (predicted == batch_label.to(device)).sum().item()
            total_predictions += batch_label.size(0)

        epoch_loss = round(running_loss / len(trainloader.dataset), 4)
        epoch_acc = round(correct_predictions / total_predictions, 4)
        print(f"Client>>>>>>>>>>>> {cid} | Epoch {epoch+1} | Loss: {epoch_loss} | Accuracy: {epoch_acc}")

        # if epoch == epochs - 1:
        #     log_train_metrics_to_csv(cid, epoch_loss, epoch_acc)
    if strategy_name == 'fedmap':
        contribution = calculate_contribution(net, trainloader, gamma, variance, device=device)

        return contribution

def test(net, testloader, device: str, cid: str, isMultiClass=False):
    """Validate the model on the test set."""
    if isMultiClass:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()
     
    correct, loss, total = 0, 0.0, 0
    true_labels = []
    predicted_labels = []
    net.eval()
     
    with torch.no_grad():
        for batch_data, batch_label in testloader:
            outputs = net(batch_data.to(device)).squeeze()
            labels = batch_label.to(device)
            loss += criterion(outputs, labels).item()
            predicted = outputs.round() 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend(batch_label.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            
    accuracy = round(correct / len(testloader.dataset), 4)
    loss = round(loss / len(testloader.dataset), 4)
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    if cid != "Aggregator":
        log_test_metrics_to_csv(cid, loss, balanced_accuracy)

    return loss, balanced_accuracy

def log_train_metrics_to_csv(cid, train_loss, train_acc):
    """Log training metrics to a CSV file."""
    filename = 'metrics_train.csv'
    headers = ['Client_ID', 'Train_Loss', 'Train_Accuracy']
    record = [int(cid)+1, train_loss, train_acc]

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerow(record)
    else:
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(record)

def log_test_metrics_to_csv(cid, val_loss, val_acc):
    """Log validation metrics to a CSV file."""
    filename = 'metrics_test.csv'
    headers = ['Client_ID', 'Val_Loss', 'Val_Accuracy']
    record = [int(cid) + 1, val_loss, val_acc]
    
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerow(record)
    else:
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(record)

def calculate_contribution(model, data_i, gamma, variance, device):
    """Calculate weight = P(Z_i|theta_i) * rho_{gamma_n}(theta_i) for a given data point."""
    all_datasets = []
    all_labels = []

    for batch in data_i:
        data, labels = batch
        all_datasets.append(data.to(device)) 
        all_labels.append(labels.to(device))  
        
    all_datasets = torch.cat(all_datasets, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    model.eval()
    with torch.no_grad():
        output_i = model(all_datasets)
        
    output_i = output_i.squeeze()
    
    bce_loss = F.cross_entropy(output_i, all_labels)

    total_prior_loss = 0.0

    for param, gamma in zip(model.parameters(), gamma.parameters()):
        total_prior_loss += torch.sum((param - gamma).pow(2)) / (2*variance)
        
    weight = torch.exp(-bce_loss) * torch.exp(-total_prior_loss)

    return weight.item()
