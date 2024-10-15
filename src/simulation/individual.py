import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from ..models import MLP, CNNModel
from datasets import loader

class ModelFactory:
    @staticmethod
    def create_model(is_synthetic, input_shape=None, num_classes=31):
        if is_synthetic:
            return MLP(input_shape)
        else:
            return CNNModel(num_classes)

class LossStrategy:
    def get_loss_function(self):
        raise NotImplementedError

class SyntheticLoss(LossStrategy):
    def get_loss_function(self):
        return nn.BCELoss()

class ImageLoss(LossStrategy):
    def get_loss_function(self):
        return nn.CrossEntropyLoss()

class OptimizerStrategy:
    def get_optimizer(self, net):
        raise NotImplementedError

class SyntheticOptimizer(OptimizerStrategy):
    def get_optimizer(self, net):
        return optim.Adam(net.parameters(), lr=0.0001)

class ImageOptimizer(OptimizerStrategy):
    def get_optimizer(self, net):
        return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def get_data_loaders(is_synthetic, client_id):
    if is_synthetic:
        return loader.load_individual_dataset(client_id)
    else:
        return loader.load_image_data(client_id)

def train_val(is_synthetic=True, epochs=100, num_classes=31, num_clients=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for client_id in range(num_clients):
        # Data Loaders
        trainloader, testloader = get_data_loaders(is_synthetic, client_id)

        # Model Factory
        input_shape = trainloader.dataset[0][0].shape[0] if is_synthetic else None
        net = ModelFactory.create_model(is_synthetic, input_shape, num_classes)
        net = net.to(device)

        # Loss and Optimizer Strategy
        loss_strategy = SyntheticLoss() if is_synthetic else ImageLoss()
        optimizer_strategy = SyntheticOptimizer() if is_synthetic else ImageOptimizer()

        criterion = loss_strategy.get_loss_function()
        optimizer = optimizer_strategy.get_optimizer(net)

        for epoch in range(epochs):
            train_one_epoch(net, trainloader, criterion, optimizer, device, epoch, epochs, is_synthetic)

            if epoch % 4 == 0:
                evaluate_model(net, testloader, criterion, client_id, is_synthetic, device)

def train_one_epoch(net, trainloader, criterion, optimizer, device, epoch, epochs, is_synthetic):
    net.train()
    running_loss = 0.0
    correct_predictions = 0.0
    total_predictions = 0.0

    for batch_data, batch_label in trainloader:
        batch_data, batch_label = batch_data.to(device), batch_label.to(device)

        optimizer.zero_grad()
        outputs = net(batch_data).squeeze()
        loss = criterion(outputs, batch_label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if is_synthetic:
            predicted = outputs.round()
        else:
            _, predicted = torch.max(outputs, dim=1)

        correct_predictions += (predicted == batch_label).sum().item()
        total_predictions += batch_label.size(0)

    epoch_loss = round(running_loss / len(trainloader.dataset), 4)
    epoch_acc = round(correct_predictions / total_predictions, 4)

    print(f"Epoch {epoch + 1}/{epochs}: train-loss: {epoch_loss}, train-accuracy: {epoch_acc}")

def evaluate_model(net, testloader, criterion, client_id, is_synthetic, device):
    net.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    val_true_labels = []
    val_predicted_labels = []

    with torch.no_grad():
        for batch_data, batch_label in testloader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)

            outputs = net(batch_data).squeeze()
            val_loss += criterion(outputs, batch_label).item()

            if is_synthetic:
                predicted = outputs.round()
            else:
                _, predicted = torch.max(outputs, dim=1)

            correct_predictions += (predicted == batch_label).sum().item()
            total_predictions += batch_label.size(0)
            val_true_labels.extend(batch_label.cpu().numpy())
            val_predicted_labels.extend(predicted.cpu().numpy())

    val_epoch_loss = round(val_loss / len(testloader.dataset), 4)

    if is_synthetic:
        val_epoch_acc = round(correct_predictions / total_predictions, 4)
        val_balanced_acc = balanced_accuracy_score(val_true_labels, val_predicted_labels)
        log_benchmark_train_metrics_to_csv(client_id, val_epoch_loss, val_balanced_acc)
    else:
        val_epoch_acc = accuracy_score(val_true_labels, val_predicted_labels)
        log_benchmark_train_metrics_to_csv(client_id, val_epoch_loss, val_epoch_acc)

def log_benchmark_train_metrics_to_csv(cid, val_loss, val_acc):
    filename = './results/metrics_individual.csv'
    headers = ['Client_ID', 'Validation_Loss', 'Validation_Accuracy']
    record = [int(cid)+1, val_loss, val_acc]

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerow(record)
    else:
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(record)
