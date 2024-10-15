import torch
import torch.nn as nn

class MAPLoss(nn.Module):
    def __init__(self, loss_fn, gamma, variance):
        super(MAPLoss, self).__init__()
        
        self.gamma = gamma  
        self.ce_loss = loss_fn
        self.variance = variance + 1e-8  # Ensure non-zero to avoid division by zero

    def forward(self, outputs, targets, model_parameters):
        """
            Calculate the combined loss.

            :param outputs: The model's outputs.
            :param targets: The true targets.
            :param model_parameters: An iterable of model parameters.
            :return: The combined BCE and Gaussian prior loss.
        """
        bce_loss = self.ce_loss(outputs, targets)
        
        total_prior_loss = 0.0

        for param, gamma in zip(model_parameters.parameters(), self.gamma.parameters()):
            # Compute the prior loss as the squared difference between model parameters and gamma
            total_prior_loss += torch.sum((param - gamma).pow(2)) / (2*self.variance)
              
        # The total loss is the sum of BCE loss and the Gaussian prior loss
        total_loss = bce_loss + total_prior_loss
        
        return total_loss