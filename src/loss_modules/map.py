import torch
import torch.nn as nn
from typing import Iterable, Optional, List

class BasePrior(nn.Module):
    """
    A prior defines a (negative log) probability distribution over the
    local model parameters (model) given the global parameters (gamma).
    """
    def forward(self, model: nn.Module, gamma: nn.Module) -> torch.Tensor:
        """
        Calculates the prior loss.
        
        Args:
            model (nn.Module): The local model (theta).
            gamma (nn.Module): The global model (gamma).

        Returns:
            torch.Tensor: A scalar tensor representing the prior loss.
        """
        raise NotImplementedError


class ICNNPrior(BasePrior):
    """
    Implements the Input Convex Neural Network (ICNN) prior.
    This uses a trained set of ICNNs to define a complex, adaptive prior.
    """
    def __init__(self, cnnet_modules):
        """
        Args:
            cnnet_modules: Either nn.ModuleList or nn.ModuleDict containing ICNN modules
        """
        super().__init__()
        self.cnnet_modules = cnnet_modules
        self.is_module_dict = isinstance(cnnet_modules, nn.ModuleDict)

    @torch.no_grad()
    def _check_shapes_list(self, model: nn.Module, gamma: nn.Module):
        mp = list(model.parameters())
        gp = list(gamma.parameters())
        if len(mp) != len(self.cnnet_modules) or len(gp) != len(mp):
            raise ValueError(
                f"ICNNPrior: mismatch — params={len(mp)}, gamma={len(gp)}, cnns={len(self.cnnet_modules)}"
            )

    def forward(self, model: nn.Module, gamma: nn.Module) -> torch.Tensor:
        """
        Compute prior term: R(θ; μ, ψ) = sum over all params of ICNN(θ_i, μ_i)
        
        Args:
            model: Local model with parameters θ
            gamma: Global model with parameters μ
            
        Returns:
            Prior loss (scalar tensor)
        """
        device = next(model.parameters()).device
        total = torch.zeros((), device=device)
        
        if self.is_module_dict:
            return self._forward_module_dict(model, gamma, device)
        else:
            return self._forward_module_list(model, gamma, device)
    
    def _forward_module_list(self, model: nn.Module, gamma: nn.Module, device: torch.device) -> torch.Tensor:
        """Forward pass for ModuleList (original implementation)."""
        self._check_shapes_list(model, gamma)
        total = torch.zeros((), device=device)
        
        for (p, g), cnnet in zip(zip(model.parameters(), gamma.parameters()), self.cnnet_modules):
            cnnet = cnnet.to(device)
            model_flat = p.view(-1).unsqueeze(0).to(device)
            gamma_flat = g.view(-1).unsqueeze(0).to(device)
            total = total + cnnet(model_flat, gamma_flat).sum()
            
        return total
    
    def _forward_module_dict(self, model: nn.Module, gamma: nn.Module, device: torch.device) -> torch.Tensor:
        """Forward pass for ModuleDict (with sanitized parameter names)."""
        total = torch.zeros((), device=device)
        
       
        model_params = {name.replace('.', '__'): param 
                       for name, param in model.named_parameters()}
        gamma_params = {name.replace('.', '__'): param 
                       for name, param in gamma.named_parameters()}
        
  
        for sanitized_name, cnnet in self.cnnet_modules.items():
            if sanitized_name not in model_params or sanitized_name not in gamma_params:
                continue
            
            cnnet = cnnet.to(device)
            
            local_param = model_params[sanitized_name]
            global_param = gamma_params[sanitized_name]
            

            model_flat = local_param.view(-1).unsqueeze(0).to(device)
            gamma_flat = global_param.view(-1).unsqueeze(0).to(device)
            
            total = total + cnnet(model_flat, gamma_flat).sum()
        
        return total


class FedMAPLoss(nn.Module):
    """
    A generic Maximum A Posteriori (MAP) loss function for federated learning.
    
    This loss is a combination of:
    1. A data likelihood loss (e.g., CrossEntropy, MSE)
    2. A prior loss (e.g., ICNNPrior, L2Prior)
    
    Loss = pred_loss(outputs, targets) + prior(model, gamma)
    """
    def __init__(self, pred_loss: nn.Module, prior: BasePrior, gamma: nn.Module):
        """
        Args:
            pred_loss (nn.Module): The data likelihood loss (e.g., nn.CrossEntropyLoss()).
            prior (BasePrior): The prior loss module (e.g., ICNNPrior()).
            gamma (nn.Module): A reference to the client's global model (gamma).
        """
        super().__init__()
        self.pred_loss = pred_loss
        self.prior = prior
        self.gamma = gamma
        
        self._bound_model: Optional[nn.Module] = None
        
    def bind_model(self, model: nn.Module) -> None:
        """Bind a model to this loss for convenience."""
        self._bound_model = model

    def unbind_model(self) -> None:
        """Unbind the currently bound model."""
        self._bound_model = None

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Calculates the total FedMAP loss.
        
        Args:
            outputs (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The true labels.
            model (nn.Module, optional): The local model (theta) being trained.
                                        If None, uses the bound model.

        Returns:
            torch.Tensor: The total scalar loss.
        """
        if model is None:
            model = self._bound_model
        if model is None:
            raise ValueError(
                "FedMAPLoss.forward requires `model` argument or a bound model via `bind_model()`."
            )
        
        # Data likelihood term
        data_loss = self.pred_loss(outputs, targets)
        
        # Prior term
        prior_loss = self.prior(model, self.gamma)
        
        return data_loss + prior_loss