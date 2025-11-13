# iron_classifier.py
from typing import List
import torch
import torch.nn as nn


def _weights_init_he(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class DenseNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        assert len(hidden_dims) > 0, "At least one hidden layer dimension must be specified."

        layers: List[nn.Module] = []

  
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

       
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))


        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def initialise_weights(self) -> None:
        self.apply(_weights_init_he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DenseClassifier(DenseNetwork):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            input_dim=18,
            output_dim=2,
            hidden_dims=[64, 64],
            dropout=0.3,
            use_batchnorm=True,
        )
        self.output_dim = 2
        self.initialise_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
