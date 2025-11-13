import torch
import torch.nn as nn

class MultimodalFFN(nn.Module):
    def __init__(self, input_dim_drugs, input_dim_dx, input_dim_physio):
        """
        Initializes the multi-modal feed-forward network.

        Args:
            input_dim_drugs (int): Number of features for the medication data.
            input_dim_dx (int): Number of features for the diagnosis data.
            input_dim_physio (int): Number of features for the physiology data.
        """
        super().__init__()
        self.input_dim_drugs = input_dim_drugs
        self.input_dim_dx = input_dim_dx
        self.input_dim_physio = input_dim_physio
        

        self.FF_meds = nn.Sequential(
            nn.Linear(self.input_dim_drugs, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        

        self.FF_dx = nn.Sequential(
            nn.Linear(self.input_dim_dx, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        

        self.FF_physio = nn.Sequential(
            nn.Linear(self.input_dim_physio, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        

        self.FF_multihead = nn.Sequential(
            nn.Linear(15, 15), 
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2) 
        )

    def forward(self, x_drugs, x_dx, x_physio):
        """
        Forward pass of the model.

        Args:
            x_drugs (torch.Tensor): Input tensor for medication data.
            x_dx (torch.Tensor): Input tensor for diagnosis data.
            x_physio (torch.Tensor): Input tensor for physiology data.

        Returns:
            torch.Tensor: The output logits (size [batch_size, 2]).
        """

        meds = self.FF_meds(x_drugs)
        dx = self.FF_dx(x_dx)
        physio = self.FF_physio(x_physio)
        
        x_concat = torch.cat((meds, dx, physio), dim=1)

        logits = self.FF_multihead(x_concat)
        
        return logits