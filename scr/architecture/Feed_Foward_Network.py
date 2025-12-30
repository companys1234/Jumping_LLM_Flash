import torch
import torch.nn as nn
import torch.nn.functional as F


class Feed_Forward_Network(nn.Module):
    def __init__(self, dim, hidden_dim, activation, dropout=None):
        super().__init__()
        if dropout == True:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),  # Удваиваем для split на SwigLU
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),
                activation,
                nn.Linear(hidden_dim,dim)
            )

    def forward(self, x):
        return self.net(x)

