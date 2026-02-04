"""
GRU Built from scratch without using nn.gru
"""
import math
from typing import Optional, Tuple
from torch.nn.modules.module import Module

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """
    Single GRU cell: h' = (1-z)*h + z*tilde_h
    z = sigmoid(W_z @ [x, h])
    r = sigmoid(W_r @ [x, h])
    tilde_h = tanh(W @ [x, r*h])
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
            elif "W_h" in name and p.dim() == 2:
                nn.init.orthogonal_(p[..., : self.hidden_size])
                nn.init.xavier_uniform_(p[..., self.hidden_size :])
            else:
                nn.init.xavier_uniform_(p)

    def forward(self,
                x: torch.Tensor,
                h: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # x: (B, input_size), h: (B, hidden_size)
        if h is None:
            h = x.new_zeros(x.size(0), self.hidden_size)
        xh = torch.cat([x, h], dim = 1)
        z = torch.sigmoid(self.W_z(xh))
        r = torch.sigmoid(self.W_r(xh))
        tilde_h = torch.tanh(self.W_h(torch.cat([x, r*h], dim = 1)))
        return (1-z) * h + z * tilde_h 

class GRULayer(nn.Module):
    """
    Single layer GRU over sequence, built from GRUCell
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self,
                x: torch.Tensor,
                h_0: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, input_size)
        B, T, _ = x.shape
        out = []
        h = h_0
        for t in range(T):
            h = self.cell(x[:, t], h)
            out.append(h)
        return torch.stack(out, dim = 1), h # (B, T, H), (B, H)
    
class GRU(nn.Module):
    """
    Multi layer GRU from scratch (stacked GRULayers)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.layers.append(GRULayer(in_dim, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                h_0: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        #X: (B, T, input_size)
        h_all = []
        h = h_0
        for i, layer in enumerate[Module](self.layers):
            x, h = layer(x, h)
            h_all.append(h)
            if i < self.num_layers - 1:
                x = self.dropout(x)
        
        return x, torch.stack(h_all, dim=0)