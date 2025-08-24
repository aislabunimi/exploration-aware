from typing import Any

import torch.nn as nn
import torch

class MLPHead(nn.Module):
    '''
    Simple MLP configurable with the sizes of hidden dims and the dropout rate
    '''
    def __init__(self, input_size: int, out_size: int, config: dict[str, Any]):
        super().__init__()
        
        # check that the config has all the required keys
        required_keys = ['hidden_dims', 'dropout_rate']

        assert all(key in config for key in required_keys), f"Config needs all of the following keys: {required_keys}"
        
        layers = nn.ModuleList()
        
        for hidden_dim in config['hidden_dims']:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())

            if (rate := config['dropout_rate'] > 0.0):
                layers.append(nn.Dropout(p = rate))
            
            input_size = hidden_dim
        
        layers.append(nn.Linear(input_size, out_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    