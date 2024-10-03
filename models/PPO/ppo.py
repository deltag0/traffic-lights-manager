import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class PPO(nn.Module):
    """
    TO DO
    """

    def __init__(self, input_layers: int, output_layers: int, lanes: int, max_lane_occupancy: int):
        super(PPO, self).__init__()
        hidden_layer = lanes * max_lane_occupancy

        self.layer1 = nn.Linear(input_layers, hidden_layer)
        self.layer2 = nn.Linear(hidden_layer, hidden_layer)
        self.layer3 = nn.Linear(hidden_layer, output_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        return self.layer3(F.relu(self.layer2(F.relu(self.layer1(x)))))
