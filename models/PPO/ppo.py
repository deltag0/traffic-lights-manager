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
        # total intersections
        batches, num_intersections_in_batch, obs_space = x.shape

        # flatten x
        x = x.view(batches * num_intersections_in_batch, obs_space)

        # return a tensor of shape (batch_size, num_intersections, output_layers)
        return self.layer3(F.relu(self.layer2(F.relu(self.layer1(x))))).view(batches, num_intersections_in_batch, -1)
