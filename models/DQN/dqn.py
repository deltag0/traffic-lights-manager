import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    The DQN class is the neural network for the target network and the policy network.
    Layers are... TO DO
    """

    def __init__(self, input_layers: int, output_layers: int, lanes: int, max_lane_occupancy: int):
        hidden_layer = lanes * max_lane_occupancy

        self.layer1 = nn.Linear(input_layers, hidden_layer)
        self.layer2 = nn.Linear(hidden_layer, output_layers)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
