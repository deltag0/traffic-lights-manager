import os
import random
from datetime import datetime, timedelta


import yaml  # module used to retrieve data from .yaml files
import torch
from torch import nn
import matplotlib
from matplotlib import pyplot as plt
from typing import Type

from constants import DEVICE, DATE_FORMAT, DIR_PATH
from envs import Four_way
from dqn import DQN


class DQN_Agent():
    """
    Class for the DQN agent. There are 2 modes, training mode and evaluating mode.

    In training mode, the agent will update the weights of its neural network using
    the epsilon-greedy algorithm, will log its results, plot a reward graph and loss
    graph.

    Evaluating mode can be used to perform tests on the trained agent, and compare.
    """

    def __init__(self, intersection_type: str):

        # getting hyperparameters from .yml as a dictionary
        with open("hyperparameters.yml", 'r') as f:
            hyperparameters = yaml.safe_load(f)
            hyperparameters = hyperparameters[intersection_type]

        self.lr = hyperparameters["lr"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.min_epsilon = hyperparameters["min_epsilon"]
        self.init_epsilon = hyperparameters["init_epsilon"]
        self.model_version = hyperparameters["model_version"]
        self.model = hyperparameters["model"]

        self.log_file = DIR_PATH + "\\models\\" + self.model + "\\logs\\" + self.model_version + ".txt"
        self.graph_file = DIR_PATH + "\\models\\" + self.model + "\\graphs\\" + self.model_version + ".png"
        self.model_file = DIR_PATH + "\\models\\" + self.model + "\\trained_models\\" + self.model_version + ".pth"

        self.loss_fn = nn.MSELoss()

    def run(self, is_training: bool=True):
        if is_training:
            start_time = datetime.now()

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Starting training..."
            print(log_message)
            with open(self.log_file, 'w') as f:
                f.write(log_message + '\n')

        intersection = Four_way(True)

        env = intersection.env

        state = env.reset()[0]

        input_layers = len(state)
        output_layers = env.action_space.n

        policy = DQN(input_layers, output_layers, intersection.lanes, intersection.max_occupancy)

        rewards_per_episode = []

        if is_training:
            epsilon = self.epsilon_init

            target_newtwork = DQN(input_layers, output_layers, intersection.lanes, intersection.max_occupancy)
            target_newtwork.load_state_dict(policy.state_dict())

            self.optimizer = torch.optim.Adam(policy.parameters(), self.lr)

            step_count = 0

            best_reward = float('-inf')
            policy.train()
        else:
            policy.load_state_dict(torch.load(self.model_file))
            policy.eval()

        while True:
            state = env.reset()[0]
            terminated = False
            episode_reward = 0.0

            state = torch.tensor(state, dtype=torch.float64, device=DEVICE)

            while (not terminated):
                pass
