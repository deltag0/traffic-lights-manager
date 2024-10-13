import tkinter as tk
import numpy as np
import math
import copy
import random
import gymnasium as gym
from sympy import FF_python
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from envs import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.00
EPS_DECAY = 12000

TAU = 0.005
LR = 1e-3

intersection = Four_way(False)
env = intersection.env
n_actions = env.action_space.n

state, info = env.reset()
n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
