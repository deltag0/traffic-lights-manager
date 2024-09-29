import os
import sys
import random
from datetime import datetime, timedelta


import yaml  # module used to retrieve data from .yaml files
import torch
import numpy as np
from torch import nn
import matplotlib
from matplotlib import pyplot as plt
from typing import Type
from itertools import count


from constants import DEVICE, DATE_FORMAT, DIR_PATH
sys.path.insert(0, DIR_PATH + "\\models\\DQN")  # allow imports from DQN
from dqn import DQN
from experience import Saved_Memories
from envs import Four_way


class DQN_Agent():
    """
    Class for the DQN agent. There are 2 modes, training mode and evaluating mode.

    In training mode, the agent will update the weights of its neural network using
    the epsilon-greedy algorithm, will log its results, plot a reward graph and loss
    graph.

    Evaluating mode can be used to perform tests on the trained agent, and compare.
    """

    # intersection_type is one of: 4_way, TO_DO
    def __init__(self, intersection_type: str):

        # getting hyperparameters from .yml as a dictionary
        with open("hyperparameters.yml", 'r') as f:
            hyperparameters = yaml.safe_load(f)
            hyperparameters = hyperparameters[intersection_type]

        # initialize hyperparameters
        self.lr = hyperparameters["lr"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.min_epsilon = hyperparameters["min_epsilon"]
        self.epsilon = hyperparameters["init_epsilon"]
        self.model_version = hyperparameters["model_version"]
        self.model = hyperparameters["model"]
        self.max_len = hyperparameters["max_len"]
        self.sync_steps = hyperparameters["sync_steps"]
        self.batch_size = hyperparameters["batch_size"]

        # initialize path files
        self.log_file = DIR_PATH + "\\models\\" + self.model + "\\logs\\" + self.model_version + ".txt"
        self.graph_file = DIR_PATH + "\\models\\" + self.model + "\\graphs\\" + self.model_version + ".png"
        self.model_file = DIR_PATH + "\\models\\" + self.model + "\\trained_models\\" + self.model_version + ".pth"

        self.loss_fn = nn.MSELoss()

    def run(self, is_training: bool = True):
        """
        Defines main loop for running the model in training or evaluating mode.

        Running the function with "is_training = False" will load the last
        saved model unless the model file is explicitly changed. Logging will
        be turned off.

        Running in training mode will save the model every every episode in
        which the current reward was greater than the previous greatest.
        """

        if is_training:
            start_time = datetime.now()
            last_update = start_time

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
        loss_history = []

        # initialize all variables needed for training
        if is_training:
            epsilon = self.epsilon_init

            memories = Saved_Memories(self.max_len)

            target_newtwork = DQN(input_layers, output_layers, intersection.lanes, intersection.max_occupancy)
            target_newtwork.load_state_dict(policy.state_dict())

            self.optimizer = torch.optim.Adam(policy.parameters(), self.lr)

            step_count = 0

            best_reward = float('-inf')
            policy.train()
        else:
            policy.load_state_dict(torch.load(self.model_file))
            policy.eval()

        # main episode loop. Up to user to decide when to terminate training/evaluating
        for episode in range(float('inf')):
            state = env.reset()[0]
            terminated = False
            episode_reward = 0.0

            state = torch.tensor(state, dtype=torch.float64, device=DEVICE)

            # steps in episode
            while (not terminated):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.float64, device=DEVICE)
                else:
                    # don't need extra computation
                    with torch.no_grad:
                        action = policy(state.unsqueeze(0)).squeeze().argmax()
                        action = torch.tensor(action, dtype=torch.float64, device=DEVICE)

                new_state, reward, terminated, _, _ = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float64, device=DEVICE)
                reward = torch.tensor(reward, dtype=torch.float64, device=DEVICE)

                # adding observations to memory
                if is_training:
                    memories.add([new_state, reward, action, state, terminated])

                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    increase = (episode_reward - best_reward) / best_reward * 100
                    best_reward = episode_reward

                    torch.save(policy.state_dict(), self.model_file)

                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: new best reward of {best_reward} | Increase of {increase}"
                    print(log_message)
                    with open(self.log_file, "w") as f:
                        f.write(log_message + '\n')

                # update policy, update epsilon, and possibly update target network
                if len(memories) >= self.batch_size:
                    epsilon = max(epsilon * self.epsilon_decay, self.min_epsilon)

                    sample = memories.sample(self.batch_size)

                    if step_count >= self.sync_steps:
                        target_newtwork.load_state_dict(policy.state_dict())
                        step_count = 0

                # save graphs
                if datetime.now() - last_update > timedelta(seconds=10):
                    last_update = datetime.now()
                    self.save_graph(rewards_per_episode, loss_history)

    # saves graph of the rewards and loss per episode
    def save_graph(self, rewards_per_episode, loss):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        plt.subplot(121)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)

        plt.ylabel('Epsilon Decay')
        plt.plot(loss)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


if __name__ == "__main__":
    agent = DQN_Agent("4_way")
