import os
import sys
import random
import time
from datetime import datetime, timedelta


import yaml  # module used to retrieve data from .yaml files
import torch
import numpy as np
import matplotlib
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter  # using tensorboard for easier visualization
from matplotlib import pyplot as plt
from typing import Type
from itertools import count  # used for infinite loop


from constants import DEVICE, DATE_FORMAT, DIR_PATH
sys.path.insert(0, os.path.join(DIR_PATH, "models", "DQN"))  # allow imports from DQN
sys.path.insert(0, os.path.join(DIR_PATH, "models", "PPO"))  # allow imports from PPO
from dqn import DQN
from ppo import PPO
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
        self.writer = SummaryWriter()
        self.model = "DQN"

        # getting hyperparameters from .yml as a dictionary
        with open("hyperparameters.yml", 'r') as f:
            hyperparameters = yaml.safe_load(f)
            hyperparameters = hyperparameters[self.model][intersection_type]

        # initialize hyperparameters
        self.lr = hyperparameters["lr"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.min_epsilon = hyperparameters["min_epsilon"]
        self.epsilon = hyperparameters["init_epsilon"]
        self.model_version = hyperparameters["model_version"]
        self.max_len = hyperparameters["max_len"]
        self.sync_steps = hyperparameters["sync_steps"]
        self.batch_size = hyperparameters["batch_size"]

        # initialize path files
        self.log_file = DIR_PATH + "\\models\\" + self.model + "\\logs\\" + self.model_version + ".txt"
        self.graph_file = DIR_PATH + "\\models\\" + self.model + "\\graphs\\" + self.model_version + ".png"
        self.model_file = DIR_PATH + "\\models\\" + self.model + "\\trained_models\\" + self.model_version + ".pth"

        self.loss_fn = nn.MSELoss()

    def run(self, is_training: bool = True) -> None:
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

            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

            # steps in episode
            while (not terminated):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.float32, device=DEVICE)
                else:
                    # don't need extra computation
                    with torch.no_grad:
                        action = policy(state.unsqueeze(0)).squeeze().argmax()
                        action = torch.tensor(action, dtype=torch.float32, device=DEVICE)

                new_state, reward, terminated, _, _ = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float32, device=DEVICE)
                reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)

                # adding observations to memory
                if is_training:
                    memories.add([new_state, reward, action, state, terminated])

                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)
            self.writer.add_scalar("Episode Reward", episode_reward, episode)

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

        self.writer.flush()

    def update_policy(samples: list[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]) -> float:
        """
        TO DO

        returns loss
        """
        pass

    # saves graph of the rewards and loss per episode
    def save_graph(self, rewards_per_episode, loss) -> None:
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


class PPO_Agent():
    def __init__(self, intersection_type: str):
        self.model = "PPO"
        self.writer = SummaryWriter()

        # getting hyperparameters from .yml as a dictionary
        with open("hyperparameters.yml", 'r') as f:
            hyperparameters = yaml.safe_load(f)
            hyperparameters = hyperparameters[self.model][intersection_type]

        # initializing hyperparameters
        self.clip = hyperparameters["clip"]
        self.batches = hyperparameters["batches"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.model_version = hyperparameters["model_version"]
        self.episode_time = hyperparameters["episode_time"]

        # initialize path files
        self.log_file = os.path.join(DIR_PATH, "models", self.model, "logs", self.model_version + ".txt")
        self.graph_file = os.path.join(DIR_PATH, "models", self.model, "graphs", self.model_version + ".png")
        self.model_file = os.path.join(DIR_PATH, "models", self.model, "trained_models", self.model_version + ".pth")

    def run(self, is_training: bool = True) -> None:

        # log messages of training
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

        self.cov_var = torch.full(size=(output_layers, ), fill_value=0.5)

        self.policy = PPO(input_layers, output_layers, intersection.lanes, intersection.max_occupancy)

        # initialize the critic network
        if is_training:
            critic = PPO(input_layers, 1, intersection.lanes, intersection.max_occupancy)

            loss_history = []

        # for loop for every batch round (i.e episodes = batch_size * i)
        for i in count(0):
            state = env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

            # initialize batch data which will be used to update the policy
            if is_training:
                rewards_per_episode = []
                state_batch = []
                action_batch = []
                prob_batch = []
                batch_len = []

            # for loop for every batch
            for batch in range(self.batches):
                state = env.reset()[0]
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

                if is_training:
                    batch_rewards = []
                    steps = 0

                terminated = False

                episode_reward = 0.0

                # episode loop
                end_time = time.time() + self.episode_time
                while time.time() < end_time:
                    action, log_prob = self.get_action(state.unsqueeze(0))

                    new_state, reward, terminated, _, _ = env.step(action)

                    # collect data from batch to update policy
                    if is_training:
                        batch_rewards.append(reward)
                        prob_batch.append(log_prob)
                        state_batch.append(state)
                        action_batch.append(action)
                        steps += 1
                    episode_reward += reward

                    state = new_state
                    state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

                if is_training:
                    rewards_per_episode.append(batch_rewards)
                    batch_len.append(steps)

            if is_training:
                state_batch = torch.stack(state_batch, dim=0)
                action_batch = torch.tensor(action_batch, dtype=torch.float32, device=DEVICE)
                prob_batch = torch.tensor(prob_batch, dtype=torch.float32, device=DEVICE)
                rewards_per_episode = self.compute(rewards_per_episode)

            # Evaluate 
            V = self.evaluate(state_batch, critic)

    def evaluate(self, state_batch: torch.Tensor, critic: torch.nn.Module) -> torch.Tensor:
        return critic(state_batch.unsqueeze(0)).squeeze()

    # returns a list of shape (batches, episode_timesteps)
    def compute(self, reward_batch: list[list[float]]) -> list[list[float]]:
        batch = []

        for ep_reward in reward_batch[::-1]:
            discounted_reward = 0

            # reverse the rewards from the espisode because future rewards are less important
            for reward in ep_reward[::-1]:
                discounted_reward = reward + discounted_reward * self.discount_factor
                batch.insert(0, discounted_reward)
        batch = torch.tensor(batch, dtype=torch.float, device=DEVICE)

        return batch

    def get_action(self, state: torch.Tensor) -> tuple[int, float]:
        """
        get action from policy in state state TO DO
        """

        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach()


if __name__ == "__main__":
    agent = DQN_Agent("4_way")
    agent2 = PPO_Agent("4_way")
    agent2.run()
