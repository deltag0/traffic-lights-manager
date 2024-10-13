import os
import sys
import random
import time
from datetime import datetime, timedelta
from collections import defaultdict
from collections import namedtuple


import torch.optim
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


from constants import DEVICE, DATE_FORMAT, DIR_PATH, PHASE_TIME
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
        self.loss_fn = torch.nn.MSELoss()
        self.start = False

        # getting hyperparameters from .yml as a dictionary
        with open("hyperparameters.yml", 'r') as f:
            hyperparameters = yaml.safe_load(f)
            hyperparameters = hyperparameters[self.model][intersection_type]

        # initializing hyperparameters
        self.clip = hyperparameters["clip"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.model_version = hyperparameters["model_version"]
        self.max_steps = hyperparameters["max_steps"]
        self.overload = hyperparameters["overload"]
        self.batches = hyperparameters["batches"]
        self.iteration_updates = hyperparameters["iteration_updates"]
        self.lr = hyperparameters["lr"]

        # initialize path files
        self.log_file = os.path.join(DIR_PATH, "models", self.model, "logs", self.model_version + ".txt")
        self.graph_file = os.path.join(DIR_PATH, "models", self.model, "graphs", self.model_version + ".png")
        self.model_file = os.path.join(DIR_PATH, "models", self.model, "trained_models", self.model_version + ".pth")
        self.waiting_file = os.path.join(DIR_PATH, "models", self.model, "graphs", self.model_version + "_waiting.png")

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

        update_num = 0

        state = env.reset()[0]
        input_layers = len(state)
        output_layers = env.action_space.n

        self.cov_var = torch.full(size=(output_layers, ), fill_value=0.5)

        self.policy = PPO(input_layers, output_layers, intersection.lanes, intersection.max_occupancy)

        # initialize the critic network
        if is_training:
            self.batch_size = self.max_steps // self.batches

            critic = PPO(input_layers, 1, intersection.lanes, intersection.max_occupancy)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
            optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

            loss_history = []
            steps = 0

            mean_waiting_time = []
            accumulated_rewards = []

        # main loop
        for i in count(0):
            state = env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

            # initialize data for collecting information from steps
            if is_training:
                rewards_buffer = []
                state_buffer = []
                action_buffer = []
                prob_buffer = []
                steps = 0

            terminated = False

            rewards_accumulated = 0.0

            curr_action = 0
            info = defaultdict(lambda: -1)

            # continuing task loop
            while not terminated:
                print(state[1])
                if int(info["step"]) % 10 == 0:
                    curr_action, log_prob = self.get_action(state.unsqueeze(0))
                    self.start = True

                new_state, reward, terminated, _, info = env.step(curr_action)

                rewards_accumulated += reward

                # track data
                if is_training and self.start:
                    rewards_buffer.append(reward)
                    state_buffer.append(state)
                    action_buffer.append(curr_action)
                    prob_buffer.append(log_prob)
                    mean_waiting_time.append(info["system_mean_waiting_time"])
                    steps += 1

                if self.lane_overload(new_state):
                    break

                state = new_state
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

                # trigger update
                if steps == self.max_steps and self.start:
                    steps = 0

                    update_num += 1

                    # batchify data
                    state_buffer = torch.stack(state_buffer)
                    action_buffer = torch.tensor(action_buffer, dtype=torch.float32, device=DEVICE)
                    prob_buffer = torch.tensor(prob_buffer, dtype=torch.float32, device=DEVICE)

                    # Advantage function
                    V, _ = self.evaluate(state_buffer, critic, action_buffer)

                    # compute discounted rewards
                    rewards_buffer = self.batchify(rewards_buffer)
                    rewards_buffer = self.compute(rewards_buffer)

                    advantage = rewards_buffer - V.detach()

                    # normalize advantage
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

                    # update policy and critic
                    for _ in range(self.iteration_updates):
                        V, curr_log_prob = self.evaluate(state_buffer, critic, action_buffer)

                        ratios = torch.exp(curr_log_prob - prob_buffer)

                        surr1 = ratios * advantage
                        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage

                        policy_loss = (-torch.min(surr1, surr2)).mean()
                        self.policy.zero_grad()
                        policy_loss.backward(retain_graph=True)
                        optimizer.step()
                        loss_history.append(policy_loss.item())

                        critic_loss = self.loss_fn(V, rewards_buffer)
                        critic.zero_grad()  # critic estimates average reward we get in a state
                        critic_loss.backward()
                        critic_optimizer.step()

                    rewards_buffer = []
                    state_buffer = []
                    action_buffer = []
                    prob_buffer = []

                    print(f"At update {update_num}, Accumulated reward of: {rewards_accumulated:0.2f} \n ---- Policy loss of {policy_loss:0.2f}")

                    accumulated_rewards.append(rewards_accumulated)

                    if datetime.now() - last_update > timedelta(seconds=10) and update_num > 10:
                        last_update = datetime.now()
                        self.save_graph(accumulated_rewards, loss_history)
                        self.save_times(mean_waiting_time)

                    accumulated_rewards.append(rewards_accumulated)
                    rewards_accumulated = 0.0

    def save_times(self, waiting_time: list[float]) -> None:
        fig = plt.figure(1)
        waiting_time = np.array(waiting_time)
        plt.ylabel('Mean Waiting Time')
        plt.plot(waiting_time)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.waiting_file)
        plt.close(fig)

    def save_graph(self, rewards_per_episode: list[float], loss: list[float]) -> None:
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        print(mean_rewards)
        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)

        loss = np.array(loss)
        plt.ylabel('Loss')
        plt.plot(loss)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.graph_file)
        plt.close(fig)

    # returns true if there are too many filled lanes (based off of hyperparameter)
    def lane_overload(self, state: list[float]):
        score = 0
        for i in range(5, 12):
            score += state[i]

        if score >= self.overload:
            return True
        else:
            return False

    # turns buffers into batches with the size specified in the hyperparameters
    def batchify(self, data: list[torch.Tensor]) -> torch.Tensor:
        batch = []

        for i in range(self.batches):
            batch.append(data[i * self.batch_size:(i + 1) * self.batch_size])

        return batch

    # evaluates the states according to the critic network
    def evaluate(self, state_batch: torch.Tensor, critic: torch.nn.Module, action_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        V = critic(state_batch).squeeze()
        logits = self.policy(state_batch)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_batch)

        return V, log_probs

    # returns a list of shape (episode_timesteps)
    def compute(self, reward_batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
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
