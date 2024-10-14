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
        self.update_steps = hyperparameters["update_steps"]
        self.overload = hyperparameters["overload"]
        self.batches = hyperparameters["batches"]
        self.iteration_updates = hyperparameters["iteration_updates"]
        self.lr = hyperparameters["lr"]
        self.beta = hyperparameters["beta"]
        self.overload_penalty = hyperparameters["overload_penalty"]
        self.max_steps = hyperparameters["max_steps"]
        self.max_grad_norm = hyperparameters["max_grad_norm"]
        self.lam = hyperparameters["lambda"]

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

        state_dict = env.reset()

        # num_intersections is the number of intersection PER meeting point
        num_intersections = len(state_dict.keys())

        if num_intersections == 1:
            action_dict = {'t': 0}
            key = 't'
        else:
            action_dict = {}
            for i in range(num_intersections):
                action_dict[str(i)] = 0
                key = 0

        input_layers = len(state_dict[str(key)])
        output_layers = env.action_space.n

        self.cov_var = torch.full(size=(output_layers, ), fill_value=0.5)

        self.policy = PPO(input_layers, output_layers, intersection.lanes, intersection.max_occupancy)

        # initialize the critic network
        if is_training:
            self.batch_size = self.update_steps // self.batches

            critic = PPO(input_layers, 1, intersection.lanes, intersection.max_occupancy)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
            optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

            loss_history = []
            steps = 0
            accumulated_steps = 0
            simulation_steps = 0  # used for choosing a new action every 10 steps

            # initialize data for collecting information from steps
            mean_waiting_time = []
            accumulated_rewards = []
            rewards_buffer = []
            state_buffer = []
            action_buffer = []
            prob_buffer = []
            value_buffer = []

        # main loop
        for i in count(0):
            state_dict = env.reset()
            state = self.dict_to_tensor(state_dict)
            curr_action = self.dict_to_tensor(action_dict)

            terminated = False

            if is_training:
                rewards_accumulated = 0.0
                accumulated_steps = 0

            self.start = False

            info = defaultdict(lambda: -1)

            # continuing task loop
            while not terminated:
                if simulation_steps % 10 == 0:
                    action_dict, log_prob = self.get_action(state.unsqueeze(0), num_intersections)
                    curr_action = self.dict_to_tensor(action_dict)
                    self.start = True

                if is_training:
                    val = critic(state.unsqueeze(0))

                new_state_dict, reward, terminated, info = env.step(action_dict)
                terminated = terminated["__all__"]

                simulation_steps += 5

                # track data
                if is_training and self.start:
                    rewards_accumulated += self.calculate_rewards(reward)

                    rewards_buffer.append(self.dict_to_tensor(reward))
                    state_buffer.append(state)
                    action_buffer.append(curr_action)
                    prob_buffer.append(log_prob)
                    value_buffer.append(val.flatten()) # MIGHT need to change this
                    mean_waiting_time.append(info["system_mean_waiting_time"])

                    steps += 1
                    accumulated_steps += 1

                state = self.dict_to_tensor(new_state_dict)

                # check for extremely poor model performance
                # if self.lane_overload(state):
                #     # rewards_buffer[-1] -= self.overload_penalty
                #     break

                # trigger update
                if steps == self.update_steps and self.start:
                    steps = 0

                    update_num += 1

                    # batchify data
                    state_buffer = torch.stack(state_buffer)
                    state_buffer = state_buffer.view(self.batches,
                                                     self.batch_size * num_intersections,
                                                     input_layers)

                    action_buffer = torch.stack(action_buffer)
                    action_buffer = action_buffer.view(self.batches,
                                                       self.batch_size * num_intersections,
                                                       1)

                    prob_buffer = torch.tensor(prob_buffer, dtype=torch.float32, device=DEVICE)
                    rewards_buffer = self.batchify(rewards_buffer)
                    # rewards_buffer = self.compute(rewards_buffer)
                    value_buffer = self.batchify(value_buffer)

                    # V, _, _ = self.evaluate(state_buffer, critic, action_buffer)

                    # A_k = rewards_buffer - V.detach()

                    # compute advantages
                    A_k = self.compute_gae(rewards_buffer, value_buffer)

                    # normalize advantage
                    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                    # # calculate advantage function
                    V = critic(state_buffer).flatten()

                    rewards_buffer = A_k + V.detach()

                    # update policy and critic
                    for _ in range(self.iteration_updates):
                        frac = accumulated_steps / self.max_steps
                        # self.lr = max(self.lr * (1.0 - frac), 0.0)  # learning rate decay, formula can be adjusted
                        # optimizer.param_groups[0]["lr"] = self.lr
                        # critic_optimizer.param_groups[0]["lr"] = self.lr

                        V, entropy_bonus, curr_log_prob = self.evaluate(state_buffer, critic, action_buffer.squeeze(-1))

                        log_ratio = curr_log_prob.flatten() - prob_buffer
                        ratios = torch.exp(log_ratio)

                        surr1 = ratios * A_k
                        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                        policy_loss = (-torch.min(surr1, surr2)).mean()

                        total_loss = policy_loss - entropy_bonus

                        self.policy.zero_grad()
                        total_loss.backward(retain_graph=True)
                        # nn.utils.clip_grad_norm(self.policy.parameters(), self.max_grad_norm)
                        optimizer.step()
                        loss_history.append(policy_loss.item())

                        critic_loss = self.loss_fn(V, rewards_buffer)
                        critic.zero_grad()  # critic estimates average reward we get in a state
                        critic_loss.backward()
                        nn.utils.clip_grad_norm(critic.parameters(), self.max_grad_norm)
                        critic_optimizer.step()

                    rewards_buffer = []
                    state_buffer = []
                    action_buffer = []
                    prob_buffer = []
                    value_buffer = []
                    if is_training:
                        print(f"At update {update_num}, Accumulated reward of: {rewards_accumulated:0.2f} \n ---- Policy loss of {policy_loss:0.2f}")

                        if datetime.now() - last_update > timedelta(seconds=10) and update_num > 10:
                            last_update = datetime.now()
                            self.save_graph(accumulated_rewards, loss_history)
                            self.save_times(mean_waiting_time)

                        accumulated_rewards.append(rewards_accumulated)
                        rewards_accumulated = 0.0

    def calculate_rewards(self, rew_dict: dict[str: torch.Tensor]) -> float:
        total = 0

        for reward in list(rew_dict.values()):
            total += reward

        return total

    # 
    def dict_to_tensor(self, val_dict: dict[str: int]) -> torch.Tensor:
        """
        val_dict: dictionary with keys and values for the state of each intersection

        returns a tensor of size (number_of_intersections, observation_space)
        """
        state = []

        for intersection_state in list(val_dict.values()):
            state.append(intersection_state)

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

        return state

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
        V = critic(state_batch)
        logits = self.policy(state_batch)

        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()
        entropy_bonus = self.beta * entropy

        log_probs = dist.log_prob(action_batch)

        return V, entropy_bonus, log_probs

    # returns a list of shape (episode_timesteps)
    def compute(self, reward_batch: list[list[float]]) -> list[torch.Tensor]:
        batch = []

        for ep_reward in reward_batch[::-1]:
            discounted_reward = 0

            # reverse the rewards from the espisode because future rewards are less important
            for reward in ep_reward[::-1]:
                discounted_reward = reward + discounted_reward * self.discount_factor
                batch.insert(0, discounted_reward)

        batch = torch.tensor(batch, dtype=torch.float, device=DEVICE)

        return batch

    def compute_gae(self, reward_batch: list[list[float]], value_batch: list[list[float]]) -> list[torch.Tensor]:
        """
        reward_batch: list of rewards for the data collected within update_steps.
                      Has dimension (batches, update_steps / batches)
        value_batch: list of values (expected reward) for the data collected within update_steps.
                      Has dimension (batches, update_steps / batches)

        returns a list of tensors of shape (update_steps) with the advantages
        """
        batch_advantages = []
        for seq_rewards, seq_vals in zip(reward_batch, value_batch):
            advantages = []
            last_advantage = 0

            seq_len = len(seq_rewards)
            # compute advantages for every batch
            for i in reversed(range(seq_len)):
                if i + 1 < seq_len:
                    delta = seq_rewards[i] + self.discount_factor * seq_vals[i + 1] - seq_vals[i]
                else:
                    delta = seq_rewards[i] - seq_vals[i]

                advantage = delta + self.discount_factor * self.lam * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)  # reorder advantages in batch in original order

            batch_advantages.extend(advantages)  # flattened and transformed reward_batch into advantages

        batch_advantages = torch.tensor(batch_advantages, dtype=torch.float, device=DEVICE)

        return batch_advantages

    def create_action_dict(self, size: int, items: torch.Tensor) -> dict[str: int]:
        """
        size: number of intersections of the current environemnt as an integer
        items: tensor of shape (number_of_intersections, 1) for the actions of each intersection

        returns a dictionary format of the actions
        """
        if size == 1:
            return {'t': items.item()}
        else:
            d = {}

            for i in range(size):
                d[str(i)] = items[i].item()

            return d

    def get_action(self, state: torch.Tensor, num_intersections: int) -> tuple[dict[str: int], float]:
        """
        state: tensor of shape (number_of_intersections, observation_space)
        num_intersections: number_of_intersections as an integer

        returns the actions for the current state as a dictionary and the log probability of taking that action as a float
        """

        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return self.create_action_dict(num_intersections, action), log_prob.detach()


if __name__ == "__main__":
    agent = DQN_Agent("4_way")
    agent2 = PPO_Agent("4_way")
    agent2.run()
