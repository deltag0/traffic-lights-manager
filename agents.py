import os
import sys
import random
import time
from datetime import datetime, timedelta
from collections import defaultdict


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


from constants import DEVICE, DATE_FORMAT, DIR_PATH, EVAL_STEPS
sys.path.insert(0, os.path.join(DIR_PATH, "models", "DQN"))  # allow imports from DQN
sys.path.insert(0, os.path.join(DIR_PATH, "models", "PPO"))  # allow imports from PPO
from dqn import DQN
from ppo import PPO
from experience import Saved_Memories
from envs import Four_way, FourxFour, TwoxTwo


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
    """
    PPO model has 2 training modes, training mode and evaluating mode

    initializing arguments:

    intersection_type: str that can either be 4_way, FourxFour, TwoxTwo (for now)

    training mode updates the weights of the model using the the default algorithms in PPO,
    adapted for the continuing task nature of the task

    evaluating mode can be used to compare results between the trained model and the standard_cycle agent
    """
    def __init__(self, intersection_type: str):
        self.model = "PPO"
        self.writer = SummaryWriter()
        self.loss_fn = torch.nn.MSELoss()
        self.start = False

        # getting hyperparameters from .yml as a dictionary
        with open("hyperparameters.yml", 'r') as f:
            hyperparameters = yaml.safe_load(f)
            self.save_steps = hyperparameters[self.model]["save_steps"]
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

    def run(self, is_training: bool = True) -> list[float] | None:
        """
        main function to run the PPO for testing/training
        """

        # log messages of training
        if is_training:
            start_time = datetime.now()
            last_update = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Starting training..."
            print(log_message)
            with open(self.log_file, 'w') as f:
                f.write(log_message + '\n')

        intersection = TwoxTwo(display=False)

        env = intersection.env

        update_num = 0
        simulation_steps = 0  # used for choosing a new action every 10 steps
        steps = 0

        mean_waiting_time = []

        state_dict = env.reset()
        self.keys = list(state_dict.keys())

        # num_intersections is the number of intersection PER meeting point
        num_intersections = len(self.keys)
        action_dict = {}

        for key in self.keys:
            action_dict[key] = 0

        input_layers = len(state_dict[str(key)])
        output_layers = env.action_space.n

        self.policy = PPO(input_layers, output_layers, intersection.lanes, intersection.max_occupancy)

        if not is_training:
            self.policy.load_state_dict(torch.load(self.model_file))

        # initialize the critic network
        if is_training:
            self.batch_size = self.update_steps // self.batches

            critic = PPO(input_layers, 1, intersection.lanes, intersection.max_occupancy)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
            optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

            loss_history = []
            accumulated_steps = 0

            # initialize data for collecting information from steps
            accumulated_rewards = []
            rewards_buffer = []
            state_buffer = []
            action_buffer = []
            prob_buffer = []
            value_buffer = []

        # main loop
        for i in count(0):
            state_dict = env.reset()

            # state has a shape of (num_intersections, obs_space)
            state = self.dict_to_tensor(state_dict)

            # action has shape (num_intersections,)
            curr_action = self.dict_to_tensor(action_dict)

            terminated = False

            if is_training:
                rewards_accumulated = 0.0
                accumulated_steps = 0

            self.start = False

            info = defaultdict(lambda: -1)

            # continuing task loop
            while not terminated:
                if not is_training and steps == EVAL_STEPS:
                    return mean_waiting_time

                if simulation_steps % 10 == 0:
                    action_dict, log_prob = self.get_action(state.unsqueeze(0), num_intersections)
                    curr_action = self.dict_to_tensor(action_dict)
                    self.start = True

                if is_training:
                    val = critic(state.unsqueeze(0)).squeeze()

                new_state_dict, reward_dict, terminated, info = env.step(action_dict)
                terminated = terminated["__all__"]

                simulation_steps += 5

                # track data
                if is_training and self.start:
                    rewards_accumulated += self.calculate_rewards(reward_dict)

                    # reward, log_prob, val have shape (num_intersections, )
                    rewards_buffer.append(self.dict_to_tensor(reward_dict))
                    prob_buffer.append(log_prob)
                    value_buffer.append(val.flatten().detach())

                    state_buffer.append(state)
                    action_buffer.append(curr_action)

                    accumulated_steps += 1

                mean_waiting_time.append(info['system_mean_waiting_time'])

                steps += 1

                state = self.dict_to_tensor(new_state_dict)

                # check for extremely poor model performance
                # if self.lane_overload(state, output_layers):
                #     # rewards_buffer[-1] -= self.overload_penalty
                #     break

                # trigger update
                if steps == self.update_steps and self.start and is_training:
                    steps = 0

                    update_num += 1

                    # batchify data

                    # state_buffer has shape (num_batches, batch_size, num_intersections, obs_space)
                    state_buffer = torch.stack(state_buffer)
                    state_buffer = state_buffer.view(self.batches,
                                                     self.batch_size,
                                                     num_intersections,
                                                     input_layers)

                    # action_buffer, prob_buffer have shape (num_batches * batch_size, num_intersections)
                    action_buffer = torch.stack(action_buffer)
                    prob_buffer = torch.stack(prob_buffer)

                    # rewards_buffer, value_buffer are lists of "shape" (num_batches, batch_size, num_intersections)
                    rewards_buffer = self.batchify(rewards_buffer)
                    value_buffer = self.batchify(value_buffer)

                    # compute advantages

                    # A_k has shape (num_batches * batch_size, num_intersections)
                    A_k = self.compute_gae(rewards_buffer, value_buffer)

                    # normalize advantage
                    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                    V = []

                    # calculate advantage function batches
                    for state_batch in state_buffer:
                        V.append(critic(state_batch))

                    V = torch.cat(V, dim=0).squeeze()

                    discounted_rewards = A_k + V.detach()

                    # update policy and critic
                    for _ in range(self.iteration_updates):
                        frac = accumulated_steps / self.max_steps
                        # self.lr = max(self.lr * (1.0 - frac), 0.0)  # learning rate decay, formula can be adjusted
                        # optimizer.param_groups[0]["lr"] = self.lr
                        # critic_optimizer.param_groups[0]["lr"] = self.lr

                        V, entropy_bonus, curr_log_prob = self.evaluate(state_buffer, critic, action_buffer)

                        log_ratio = curr_log_prob - prob_buffer
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

                        critic_loss = self.loss_fn(V, discounted_rewards)
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
                        log_message = f" At update {update_num} \n ---- Accumulated reward of: {rewards_accumulated:0.2f} \n ---- Policy loss of: {policy_loss:0.2f} \n ---- Mean waiting time of: {mean_waiting_time[-1]}"
                        print(log_message)

                        with open(self.log_file, 'a') as f:
                            f.write(log_message + '\n')

                        if datetime.now() - last_update > timedelta(seconds=10) and update_num > 10:
                            last_update = datetime.now()
                            self.save_graph(accumulated_rewards, loss_history)
                            self.save_times(mean_waiting_time)

                        accumulated_rewards.append(rewards_accumulated)
                        rewards_accumulated = 0.0

                    if update_num % self.save_steps == 0 and update_num > 0:
                        torch.save(self.policy.state_dict(), self.model_file)

        if not is_training:
            return mean_waiting_time

    def calculate_rewards(self, rew_dict: dict[str: torch.Tensor]) -> float:
        """
        rew_dict: dictionary with intersecionts indexed from 0 to num_intersections having values of the rewards received
        for the current state

        returns: the total reward of the environment
        """
        total = 0

        for reward in list(rew_dict.values()):
            total += reward

        return total

    def dict_to_tensor(self, val_dict: dict[str: int]) -> torch.Tensor:
        """
        val_dict: dictionary with keys and values for the state of each intersection

        returns: a tensor of size (number_of_intersections, observation_space)
        """
        vals = []

        for val in list(val_dict.values()):
            vals.append(val)

        state = torch.tensor(np.array(vals), dtype=torch.float32, device=DEVICE)

        return state

    def save_times(self, waiting_time: list[float]) -> None:
        """
        waiting_time: list of average waiting time of the environment after each timestep
            Shape of (num_timesteps)

        effects: saves graph of average waiting time to steps
        """

        fig = plt.figure(1)
        waiting_time = np.array(waiting_time)
        plt.ylabel('Mean Waiting Time')
        plt.plot(waiting_time)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.waiting_file)
        plt.close(fig)

    def save_graph(self, rewards_per_step: list[float], loss: list[float]) -> None:
        """
        rewards_per_episode: list with rewards per time step of model
            Shape of (num_timesteps)
        loss: list with losses of model per time step
            Shape of (num_timesteps)

        effects: saves an image with 2 graphs
        """
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_step))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_step[max(0, x-99):(x+1)])

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

    def lane_overload(self, state: torch.Tensor, num_actions: int) -> bool:
        """
        state: shape of (num_intersections, obs_space)
        num_actions: number of possible actions of the environment

        returns True for early environment reset, False if model is performing ok
        """

        score = 0
        start = num_actions + 1
        end = (len(state[0]) - start) // 2 + 3

        for intersection in state:
            # start based on structure of observation space
            for i in range(start, end):
                score += intersection[i]

        return score >= self.overload

    # turns buffers into batches with the size specified in the hyperparameters
    def batchify(self, data: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        """
        data: list of shape (num_batches * batch_size)

        returns a list of shape (num_batches, batch_size)
        """

        batch = []

        for i in range(self.batches):
            batch.append(data[i * self.batch_size:(i + 1) * self.batch_size])

        return batch

    # evaluates the states according to the critic network
    def evaluate(self, state_batch: torch.Tensor, critic: torch.nn.Module, action_batch: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
        """
        state_batch: tensor of shape (num_batches, batch_size, num_intersections, obs_space)
        critic: critic network
        action_batch: tensor of shape (num_batches, batch_size, num_intersections)

        returns: tuple of size 3 containing:
            V (advantage values): tensor of shape (num_batches * batch_size, num_intersections)
            entropy_bonus: float
            log_probs: tensor of shape (num_batches * batch_size, num_intersections)

        """

        V = []
        logits = []

        # calculate advantage function batches
        for state in state_batch:
            V.append(critic(state))
            logits.append(self.policy(state))

        V = torch.cat(V, dim=0).squeeze()
        logits = torch.cat(logits, dim=0)

        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()
        entropy_bonus = self.beta * entropy

        log_probs = dist.log_prob(action_batch)

        return V, entropy_bonus, log_probs

    def compute(self, reward_batch: list[list[float]]) -> torch.Tensor:
        """
        reward_batch: list of shape (num_batches, batch_size, num_intersections)

        returns: OUTDATED
        """
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
                      List of dimension (num_batches, batch_size, num_intersections)
        value_batch: list of values (expected reward) for the data collected within update_steps.
                      List of dimension (num_batches, batch_size, num_intersections)

        returns: list of tensors of shape (update_steps) with the advantages
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

        batch_advantages = torch.stack(batch_advantages)

        return batch_advantages

    def create_action_dict(self, size: int, items: torch.Tensor) -> dict[str: int]:
        """
        size: number of intersections of the current environemnt as an integer
        items: tensor of shape (number_of_intersections, 1) for the actions of each intersection

        returns a dictionary format of the actions
        """

        d = {}
        i = 0

        for key in self.keys:
            d[key] = items[i]
            i += 1

        return d

    def get_action(self, state: torch.Tensor, num_intersections: int) -> tuple[dict[str: int], float]:
        """
        state: tensor of shape (number_of_intersections, observation_space)
        num_intersections: number_of_intersections as an integer

        returns: actions for the current state as a dictionary and the log probability of taking that action as a float
        """

        logits = self.policy(state).squeeze(0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return self.create_action_dict(num_intersections, action), log_prob.detach()


class Standard_Cycle():
    """
    Standard_cycle is meant to replicate real life behavior of traffic lights, changing the traffic signal
    after certain intervals
    """

    def __init__(self, intersection_type: str):

        # getting parameters from .yml as a dictionary
        with open("hyperparameters.yml", 'r') as f:
            parameters = yaml.safe_load(f)
            parameters = parameters["Regular"][intersection_type]

        # setting up cycle parameters
        self.test_version = parameters["test_version"]
        self.max_steps = parameters["max_steps"]

        self.waiting_file = os.path.join(DIR_PATH, "models", "CYCLE", "graphs", self.test_version + "_waiting.png")

    def run(self) -> list[float]:
        """
        main function to run the standard cycle for the environment
        """
        last_update = datetime.now()

        intersection = TwoxTwo()

        env = intersection.env

        simulation_steps = 0
        update_num = 0

        state_dict = env.reset()
        self.keys = list(state_dict.keys())

        num_actions = env.action_space.n
        curr_action = {}
        action_idx = 0

        for key in self.keys:
            curr_action[key] = 0

        average_waiting_times = []

        for i in count(0):
            if update_num == EVAL_STEPS:
                return average_waiting_times

            if simulation_steps % 10 == 0:
                curr_action, action_idx = self.find_next_action(action_idx, num_actions)

            simulation_steps += 5

            _, _, _, info = env.step(curr_action)
            average_waiting_times.append(info["system_mean_waiting_time"])

            update_num += 1

            if update_num % 20 == 0:
                print(average_waiting_times[-1])

            if datetime.now() - last_update > timedelta(seconds=10) and update_num > 10:
                last_update = datetime.now()
                self.save_times(average_waiting_times)

        return average_waiting_times

    def find_next_action(self, curr: int, actions: int) -> tuple[dict[str: int], int]:
        """
        curr: the current action as an integer representing the index of the action space
        actions: number of actions in the action space, used to limit the index of the new action

        returns: new action dictionary
        """

        d = {}

        if curr + 1 == actions:
            action_idx = 0
            for key in self.keys:
                d[key] = action_idx
        else:
            action_idx = curr + 1
            for key in self.keys:
                d[key] = action_idx

        return d, action_idx

    def save_times(self, waiting_time: list[float]) -> None:
        """
        waiting_time: list of average waiting time of the environment after each timestep
            Shape of (num_timesteps)

        effects: saves graph of average waiting time to steps
        """

        fig = plt.figure(1)
        waiting_time = np.array(waiting_time)
        plt.ylabel('Mean Waiting Time')
        plt.plot(waiting_time)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.waiting_file)
        plt.close(fig)


if __name__ == "__main__":
    agent = DQN_Agent("4_way")
    agent2 = PPO_Agent("2x2")
    standard = Standard_Cycle("4_way")
    agent2.run()
