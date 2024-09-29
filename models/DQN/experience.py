import torch
from collections import deque
from random import sample


class Saved_Memories():
    """
    class Saved_Memories serves as the experience replay of the model,
    saving previous experiences to be used for updating the policy
    """

    def __init__(self, maxlen):
        self.memories = deque([], maxlen)
        self.len = 0

    # returns a sample from the experiences of size batch_size
    def sample(self, batch_size: int):
        return sample(self.memories, batch_size)

    # adds an experience of the format: [new_state, reward, state, action, terminated]
    def add(self, experience: torch.Tensor):
        self.memories.append(experience)
        self.len += 1

    def __len__(self):
        return self.len
