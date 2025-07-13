import random
from collections import deque
from typing import TypedDict

from torch import Tensor


class Experience(TypedDict):
    rocks: Tensor  # (N, N)
    probs: Tensor  # (N, N)
    action: Tensor  # (2,)
    reward: Tensor  # ()
    next_rocks: Tensor  # (N, N)
    next_probs: Tensor  # (N, N)
    done: Tensor  # (1,)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
