import random

from torch import Tensor

from app.const import N
from app.policies.base import BasePolicy
from app.policies.utils import epsilon_random, greedy
from app.types.env import Point


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def sample(self, q_values: Tensor, mask: Tensor) -> Point:
        assert q_values.shape == mask.shape == (N, N)

        if random.random() < self.epsilon:
            print("epsilon_random")
            return epsilon_random(mask)
        else:
            print("greedy")
            return greedy(q_values, mask)
