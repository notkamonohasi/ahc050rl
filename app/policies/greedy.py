from torch import Tensor

from app.policies.base import BasePolicy
from app.policies.utils import greedy
from app.types.env import Point


class GreedyPolicy(BasePolicy):
    def __init__(self) -> None:
        pass

    def sample(self, q_values: Tensor, mask: Tensor) -> Point:
        return greedy(q_values, mask)
