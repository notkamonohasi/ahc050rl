from abc import ABC, abstractmethod

from torch import Tensor

from app.exceptions.utils import NotImplementedError
from app.types.env import Point


class BasePolicy(ABC):
    @abstractmethod
    def sample(self, q_values: Tensor, mask: Tensor) -> Point:
        raise NotImplementedError
