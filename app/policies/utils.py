import torch
from torch import Tensor

from app.const import N
from app.types.env import Point


def greedy(q_values: Tensor, mask: Tensor) -> Point:
    assert q_values.shape == mask.shape == (N, N)
    assert (~mask).any()
    masked_q_values = q_values.masked_fill(mask, -float("inf"))
    idx = torch.argmax(masked_q_values).item()
    i, j = divmod(idx, N)
    return Point(y=i, x=j)


def epsilon_random(mask: Tensor) -> Point:
    assert mask.shape == (N, N)
    assert (~mask).any()
    false_indices = torch.where(~mask.view(-1))[0]
    idx = false_indices[torch.randint(false_indices.size(0), (1,))].item()
    i, j = divmod(idx, N)
    return Point(y=i, x=j)
