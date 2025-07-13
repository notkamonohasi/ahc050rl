import torch
from torch import Tensor, nn

from app.const import N
from app.models.base import BaseRLModel
from app.types.config import ModelConfig


class SimpleNNModel(BaseRLModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.first_layer = nn.Linear(N * N * 2 + 2, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.num_layers)
            ]
        )
        self.last_layer = nn.Linear(config.hidden_size, N * N)

    def forward(self, rocks: Tensor, probs: Tensor) -> Tensor:
        assert rocks.size(dim=-1) == probs.size(dim=-1) == N
        assert rocks.size(dim=-2) == probs.size(dim=-2) == N
        assert rocks.size() == probs.size()

        x: Tensor
        if rocks.dim() == 2:
            n_falses = rocks.view(-1).sum()
            n_remain_probs = probs.view(-1).sum()
            x = torch.cat(
                [
                    rocks.view(-1),
                    probs.view(-1),
                    n_falses.unsqueeze(-1),
                    n_remain_probs.unsqueeze(-1),
                ],
                dim=-1,
            )
        elif rocks.dim() == 3:
            n_falses = rocks.view(-1, N * N).sum(dim=-1)
            n_remain_probs = probs.view(-1, N * N).sum(dim=-1)
            x = torch.cat(
                [
                    rocks.view(-1, N * N),
                    probs.view(-1, N * N),
                    n_falses.unsqueeze(-1),
                    n_remain_probs.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid shape: {rocks.shape} {probs.shape}")

        x = torch.relu(self.first_layer(x))
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.last_layer(x)

        if rocks.dim() == 2:
            assert x.size() == (N * N,)
            x = x.view(N, N)
        elif rocks.dim() == 3:
            assert x.size() == (rocks.size(dim=0), N * N)
            x = x.view(-1, N, N)
        else:
            raise ValueError(f"Invalid shape: {rocks.shape} {probs.shape}")

        return x
