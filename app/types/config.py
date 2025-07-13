import torch
from pydantic import BaseModel


class TrainConfig(BaseModel):
    num_episodes: int = 3000
    num_steps: int = 100
    learning_rate: float = 0.0001
    replay_buffer_capacity: int = 10000
    batch_size: int = 3
    gamma: float = 0.9999
    tau: float = 1e-4
    epsilon: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModelConfig(BaseModel):
    hidden_size: int = 256
    num_layers: int = 5
