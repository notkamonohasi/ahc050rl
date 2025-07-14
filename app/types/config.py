import datetime
from pathlib import Path

import torch
from pydantic import BaseModel

from app.const import RESULT_DIR


class TrainConfig(BaseModel):
    num_episodes: int = 3000
    num_steps: int = 100
    learning_rate: float = 1e-3
    replay_buffer_capacity: int = 10000
    batch_size: int = 3
    gamma: float = 0.9999
    tau: float = 1e-3
    epsilon: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    result_dir: Path = RESULT_DIR.joinpath(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def __post_init__(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)


class ModelConfig(BaseModel):
    hidden_size: int = 256
    num_layers: int = 5
