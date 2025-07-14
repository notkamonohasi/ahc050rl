from pathlib import Path

import torch

from app.models import BaseRLModel
from app.policies import GreedyPolicy
from app.simulator import Simulator
from app.types.config import TrainConfig
from app.visualizer.visualize import visualize_one_episode


def eval_one_episode(
    model: BaseRLModel,
    train_config: TrainConfig,
    first_rocks: list[list[bool]],
    save_path: Path,
) -> None:
    print(first_rocks)
    simulator = Simulator(first_rocks)
    policy = GreedyPolicy()
    device = torch.device(train_config.device)

    while not simulator.is_finished():
        rocks, probs = simulator.get_tensors(device)
        q_values = model.forward(rocks, probs)
        simulator.update(policy.sample(q_values, rocks))

        print("=" * 100)
        print("Q values")
        print(q_values)
        print()
        print("rocks")
        print(simulator.rocks)
        print()
        print("probs")
        print(simulator.probs)
        print()
        print("score")
        print(simulator.score)
        print()
        print("is_finished")
        print(simulator.is_finished())
        simulator.dump()
        print("=" * 100)
        print()

    visualize_one_episode(simulator.rocks_records, simulator.probs_records, save_path)
