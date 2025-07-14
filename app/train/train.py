import copy
import random

import torch
from torch import Tensor

from app.const import N
from app.models import BaseRLModel
from app.policies import BasePolicy, EpsilonGreedyPolicy
from app.simulator import Simulator
from app.train.buffer import Experience, ReplayBuffer
from app.train.eval import eval_one_episode
from app.train.reward import calc_reward
from app.types.config import TrainConfig


def train(
    model: BaseRLModel,
    train_config: TrainConfig,
) -> None:
    fixed_model = copy.deepcopy(model)
    buffer = ReplayBuffer(capacity=train_config.replay_buffer_capacity)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)

    rocks = [[False] * N for _ in range(N)]
    for _ in range(int(N / 2)):
        y = random.randint(0, N - 1)
        x = random.randint(0, N - 1)
        rocks[y][x] = True

    for episode in range(1, train_config.num_episodes + 1):
        print(f"episode: {episode}")
        train_one_episode(
            model,
            fixed_model,
            train_config,
            buffer,
            criterion,
            optimizer,
            rocks,
        )

        if episode % 100 == 0:
            save_dir = train_config.result_dir.joinpath(
                f"episode_{str(episode).zfill(4)}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir.joinpath("episode.gif")
            print(f"save_path: {save_path}")
            eval_one_episode(
                model,
                train_config,
                rocks,
                save_path,
            )


def train_one_episode(
    model: BaseRLModel,
    fixed_model: BaseRLModel,
    train_config: TrainConfig,
    buffer: ReplayBuffer,
    criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    rocks: list[list[bool]],
) -> None:
    simulator = Simulator(rocks)
    policy = EpsilonGreedyPolicy(epsilon=0.10)

    while simulator.n_remain_points > 0:
        train_one_turn(
            model,
            fixed_model,
            train_config,
            buffer,
            simulator,
            policy,
            criterion,
            optimizer,
        )


def train_one_turn(
    model: BaseRLModel,
    fixed_model: BaseRLModel,
    train_config: TrainConfig,
    buffer: ReplayBuffer,
    simulator: Simulator,
    policy: BasePolicy,
    criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Optimizer,
) -> None:
    device = torch.device(train_config.device)
    rocks, probs = simulator.get_tensors(device)
    q_values = model.forward(rocks, probs)
    assert q_values.size() == (N, N)

    action_point = policy.sample(q_values, rocks)
    simulator.update(action_point)
    # simulator.dump()

    # バッファに経験を追加する
    next_rocks, next_probs = simulator.get_tensors(device)
    experience = Experience(
        rocks=rocks,
        probs=probs,
        action=torch.tensor([action_point.y, action_point.x], device=device),
        reward=torch.tensor(calc_reward(simulator.probs), device=device),
        next_rocks=next_rocks,
        next_probs=next_probs,
        done=torch.tensor(simulator.is_finished(), device=device, dtype=torch.bool),
    )
    buffer.push(experience)

    # バッファが十分に大きくなったら学習する
    if len(buffer) < train_config.batch_size * 3:
        return

    experiences = buffer.sample(batch_size=train_config.batch_size)
    batch_rocks = torch.stack([t["rocks"] for t in experiences])
    batch_probs = torch.stack([t["probs"] for t in experiences])
    batch_actions = torch.stack([t["action"] for t in experiences])  # (batch_size, 2)
    batch_rewards = torch.stack([t["reward"] for t in experiences])
    batch_next_rocks = torch.stack([t["next_rocks"] for t in experiences])
    batch_next_probs = torch.stack([t["next_probs"] for t in experiences])
    batch_done = torch.stack([t["done"] for t in experiences])

    # q_valuesから対応するQ値を取得
    q_values = model.forward(batch_rocks, batch_probs)  # (batch_size, N, N)
    batch_y = batch_actions[:, 0]
    batch_x = batch_actions[:, 1]
    ids = torch.arange(batch_y.size(0), device=device)
    q_values = q_values[ids, batch_y, batch_x]

    # q_values_nextを計算
    q_values_next = fixed_model.forward(batch_next_rocks, batch_next_probs)
    q_values_next = (
        q_values_next.masked_fill(batch_next_rocks, -float("inf"))
        .view(-1, N * N)
        .max(dim=-1)
        .values
    )  # (batch_size,)
    q_values_next = q_values_next.masked_fill(batch_done, 0.0)

    td_target = (batch_rewards + train_config.gamma * q_values_next).detach()
    loss: Tensor = criterion(td_target, q_values)

    """
    print()
    print(f"loss: {loss.item()}")
    print(f"q_value: {q_values.mean().item()}")
    print(f"q_value_next: {q_values_next.mean().item()}")
    print(f"reward: {batch_rewards.mean().item()}")
    print(f"td_target: {td_target.mean().item()}")
    print()
    """

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # fixed_model を model に近づける
    for param, fixed_param in zip(
        model.parameters(), fixed_model.parameters(), strict=True
    ):
        fixed_param.data = param.data * train_config.tau + fixed_param.data * (
            1.0 - train_config.tau
        )


if __name__ == "__main__":
    from app.models import SimpleNNModel, UNetModel
    from app.types.config import ModelConfig, TrainConfig

    model_name = "unet"

    model_config = ModelConfig()
    train_config = TrainConfig()

    model: BaseRLModel
    if model_name == "simple_nn":
        model = SimpleNNModel(ModelConfig())
    elif model_name == "unet":
        model = UNetModel(channels=(N, N * 2, N * 4))
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    train(model, train_config)
