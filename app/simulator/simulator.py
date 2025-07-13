from copy import deepcopy

import torch
from torch import Tensor

from app.const import NEIGHBORS, N
from app.types.env import Point
from app.utils import calc_prob_sum, is_inside


class Simulator:
    def __init__(self, rocks: list[list[bool]]) -> None:
        self.rocks = rocks
        self.n_remain_points = sum(row.count(False) for row in rocks)
        self.probs = [[1.0 / float(self.n_remain_points)] * N for _ in range(N)]
        self.score = 0.0

        self.probs = self.simulate(None)

    def simulate(self, rock_point: Point | None) -> list[list[float]]:
        """
        point に置いた後の確率を計算する\n
        シミュレーションを行うだけで、状態は変更しない
        """
        next_rocks = deepcopy(self.rocks)
        if rock_point is not None:
            assert is_inside(rock_point) is True
            next_rocks[rock_point.y][rock_point.x] = True

        next_probs = [[0.0] * N for _ in range(N)]

        starts: dict[tuple[int, int], list[Point]] = {
            (0, 1): [Point(y=i, x=0) for i in range(N)],
            (1, 0): [Point(y=0, x=i) for i in range(N)],
            (0, -1): [Point(y=i, x=N - 1) for i in range(N)],
            (-1, 0): [Point(y=N - 1, x=i) for i in range(N)],
        }
        for neighbor in NEIGHBORS:
            for start in starts[neighbor]:
                point = start
                prob = 0.0
                while is_inside(point) is True:
                    next_point = point + neighbor
                    if next_rocks[point.y][point.x] is False:
                        prob += self.get_prob(point) / 4.0
                    if (
                        is_inside(next_point) is False
                        or next_rocks[next_point.y][next_point.x] is True
                    ):
                        next_probs[point.y][point.x] += prob
                        prob = 0.0
                    point = next_point
                assert prob <= 1e-10

        return next_probs

    def update(self, rock_point: Point) -> None:
        """
        rock_point に岩を置く
        """
        print(f"update: {rock_point}")
        assert is_inside(rock_point) is True
        assert self.rocks[rock_point.y][rock_point.x] is False
        assert self.n_remain_points > 0
        self.n_remain_points -= 1
        self.rocks[rock_point.y][rock_point.x] = True
        self.probs = self.simulate(rock_point)
        self.score += calc_prob_sum(self.probs)

    def get_tensors(self, device: torch.device) -> tuple[Tensor, Tensor]:
        return (
            torch.tensor(self.rocks, device=device, dtype=torch.bool),
            torch.tensor(self.probs, device=device),
        )

    def get_prob(self, point: Point) -> float:
        assert is_inside(point) is True
        return self.probs[point.y][point.x]

    def is_finished(self) -> bool:
        return self.n_remain_points == 0

    def dump(self) -> None:
        print()
        remain_prob = calc_prob_sum(self.probs)
        print(f"remain_prob: {remain_prob:.4f}")
        for row in self.rocks:
            for col in row:
                print("#" if col else ".", end=" ")
            print()
        print()
        for row in self.probs:  # type: ignore
            for col in row:
                print(f"{col:.4f}", end=" ")
            print()
        print()


if __name__ == "__main__":
    simulator = Simulator([[False] * N for _ in range(N)])
    simulator.dump()
    simulator.update(Point(y=0, x=0))
    simulator.dump()
    simulator.update(Point(y=1, x=1))
    simulator.dump()
