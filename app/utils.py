from app.const import N
from app.types.env import Point


def is_inside(point: Point) -> bool:
    return 0 <= point.x < N and 0 <= point.y < N


def calc_prob_sum(probs: list[list[float]]) -> float:
    return sum(sum(row) for row in probs)
