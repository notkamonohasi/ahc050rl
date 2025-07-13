def calc_reward(probs: list[list[float]]) -> float:
    return sum([sum(row) for row in probs])
