from pathlib import Path
from typing import Final

N: Final[int] = 8
NEIGHBORS: Final[list[tuple[int, int]]] = [(-1, 0), (0, -1), (0, 1), (1, 0)]

ROOT_DIR: Final[Path] = Path(__file__).parent
RESULT_DIR: Final[Path] = ROOT_DIR.parent.joinpath("Result")
