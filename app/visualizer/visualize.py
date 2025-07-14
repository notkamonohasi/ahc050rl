from pathlib import Path

import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from app.const import N


def visualize_one_turn(
    rocks: list[list[bool]], probs: list[list[float]], save_path: Path
) -> None:
    plt.figure(figsize=(N, N))

    # Create a heatmap for probabilities (white to red)
    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    plt.imshow(probs, cmap=cmap, vmin=0, vmax=0.25)

    # Overlay black squares for rocks (where rocks[i][j] is True)
    for i in range(len(rocks)):
        for j in range(len(rocks[i])):
            if rocks[i][j] is True:
                plt.fill(
                    [j - 0.5, j + 0.5, j + 0.5, j - 0.5],
                    [i - 0.5, i - 0.5, i + 0.5, i + 0.5],
                    color="black",
                )

    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


def visualize_one_episode(
    rocks_records: list[list[list[bool]]],
    probs_records: list[list[list[float]]],
    save_path: Path,
) -> None:
    tmp_dir = save_path.parent.joinpath("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_paths: list[Path] = []
    for i, (rocks, probs) in enumerate(zip(rocks_records, probs_records, strict=True)):
        tmp_path = tmp_dir.joinpath(f"{str(i).zfill(3)}.png")
        visualize_one_turn(rocks, probs, tmp_path)
        img_paths.append(tmp_path)

    images = [imageio.imread(img_path) for img_path in img_paths]
    imageio.mimsave(save_path, images, duration=1000, loop=0)  # type: ignore


if __name__ == "__main__":
    import random

    visualize_one_episode(
        [
            [[random.random() < 0.1 for _ in range(N)] for _ in range(N)]
            for _ in range(10)
        ],
        [[[random.random() for _ in range(N)] for _ in range(N)] for _ in range(10)],
        Path("test.gif"),
    )
