from os.path import join
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
OUT_DIR = "figures_task3"


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    sample_ids = building_ids[:4]

    fig, axes = plt.subplots(3, len(sample_ids), figsize=(4 * len(sample_ids), 10))

    for j, bid in enumerate(sample_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi(u0, interior_mask, max_iter=20_000, atol=1e-4)

        axes[0, j].imshow(u0[1:-1, 1:-1], cmap="inferno")
        axes[0, j].set_title(f"ID {bid} - initial")
        axes[0, j].axis("off")

        axes[1, j].imshow(interior_mask, cmap="gray")
        axes[1, j].set_title(f"ID {bid} - mask")
        axes[1, j].axis("off")

        axes[2, j].imshow(u[1:-1, 1:-1], cmap="inferno")
        axes[2, j].set_title(f"ID {bid} - result")
        axes[2, j].axis("off")

    plt.tight_layout()
    out_path = join(OUT_DIR, "task3_simulation_results_horizontal.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()