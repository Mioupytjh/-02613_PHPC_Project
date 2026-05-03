from os.path import join
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
OUT_DIR = "figures_task1"


def load_data(load_dir, bid):
    size = 512
    u = np.zeros((size + 2, size + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    print(f"Number of buildings: {len(building_ids)}")
    print("First 5 IDs:", building_ids[:5])

    bid = building_ids[0]
    u0, interior_mask = load_data(LOAD_DIR, bid)

    print("\nSingle building inspection")
    print("Building ID:", bid)
    print("u0 shape:", u0.shape)
    print("interior_mask shape:", interior_mask.shape)
    print("Unique values in domain:", np.unique(u0))
    print("Interior pixels:", interior_mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(u0[1:-1, 1:-1], cmap="inferno")
    axes[0].set_title(f"ID {bid} - domain")
    axes[0].axis("off")

    axes[1].imshow(interior_mask, cmap="gray")
    axes[1].set_title(f"ID {bid} - interior mask")
    axes[1].axis("off")

    plt.tight_layout()
    single_path = join(OUT_DIR, f"{bid}_single.png")
    plt.savefig(single_path, dpi=200, bbox_inches="tight")
    plt.close()

    sample_ids = building_ids[:4]
    fig, axes = plt.subplots(2, len(sample_ids), figsize=(4 * len(sample_ids), 8))

    for col, bid in enumerate(sample_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)

        axes[0, col].imshow(u0[1:-1, 1:-1], cmap="inferno")
        axes[0, col].set_title(f"ID {bid} - domain")
        axes[0, col].axis("off")

        axes[1, col].imshow(interior_mask, cmap="gray")
        axes[1, col].set_title(f"ID {bid} - mask")
        axes[1, col].axis("off")

    plt.tight_layout()
    multi_path = join(OUT_DIR, "sample_floorplans_horizontal.png")
    plt.savefig(multi_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved figures:")
    print(single_path)
    print(multi_path)


if __name__ == "__main__":
    main()