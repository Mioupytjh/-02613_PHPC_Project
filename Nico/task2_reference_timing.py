from os.path import join
import time
import numpy as np

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"


def load_data(bid):
    u = np.zeros((514, 514))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, mask


def jacobi(u, mask, max_iter=20_000, atol=1e-4):
    u = u.copy()

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1]
        )
        new_vals = u_new[mask]
        delta = np.abs(u[1:-1, 1:-1][mask] - new_vals).max()
        u[1:-1, 1:-1][mask] = new_vals

        if delta < atol:
            break

    return u


with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
    building_ids = f.read().splitlines()

N = 10
building_ids = building_ids[:N]

start = time.perf_counter()

for bid in building_ids:
    u, mask = load_data(bid)
    jacobi(u, mask)

elapsed = time.perf_counter() - start

print(f"Processed {N} floorplans in {elapsed:.2f} s")
print(f"Average per floorplan: {elapsed / N:.2f} s")
print(f"Estimated time for all 4571 floorplans: {(elapsed / N) * 4571 / 3600:.2f} hours")