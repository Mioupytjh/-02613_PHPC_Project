from os.path import join
import sys
import time
import numpy as np
from numba import njit

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"


def load_data(load_dir, bid):
    size = 512
    u = np.zeros((size + 2, size + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior_mask


@njit
def jacobi_numba(u, interior_mask, max_iter, atol):
    u_old = u.copy()
    u_new = u.copy()

    h, w = u_old.shape

    for _ in range(max_iter):
        delta = 0.0

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if interior_mask[i - 1, j - 1]:
                    new_val = 0.25 * (
                        u_old[i, j - 1] +
                        u_old[i, j + 1] +
                        u_old[i - 1, j] +
                        u_old[i + 1, j]
                    )
                    u_new[i, j] = new_val

                    diff = abs(u_old[i, j] - new_val)
                    if diff > delta:
                        delta = diff
                else:
                    u_new[i, j] = u_old[i, j]

        if delta < atol:
            break

        tmp = u_old
        u_old = u_new
        u_new = tmp

    return u_old


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


if __name__ == "__main__":
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 10
    else:
        N = int(sys.argv[1])

    building_ids = building_ids[:N]

    max_iter = 20_000
    abs_tol = 1e-4

    # warm-up compilation
    u0, mask0 = load_data(LOAD_DIR, building_ids[0])
    jacobi_numba(u0, mask0, 1, abs_tol)

    start = time.perf_counter()

    all_stats = []
    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi_numba(u0, interior_mask, max_iter, abs_tol)
        stats = summary_stats(u, interior_mask)
        all_stats.append((bid, stats))

    elapsed = time.perf_counter() - start

    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id," + ",".join(stat_keys))
    for bid, stats in all_stats:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))

    print(f"\nProcessed {N} floorplans in {elapsed:.2f} s", file=sys.stderr)
    print(f"Average per floorplan: {elapsed / N:.2f} s", file=sys.stderr)
    print(
        f"Estimated time for all 4571 floorplans: {(elapsed / N) * 4571 / 3600:.2f} hours",
        file=sys.stderr,
    )