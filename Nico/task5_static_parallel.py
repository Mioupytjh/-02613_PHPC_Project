from os.path import join
import sys
import time
import multiprocessing as mp
import numpy as np

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"


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


def process_building(bid):
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi(u0, interior_mask, max_iter=20_000, atol=1e-4)
    stats = summary_stats(u, interior_mask)
    return bid, stats


if __name__ == "__main__":
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 3:
        N = 20
        n_workers = 2
    else:
        N = int(sys.argv[1])
        n_workers = int(sys.argv[2])

    building_ids = building_ids[:N]

    start = time.perf_counter()

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_building, building_ids)

    elapsed = time.perf_counter() - start

    print(f"Processed {N} floorplans with {n_workers} workers in {elapsed:.2f} s")
    print(f"Average per floorplan: {elapsed / N:.2f} s")

    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id," + ",".join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))