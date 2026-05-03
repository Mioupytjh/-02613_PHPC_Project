from os.path import join
import sys
import time
import numpy as np
import cupy as cp

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"


def load_data(load_dir, bid):
    size = 512
    u = np.zeros((size + 2, size + 2), dtype=np.float32)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior_mask


def jacobi_cupy_optimized(u_cpu, mask_cpu, max_iter):
    u = cp.asarray(u_cpu)
    mask = cp.asarray(mask_cpu)

    for _ in range(max_iter):
        u_core = u[1:-1, 1:-1]

        u_new = 0.25 * (
            u[1:-1, :-2]
            + u[1:-1, 2:]
            + u[:-2, 1:-1]
            + u[2:, 1:-1]
        )

        u[1:-1, 1:-1] = cp.where(mask, u_new, u_core)

    return u


def summary_stats(u_cpu, mask_cpu):
    u_interior = u_cpu[1:-1, 1:-1][mask_cpu]
    return {
        "mean_temp": u_interior.mean(),
        "std_temp": u_interior.std(),
        "pct_above_18": np.sum(u_interior > 18) / u_interior.size * 100,
        "pct_below_15": np.sum(u_interior < 15) / u_interior.size * 100,
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

    # Warm-up
    u0, mask0 = load_data(LOAD_DIR, building_ids[0])
    jacobi_cupy_optimized(u0, mask0, 1)
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()

    results = []
    for bid in building_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        u_gpu = jacobi_cupy_optimized(u0, mask, max_iter)
        cp.cuda.Stream.null.synchronize()

        u = cp.asnumpy(u_gpu)
        stats = summary_stats(u, mask)
        results.append((bid, stats))

    elapsed = time.perf_counter() - start

    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id," + ",".join(stat_keys))

    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))

    print(f"\nProcessed {N} floorplans in {elapsed:.2f} s", file=sys.stderr)
    print(f"Average per floorplan: {elapsed / N:.2f} s", file=sys.stderr)
    print(
        f"Estimated time for all 4571 floorplans: {(elapsed / N) * 4571 / 3600:.2f} hours",
        file=sys.stderr,
    )