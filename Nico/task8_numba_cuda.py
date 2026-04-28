from os.path import join
import sys
import time
import numpy as np
from numba import cuda

# Reminder: Run as: python task8_numba_cuda.py N (with N being the number of iterations we would like to run, default 10)

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"

def load_data(load_dir, bid):
    size = 512
    u = np.zeros((size + 2, size + 2), dtype=np.float32)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy")).astype(np.float32)
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior_mask


@cuda.jit
def jacobi_kernel(u_old, u_new, interior_mask):
    i, j = cuda.grid(2)

    # i and j refer to coordinates inside the 512x512 mask
    if i < interior_mask.shape[0] and j < interior_mask.shape[1]:
        ui = i + 1
        uj = j + 1

        if interior_mask[i, j]:
            u_new[ui, uj] = 0.25 * (
                u_old[ui, uj - 1]
                + u_old[ui, uj + 1]
                + u_old[ui - 1, uj]
                + u_old[ui + 1, uj]
            )
        else:
            u_new[ui, uj] = u_old[ui, uj]


def jacobi_cuda(u_cpu, interior_mask_cpu, max_iter):
    u_old = cuda.to_device(u_cpu)
    u_new = cuda.to_device(u_cpu)
    interior_mask = cuda.to_device(interior_mask_cpu)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (interior_mask_cpu.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (interior_mask_cpu.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](u_old, u_new, interior_mask)

        tmp = u_old
        u_old = u_new
        u_new = tmp

    cuda.synchronize()

    return u_old.copy_to_host()


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
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

    # Warm-up compilation
    u0, mask0 = load_data(LOAD_DIR, building_ids[0])
    jacobi_cuda(u0, mask0, 1)

    start = time.perf_counter()

    results = []
    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi_cuda(u0, interior_mask, max_iter)
        stats = summary_stats(u, interior_mask)
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