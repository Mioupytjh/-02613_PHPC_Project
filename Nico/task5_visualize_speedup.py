import matplotlib.pyplot as plt

workers = [1, 2, 4, 8, 16]
times = [14.79, 7.07, 4.33, 2.56, 2.47 ]   # replace with your measured times
speedups = [times[0] / t for t in times]

plt.figure(figsize=(6, 4))
plt.plot(workers, speedups, marker="o")
plt.xlabel("Number of workers")
plt.ylabel("Speed-up")
plt.title("Static scheduling speed-up")
plt.grid(True)
plt.savefig("task5_speedup.png", dpi=200, bbox_inches="tight")
plt.show()