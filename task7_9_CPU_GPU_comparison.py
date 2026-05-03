# compare_cpu_gpu.py
import subprocess

print("CPU:")
subprocess.run(["python", "task7_numba_cpu.py", "10"])

print("\nGPU (CuPy):")
subprocess.run(["python", "task9_cupy.py", "10"])