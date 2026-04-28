#!/bin/bash
#BSUB -q gpua100
#BSUB -J task10_cupy_profile
#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -o task10_%J.out
#BSUB -e task10_%J.err

module load cuda
module load python3

nsys profile \
  -o task10_optimized_cupy_profile \
  --force-overwrite=true \
  python task10_cupy_optimized.py 10