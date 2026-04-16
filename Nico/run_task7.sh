#!/bin/bash
#BSUB -q hpc
#BSUB -J wallheat_numba
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 01:00
#BSUB -oo task7_out.txt
#BSUB -eo task7_err.txt

source /dtu/projects/02613_2025/conda/miniconda3/etc/profile.d/conda.sh
conda activate 02613

python task7_numba_cpu.py 10 > task7_results_10.csv 2> task7_timing_10.txt