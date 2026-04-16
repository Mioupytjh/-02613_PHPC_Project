#!/bin/bash
#BSUB -q hpc
#BSUB -J wallheat_static
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 01:00
#BSUB -oo task5_out.txt
#BSUB -eo task5_err.txt

python task5_static_parallel.py 100 4 > task5_workers4.txt