#!/bin/bash
#BSUB -q hpc
#BSUB -J wallheat_dynamic4
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 01:00
#BSUB -oo task6_out.txt
#BSUB -eo task6_err.txt

python task6_dynamic_parallel.py 100 16 > task6_workers16.txt