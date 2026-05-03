#!/bin/bash
#BSUB -q hpc
#BSUB -J wallheat_ref
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -W 01:00
#BSUB -oo task2_out.txt
#BSUB -eo task2_err.txt

python task2_reference_timing.py 10 > results_10.csv 2> timing_10.txt