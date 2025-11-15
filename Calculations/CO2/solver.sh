#!/bin/bash
#SBATCH --job-name= CO2_calc
#SBATCH --partition=west
#SBATCH --cpus-per-task=16

deactivate
conda deactivate
conda activate pynta_fairchem
python /projects/westgroup/lekia.p/NEB/Calculations/CO2/solver.py