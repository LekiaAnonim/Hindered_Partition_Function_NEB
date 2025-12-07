#!/bin/bash
#SBATCH --job-name=CH3_calc
#SBATCH --partition=west
#SBATCH --cpus-per-task=8


conda deactivate
conda activate pynta_fairchem
python /projects/westgroup/lekia.p/NEB/Adsorbates/CH3/script.py