#!/bin/bash
#SBATCH --job-name=NH3_calc
#SBATCH --partition=west
#SBATCH --cpus-per-task=2


conda deactivate
conda activate pynta_fairchem
python /projects/westgroup/lekia.p/NEB/Adsorbates/NH3/script2.py