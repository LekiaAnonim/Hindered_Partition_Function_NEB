#!/bin/bash
#SBATCH --job-name=OH_calc
#SBATCH --partition=short
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda deactivate
conda activate pynta_fairchem_dev 
python /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/NH3/solver.py
