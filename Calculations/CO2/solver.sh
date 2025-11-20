#!/bin/bash
#SBATCH --job-name=CO2_calc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda deactivate
conda activate mace_dev
python /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/NH3/solver.py