#!/bin/bash
#SBATCH --job-name= CH2_calc
#SBATCH --partition=west
#SBATCH --cpus-per-task=16

deactivate
conda deactivate
conda activate mace_dev
python /projects/westgroup/akinyemi.az/Hindered_/Hindered_Partition_Function_NEB/Calculations/NH3/solver.py