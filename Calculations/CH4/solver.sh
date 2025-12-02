#!/bin/bash
#SBATCH --job-name=CH4
#SBATCH --partition=west
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda deactivate
conda activate pynta_fairchem_dev 
python /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/NH3/solver.py