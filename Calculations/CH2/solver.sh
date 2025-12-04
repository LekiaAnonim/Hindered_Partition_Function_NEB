#!/bin/bash
#SBATCH --job-name=CH2_calc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --output=log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=log/%j.err                  # where to store error messages

#source ~/.bashrc
#conda deactivate
#conda activate pynta_fairchem_dev 
python /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/CH2/solver.py