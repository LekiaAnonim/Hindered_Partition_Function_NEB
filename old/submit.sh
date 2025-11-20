#!/bin/bash
#SBATCH --job-name=Submit_Jobs
#SBATCH --partition=west
#SBATCH --cpus-per-task=16

sbatch /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/CH2/solver.sh
sbatch /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/CH4/solver.sh
sbatch /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/CH3/solver.sh
sbatch /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/CO/solver.sh
sbatch /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/CO2/solver.sh
sbatch /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/OH/solver.sh
sbatch /projects/westgroup/akinyemi.az/Hindered_Partition_Function_NEB/Calculations/NH3/solver.sh
