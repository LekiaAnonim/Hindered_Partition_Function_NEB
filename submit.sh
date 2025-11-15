#!/bin/bash
#SBATCH --job-name= Submit_Jobs
#SBATCH --partition=west
#SBATCH --cpus-per-task=16

sbatch /projects/westgroup/lekia.p/NEB/Calculations/CH2/solver.sh
sbatch /projects/westgroup/lekia.p/NEB/Calculations/CH4/solver.sh
sbatch /projects/westgroup/lekia.p/NEB/Calculations/CH3/solver.sh
sbatch /projects/westgroup/lekia.p/NEB/Calculations/CO2/solver.sh
sbatch /projects/westgroup/lekia.p/NEB/Calculations/OH/solver.sh
sbatch /projects/westgroup/lekia.p/NEB/Calculations/NH3/solver.sh