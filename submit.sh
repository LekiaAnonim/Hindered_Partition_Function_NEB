#!/bin/bash
#SBATCH --job-name= Submit_Jobs
#SBATCH --partition=west
#SBATCH --cpus-per-task=16

sbatch Calculations/CH2/solver.sh
sbatch Calculations/CH4/solver.sh
sbatch Calculations/CH3/solver.sh
sbatch Calculations/CO/solver.sh
sbatch Calculations/CO2/solver.sh
sbatch Calculations/OH/solver.sh
sbatch Calculations/NH3/solver.sh