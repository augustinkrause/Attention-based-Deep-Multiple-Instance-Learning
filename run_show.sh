#!/bin/bash
#SBATCH --job-name=show_PML
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/show-%j.out

# run script with apptainer
apptainer run ./environment.sif python -m data.data
