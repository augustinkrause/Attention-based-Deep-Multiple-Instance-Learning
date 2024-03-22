#!/bin/bash
#SBATCH --job-name=run_prediction
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=run_prediction.log

# run script with apptainer
apptainer run --nv ../../pml.sif python -m run_prediction
