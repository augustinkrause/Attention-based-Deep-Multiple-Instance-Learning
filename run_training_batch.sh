#!/bin/bash
#SBATCH --job-name=training_test
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/training-ELEPHANT-instance-mean-%j.out

# run script with apptainer
apptainer run --nv ./environment.sif python -m train_apply --dataset "ELEPHANT" --n-epochs 100 --n-test 40 --n-train 160 --weight-decay 0.005 --learning-rate 0.0001 --mil-type "instance_based" --pooling-type "mean"
