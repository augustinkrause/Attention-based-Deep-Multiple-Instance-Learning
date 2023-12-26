#!/bin/bash
#SBATCH --job-name=training_test
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/cv/training-ELEPHANT-instance-mean-%j.out

# run script with apptainer
apptainer run --nv ./environment.sif python -m train_apply --dataset "ELEPHANT" \
--n-epochs 1 10 100 \
--weight-decay 0 0.0005 0.005 0.05 \
--learning-rate 0.0001 0.001 0.01 0.1\
--mil-type "instance_based" \
--pooling-type "mean" \
--momentum 0 0.09 0.9 \
--optimizer "Adam" "SGD"
