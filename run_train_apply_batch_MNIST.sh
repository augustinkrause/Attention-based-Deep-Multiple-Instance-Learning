#!/bin/bash
#SBATCH --job-name=MNIST_cv
#SBATCH --partition=gpu-7d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/cv/MNIST-training-%j.out

# run script with apptainer
echo "MNIST" $1 $2
echo ""
apptainer run --nv ./environment.sif python -m train_apply --dataset "MNIST" \
--n-epochs 1 10 100 \
--weight-decay 0 0.0005 0.005 0.05 \
--learning-rate 0.0001 0.001 0.01 0.1 \
--mil-type $1 \
--pooling-type $2 \
--momentum 0 0.09 0.9 \
--optimizer "Adam" "SGD"
