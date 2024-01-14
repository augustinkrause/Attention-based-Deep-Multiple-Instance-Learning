#!/bin/bash
#SBATCH --job-name="MIL-training-cv"
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output="logs/cv/MIL-training-%j.out"

# run script with apptainer
echo $1 $2 $3
echo ""
apptainer run --nv ./environment.sif python -m train_apply --dataset $1 \
--n-epochs 1 10 100 \
--weight-decay 0 0.0005 0.005 0.05 \
--learning-rate 0.0001 0.001 0.01 0.1 \
--mil-type $2 \
--pooling-type $3 \
--momentum 0 0.09 0.9 \
--optimizer "Adam" "SGD"
