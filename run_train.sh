#!/bin/bash
#SBATCH --job-name=art_resnet
#SBATCH --output=results/.slurm/%x-%j.out
#SBATCH --error=results/.slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=24:00:00

mkdir -p results/.slurm

python train.py