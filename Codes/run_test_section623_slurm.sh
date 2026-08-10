#!/bin/bash
#SBATCH --job-name=test_sec623
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Évaluation SLURM Job Section 6.2.3 (ResNet50 vs ViT) sur GPU ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

python3 -u test_section623.py
