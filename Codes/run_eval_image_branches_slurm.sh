#!/bin/bash
#SBATCH --job-name=eval_img_branches
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Évaluation SLURM Job sur GPU - Branches Image Seules (Section 6.2.4.3) ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

python3 -u finetune_image_branch_standalone.py
