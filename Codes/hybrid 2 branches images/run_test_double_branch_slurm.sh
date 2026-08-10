#!/bin/bash
#SBATCH --job-name=test_double_branch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Évaluation Test Final Modèle Double Branche (CC + MLO 1024D) ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

python3 -u test_hybrid_model.py
