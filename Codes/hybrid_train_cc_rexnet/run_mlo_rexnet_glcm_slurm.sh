#!/bin/bash
#SBATCH --job-name=glcm_mlo_rexnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Évaluation Modèle Hybride ReXNet150 + GLCM sur Vues MLO ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

python3 -u hybrid_finetuning_glcm.py MLO cnn
