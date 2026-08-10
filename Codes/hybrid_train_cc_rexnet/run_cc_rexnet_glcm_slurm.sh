#!/bin/bash
#SBATCH --job-name=cc_rexnet_glcm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Étape 1 : Entraînement & Test Modèle Hybride ReXNet150 + GLCM sur Vues CC ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

python3 -u hybrid_finetuning_glcm.py CC cnn
