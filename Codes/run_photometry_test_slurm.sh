#!/bin/bash
#SBATCH --job-name=dicom_photometry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Évaluation et Diagnostic Photométrie DICOM (MONOCHROME1 vs MONOCHROME2) ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

python3 -u test_dicom_photometry.py
