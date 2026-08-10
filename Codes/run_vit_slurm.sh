#!/bin/bash
#SBATCH --job-name=train_vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Démarrage SLURM Job pour ViT ==="
date
hostname

# Exécuter l'entraînement sur GPU
python3 training.py \
    --mode finetune \
    --backbone vit \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --use_augmentation \
    --annotations_csv /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv \
    --image_root /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0
