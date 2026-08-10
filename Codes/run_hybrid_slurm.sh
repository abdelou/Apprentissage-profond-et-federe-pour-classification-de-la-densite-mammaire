#!/bin/bash
#SBATCH --job-name=train_hybrid_etape1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

echo "=== Démarrage SLURM Job - Modèle Hybride Étape 1 (Image + Histogramme) ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

# Entraînement du modèle hybride sur GPU (unbuffered -u)
python3 -u hybrid_finetuning.py \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-4 \
    --use_augmentation \
    --data_csv /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv \
    --image_root /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0
