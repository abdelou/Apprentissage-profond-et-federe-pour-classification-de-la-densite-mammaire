#!/bin/bash
#SBATCH --job-name=train_hybrid_rexnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Reprise SLURM Job pour Hybride RexNet MLO (Époque 2/5 à 5/5) ==="
date
hostname

# Exécuter l'entraînement sur GPU avec reprise depuis le checkpoint
python3 -u hybrid_finetuning.py \
    --epochs 5 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_augmentation \
    --resume_from featuresfinetuned_weights/hybrid_model_checkpoint_cnn.pth \
    --data_csv /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv \
    --image_root /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0
