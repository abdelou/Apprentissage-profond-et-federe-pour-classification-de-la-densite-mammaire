#!/bin/bash
#SBATCH --job-name=train_double_branch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Démarrage SLURM Job - Modèle Double Branche (5 époques express) ==="
date
hostname

# Se placer dans le dossier depuis lequel la commande sbatch a été lancée
cd "$SLURM_SUBMIT_DIR"

# Exécution de l'entraînement du modèle à double branche avec reprise automatique
python3 -u hybrid_finetuning.py \
    --backbone resnet50 \
    --epochs 5 \
    --batch_size 8 \
    --lr 1e-4 \
    --use_augmentation \
    --resume_from featuresfinetuned_weights/hybrid_model_checkpoint_resnet50.pth \
    --data_csv /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv \
    --image_root_cc /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0 \
    --image_root_mlo /home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0
