#!/bin/bash
#SBATCH --job-name=eval_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Exécution & Évaluation SLURM Job - Pipeline Hiérarchique (AB vs CD -> A vs B / C vs D) ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

echo "--- 1. Extraction des features si besoin ---"
python3 -u training.py --mode dump_features --backbone resnet50 --annotations_csv "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv" --image_root "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"

echo "--- 2. Entraînement & Évaluation Pipeline Hiérarchique (ResNet50) ---"
python3 -u three_stage_pipeline.py --backbone resnet50 --mode train

echo "--- 3. Entraînement & Évaluation Pipeline Hiérarchique (ViT) ---"
python3 -u three_stage_pipeline.py --backbone vit --mode train
