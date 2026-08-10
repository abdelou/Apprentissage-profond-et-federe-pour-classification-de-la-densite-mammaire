#!/bin/bash
#SBATCH --job-name=three_stage_pipeline
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

echo "=== Démarrage SLURM Job - Pipeline Hiérarchique (AB vs CD / A vs B / C vs D) ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

# Étape 1 : Extraction des caractéristiques (dump_features)
echo "--- Étape 1 : Extraction des features avec dump_features ---"
python3 -u training.py --mode dump_features --backbone resnet50 --annotations_csv "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv" --image_root "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"

# Étape 2 : Entraînement de la pipeline hiérarchique en 3 étapes (unbuffered -u)
echo "--- Étape 2 : Entraînement de la pipeline hiérarchique ---"
python3 -u three_stage_pipeline.py --backbone resnet50 --mode train

