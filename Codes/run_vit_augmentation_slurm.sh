#!/bin/bash
#SBATCH --job-name=vit_aug_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

echo "=== Exécution SLURM Job - Pipeline Hiérarchique avec Data Augmentation Rééquilibrée (ViT & ResNet50) ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

echo "--- 1. Génération du CSV d'annotations rééquilibré 4000 images (800 A, 960 B, 1440 C, 800 D) ---"
python3 -u generate_augmented_dataset.py \
    --annotations_csv "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv" \
    --image_root "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0" \
    --out_csv "annotations_augmented_target4000.csv"

echo "--- 2. Extraction des caractéristiques ViT avec Data Augmentation Ciblée ---"
rm -f featuresextracted/vit_*.npy
python3 -u training.py --mode dump_features --backbone vit --use_augmentation --batch_size 16 --epochs 30 \
    --annotations_csv "annotations_augmented_target4000.csv" \
    --image_root "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"

echo "--- 3. Entraînement & Évaluation du Pipeline Hiérarchique ViT ---"
rm -f vit_three_stage_model*.pth resnet50_three_stage_model*.pth
python3 -u three_stage_pipeline.py --backbone vit --mode train --use_augmentation

echo "--- 4. Extraction des caractéristiques ResNet50 avec Data Augmentation Ciblée ---"
rm -f featuresextracted/resnet50_*.npy
python3 -u training.py --mode dump_features --backbone resnet50 --use_augmentation --batch_size 16 --epochs 30 \
    --annotations_csv "annotations_augmented_target4000.csv" \
    --image_root "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"

echo "--- 5. Entraînement & Évaluation du Pipeline Hiérarchique ResNet50 ---"
python3 -u three_stage_pipeline.py --backbone resnet50 --mode train --use_augmentation

echo "Tâches terminées avec succès !"
date
