#!/bin/bash
#SBATCH --job-name=finetune_backbones
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --mail-user=abdelouahada@umons.ac.be
#SBATCH --mail-type=ALL

# Fine-tune les backbones ViT et ResNet50 de bout en bout (script séparé de
# run_vit_augmentation_slurm.sh : le fine-tuning est une étape lourde et isolée).
# Une fois ce job terminé, featuresfinetuned_weights/{backbone}_finetuned_best.pth
# existe pour vit et resnet50 ; run_vit_augmentation_slurm.sh les détectera et les
# utilisera automatiquement (voir FeatureManager.get_feature_extractor) sans aucune
# modification nécessaire de ce script-là.

echo "=== SLURM Job - Fine-tuning des backbones (ViT & ResNet50) ==="
date
hostname

cd "$SLURM_SUBMIT_DIR"

DATA_CSV="/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv"
IMAGE_ROOT="/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"

echo "--- 1. (Re)génération du CSV d'entraînement rééquilibré (test non retouché) ---"
python3 -u generate_augmented_dataset.py \
    --annotations_csv "$DATA_CSV" \
    --image_root "$IMAGE_ROOT" \
    --out_csv "annotations_augmented_target4000.csv"

echo "--- 2. Fine-tuning ViT ---"
python3 -u training.py \
    --mode finetune \
    --backbone vit \
    --epochs 30 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --patience 5 \
    --use_augmentation \
    --annotations_csv "annotations_augmented_target4000.csv" \
    --image_root "$IMAGE_ROOT"

echo "--- 3. Fine-tuning ResNet50 ---"
python3 -u training.py \
    --mode finetune \
    --backbone resnet50 \
    --epochs 30 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --patience 5 \
    --use_augmentation \
    --annotations_csv "annotations_augmented_target4000.csv" \
    --image_root "$IMAGE_ROOT"

echo "Fine-tuning terminé pour vit et resnet50 !"
echo "Poids sauvegardés dans featuresfinetuned_weights/{vit,resnet50}_finetuned_best.pth"
date
