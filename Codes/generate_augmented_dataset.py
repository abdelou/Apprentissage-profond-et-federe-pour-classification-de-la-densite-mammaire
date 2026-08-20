#!/usr/bin/env python3
# Script : generate_augmented_dataset.py
import argparse
import os
import pandas as pd
import numpy as np

from training import DataAugmentationManager

# Targets d'entraînement (propres à ce projet)
TRAIN_TARGETS = {
  'DENSITY A': 1800,
  'DENSITY B': 2200,
  'DENSITY C': 5000,
  'DENSITY D': 2200
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--annotations_csv', type=str, required=True, help='CSV d\'annotations original')
  parser.add_argument('--image_root', type=str, required=True, help='Racine des images')
  parser.add_argument('--out_csv', type=str, default='annotations_augmented_target4000.csv', help='Chemin de sortie du CSV augmenté')
  args = parser.parse_args()

  if not os.path.exists(args.annotations_csv):
    raise FileNotFoundError(f"Annotations CSV introuvable: {args.annotations_csv}")

  dam = DataAugmentationManager(args.image_root, args.annotations_csv)

  print('[generate_augmented_dataset] Génération initiale des augmentations ciblées...')
  augmented_df = dam.create_augmented_dataset_for_finetuning(target_samples_per_class=5000)

  print('[generate_augmented_dataset] Réajustement des splits (Train & Test)...')
  
  rows = []

  # 1. Split TEST : distribution naturelle, non retouchée. On garde uniquement les
  # images réelles (is_augmented == False) du split 'test' d'origine, sans les
  # dupliquer ni les rééquilibrer par classe — l'évaluation doit refléter la
  # vraie distribution du dataset, pas une distribution artificielle.
  test_df = augmented_df[(augmented_df['split'] == 'test') & (augmented_df['is_augmented'] == False)].copy()
  rows.append(test_df)
  print("Distribution du test (naturelle, non retouchée):")
  print(test_df.groupby('breast_density').size())

  # 2. Équilibrage du split TRAINING (1800 A, 2200 B, 5000 C, 2200 D)
  for density, target in TRAIN_TARGETS.items():
    subset = augmented_df[(augmented_df['breast_density'] == density) & (augmented_df['split'] == 'training')].copy()
    if len(subset) >= target:
      sampled = subset.sample(n=target, random_state=42)
    else:
      sampled = subset.sample(n=target, replace=True, random_state=42)
    rows.append(sampled)

  final_df = pd.concat(rows, ignore_index=True)

  # Sauvegarde
  out_dir = os.path.dirname(args.out_csv)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

  final_df.to_csv(args.out_csv, index=False)
  print(f"[generate_augmented_dataset] CSV augmenté sauvegardé avec succès: {args.out_csv}")
  print("Distribution finale par split et par densité:")
  print(final_df.groupby(['split', 'breast_density']).size())


if __name__ == '__main__':
  main()
