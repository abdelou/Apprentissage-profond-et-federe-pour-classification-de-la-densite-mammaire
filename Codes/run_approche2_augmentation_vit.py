#!/usr/bin/env python3
"""
Idem que run_approche2_augmentation_resnet50.py mais pour backbone ViT (ou autre CVT/ViT).
"""
import argparse
import os
import torch
import pandas as pd

from models import FeatureDumper, FeatureExtractor
from three_stage_pipeline import ThreeStagePipeline


def generate_augmented_csv(annotations_csv, image_root, out_csv):
    cmd = f"python3 generate_augmented_dataset.py --annotations_csv {annotations_csv} --image_root {image_root} --out_csv {out_csv}"
    print('[run_vit] Génération CSV augmenté: ' + cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError('Erreur lors de la génération du CSV augmenté')


def extract_features_from_csv(backbone, augmented_csv, image_root, device, out_features_prefix):
    print(f"[extract_features] Extraction des features pour backbone={backbone} -> prefix={out_features_prefix}")
    df = pd.read_csv(augmented_csv)
    # Charger l'extracteur
    fm = FeatureExtractor(backbone, pretrained=True)
    fm.to(device)
    dumper = FeatureDumper(fm, device)

    # Dump features train et test
    train_feat_path = f"{out_features_prefix}_features_train.npy"
    train_labels_path = f"{out_features_prefix}_labels_train.npy"
    test_feat_path = f"{out_features_prefix}_features_test.npy"
    test_labels_path = f"{out_features_prefix}_labels_test.npy"

    dumper.dump_features(df, image_root, train_feat_path, train_labels_path, label_map=None, split='training')
    dumper.dump_features(df, image_root, test_feat_path, test_labels_path, label_map=None, split='test')

    return {
        'train_features': train_feat_path,
        'train_labels': train_labels_path,
        'test_features': test_feat_path,
        'test_labels': test_labels_path
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_csv', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--out_aug_csv', type=str, default='DDSM/annotations_augmented_target4000.csv')
    parser.add_argument('--backbone', type=str, default='vit')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) Générer CSV augmenté
    generate_augmented_csv(args.annotations_csv, args.image_root, args.out_aug_csv)

    # 2) Extraire features (ViT)
    out_prefix = f'featuresextracted/{args.backbone}_augmented'
    feature_paths = extract_features_from_csv(args.backbone, args.out_aug_csv, args.image_root, device, out_prefix)

    # 3) Lancer la pipeline
    pipeline = ThreeStagePipeline(args.backbone, device, feature_paths)
    pipeline.train_all_models()


if __name__ == '__main__':
    main()
