
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import pandas as pd
from hybrid_model import SiameseDoubleBranchClassifier
from hybrid_finetuning import HybridMammographyDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import argparse

CLASS_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
CLASS_NAMES = list(CLASS_MAP.keys())

DEFAULT_WEIGHTS_PATH = "featuresfinetuned_weights/hybrid_model_best_efficientnet_b0_2branches.pth"
DEFAULT_ANNOTATIONS_CSV = '/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/breast-level_annotations.csv'
DEFAULT_IMAGE_ROOT = '/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0'

parser = argparse.ArgumentParser(description="Évaluation du modèle Siamese à branche partagée (EfficientNet-B0, CC+MLO)")
parser.add_argument('--weights_path', default=DEFAULT_WEIGHTS_PATH, help="Chemin vers les poids du modèle hybride 2 branches")
parser.add_argument('--annotations_csv', default=DEFAULT_ANNOTATIONS_CSV, help="Chemin vers le CSV d'annotations")
parser.add_argument('--image_root_cc', default=DEFAULT_IMAGE_ROOT, help="Chemin vers le dossier des images CC")
parser.add_argument('--image_root_mlo', default=DEFAULT_IMAGE_ROOT, help="Chemin vers le dossier des images MLO")
parser.add_argument('--batch_size', type=int, default=8, help="Taille du batch")
args = parser.parse_args()

BEST_WEIGHTS_PATH = args.weights_path
ANNOTATIONS_CSV = args.annotations_csv
IMAGE_ROOT_CC = args.image_root_cc
IMAGE_ROOT_MLO = args.image_root_mlo
BATCH_SIZE = args.batch_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPHES_DIR = "graphes"
os.makedirs(GRAPHES_DIR, exist_ok=True)

def save_plot(plt, filename, graphes_dir=GRAPHES_DIR):
    filepath = os.path.join(graphes_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[INFO] Graphique sauvegardé: {filepath}")
    plt.close()

def save_text_report(report, filename, graphes_dir=GRAPHES_DIR):
    filepath = os.path.join(graphes_dir, filename)
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"[INFO] Rapport texte sauvegardé: {filepath}")

# === CHARGEMENT DES DONNÉES ===
# Réutilise HybridMammographyDataset (hybrid_finetuning.py), qui apparie correctement
# les vues CC/MLO d'un même sein par (study_id, laterality) — voir hybrid_finetuning.py.
df = pd.read_csv(ANNOTATIONS_CSV)
df_test = df[df['split'] == 'test'].reset_index(drop=True)
test_dataset = HybridMammographyDataset(
    df_test, IMAGE_ROOT_CC, IMAGE_ROOT_MLO,
    label_map=CLASS_MAP, use_augmentation=False, split='test'
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === CHARGEMENT DU MODÈLE ===
model = SiameseDoubleBranchClassifier(
    backbone='efficientnet_b0',
    input_channels=1,
    image_feature_dim=512,
    num_classes=4,
    dropout=0.3,
    pretrained=False,
)
model.load_finetuned_weights(BEST_WEIGHTS_PATH, device=DEVICE)
model.to(DEVICE)
model.eval()

# === ÉVALUATION ===
all_preds = []
all_labels = []

with torch.no_grad():
    for mlo_images, cc_images, labels in test_loader:
        mlo_images = mlo_images.to(DEVICE)
        cc_images = cc_images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(mlo_images, cc_images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
print("=== Résultats sur le set de test ===")
print("la taille d'une sortie du modele est :", outputs.shape)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Rapport de classification avec les 4 classes garanties (A, B, C, D)
report = classification_report(all_labels, all_preds, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, zero_division=0)
print("=== Résultats sur le set de test ===")
print(report)
save_text_report(report, "hybrid_model_siamese_efficientnet_classification_report.txt")

# Matrice de confusion 4x4 garantie
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
print("Matrice de confusion (4x4) :")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.title('Matrice de confusion - Modèle Siamese EfficientNet-B0 (4 classes)')
save_plot(plt, "hybrid_model_siamese_efficientnet_confusion_matrix.png")