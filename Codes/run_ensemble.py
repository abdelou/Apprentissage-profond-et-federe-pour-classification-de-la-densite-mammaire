"""
Ensemble (vote souple / fusion tardive) entre deux modeles deja entraines
independamment :
    - Approche 3 : ResNet50 + Histogramme, vues MLO (85% Test Acc, 0% recall A)
    - Approche 2 : ResNet50 fine-tuning direct (74% Test Acc, 95% recall A)

On moyenne leurs probabilites softmax sur le meme sous-ensemble d'images
(vues MLO du test, 2000 images -- Approche 3 ne fonctionne que sur cette
vue), puis on evalue le resultat combine. Aucun reentrainement : les deux
modeles sont charges tels quels depuis leurs checkpoints existants.
"""
import sys
import os

CODES_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CODES_DIR, "hybrid train mlo resnet")
sys.path.insert(0, CODES_DIR)
sys.path.insert(0, A3_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from training import MammographyDataset, CONFIG as CONFIG_A2
from models import FineTunedFeatureExtractor
from hybrid_model import HybridMammographyClassifier
from hybrid_finetuning import HybridMammographyDataset, VINDR_ROOT, CONFIG as CONFIG_A3

# Checkpoints -- chemins absolus pour ne pas dependre du repertoire courant
A3_WEIGHTS = os.path.join(A3_DIR, "featuresfinetuned_weights", "hybrid_model_best_cnn.pth")
A2_WEIGHTS = os.path.join(CODES_DIR, "featuresfinetuned_weights", "resnet50_finetuned_best.pth")

CLASS_NAMES = ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Jeu de test commun : vues MLO uniquement (2000 images), meme
# filtrage que celui utilise par l'Approche 3
# ============================================================
df = pd.read_csv(CONFIG_A3["ANNOTATIONS_CSV"])
df_test_mlo = df[df["view_position"] == "MLO"]
df_test_mlo = df_test_mlo[df_test_mlo["split"] == "test"].reset_index(drop=True)
print(f"[INFO] {len(df_test_mlo)} images de test (vues MLO)")

# ============================================================
# Modele Approche 3 : ResNet50 + Histogramme (MLO)
# ============================================================
print("[INFO] Chargement Approche 3 (ResNet50 + Histogramme, MLO)...")
model_a3 = HybridMammographyClassifier(
    backbone="cnn", input_channels=1, image_feature_dim=512,
    hist_hidden_dims=[128, 64], num_classes=4, dropout=0.3, pretrained=False,
)
model_a3.load_state_dict(torch.load(A3_WEIGHTS, map_location=device))
model_a3.to(device)
model_a3.eval()

dataset_a3 = HybridMammographyDataset(
    df_test_mlo, VINDR_ROOT, label_map=CONFIG_A3["CLASS_MAP"], use_augmentation=False, split="test",
)
loader_a3 = DataLoader(dataset_a3, batch_size=16, shuffle=False, num_workers=0)

# ============================================================
# Modele Approche 2 : ResNet50 fine-tuning direct
# ============================================================
print("[INFO] Chargement Approche 2 (ResNet50 fine-tuning)...")
model_a2 = FineTunedFeatureExtractor("resnet", num_classes=4, pretrained=False)
model_a2.load_finetuned_weights(A2_WEIGHTS, device)
model_a2.to(device)
model_a2.eval()

dataset_a2 = MammographyDataset(
    df_test_mlo, CONFIG_A2["IMAGE_ROOT"], split="test", use_augmentation=False, label_map=CONFIG_A2["CLASS_MAP"],
)
loader_a2 = DataLoader(dataset_a2, batch_size=16, shuffle=False, num_workers=0)

assert len(dataset_a3) == len(dataset_a2), "Les deux jeux de test ne sont pas alignes !"

# ============================================================
# Inference des deux modeles + moyenne des probabilites
# ============================================================
print("[INFO] Inference Approche 3...")
probs_a3, labels_a3 = [], []
with torch.no_grad():
    for images, hists, labels in loader_a3:
        images, hists = images.to(device), hists.to(device)
        outputs = model_a3(images, hists)
        probs_a3.append(F.softmax(outputs, dim=1).cpu().numpy())
        labels_a3.extend(labels.numpy().tolist())
probs_a3 = np.concatenate(probs_a3, axis=0)

print("[INFO] Inference Approche 2...")
probs_a2, labels_a2 = [], []
with torch.no_grad():
    for images, labels in loader_a2:
        images = images.to(device)
        outputs = model_a2(images)
        probs_a2.append(F.softmax(outputs, dim=1).cpu().numpy())
        labels_a2.extend(labels.numpy().tolist())
probs_a2 = np.concatenate(probs_a2, axis=0)

assert labels_a3 == labels_a2, "Les labels des deux jeux de test ne correspondent pas -- desalignement !"
labels_true = np.array(labels_a3)

print(f"\n[VERIF] {len(labels_true)} labels compares, alignement OK")

# ============================================================
# Evaluation individuelle (verification) + ensemble (moyenne simple)
# ============================================================
for name, probs in [("Approche 3 seule", probs_a3), ("Approche 2 seule", probs_a2)]:
    preds = probs.argmax(axis=1)
    print(f"\n=== {name} (verification, doit matcher les resultats deja connus) ===")
    print(classification_report(labels_true, preds, target_names=CLASS_NAMES, zero_division=0))

probs_ensemble = (probs_a3 + probs_a2) / 2.0
preds_ensemble = probs_ensemble.argmax(axis=1)

print("\n=======================================================")
print("=== ENSEMBLE (moyenne simple des probabilites) ===")
print("=======================================================")
print(classification_report(labels_true, preds_ensemble, target_names=CLASS_NAMES, zero_division=0))
print("Matrice de confusion (A/B/C/D):")
print(confusion_matrix(labels_true, preds_ensemble, labels=[0, 1, 2, 3]))

print("\nTermine.")
