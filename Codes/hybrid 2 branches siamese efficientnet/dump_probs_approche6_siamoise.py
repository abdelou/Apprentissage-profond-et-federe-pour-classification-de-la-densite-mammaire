"""
Calcule et sauvegarde les probabilites softmax du modele Approche 6 -
variante siamoise (EfficientNet-B0, poids partages, CC+MLO), avec une cle
d'alignement (study_id + laterality) pour pouvoir combiner ces
probabilites avec celles d'un autre modele qui n'utilise pas le meme
regroupement par paire (ex: Approche 2, qui evalue les images individuellement).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hybrid_model import SiameseDoubleBranchClassifier
from hybrid_finetuning import HybridMammographyDataset

CLASS_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
VINDR_ROOT = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
ANNOTATIONS_CSV = os.path.join(VINDR_ROOT, "breast-level_annotations.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(ANNOTATIONS_CSV)
df_test = df[df["split"] == "test"].reset_index(drop=True)
test_dataset = HybridMammographyDataset(
    df_test, VINDR_ROOT, VINDR_ROOT, label_map=CLASS_MAP, use_augmentation=False, split="test",
)
loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
print(f"[INFO] {len(test_dataset)} paires CC+MLO de test")

model = SiameseDoubleBranchClassifier(
    backbone="efficientnet_b0", input_channels=1, image_feature_dim=512,
    num_classes=4, dropout=0.3, pretrained=False,
)
model.load_finetuned_weights("featuresfinetuned_weights/hybrid_model_best_efficientnet_b0_2branches.pth", device=device)
model.to(device)
model.eval()

print("[INFO] Inference Approche 6 (siamoise)...")
all_probs, all_labels = [], []
with torch.no_grad():
    for mlo_images, cc_images, labels in loader:
        mlo_images, cc_images = mlo_images.to(device), cc_images.to(device)
        outputs = model(mlo_images, cc_images)
        all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.numpy().tolist())

probs = np.concatenate(all_probs, axis=0)
labels = np.array(all_labels)
keys = np.array([f"{p['study_id']}_{p['laterality']}" for p in test_dataset.pairs])

assert len(keys) == len(labels) == probs.shape[0], "Desalignement entre cles/labels/probs !"

np.save("ensemble_probs_a6siam.npy", probs)
np.save("ensemble_labels_a6siam.npy", labels)
np.save("ensemble_keys_a6siam.npy", keys)
print(f"[INFO] Sauvegarde : {probs.shape[0]} paires, probs shape {probs.shape}")
print("Termine (Approche 6 siamoise).")
