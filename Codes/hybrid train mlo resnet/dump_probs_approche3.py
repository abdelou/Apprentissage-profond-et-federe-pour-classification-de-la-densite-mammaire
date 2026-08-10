"""
Calcule et sauvegarde les probabilites softmax de l'Approche 3 (ResNet50 +
Histogramme, MLO) sur le jeu de test MLO, pour un ensemble ulterieur avec
l'Approche 2. Tourne dans son propre process (son propre preprocessing.py
local), pas de conflit d'import avec le dossier racine.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hybrid_model import HybridMammographyClassifier
from hybrid_finetuning import HybridMammographyDataset, VINDR_ROOT, CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(CONFIG["ANNOTATIONS_CSV"])
df_test_mlo = df[df["view_position"] == "MLO"]
df_test_mlo = df_test_mlo[df_test_mlo["split"] == "test"].reset_index(drop=True)
print(f"[INFO] {len(df_test_mlo)} images de test (vues MLO)")

model = HybridMammographyClassifier(
    backbone="cnn", input_channels=1, image_feature_dim=512,
    hist_hidden_dims=[128, 64], num_classes=4, dropout=0.3, pretrained=False,
)
model.load_state_dict(torch.load("featuresfinetuned_weights/hybrid_model_best_cnn.pth", map_location=device))
model.to(device)
model.eval()

dataset = HybridMammographyDataset(
    df_test_mlo, VINDR_ROOT, label_map=CONFIG["CLASS_MAP"], use_augmentation=False, split="test",
)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

print("[INFO] Inference Approche 3...")
all_probs, all_labels = [], []
with torch.no_grad():
    for images, hists, labels in loader:
        images, hists = images.to(device), hists.to(device)
        outputs = model(images, hists)
        all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.numpy().tolist())

probs = np.concatenate(all_probs, axis=0)
labels = np.array(all_labels)

np.save("../ensemble_probs_a3.npy", probs)
np.save("../ensemble_labels_a3.npy", labels)
print(f"[INFO] Sauvegarde : {probs.shape[0]} images, probs shape {probs.shape}")
print("Termine (Approche 3).")
