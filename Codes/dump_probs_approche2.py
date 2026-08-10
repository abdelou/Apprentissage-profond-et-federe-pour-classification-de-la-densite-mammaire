"""
Calcule et sauvegarde les probabilites softmax de l'Approche 2 (ResNet50,
fine-tuning direct) sur le meme jeu de test MLO que l'Approche 3, pour
l'ensemble. Tourne dans son propre process, dans le dossier racine (son
propre preprocessing.py), pas de conflit avec le dossier variante.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training import MammographyDataset, CONFIG
from models import FineTunedFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(CONFIG["ANNOTATIONS_CSV"])
df_test_mlo = df[df["view_position"] == "MLO"]
df_test_mlo = df_test_mlo[df_test_mlo["split"] == "test"].reset_index(drop=True)
print(f"[INFO] {len(df_test_mlo)} images de test (vues MLO)")

model = FineTunedFeatureExtractor("resnet", num_classes=4, pretrained=False)
model.load_finetuned_weights("featuresfinetuned_weights/resnet50_finetuned_best.pth", device)
model.to(device)
model.eval()

dataset = MammographyDataset(
    df_test_mlo, CONFIG["IMAGE_ROOT"], split="test", use_augmentation=False, label_map=CONFIG["CLASS_MAP"],
)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

print("[INFO] Inference Approche 2...")
all_probs, all_labels = [], []
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.numpy().tolist())

probs = np.concatenate(all_probs, axis=0)
labels = np.array(all_labels)

np.save("ensemble_probs_a2.npy", probs)
np.save("ensemble_labels_a2.npy", labels)
print(f"[INFO] Sauvegarde : {probs.shape[0]} images, probs shape {probs.shape}")
print("Termine (Approche 2).")
