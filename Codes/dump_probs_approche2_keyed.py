"""
Variante de dump_probs_approche2.py qui sauvegarde en plus une cle
d'alignement (study_id + laterality) par image, pour pouvoir combiner ces
probabilites avec un modele qui raisonne par paire de seins (ex: Approche 6
siamoise) plutot que par image individuelle.
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
keys = (df_test_mlo["study_id"].astype(str) + "_" + df_test_mlo["laterality"].astype(str)).values

assert len(keys) == len(labels) == probs.shape[0], "Desalignement entre cles/labels/probs !"

np.save("ensemble_probs_a2_keyed.npy", probs)
np.save("ensemble_labels_a2_keyed.npy", labels)
np.save("ensemble_keys_a2_keyed.npy", keys)
print(f"[INFO] Sauvegarde : {probs.shape[0]} images, probs shape {probs.shape}")
print("Termine (Approche 2, avec cles).")
