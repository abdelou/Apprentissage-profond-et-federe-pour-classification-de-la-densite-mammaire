"""
Évalue le meilleur checkpoint GLCM déjà entraîné sur son split de VALIDATION
(même split que celui utilisé pendant l'entraînement, seed=42), sans réentraîner.
Sert à obtenir la matrice de confusion + rapport de classification sur validation
pour la figure entrainement.png (le script d'entraînement n'affiche que l'accuracy
scalaire par époque, pas de matrice).
"""
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from hybrid_finetuning_glcm import HybridGLCMClassifier, HybridGLCMDataset, VINDR_ROOT, ANNOTATIONS_CSV

view_position = sys.argv[1] if len(sys.argv) > 1 else 'CC'
backbone = sys.argv[2] if len(sys.argv) > 2 else 'cnn'
weights_path = f"featuresfinetuned_weights/hybrid_model_best_{view_position.lower()}_{backbone}_GLCM.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(ANNOTATIONS_CSV)
train_df = df[df['split'] == 'training'].copy()

train_indices = train_df.index.tolist()
np.random.seed(42)
val_size = int(0.2 * len(train_indices))
val_indices = np.random.choice(train_indices, size=val_size, replace=False)
train_df.loc[val_indices, 'temp_split'] = 'validation'

val_dataset = HybridGLCMDataset(
    train_df[train_df['temp_split'] == 'validation'], VINDR_ROOT,
    view_position=view_position, use_augmentation=False, split='training'
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

print(f"[INFO] {len(val_dataset)} images de validation ({view_position})")

model = HybridGLCMClassifier(backbone=backbone, num_classes=4, dropout=0.3)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print(f"\n ACCURACY VALIDATION ({view_position}): {accuracy_score(all_labels, all_preds)*100:.2f}%")
print("\nMatrice de confusion (validation):")
print(confusion_matrix(all_labels, all_preds))
print("\nRapport de classification (validation):")
print(classification_report(all_labels, all_preds, target_names=["Density A", "Density B", "Density C", "Density D"], zero_division=0))
