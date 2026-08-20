# Script : eval_val_report.py
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from hybrid_finetuning import DualBackboneClassifier, SingleViewDataset, VIEW, VINDR_ROOT, ANNOTATIONS_CSV, DENSITY_CLASSES

view = sys.argv[1] if len(sys.argv) > 1 else VIEW
backbone_a = sys.argv[2] if len(sys.argv) > 2 else 'rexnet_150'
backbone_b = sys.argv[3] if len(sys.argv) > 3 else 'resnet50'
tag = f"{backbone_a}_{backbone_b}_{view.lower()}"
weights_path = f"featuresfinetuned_weights/dualbackbone_best_{tag}.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(ANNOTATIONS_CSV)
train_df = df[df['split'] == 'training'].copy()

train_indices = train_df.index.tolist()
np.random.seed(42)
val_size = int(0.2 * len(train_indices))
val_indices = np.random.choice(train_indices, size=val_size, replace=False)
train_df.loc[val_indices, 'temp_split'] = 'validation'

val_dataset = SingleViewDataset(
  train_df[train_df['temp_split'] == 'validation'], VINDR_ROOT, view,
  use_augmentation=False, split='training'
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

print(f"{len(val_dataset)} images de validation ({view}, {tag})")

model = DualBackboneClassifier(view=view, backbone_a=backbone_a, backbone_b=backbone_b, num_classes=4, dropout=0.3)
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

print(f"\n ACCURACY VALIDATION ({tag}): {accuracy_score(all_labels, all_preds)*100:.2f}%")
print("\nMatrice de confusion (validation):")
print(confusion_matrix(all_labels, all_preds))
print("\nRapport de classification (validation):")
print(classification_report(all_labels, all_preds, target_names=DENSITY_CLASSES, zero_division=0))
