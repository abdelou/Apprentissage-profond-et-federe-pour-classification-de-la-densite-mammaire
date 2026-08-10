"""Calcule precision/recall/f1/support par classe pour un backbone déjà fine-tuné,
sans relancer l'entraînement. Usage: python3 eval_finetuned_report.py <backbone> <weights_path> [--test]
"""
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from training import MammographyDataset, CONFIG
from models import FineTunedFeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('backbone', type=str)
parser.add_argument('weights_path', type=str)
parser.add_argument('--annotations_csv', type=str, default=CONFIG['ANNOTATIONS_CSV'])
parser.add_argument('--image_root', type=str, default=CONFIG['IMAGE_ROOT'])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--split', type=str, default='training', choices=['training', 'test'])
parser.add_argument('--view', type=str, default=None, choices=['CC', 'MLO'],
                     help='Filtrer sur une seule vue (ex: CC, pour comparer a un sous-ensemble '
                          'identique a celui utilise cote federe, qui ne voit que des vues CC)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(args.annotations_csv)
df_split = df[df['split'] == args.split]
if args.view is not None:
    df_split = df_split[df_split['view_position'] == args.view]
df_split = df_split.reset_index(drop=True)
dataset = MammographyDataset(df_split, args.image_root, split=args.split, use_augmentation=False, label_map=CONFIG['CLASS_MAP'])
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
print(f"[INFO] {len(dataset)} images ({args.split})")

model = FineTunedFeatureExtractor(args.backbone, num_classes=4, pretrained=False)
model.load_finetuned_weights(args.weights_path, device)
model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

print(f"\n=== Rapport de classification ({args.backbone}, {args.split}) ===")
print(classification_report(all_labels, all_preds, labels=[0, 1, 2, 3], target_names=CONFIG['DENSITY_CLASSES'], zero_division=0))
print("Matrice de confusion (A/B/C/D):")
print(confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3]))
