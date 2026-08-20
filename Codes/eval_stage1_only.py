# Script : eval_stage1_only.py
import argparse
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from training import MammographyDataset, CONFIG
from models import FineTunedFeatureExtractor
from cascade_finetuned_pipeline import infer_probs, DEVICE


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--backbone', type=str, default='resnet50')
  parser.add_argument('--out_dir', default='cascade_finetuned_weights')
  args = parser.parse_args()

  df = pd.read_csv(CONFIG['ANNOTATIONS_CSV'])
  df_test = df[df['split'] == 'test'].reset_index(drop=True)

  group_map = {"DENSITY A": 0, "DENSITY B": 0, "DENSITY C": 1, "DENSITY D": 1}
  test_dataset = MammographyDataset(
    df_test, CONFIG['IMAGE_ROOT'], label_map=group_map,
    use_augmentation=False, split='test',
  )
  print(f"{len(test_dataset)} images (A+B+C+D) dans le test set")

  model = FineTunedFeatureExtractor(args.backbone, num_classes=2, pretrained=False).to(DEVICE)
  model.load_state_dict(torch.load(f"{args.out_dir}/{args.backbone}_stage1_abcd.pth", map_location=DEVICE))

  probs, labels = infer_probs(model, test_dataset)
  preds = (probs[:, 1] > 0.5).astype(int)

  accuracy = (preds == labels).mean()
  print(f"\nAccuracy (A,B) vs (C,D) sur le test set : {accuracy*100:.2f}%")
  print("\nRapport de classification :")
  print(classification_report(labels, preds, target_names=["(A,B)", "(C,D)"], zero_division=0))
  print("Matrice de confusion (lignes=vrai, colonnes=predit) [(A,B), (C,D)] :")
  print(confusion_matrix(labels, preds, labels=[0, 1]))


if __name__ == "__main__":
  main()
