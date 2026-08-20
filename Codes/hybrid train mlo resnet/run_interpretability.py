# Script : run_interpretability.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hybrid_model import HybridMammographyClassifier, create_histogram_from_image
from hybrid_finetuning import HybridMammographyDataset, VINDR_ROOT, CONFIG

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from interpretability_analysis import GradCAM, run_gradcam_grid, run_tsne, run_tsne_train_test, get_module_by_name

# on garde les deux versions de figures (pas de remplacement) :
#  1) tsne_approche3_resnet50_mlo.png    -- test seul (comme avant)
#  2) tsne_approche3_resnet50_mlo_traintest.png -- train vs test (nouveau)


class ImageOnlyWrapper(nn.Module):
  """Presente une interface a une seule entree (image) pour un modele qui
  attend normalement (image, histogramme). L'histogramme est recalcule a
  la volee a partir de l'image, exactement comme au moment de l'evaluation
  normale du modele (voir create_histogram_from_image)."""
  def __init__(self, inner_model):
    super().__init__()
    self.inner = inner_model

  def forward(self, image_tensor):
    hist = create_histogram_from_image(image_tensor)
    return self.inner(image_tensor, hist)


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  weights_path = "featuresfinetuned_weights/hybrid_model_best_cnn.pth"
  out_dir = "../graphes_interpretabilite"
  os.makedirs(out_dir, exist_ok=True)

  print(f"Chargement du modele depuis {weights_path}")
  inner_model = HybridMammographyClassifier(
    backbone="cnn",
    input_channels=1,
    image_feature_dim=512,
    hist_hidden_dims=[128, 64],
    num_classes=4,
    dropout=0.3,
    pretrained=False,
  )
  state_dict = torch.load(weights_path, map_location=device)
  inner_model.load_state_dict(state_dict)
  inner_model.to(device)
  inner_model.eval()

  model = ImageOnlyWrapper(inner_model).to(device)
  model.eval()

  target_layer = get_module_by_name(model, "inner.image_branch.cnn.layer4")

  # Jeu de test (2000 images MLO, jamais vues pendant l'entrainement) et
  # jeu d'entrainement (pour la comparaison t-SNE train vs test)
  df = pd.read_csv(CONFIG["ANNOTATIONS_CSV"])
  if "view_position" in df.columns:
    df = df[df["view_position"] == "MLO"].reset_index(drop=True)
  df_test = df[df["split"] == "test"].reset_index(drop=True)
  df_train = df[df["split"] == "training"].reset_index(drop=True)
  test_dataset = HybridMammographyDataset(
    df_test, VINDR_ROOT, label_map=CONFIG["CLASS_MAP"],
    use_augmentation=False, split="test",
  )
  train_dataset = HybridMammographyDataset(
    df_train, VINDR_ROOT, label_map=CONFIG["CLASS_MAP"],
    use_augmentation=False, split="training",
  )

  # HybridMammographyDataset retourne (image, hist, label) -- on ne garde
  # que (image, label) pour le wrapper, qui recalcule son propre histogramme
  class ImageLabelOnly(torch.utils.data.Dataset):
    def __init__(self, base):
      self.base = base
    def __len__(self):
      return len(self.base)
    def __getitem__(self, idx):
      image, _hist, label = self.base[idx]
      return image, label

  dataset = ImageLabelOnly(test_dataset)
  dataset_train = ImageLabelOnly(train_dataset)
  print(f"{len(dataset)} images de test, {len(dataset_train)} images de train chargees")

  print("\n[1/2] Generation de la grille Grad-CAM (4 classes x 3 exemples)...")
  run_gradcam_grid(
    model, target_layer, dataset, device,
    out_path=f"{out_dir}/gradcam_approche3_resnet50_mlo.png",
    n_per_class=2,
  )

  print("\n[2/3] Generation de la projection t-SNE (test seul)...")
  run_tsne(
    model, target_layer, dataset, device,
    out_path=f"{out_dir}/tsne_approche3_resnet50_mlo.png",
    n_samples=300,
    backbone_name="ResNet50 + Histogramme, MLO",
  )

  print("\n[3/3] Generation de la projection t-SNE train vs test...")
  run_tsne_train_test(
    model, target_layer, dataset_train, dataset, device,
    out_path=f"{out_dir}/tsne_approche3_resnet50_mlo_traintest.png",
    n_samples_train=300, n_samples_test=300,
    backbone_name="ResNet50 + Histogramme, MLO",
  )

  print("\nTermine.")


if __name__ == "__main__":
  main()
