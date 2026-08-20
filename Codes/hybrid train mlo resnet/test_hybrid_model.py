import torch
from torch.utils.data import DataLoader
import pandas as pd
from hybrid_model import HybridMammographyClassifier, create_histogram_from_image
from preprocessing import read_dicom, preprocess_image
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURATION (VinDr-Mammo Cluster) ===
VINDR_ROOT = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
BEST_WEIGHTS_PATH = "featuresfinetuned_weights/hybrid_model_best_cnn.pth"
ANNOTATIONS_CSV = os.path.join(VINDR_ROOT, 'breast-level_annotations.csv')
IMAGE_ROOT = VINDR_ROOT
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
CLASS_NAMES = list(CLASS_MAP.keys())

GRAPHES_DIR = "graphes"
os.makedirs(GRAPHES_DIR, exist_ok=True)

class HybridMammographyDataset(torch.utils.data.Dataset):
  def __init__(self, annotations_df, image_dir, label_map):
    self.image_dir = image_dir
    self.density_map = {
      "DENSITY A": "density_A", "DENSITY B": "density_B",
      "DENSITY C": "density_C", "DENSITY D": "density_D"
    }
    self.split_map = {"training": "train", "test": "test"}
    
    # Filtrer sur les vues MLO
    if "view_position" in annotations_df.columns:
      annotations_df = annotations_df[annotations_df["view_position"] == "MLO"].copy()
      
    self.df = annotations_df.copy()
    self.df['label'] = self.df['breast_density'].map(label_map)
    self.df = self.df.reset_index(drop=True)
    self.transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485], std=[0.229])
    ])

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    study_id = row['study_id']
    image_id = row['image_id']
    density = self.density_map.get(row["breast_density"])
    split = self.split_map.get(row["split"], "test")
    
    image_path = os.path.join(self.image_dir, split, density, study_id, f"{image_id}.dicom")
    if not os.path.exists(image_path):
      image_path = os.path.join(self.image_dir, 'images', study_id, f"{image_id}.dicom")
      if not os.path.exists(image_path):
        image_path = os.path.join(self.image_dir, study_id, f"{image_id}.dicom")
        
    try:
      image = read_dicom(image_path)
      image = preprocess_image(image, laterality=row["laterality"])
      if len(image.shape) == 3:
        image = image[:, :, 0]
      elif len(image.shape) > 3:
        image = image[0, :, :, 0] if len(image.shape) == 4 else image.squeeze()
      image = Image.fromarray(image.astype(np.uint8), mode='L')
      image = self.transforms(image)
      hist = create_histogram_from_image(image.unsqueeze(0)).squeeze(0)
      label = row['label']
      return image, hist, label
    except Exception as e:
      print(f"Erreur lors du chargement de {image_path}: {e}")
      return torch.randn(1, 224, 224), torch.randn(256), 0

def main():
  print(f"=== Évaluation Finale du Modèle ResNet50 sur MLO (Figure 15) ===")
  print(f"Device: {DEVICE}")
  print(f"Poids: {BEST_WEIGHTS_PATH}")
  
  df = pd.read_csv(ANNOTATIONS_CSV)
  df_test = df[df['split'] == 'test'].reset_index(drop=True)
  test_dataset = HybridMammographyDataset(df_test, IMAGE_ROOT, CLASS_MAP)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

  model = HybridMammographyClassifier(
    backbone='cnn',
    input_channels=1,
    image_feature_dim=512,
    hist_hidden_dims=[128, 64],
    num_classes=4,
    dropout=0.3,
    pretrained=False
  )
  if os.path.exists(BEST_WEIGHTS_PATH):
    state_dict = torch.load(BEST_WEIGHTS_PATH, map_location=DEVICE)
    if 'model_state_dict' in state_dict:
      state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    print(" Poids chargés avec succès!")
  model.to(DEVICE)
  model.eval()

  all_preds, all_labels = [], []
  with torch.no_grad():
    for images, hists, labels in test_loader:
      images, hists, labels = images.to(DEVICE), hists.to(DEVICE), labels.to(DEVICE)
      outputs = model(images, hists)
      preds = torch.argmax(outputs, dim=1)
      all_preds.append(preds.cpu().numpy())
      all_labels.append(labels.cpu().numpy())

  all_preds = np.concatenate(all_preds)
  all_labels = np.concatenate(all_labels)

  report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)
  print("\n=== RÉSULTATS DU TEST FINAL (ResNet50 MLO) ===")
  print(report)

  cm = confusion_matrix(all_labels, all_preds)
  print("Matrice de confusion :")
  print(cm)

if __name__ == '__main__':
  main()