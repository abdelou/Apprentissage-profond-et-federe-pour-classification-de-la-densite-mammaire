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

# === CONFIGURATION ===
BEST_WEIGHTS_PATH = "featuresfinetuned_weights/hybrid_model_best_cnn.pth"
ANNOTATIONS_CSV = "DDSM/output_annotations.csv"
# TODO: pointe vers ton propre export DDSM sur le cluster - à adapter.
IMAGE_ROOT = "/home_nfs/abdelouahada/dataset_extracted/DDSM/"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
CLASS_NAMES = list(CLASS_MAP.keys())

# === OUTDIR POUR LES GRAPHIQUES ===
GRAPHES_DIR = "graphes"
os.makedirs(GRAPHES_DIR, exist_ok=True)

def save_plot(plt, filename, graphes_dir=GRAPHES_DIR):
    filepath = os.path.join(graphes_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[INFO] Graphique sauvegardé: {filepath}")
    plt.close()

def save_text_report(report, filename, graphes_dir=GRAPHES_DIR):
    filepath = os.path.join(graphes_dir, filename)
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"[INFO] Rapport texte sauvegardé: {filepath}")

# === DATASET POUR LE TEST ===
class HybridMammographyDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_df, image_dir, label_map):
        self.image_dir = image_dir
        self.density_map = {
            "DENSITY A": "density_A", "DENSITY B": "density_B",
            "DENSITY C": "density_C", "DENSITY D": "density_D"
        }
        self.split_map = {"training": "train", "test": "test"}
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
        split = self.split_map.get(row["split"])
        image_path = f"{self.image_dir}/{split}/{density}/{study_id}/{image_id}.dicom"
        try:
            image = read_dicom(image_path)
            image = preprocess_image(image, laterality=row["laterality"])
            # Gestion des dimensions
            if len(image.shape) == 3:
                image = image[:, :, 0]
            elif len(image.shape) > 3:
                if len(image.shape) == 4:
                    image = image[0, :, :, 0]
                else:
                    image = image.squeeze()
            image = Image.fromarray(image.astype(np.uint8), mode='L')
            image = self.transforms(image)
            hist = create_histogram_from_image(image.unsqueeze(0)).squeeze(0)
            label = row['label']
            return image, hist, label
        except Exception as e:
            print(f"Erreur lors du chargement de {image_path}: {e}")
            return torch.randn(1, 224, 224), torch.randn(256), 0

# === CHARGEMENT DES DONNÉES ===
df = pd.read_csv(ANNOTATIONS_CSV)
df_test = df[df['split'] == 'test'].reset_index(drop=True)
test_dataset = HybridMammographyDataset(df_test, IMAGE_ROOT, CLASS_MAP)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === CHARGEMENT DU MODÈLE ===

model = HybridMammographyClassifier(
    backbone='cnn',
    input_channels=1,
    image_feature_dim=512,
    hist_hidden_dims=[128, 64],
    num_classes=4,
    dropout=0.3,
    pretrained=False
)
model.load_finetuned_weights(BEST_WEIGHTS_PATH, device=DEVICE)
model.to(DEVICE)
model.eval()

# === ÉVALUATION ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, hists, labels in test_loader:
        images = images.to(DEVICE)
        hists = hists.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images, hists)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
print ("=== Résultats sur le set de test ===")
print("la taille d'une sortie du modele est :", outputs.shape)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Rapport de classification
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
print("=== Résultats sur le set de test ===")
print(report)
save_text_report(report, "hybrid_model_cnn_classification_report.txt")

# Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
print("Matrice de confusion :")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.title('Matrice de confusion - Modèle hybride')
save_plot(plt, "hybrid_model_cnn_confusion_matrix.png")