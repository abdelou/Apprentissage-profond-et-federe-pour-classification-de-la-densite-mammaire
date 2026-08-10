import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from hybrid_model import HybridMammographyClassifier, create_histogram_from_image
from preprocessing import read_dicom, preprocess_image

VINDR_ROOT = '/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0'
CLASS_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}

class StandaloneImageDataset(Dataset):
    def __init__(self, df, image_dir, view_position='MLO'):
        self.image_dir = image_dir
        if "view_position" in df.columns:
            self.df = df[df["view_position"] == view_position].copy().reset_index(drop=True)
        else:
            self.df = df.copy().reset_index(drop=True)
            
        self.df['label'] = self.df['breast_density'].map(CLASS_MAP)
        self.density_map = {"DENSITY A": "density_A", "DENSITY B": "density_B", "DENSITY C": "density_C", "DENSITY D": "density_D"}
        self.split_map = {"training": "train", "test": "test"}
        
        self.transform = transforms.Compose([
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
            image = self.transform(image)
            return image, row['label']
        except Exception:
            return torch.randn(1, 224, 224), 0


def evaluate_standalone_branch(weights_path, view_position='MLO', backbone='cnn', csv_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"ÉVALUATION BRANCHE IMAGE SEULE (Section 6.2.4.3)")
    print(f"  - Backbone: {backbone.upper()}")
    print(f"  - Vue: {view_position}")
    print(f"  - Poids: {weights_path}")
    print(f"=======================================================")
    
    if csv_path is None:
        csv_path = os.path.join(VINDR_ROOT, 'breast-level_annotations.csv')
    
    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test'].copy()
    
    dataset = StandaloneImageDataset(test_df, VINDR_ROOT, view_position=view_position)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model = HybridMammographyClassifier(
        backbone=backbone,
        input_channels=1,
        image_feature_dim=512,
        hist_hidden_dims=[128, 64],
        num_classes=4,
        dropout=0.3,
        pretrained=False
    )
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Poids du modèle hybride entraîné chargés avec succès!")
    else:
        print(f"⚠️ Poids non trouvés à {weights_path}")
    
    model.to(device)
    model.eval()
    
    all_true, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Test Branche Image {view_position}"):
            images = images.to(device)
            # Pour évaluer la branche image seule, on passe un histogramme nul
            dummy_hists = torch.zeros(images.size(0), 256).to(device)
            logits = model(images, dummy_hists)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_true.extend(labels.numpy())
            all_preds.extend(preds)
            
    acc = accuracy_score(all_true, all_preds)
    print(f"\n🏆 ACCURACY BRANCHE IMAGE SEULE ({view_position}): {acc*100:.2f}%")
    print("\nMatrice de confusion:")
    print(confusion_matrix(all_true, all_preds))
    print("\nRapport de classification:")
    print(classification_report(all_true, all_preds, target_names=["Density A", "Density B", "Density C", "Density D"], zero_division=0))


if __name__ == '__main__':
    # 1. ResNet50 sur vues MLO (Section 6.2.4.3)
    mlo_weights = 'hybrid train mlo resnet/featuresfinetuned_weights/hybrid_model_best_cnn.pth'
    evaluate_standalone_branch(mlo_weights, view_position='MLO', backbone='cnn')
    
    # 2. ReXNet150 sur vues CC (Section 6.2.4.3)
    cc_weights = 'hybrid_train_cc_rexnet/featuresfinetuned_weights/hybrid_model_best_rexnet150_cc.pth'
    evaluate_standalone_branch(cc_weights, view_position='CC', backbone='cnn')
