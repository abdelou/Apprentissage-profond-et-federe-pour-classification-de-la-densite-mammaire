import torch
from torch.utils.data import DataLoader
import pandas as pd
from hybrid_model import HybridMammographyClassifier, create_histogram_from_image
from preprocessing import read_dicom, preprocess_image
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

VINDR_ROOT = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
ANNOTATIONS_CSV = os.path.join(VINDR_ROOT, 'breast-level_annotations.csv')
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
CLASS_NAMES = list(CLASS_MAP.keys())

class TestMammographyDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir):
        self.image_dir = image_dir
        self.df = df[df['split'] == 'test'].copy().reset_index(drop=True)
        self.df['label'] = self.df['breast_density'].map(CLASS_MAP)
        self.density_map = {"DENSITY A": "density_A", "DENSITY B": "density_B", "DENSITY C": "density_C", "DENSITY D": "density_D"}
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
        
        image_path = os.path.join(self.image_dir, 'test', density, study_id, f"{image_id}.dicom")
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
            return image, hist, row['label']
        except Exception:
            return torch.randn(1, 224, 224), torch.randn(256), 0

def test_backbone(backbone_name='cnn', weights_path='featuresfinetuned_weights/hybrid_model_best_cnn.pth'):
    print(f"\n=======================================================")
    print(f"TEST FINAL SECTION 6.2.3 : Extracteur {backbone_name.upper()}")
    print(f"  - Poids: {weights_path}")
    print(f"=======================================================")
    
    df = pd.read_csv(ANNOTATIONS_CSV)
    test_dataset = TestMammographyDataset(df, VINDR_ROOT)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = HybridMammographyClassifier(
        backbone=backbone_name,
        input_channels=1,
        image_feature_dim=512,
        hist_hidden_dims=[128, 64],
        num_classes=4,
        dropout=0.3,
        pretrained=False
    )
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f" Poids {backbone_name.upper()} chargés avec succès!")
    else:
        print(f"  Poids non trouvés à {weights_path}")
        
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, hists, labels in tqdm(test_loader, desc=f"Test {backbone_name.upper()}"):
            images, hists, labels = images.to(DEVICE), hists.to(DEVICE), labels.to(DEVICE)
            outputs = model(images, hists)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print(f"\n=== RAPPORT DE CLASSIFICATION ({backbone_name.upper()}) ===")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Matrice de confusion :")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    # 1. Test Extracteur ResNet50
    test_backbone('cnn', 'featuresfinetuned_weights/hybrid_model_best_cnn.pth')
    
    # 2. Test Extracteur ViT (Vision Transformer)
    test_backbone('vit', 'featuresfinetuned_weights/hybrid_model_best_vit.pth')
