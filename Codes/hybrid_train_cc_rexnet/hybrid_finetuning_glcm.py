import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from hybrid_model import ImageBranch, GLCMDescriptorMLP
from preprocessing import read_dicom, preprocess_image

VINDR_ROOT = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
ANNOTATIONS_CSV = os.path.join(VINDR_ROOT, 'breast-level_annotations.csv')
CLASS_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}

class HybridGLCMClassifier(nn.Module):
    def __init__(self, backbone='cnn', num_classes=4, dropout=0.3):
        super().__init__()
        self.image_branch = ImageBranch(backbone=backbone, input_channels=1, feature_dim=512, pretrained=True)
        self.glcm_branch = GLCMDescriptorMLP(hidden_dims=[64, 64], dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image_tensor):
        image_features = self.image_branch(image_tensor)
        glcm_features = self.glcm_branch(image_tensor)
        combined = torch.cat([image_features, glcm_features], dim=1)
        return self.classifier(combined)

class HybridGLCMDataset(Dataset):
    def __init__(self, df, image_dir, view_position='CC', use_augmentation=False, split='training'):
        self.image_dir = image_dir
        if "view_position" in df.columns:
            self.df = df[df["view_position"] == view_position].copy().reset_index(drop=True)
        else:
            self.df = df.copy().reset_index(drop=True)
            
        self.df['label'] = self.df['breast_density'].map(CLASS_MAP)
        self.density_map = {"DENSITY A": "density_A", "DENSITY B": "density_B", "DENSITY C": "density_C", "DENSITY D": "density_D"}
        self.split_map = {"training": "train", "test": "test"}
        
        if use_augmentation and split == 'training':
            class_counts = self.df['breast_density'].value_counts()
            n_majority = class_counts.max()
            target_count = int(n_majority * 0.5)
            aug_dfs = []
            for d_name in ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]:
                c_df = self.df[self.df['breast_density'] == d_name]
                if len(c_df) > 0:
                    if len(c_df) < target_count:
                        mult = int(np.ceil(target_count / len(c_df)))
                        aug_dfs.append(pd.concat([c_df] * mult, ignore_index=True).iloc[:target_count])
                    else:
                        aug_dfs.append(c_df)
            self.df = pd.concat(aug_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

        if use_augmentation and split == 'training':
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        else:
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
        split = self.split_map.get(row["split"], "train")
        
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

def train_and_eval_glcm(view_position='MLO', backbone='cnn', epochs=5, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"ENTRAÎNEMENT MODÈLE HYBRIDE : BACKBONE {backbone.upper()} + GLCM (Vues {view_position})")
    print(f"=======================================================")
    
    df = pd.read_csv(ANNOTATIONS_CSV)
    train_df = df[df['split'] == 'training'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    train_indices = train_df.index.tolist()
    np.random.seed(42)
    val_size = int(0.2 * len(train_indices))
    val_indices = np.random.choice(train_indices, size=val_size, replace=False)
    train_indices = list(set(train_indices) - set(val_indices))
    
    train_df.loc[train_indices, 'temp_split'] = 'training'
    train_df.loc[val_indices, 'temp_split'] = 'validation'
    
    train_dataset = HybridGLCMDataset(train_df[train_df['temp_split'] == 'training'], VINDR_ROOT, view_position=view_position, use_augmentation=True, split='training')
    val_dataset = HybridGLCMDataset(train_df[train_df['temp_split'] == 'validation'], VINDR_ROOT, view_position=view_position, use_augmentation=False, split='training')
    test_dataset = HybridGLCMDataset(test_df, VINDR_ROOT, view_position=view_position, use_augmentation=False, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = HybridGLCMClassifier(backbone=backbone, num_classes=4, dropout=0.3)
    model.to(device)

    # Pondération de la loss : Density A n'a qu'une poignée d'exemples (~10 en val/test)
    # sur 2000 images, donc sans pondération le modèle l'ignore quasi totalement (recall proche de 0).
    class_counts_train = train_dataset.df['label'].value_counts().sort_index()
    n_samples = class_counts_train.sum()
    n_classes = len(class_counts_train)
    class_weights = torch.tensor(
        [n_samples / (n_classes * class_counts_train.get(c, 1)) for c in range(4)],
        dtype=torch.float32, device=device
    )
    print(f"Poids de classes (loss) : {class_weights.cpu().numpy().round(3).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_acc = 0.0
    best_weights_path = f"featuresfinetuned_weights/hybrid_model_best_{view_position.lower()}_{backbone}_GLCM.pth"
    history_path = f"featuresfinetuned_weights/history_{view_position.lower()}_{backbone}_GLCM.json"
    os.makedirs("featuresfinetuned_weights", exist_ok=True)
    history = {"epochs": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    if os.path.exists(best_weights_path):
        print(f"✅ Modèle GLCM déjà entraîné trouvé à {best_weights_path}! Passage direct à l'évaluation de test sur les 2 000 images...")
        print("   (Pour relancer l'entraînement avec le code corrigé, supprimez ce fichier .pth d'abord.)")
    else:
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train GLCM {view_position}]"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            train_loss = running_loss / total

            # Validation
            model.eval()
            val_running_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            val_loss = val_running_loss / val_total
            print(f"Époque {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

            history["epochs"].append(epoch + 1)
            history["train_acc"].append(train_acc * 100)
            history["val_acc"].append(val_acc * 100)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_weights_path)
                print(f"✅ Meilleur modèle GLCM sauvegardé: {best_weights_path} ({val_acc*100:.2f}%)")

            scheduler.step()

    # ÉVALUATION TEST FINAL
    print(f"\n=======================================================")
    print(f"RÉSULTAT DU TEST FINAL (MODÈLE HYBRIDE {backbone.upper()} + GLCM SUR VUES {view_position})")
    print(f"=======================================================")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    print(f"\n🏆 ACCURACY TEST FINAL HYBRIDE GLCM ({view_position}): {accuracy_score(all_labels, all_preds)*100:.2f}%")
    print("\nMatrice de confusion:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nRapport de classification:")
    print(classification_report(all_labels, all_preds, target_names=["Density A", "Density B", "Density C", "Density D"], zero_division=0))

if __name__ == '__main__':
    import sys
    view = sys.argv[1] if len(sys.argv) > 1 else 'CC'
    backbone = sys.argv[2] if len(sys.argv) > 2 else 'cnn'
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    train_and_eval_glcm(view_position=view, backbone=backbone, epochs=epochs)
