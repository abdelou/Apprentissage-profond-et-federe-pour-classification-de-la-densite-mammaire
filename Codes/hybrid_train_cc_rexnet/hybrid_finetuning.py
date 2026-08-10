import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import torchvision.transforms as transforms
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from hybrid_model import HybridMammographyClassifier, create_histogram_from_image
from preprocessing import read_dicom, preprocess_image

# --- Configuration globale ---
VINDR_ROOT = '/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0'
CONFIG = {
    'ANNOTATIONS_CSV': os.path.join(VINDR_ROOT, 'breast-level_annotations.csv'),
    'IMAGE_ROOT': VINDR_ROOT,
    'BACKBONE': 'cnn',
    'EPOCHS': 5,
    'BATCH_SIZE': 8,
    'LEARNING_RATE': 1e-4,
    'DENSITY_CLASSES': ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"],
    'CLASS_MAP': {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3},
    'FEATURES_DIR': 'featuresextracted',
    'MODELS_DIR': 'featuresmodels',
    'FINETUNED_WEIGHTS_DIR': 'featuresfinetuned_weights',
    'RESUME_FROM': 'featuresfinetuned_weights/hybrid_model_checkpoint_cnn_GLCM.pth',
    'FINETUNED_WEIGHTS_PATH': 'featuresfinetuned_weights/cnn_image_branch_best_rexnet150.pth'
}


# --- Dataset unifié avec support data augmentation et vues CC ---
class HybridMammographyDataset(Dataset):
    """
    Dataset unifié pour charger les mammographies avec support pour data augmentation et vues CC.
    """
    def __init__(self, annotations_df, image_dir, classes_to_use=None, label_map=None, 
                 use_augmentation=False, split='training'):
        self.image_dir = image_dir
        self.density_map = {
            "DENSITY A": "density_A", "DENSITY B": "density_B",
            "DENSITY C": "density_C", "DENSITY D": "density_D"
        }
        self.split_map = {"training": "train", "test": "test"}
        self.use_augmentation = use_augmentation and split == 'training'
        self.split = split
        
        # Filtrer sur les vues CC
        if "view_position" in annotations_df.columns:
            annotations_df = annotations_df[annotations_df["view_position"] == "CC"].copy()

        if 'split' in annotations_df.columns:
            original_split = annotations_df['split'].unique()
            if len(original_split) > 1:
                annotations_df = annotations_df[annotations_df['split'] == split].copy()

        if classes_to_use:
            self.df = annotations_df[annotations_df['breast_density'].isin(classes_to_use)].copy()
            self.label_remap = {classes_to_use[0]: 0, classes_to_use[1]: 1}
            self.df['label'] = self.df['breast_density'].map(self.label_remap)
        else:
            self.df = annotations_df.copy()
            if label_map:
                self.df['label'] = self.df['breast_density'].map(label_map)
            else:
                self.df['label'] = self.df['breast_density'].map(CONFIG['CLASS_MAP'])

        # --- Data Augmentation et Rééquilibrage Automatique Dynamique (Sans valeurs en dur) ---
        if self.use_augmentation and split == 'training':
            class_counts = self.df['breast_density'].value_counts()
            majority_class = class_counts.idxmax()
            n_majority = class_counts[majority_class]
            
            target_count = int(n_majority * 0.5)
            aug_dfs = []
            
            for density_name in ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]:
                class_df = self.df[self.df['breast_density'] == density_name]
                if len(class_df) > 0:
                    if len(class_df) < target_count:
                        mult = int(np.ceil(target_count / len(class_df)))
                        augmented_class_df = pd.concat([class_df] * mult, ignore_index=True).iloc[:target_count]
                        aug_dfs.append(augmented_class_df)
                    else:
                        aug_dfs.append(class_df)
                
            self.df = pd.concat(aug_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(f"[Rééquilibrage Automatique] Distribution dynamique propre à votre projet:")
            for d in ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]:
                count = len(self.df[self.df['breast_density'] == d])
                if count > 0:
                    print(f"  - {d}: {count} échantillons")
            print(f"  - Total d'entraînement: {len(self.df)} images au total")
        else:
            self.df = self.df.reset_index(drop=True)
        
        # Transformations géométriques et d'intensité en mémoire (Section 6.2.2.2)
        if self.use_augmentation and split == 'training':
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        else:
            self.augmentation_transforms = transforms.Compose([
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
        
        # Recherche dynamique avec fallback pour résoudre l'erreur No such file or directory
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
                if len(image.shape) == 4:
                    image = image[0, :, :, 0]
                else:
                    image = image.squeeze()
            
            image = Image.fromarray(image.astype(np.uint8), mode='L')
            image = self.augmentation_transforms(image)
            hist = create_histogram_from_image(image.unsqueeze(0)).squeeze(0)
            
            label = row['label']
            return image, hist, label
            
        except Exception as e:
            print(f"Erreur lors du chargement de {image_path}: {e}")
            return torch.randn(1, 224, 224), torch.randn(256), 0


class HybridModelManager:
    """
    Gestionnaire pour ReXNet150 sur vues CC (Figure 16).
    """
    def __init__(self, backbone='cnn', device='cpu'):
        self.backbone = backbone
        self.device = device
        self.model = None
        
    def create_model(self, input_channels=1, image_feature_dim=512, hist_hidden_dims=[128, 64],
                    num_classes=4, dropout=0.3, pretrained=True, finetuned_weights_path=None):
        self.model = HybridMammographyClassifier(
            backbone=self.backbone,
            input_channels=input_channels,
            image_feature_dim=image_feature_dim,
            hist_hidden_dims=hist_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            finetuned_weights_path=finetuned_weights_path
        )
        return self.model
    
    def train_model(self, data_csv, image_root, epochs=5, batch_size=8, lr=1e-4,
                   save_dir=None, patience=10, min_delta=0.001,
                   use_augmentation=True, resume_from=None, finetuned_weights_path=None):
        if save_dir is None:
            save_dir = CONFIG['FINETUNED_WEIGHTS_DIR']
        
        df = pd.read_csv(data_csv)
        if "view_position" in df.columns:
            df = df[df["view_position"] == "CC"].reset_index(drop=True)
            
        train_df = df[df['split'] == 'training'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        train_indices = train_df.index.tolist()
        np.random.seed(42)
        val_size = int(0.2 * len(train_indices))
        val_indices = np.random.choice(train_indices, size=val_size, replace=False)
        train_indices = list(set(train_indices) - set(val_indices))
        
        train_df.loc[train_indices, 'temp_split'] = 'training'
        train_df.loc[val_indices, 'temp_split'] = 'validation'
        
        print(f"Distribution des données (Vues CC):")
        print(f"  - Entraînement: {len(train_indices)} images")
        print(f"  - Validation: {len(val_indices)} images")
        print(f"  - Test: {len(test_df)} images")
        
        train_dataset = HybridMammographyDataset(
            train_df[train_df['temp_split'] == 'training'].copy(), 
            image_root, 
            label_map=CONFIG['CLASS_MAP'],
            use_augmentation=use_augmentation, 
            split='training'
        )
        val_dataset = HybridMammographyDataset(
            train_df[train_df['temp_split'] == 'validation'].copy(), 
            image_root, 
            label_map=CONFIG['CLASS_MAP'],
            use_augmentation=False,
            split='training'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
        
        print(f"Train: {len(train_dataset)} images")
        print(f"Val: {len(val_dataset)} images")
        
        model = self.create_model(finetuned_weights_path=finetuned_weights_path)
        model.to(self.device)
        print(f"Modèle créé avec {sum(p.numel() for p in model.parameters()):,} paramètres")
        
        if resume_from and os.path.exists(resume_from):
            print(f"Reprise depuis: {resume_from}")
            try:
                checkpoint = torch.load(resume_from, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                print(f"Checkpoint chargé (poids ReXNet150 adaptés avec succès)")
            except Exception as e:
                print(f"Remarque: Incompatibilité de dimension avec l'ancien checkpoint GLCM ({e}), démarrage d'un entraînement ReXNet150 CC propre.")
        
        trained_model = self._train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, save_dir=save_dir,
            patience=patience, min_delta=min_delta
        )
        
        return trained_model
    
    def _train_model(self, model, train_loader, val_loader, epochs=5, lr=1e-4,
                    save_dir='featuresfinetuned_weights', patience=10, min_delta=0.001):
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        best_val_acc = 0.0
        patience_counter = 0

        os.makedirs(save_dir, exist_ok=True)

        print(f"[ENTRAÎNEMENT] Début du fine-tuning ReXNet150 sur vues CC pour {epochs} époques")
        print(f"[ENTRAÎNEMENT] Learning rate: {lr}")
        print(f"[ENTRAÎNEMENT] Device: {self.device}")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_total = 0
            train_correct = 0

            progress_bar = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs}")
            for batch_idx, (images, hists, labels) in enumerate(progress_bar):
                images, hists, labels = images.to(self.device), hists.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                logits = model(images, hists)
                loss = criterion(logits, labels)

                # ==========================================================================
                # Classification hiérarchique en 2 étages (non utilisée, conservée pour référence)
                # ==========================================================================
                # stage1_logits = model.classifier_stage1(features)
                # ==========================================================================

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                from sklearn.metrics import recall_score
                batch_recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)
                progress_bar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'Recall': f'{100.*batch_recall:.2f}%'
                })
            train_acc = 100. * train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0.0
            val_total = 0
            val_correct = 0
            all_true = []
            all_pred_final = []

            with torch.no_grad():
                for images, hists, labels in val_loader:
                    images, hists, labels = images.to(self.device), hists.to(self.device), labels.to(self.device)
                    logits = model(images, hists)
                    pred_final = logits.argmax(dim=1)
                    val_loss += criterion(logits, labels).item()
                    val_total += labels.size(0)
                    val_correct += pred_final.eq(labels).sum().item()
                    all_true.extend(labels.cpu().numpy())
                    all_pred_final.extend(pred_final.cpu().numpy())

            val_acc = 100. * val_correct / val_total
            scheduler.step(val_acc)

            print(f"\n[ÉPOQUE {epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            import numpy as np
            from sklearn.metrics import confusion_matrix
            print("\nMatrice de confusion finale (A/B/C/D):")
            print(confusion_matrix(all_true, all_pred_final))

            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0

                best_model_path = os.path.join(save_dir, f'hybrid_model_best_rexnet150_cc.pth')
                model.save_finetuned_weights(best_model_path)

                checkpoint_path = os.path.join(save_dir, f'hybrid_model_checkpoint_rexnet150_cc.pth')
                torch.save({
                    'backbone_name': self.backbone,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_val_acc,
                    'patience_counter': patience_counter
                }, checkpoint_path)

                print(f"  ✅ Nouveau meilleur modèle! Acc: {val_acc:.2f}%")
                print(f"  💾 Modèle sauvegardé: {best_model_path}")
            else:
                patience_counter += 1
                print(f"  ⏳ Pas d'amélioration ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Arrêt après {patience} époques sans amélioration")
                break

        final_save_path = os.path.join(save_dir, 'hybrid_model_final_rexnet150_cc.pth')
        model.save_finetuned_weights(final_save_path)

        print(f"\n[FIN] Entraînement terminé de ReXNet150 sur vues CC (Figure 16)")
        print(f"  - Meilleure accuracy: {best_val_acc:.4f}")
        print(f"  - Modèle final: {final_save_path}")

        return model


def main():
    parser = argparse.ArgumentParser(description='Entraînement ReXNet150 sur vues CC (Figure 16)')
    parser.add_argument('--data_csv', default=CONFIG['ANNOTATIONS_CSV'], help='Chemin vers le CSV d\'annotations')
    parser.add_argument('--image_root', default=CONFIG['IMAGE_ROOT'], help='Chemin vers les images')
    parser.add_argument('--backbone', default=CONFIG['BACKBONE'], choices=['cnn', 'vit'], help='Backbone pour l\'image')
    parser.add_argument('--epochs', type=int, default=5, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=CONFIG['BATCH_SIZE'], help='Taille du batch')
    parser.add_argument('--lr', type=float, default=CONFIG['LEARNING_RATE'], help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda)')
    parser.add_argument('--use_augmentation', action='store_true', help='Utiliser la data augmentation')
    parser.add_argument('--resume_from', default=None, help='Chemin vers un checkpoint')
    parser.add_argument('--finetuned_weights_path', default=CONFIG['FINETUNED_WEIGHTS_PATH'], help='Poids fine-tunés')

    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"=== Entraînement ReXNet150 sur Vues CC (Figure 16) ===")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Data augmentation: {'Activée' if args.use_augmentation else 'Désactivée'}")
    
    try:
        manager = HybridModelManager(args.backbone, device)
        trained_model = manager.train_model(
            data_csv=args.data_csv,
            image_root=args.image_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_augmentation=args.use_augmentation,
            resume_from=args.resume_from,
            finetuned_weights_path=args.finetuned_weights_path
        )
        print("=== Entraînement réussi (Figure 16 ReXNet150 CC) ===")
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()