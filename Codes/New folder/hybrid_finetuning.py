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
import time

from hybrid_model import HybridMammographyClassifier, create_histogram_from_image
from preprocessing import read_dicom, preprocess_image

# --- Configuration globale ---
CONFIG = {
    'ANNOTATIONS_CSV': 'DDSM/output_annotations_augmented.csv',
    # TODO: pointe vers ton propre export DDSM sur le cluster - à adapter.
    'IMAGE_ROOT': '/home_nfs/abdelouahada/dataset_extracted/DDSM/',
    'BACKBONE': 'vit',
    'EPOCHS': 5,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-4,
    'DENSITY_CLASSES': ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"],
    'CLASS_MAP': {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3},
    'FEATURES_DIR': 'featuresextracted',
    'MODELS_DIR': 'featuresmodels',
    'FINETUNED_WEIGHTS_DIR': 'featuresfinetuned_weights',
    'RESUME_FROM': 'featuresfinetuned_weights/hybrid_model_checkpoint_cnn.pth',
    'FINETUNED_WEIGHTS_PATH': 'featuresfinetuned_weights/cnn_image_branch_best.pth'
}

# --- Dataset unifié avec support data augmentation ---
class HybridMammographyDataset(Dataset):
    """
    Dataset unifié pour charger les mammographies avec support pour data augmentation et histogrammes.
    """
    def __init__(self, annotations_df, image_dir, classes_to_use=None, label_map=None, 
                 use_augmentation=False, split='training'):
        """
        Initialise le dataset avec vérification stricte de la séparation des données.
        
        Args:
            annotations_df: DataFrame contenant les annotations
            image_dir: Répertoire racine des images
            classes_to_use: Liste des classes à utiliser (pour classification binaire)
            label_map: Mapping des labels
            use_augmentation: Activer l'augmentation de données
            split: 'training' ou 'test'
        """
        self.image_dir = image_dir
        self.density_map = {
            "DENSITY A": "density_A", "DENSITY B": "density_B",
            "DENSITY C": "density_C", "DENSITY D": "density_D"
        }
        self.split_map = {"training": "train", "test": "test"}
        self.use_augmentation = use_augmentation and split == 'training'  # Augmentation uniquement sur training
        self.split = split
        
        # Vérification de la cohérence des données
        if 'split' in annotations_df.columns:
            original_split = annotations_df['split'].unique()
            if len(original_split) > 1:
                print(f"ATTENTION: Mélange de splits détecté: {original_split}")
                # Ne garder que les données du split demandé
                annotations_df = annotations_df[annotations_df['split'] == split].copy()

        # Filtrage et remapping des labels
        if classes_to_use:
            print(f"Utilisation des classes: {classes_to_use}")
            time.sleep(5)
            # Mode classification binaire
            self.df = annotations_df[annotations_df['breast_density'].isin(classes_to_use)].copy()
            self.label_remap = {classes_to_use[0]: 0, classes_to_use[1]: 1}
            self.df['label'] = self.df['breast_density'].map(self.label_remap)
        else:
            # Mode classification multiclasse
            self.df = annotations_df.copy()
            if label_map:
                self.df['label'] = self.df['breast_density'].map(label_map)
            else:
                self.df['label'] = self.df['breast_density'].map(CONFIG['CLASS_MAP'])

        self.df = self.df.reset_index(drop=True)
        
        # Transforms pour l'augmentation
        if self.use_augmentation and split == 'training':
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=9),  # Rotation légère
                 # Ajustement luminosité/contraste
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),  # Crop aléatoire
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])  # Normalisation pour 1 canal
            ])
        else:
            # Transforms de base sans augmentation
            self.augmentation_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])  # Normalisation pour 1 canal
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Vérifier si c'est une image augmentée ou originale
        is_augmented = row.get('is_augmented', False)
        
        if is_augmented:
            # Pour les images augmentées, on utilise une transformation aléatoire
            study_id = row['study_id'].replace('_aug', '')
            image_id = row['image_id'].replace('_aug', '')
            density = self.density_map.get(row["breast_density"])
            split = self.split_map.get(row["split"])
            
            # Chemin vers l'image originale
            image_path = os.path.join(self.image_dir, split, density, study_id, f"{image_id}.dicom")
        else:
            # Pour les images originales
            study_id = row['study_id']
            image_id = row['image_id']
            density = self.density_map.get(row["breast_density"])
            split = self.split_map.get(row["split"])
            
            # Chemin vers l'image
            image_path = os.path.join(self.image_dir, split, density, study_id, f"{image_id}.dicom")
        
        try:
            # Chargement et prétraitement de l'image
            image = read_dicom(image_path)
            image = preprocess_image(image, laterality=row["laterality"])
            
            # S'assurer que l'image est en 2D (niveaux de gris)
            if len(image.shape) == 3:
                # Si l'image a 3 dimensions, prendre le premier canal
                image = image[:, :, 0]
            elif len(image.shape) > 3:
                # Pour les images 4D, prendre la première dimension et le premier canal
                if len(image.shape) == 4:
                    image = image[0, :, :, 0]  # Prendre (0, :, :, 0)
                else:
                    image = image.squeeze()
            
            # Conversion en PIL Image en niveaux de gris
            image = Image.fromarray(image.astype(np.uint8), mode='L')
            
            # Application des transformations
            image = self.augmentation_transforms(image)
            
            # Création de l'histogramme
            hist = create_histogram_from_image(image.unsqueeze(0))  # Ajouter dimension batch
            hist = hist.squeeze(0)  # Retirer dimension batch
            
            # Label
            label = row['label']
            
            return image, hist, label
            
        except Exception as e:
            print(f"Erreur lors du chargement de {image_path}: {e}")
            # Retourner des données factices en cas d'erreur
            dummy_image = torch.randn(1, 224, 224)
            dummy_hist = torch.randn(256)
            return dummy_image, dummy_hist, 0
        


class HybridModelManager:
    """
    Gestionnaire pour l'entraînement du modèle hybride.
    """
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
        self.model = None
        
    def create_model(self, input_channels=1, image_feature_dim=512, hist_hidden_dims=[128, 64],
                    num_classes=4, dropout=0.3, pretrained=True, finetuned_weights_path=None):
        """Crée le modèle hybride."""
        self.model = HybridMammographyClassifier(
            backbone=self.backbone,
            input_channels=input_channels,
            image_feature_dim=image_feature_dim,
            hist_hidden_dims=hist_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            finetuned_weights_path=finetuned_weights_path #chemin vers les poids fine-tunés de la branche image
        )
        return self.model
    
    def train_model(self, data_csv, image_root, epochs=50, batch_size=8, lr=1e-4,
                   save_dir=None, patience=10, min_delta=0.001,
                   use_augmentation=True, resume_from=None, finetuned_weights_path=None):
        """
        Entraîne le modèle hybride.
        """
        if save_dir is None:
            save_dir = CONFIG['FINETUNED_WEIGHTS_DIR']
        
        # Chargement des données
        df = pd.read_csv(data_csv)
        
        # S'assurer que les données de test sont complètement séparées
        train_df = df[df['split'] == 'training'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        # Garder les données de test complètement séparées pour l'évaluation finale
        
        # Création d'un ensemble de validation à partir des données d'entraînement uniquement
        train_indices = train_df.index.tolist()
        np.random.seed(42)  # Pour la reproductibilité
        val_size = int(0.2 * len(train_indices))
        val_indices = np.random.choice(train_indices, size=val_size, replace=False)
        train_indices = list(set(train_indices) - set(val_indices))
        
        # Créer une nouvelle colonne pour le split train/val
        train_df.loc[train_indices, 'temp_split'] = 'training'
        train_df.loc[val_indices, 'temp_split'] = 'validation'
        
        print(f"Distribution des données:")
        print(f"  - Entraînement: {len(train_indices)} images")
        print(f"  - Validation: {len(val_indices)} images")
        print(f"  - Test (non utilisé pendant l'entraînement): {len(test_df)} images")
        time.sleep(10)
        # Création des datasets avec une séparation stricte
        train_dataset = HybridMammographyDataset(
            train_df[train_df['temp_split'] == 'training'].copy(), 
            image_root, 
            label_map=CONFIG['CLASS_MAP'],
            use_augmentation=use_augmentation, 
            split='training'
        )
        time.sleep(10)
        val_dataset = HybridMammographyDataset(
            train_df[train_df['temp_split'] == 'validation'].copy(), 
            image_root, 
            label_map=CONFIG['CLASS_MAP'],
            use_augmentation=False,  # Pas d'augmentation sur la validation
            split='training'
        )
        
        # Vérification qu'il n'y a pas de chevauchement entre train et validation
        train_ids = set(train_df[train_df['temp_split'] == 'training']['image_id'])
        val_ids = set(train_df[train_df['temp_split'] == 'validation']['image_id'])
        overlap = train_ids.intersection(val_ids)
        if overlap:
            print("ATTENTION: Chevauchement détecté entre train et validation!")
            print(f"Images en double: {overlap}")
        
        # Création des dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=0)
        
        print(f"Train: {len(train_dataset)} images")
        print(f"Val: {len(val_dataset)} images")
        time.sleep(10)
        # Création du modèle
        model = self.create_model(finetuned_weights_path=finetuned_weights_path) #chemin vers les poids fine-tunés de la branche image
        model.to(self.device)
        time.sleep(10)
        print(f"Modèle créé avec {sum(p.numel() for p in model.parameters()):,} paramètres")
        time.sleep(10)
        # Reprise d'entraînement si spécifié
        if resume_from and os.path.exists(resume_from):
            print(f"Reprise depuis: {resume_from}")
            time.sleep(5)
            checkpoint = torch.load(resume_from, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Checkpoint chargé (epoch {checkpoint.get('epoch', 0)})")
            if finetuned_weights_path:
                print(f"Chargement des poids fine-tunés de la branche image apres le checkpoint: {finetuned_weights_path}")
                time.sleep(5)
                model.image_branch.load_finetuned_weights(finetuned_weights_path, device=self.device)
        
        # Entraînement
        trained_model = self._train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, save_dir=save_dir,
            patience=patience, min_delta=min_delta
        )
        
        return trained_model
    
    def _train_model(self, model, train_loader, val_loader, epochs=50, lr=1e-4,
                    save_dir='featuresfinetuned_weights', patience=10, min_delta=0.001):
        """
        Entraînement du modèle hybride avec validation stricte.
        Note: Les données de test ne sont JAMAIS utilisées pendant l'entraînement.
        """
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        # Créer le dossier de sauvegarde
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[ENTRAÎNEMENT] Début du fine-tuning pour {epochs} époques")
        print(f"[ENTRAÎNEMENT] Learning rate: {lr}")
        print(f"[ENTRAÎNEMENT] Device: {self.device}")
        
        for epoch in range(epochs):
            # Entraînement
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs}")
            for batch_idx, (images, hists, labels) in enumerate(progress_bar):
                images, hists, labels = images.to(self.device), hists.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images, hists)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_acc = 100. * train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, hists, labels in val_loader:
                    images, hists, labels = images.to(self.device), hists.to(self.device), labels.to(self.device)
                    outputs = model(images, hists)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            print(f"\n[ÉPOQUE {epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Sauvegarde du meilleur modèle
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Sauvegarde du meilleur modèle
                best_model_path = os.path.join(save_dir, f'hybrid_model_best_{self.backbone}.pth')
                model.save_finetuned_weights(best_model_path)
                
                # Sauvegarde du checkpoint pour reprise
                checkpoint_path = os.path.join(save_dir, f'hybrid_model_checkpoint_{self.backbone}.pth')
                torch.save({
                    'backbone_name': self.backbone,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_val_acc,
                    'patience_counter': patience_counter
                }, checkpoint_path)
                
                print(f"  Nouveau meilleur modèle! Acc: {val_acc:.2f}%")
                print(f"  Modèle sauvegardé: {best_model_path}")
            else:
                patience_counter += 1
                print(f"  ⏳ Pas d'amélioration ({patience_counter}/{patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Arrêt après {patience} époques sans amélioration")
                break
        
        # Sauvegarde finale
        final_save_path = os.path.join(save_dir, f'hybrid_model_final_{self.backbone}.pth')
        model.save_finetuned_weights(final_save_path)
        
        print(f"\n[FIN] Entraînement terminé.")
        print(f"  - Meilleure accuracy: {best_val_acc:.4f}")
        print(f"  - Modèle final: {final_save_path}")
        print(f"  - Meilleur modèle: {os.path.join(save_dir, f'hybrid_model_best_{self.backbone}.pth')}")
        
        return model


def main():
    parser = argparse.ArgumentParser(description='Entraînement du modèle hybride avec fine-tuning')
    parser.add_argument('--data_csv', default=CONFIG['ANNOTATIONS_CSV'], help='Chemin vers le CSV d\'annotations')
    parser.add_argument('--image_root', default=CONFIG['IMAGE_ROOT'], help='Chemin vers les images')
    parser.add_argument('--backbone', default=CONFIG['BACKBONE'], choices=['cnn', 'vit'], help='Backbone pour l\'image')
    parser.add_argument('--epochs', type=int, default=CONFIG['EPOCHS'], help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=CONFIG['BATCH_SIZE'], help='Taille du batch')
    parser.add_argument('--lr', type=float, default=CONFIG['LEARNING_RATE'], help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda)')
    parser.add_argument('--use_augmentation', action='store_true', help='Utiliser la data augmentation')
    parser.add_argument('--resume_from', default=None, help='Chemin vers un checkpoint pour reprendre l\'entraînement')
    parser.add_argument('--finetuned_weights_path', default=None, help='Chemin vers les poids fine-tunés de la branche image')
    
    args = parser.parse_args()
    
    # Configuration du device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"=== Entraînement du modèle hybride ===")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Data augmentation: {'Activée' if args.use_augmentation else 'Désactivée'}")
    
    try:
        # Création du gestionnaire
        manager = HybridModelManager(args.backbone, device)
        
        # Entraînement
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
        
        print(f"\n Entraînement terminé!")
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 