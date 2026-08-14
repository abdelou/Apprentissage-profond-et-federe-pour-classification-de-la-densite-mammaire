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

from hybrid_model import SiameseDoubleBranchClassifier
from preprocessing import read_dicom, preprocess_image

# --- Configuration globale ---
VINDR_ROOT = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
CONFIG = {
    'ANNOTATIONS_CSV': os.path.join(VINDR_ROOT, 'breast-level_annotations.csv'),
    'IMAGE_ROOT_CC': VINDR_ROOT,
    'IMAGE_ROOT_MLO': VINDR_ROOT,
    'EPOCHS': 10,
    'BACKBONE': 'efficientnet_b0',  # branche unique à poids partagés (Siamese)
    'BATCH_SIZE': 10,
    'LEARNING_RATE': 1e-4,
    'DENSITY_CLASSES': ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"],
    'CLASS_MAP': {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3},
    'FEATURES_DIR': 'featuresextracted',
    'MODELS_DIR': 'featuresmodels',
    'FINETUNED_WEIGHTS_DIR': 'featuresfinetuned_weights',
    'SHARED_WEIGHTS_PATH': "featuresfinetuned_weights/efficientnet_b0_shared_branch_best.pth",
}

# --- Dataset unifié avec support data augmentation ---
class HybridMammographyDataset(Dataset):
    
    def __init__(self, annotations_df, image_dir_cc, image_dir_mlo, classes_to_use=None, label_map=None, 
                 use_augmentation=False, split='training'):
      
        self.image_dir_cc = image_dir_cc
        self.image_dir_mlo = image_dir_mlo
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
        # Regrouper par study_id et laterality pour trouver les couples CC/MLO
        self.pairs = []
        grouped = self.df.groupby(['study_id', 'laterality'])
        for (study_id, laterality), group in grouped:
            cc_row = group[group['view_position'] == 'CC']
            mlo_row = group[group['view_position'] == 'MLO']
            if not cc_row.empty and not mlo_row.empty:
                cc_row = cc_row.iloc[0]
                mlo_row = mlo_row.iloc[0]
                label = cc_row['label']  # ou mlo_row['label'], c'est le même sein
                pair_entry = {
                    'study_id': study_id,
                    'laterality': laterality,
                    'cc_image_id': cc_row['image_id'],
                    'mlo_image_id': mlo_row['image_id'],
                    'density': cc_row['breast_density'],
                    'split': cc_row['split'],
                    'label': label
                }
                self.pairs.append(pair_entry)
        
        # --- Suréchantillonnage ciblé des classes minoritaires (A et D) pour l'entraînement ---
        if self.split == 'training':
            initial_count = len(self.pairs)
            augmented_pairs = []
            for pair in self.pairs:
                augmented_pairs.append(pair)
                # Classe A (label 0) : multiplier par 10 pour rééquilibrer
                if pair['label'] == 0:
                    for _ in range(9):
                        augmented_pairs.append(pair)
                # Classe D (label 3) : multiplier par 2
                elif pair['label'] == 3:
                    for _ in range(1):
                        augmented_pairs.append(pair)
            self.pairs = augmented_pairs
            print(f"[DATASET] Suréchantillonnage ciblé appliqué : {initial_count} couples -> {len(self.pairs)} couples d'entraînement.")
        
        # Transforms pour l'augmentation anatomique (Inspiré de la recherche médicale)
        if self.use_augmentation and split == 'training':
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),             # Miroir horizontal (inter-patientes)
                transforms.RandomRotation(degrees=10),              # Rotation légère (+/- 10°)
                transforms.ColorJitter(brightness=0.1, contrast=0.1),# Ajustement exposition (+/- 10%)
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)), # Crop aléatoire
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])     # Normalisation
            ])
        else:
            # Transforms de base sans augmentation
            self.augmentation_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])     # Normalisation
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        study_id = pair['study_id']
        laterality = pair['laterality']
        density = self.density_map.get(pair['density'])
        split = self.split_map.get(pair['split'])
        cc_image_id = pair['cc_image_id']
        mlo_image_id = pair['mlo_image_id']
        label = pair['label']

        # Chemins vers les images CC et MLO (avec fallbacks pour le format standard VinDr-Mammo)
        cc_img, mlo_img = None, None
        cc_path = os.path.join(self.image_dir_cc, split, density, study_id, f"{cc_image_id}.dicom")
        if not os.path.exists(cc_path):
            cc_path = os.path.join(self.image_dir_cc, 'images', study_id, f"{cc_image_id}.dicom")
            if not os.path.exists(cc_path):
                cc_path = os.path.join(self.image_dir_cc, study_id, f"{cc_image_id}.dicom")

        mlo_path = os.path.join(self.image_dir_mlo, split, density, study_id, f"{mlo_image_id}.dicom")
        if not os.path.exists(mlo_path):
            mlo_path = os.path.join(self.image_dir_mlo, 'images', study_id, f"{mlo_image_id}.dicom")
            if not os.path.exists(mlo_path):
                mlo_path = os.path.join(self.image_dir_mlo, study_id, f"{mlo_image_id}.dicom")

        # Chargement CC
        try:
            cc_img_raw = read_dicom(cc_path)
            cc_img_raw = preprocess_image(cc_img_raw, laterality=laterality)
            if len(cc_img_raw.shape) == 3:
                cc_img_raw = cc_img_raw[:, :, 0]
            elif len(cc_img_raw.shape) > 3:
                if len(cc_img_raw.shape) == 4:
                    cc_img_raw = cc_img_raw[0, :, :, 0]
                else:
                    cc_img_raw = cc_img_raw.squeeze()
            cc_img = Image.fromarray(cc_img_raw.astype(np.uint8), mode='L')
            cc_img = self.augmentation_transforms(cc_img)
        except Exception as e:
            cc_img = None
            print(f"Vue CC absente ou erreur: {cc_path} ({e})")

        # Chargement MLO
        try:
            mlo_img_raw = read_dicom(mlo_path)
            mlo_img_raw = preprocess_image(mlo_img_raw, laterality=laterality)
            if len(mlo_img_raw.shape) == 3:
                mlo_img_raw = mlo_img_raw[:, :, 0]
            elif len(mlo_img_raw.shape) > 3:
                if len(mlo_img_raw.shape) == 4:
                    mlo_img_raw = mlo_img_raw[0, :, :, 0]
                else:
                    mlo_img_raw = mlo_img_raw.squeeze()
            mlo_img = Image.fromarray(mlo_img_raw.astype(np.uint8), mode='L')
            mlo_img = self.augmentation_transforms(mlo_img)
        except Exception as e:
            mlo_img = None
            print(f"Vue MLO absente ou erreur: {mlo_path} ({e})")

        # Si une vue est absente, dupliquer l'autre
        if cc_img is None and mlo_img is not None:
            cc_img = mlo_img.clone()
        if mlo_img is None and cc_img is not None:
            mlo_img = cc_img.clone()
        if cc_img is None and mlo_img is None:
            print(f"Aucune vue disponible pour {study_id} {laterality}, sélection d'images aléatoires du même label.")
            # Charger les fichiers d'annotations externes avec gestion robuste des chemins
            import random
            mlo_csv = 'DDSM/output_annotations_MLO.csv' if os.path.exists('DDSM/output_annotations_MLO.csv') else '../DDSM/output_annotations_MLO.csv'
            cc_csv = 'DDSM/output_annotations.csv' if os.path.exists('DDSM/output_annotations.csv') else '../DDSM/output_annotations.csv'
            
            if os.path.exists(mlo_csv):
                mlo_annots = pd.read_csv(mlo_csv)
            else:
                mlo_annots = pd.DataFrame()

            if os.path.exists(cc_csv):
                cc_annots = pd.read_csv(cc_csv)
            else:
                cc_annots = pd.DataFrame()
            # Filtrer par label
            label_name = None
            for k, v in CONFIG['CLASS_MAP'].items():
                if v == label:
                    label_name = k
                    break
            # Sélectionner une image MLO aléatoire du même label
            mlo_candidates = mlo_annots[mlo_annots['breast_density'] == label_name]
            if not mlo_candidates.empty:
                mlo_row = mlo_candidates.sample(1).iloc[0]
                mlo_path_rand = os.path.join(self.image_dir_mlo, self.split_map.get(mlo_row['split'], 'train'), self.density_map.get(mlo_row['breast_density']), mlo_row['study_id'], f"{mlo_row['image_id']}.dicom")
                try:
                    mlo_img_raw = read_dicom(mlo_path_rand)
                    mlo_img_raw = preprocess_image(mlo_img_raw, laterality=mlo_row['laterality'])
                    if len(mlo_img_raw.shape) == 3:
                        mlo_img_raw = mlo_img_raw[:, :, 0]
                    elif len(mlo_img_raw.shape) > 3:
                        if len(mlo_img_raw.shape) == 4:
                            mlo_img_raw = mlo_img_raw[0, :, :, 0]
                        else:
                            mlo_img_raw = mlo_img_raw.squeeze()
                    mlo_img = Image.fromarray(mlo_img_raw.astype(np.uint8), mode='L')
                    mlo_img = self.augmentation_transforms(mlo_img)
                except Exception as e:
                    print(f"Impossible de charger MLO random: {mlo_path_rand} ({e})")
                    mlo_img = torch.randn(1, 224, 224)
            else:
                mlo_img = torch.randn(1, 224, 224)
            # Sélectionner une image CC aléatoire du même label
            cc_candidates = cc_annots[cc_annots['breast_density'] == label_name]
            if not cc_candidates.empty:
                cc_row = cc_candidates.sample(1).iloc[0]
                cc_path_rand = os.path.join(self.image_dir_cc, self.split_map.get(cc_row['split'], 'train'), self.density_map.get(cc_row['breast_density']), cc_row['study_id'], f"{cc_row['image_id']}.dicom")
                try:
                    cc_img_raw = read_dicom(cc_path_rand)
                    cc_img_raw = preprocess_image(cc_img_raw, laterality=cc_row['laterality'])
                    if len(cc_img_raw.shape) == 3:
                        cc_img_raw = cc_img_raw[:, :, 0]
                    elif len(cc_img_raw.shape) > 3:
                        if len(cc_img_raw.shape) == 4:
                            cc_img_raw = cc_img_raw[0, :, :, 0]
                        else:
                            cc_img_raw = cc_img_raw.squeeze()
                    cc_img = Image.fromarray(cc_img_raw.astype(np.uint8), mode='L')
                    cc_img = self.augmentation_transforms(cc_img)
                except Exception as e:
                    print(f"Impossible de charger CC random: {cc_path_rand} ({e})")
                    cc_img = torch.randn(1, 224, 224)
            else:
                cc_img = torch.randn(1, 224, 224)

        return mlo_img, cc_img, label
        


class HybridModelManager:
    """
    Gestionnaire pour l'entraînement du modèle hybride.
    """
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
        self.model = None
        
    def create_model(self, input_channels=1, image_feature_dim=512,
                    num_classes=4, dropout=0.3, pretrained=True,
                    shared_weights=None):
        """Crée le modèle Siamese à branche unique EfficientNet-B0 partagée (CC et MLO)."""
        self.model = SiameseDoubleBranchClassifier(
            backbone=self.backbone,
            input_channels=input_channels,
            image_feature_dim=image_feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            shared_weights=shared_weights
        )
        return self.model

    def train_model(self, data_csv, image_root_cc, image_root_mlo, epochs=50, batch_size=8, lr=1e-4,
                   save_dir=None, patience=10, min_delta=0.001,
                   use_augmentation=True, resume_from=None,
                   shared_weights=None,
                   freeze_shared_branch=False):
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
        
        # Création des datasets avec une séparation stricte
        train_dataset = HybridMammographyDataset(
            train_df[train_df['temp_split'] == 'training'].copy(),
            image_root_cc,
            image_root_mlo,
            label_map=CONFIG['CLASS_MAP'],
            use_augmentation=use_augmentation,
            split='training'
        )
        val_dataset = HybridMammographyDataset(
            train_df[train_df['temp_split'] == 'validation'].copy(),
            image_root_cc,
            image_root_mlo,
            label_map=CONFIG['CLASS_MAP'],
            use_augmentation=False,
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
                                shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
        
        print(f"Train: {len(train_dataset)} images")
        print(f"Val: {len(val_dataset)} images")
        
        # Création du modèle Siamese (branche unique partagée entre CC et MLO)
        model = self.create_model(
            input_channels=1,
            image_feature_dim=512,
            num_classes=4,
            dropout=0.3,
            pretrained=True,
            shared_weights=shared_weights
        )
        model.to(self.device)
        # Geler la branche partagée si demandé (désactivé par défaut: contrairement au
        # modèle hétérogène, il n'y a pas de pré-fine-tuning séparé de branche EfficientNet-B0
        # ici, donc on entraîne bout-en-bout depuis ImageNet directement).
        if freeze_shared_branch:
            print("[INFO] Gel de la branche partagée EfficientNet-B0")
            model.freeze_shared_branch()
        print(f"Modèle créé avec {sum(p.numel() for p in model.parameters()):,} paramètres")
        
        # Reprise d'entraînement si disponible
        checkpoint_data = None
        if resume_from and os.path.exists(resume_from):
            print(f"[REPRISE] Chargement du checkpoint spécifié: {resume_from}")
            checkpoint_data = torch.load(resume_from, map_location=self.device)
        else:
            default_ckpt = os.path.join(save_dir, f'hybrid_model_checkpoint_{self.backbone}_2branches.pth')
            if os.path.exists(default_ckpt):
                print(f"[REPRISE] Checkpoint automatique trouvé: {default_ckpt}")
                checkpoint_data = torch.load(default_ckpt, map_location=self.device)

        # Entraînement
        trained_model = self._train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, save_dir=save_dir,
            patience=patience, min_delta=min_delta,
            checkpoint_data=checkpoint_data
        )
        return trained_model
    
    def _train_model(self, model, train_loader, val_loader, epochs=50, lr=1e-4,
                    save_dir='featuresfinetuned_weights', patience=10, min_delta=0.001,
                    checkpoint_data=None):
        """
        Entraînement du modèle hybride avec validation stricte et reprise automatique.
        Note: Les données de test ne sont JAMAIS utilisées pendant l'entraînement.
        """
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        start_epoch = 0
        best_val_acc = 0.0
        patience_counter = 0

        # Restauration du checkpoint si disponible
        if checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            start_epoch = checkpoint_data.get('epoch', -1) + 1
            best_val_acc = checkpoint_data.get('best_accuracy', 0.0)
            patience_counter = checkpoint_data.get('patience_counter', 0)
            print(f"  Checkpoint restauré! Reprise à l'époque {start_epoch + 1}/{epochs} (Meilleure accuracy: {best_val_acc:.2f}%)")

        # Créer le dossier de sauvegarde
        os.makedirs(save_dir, exist_ok=True)

        print(f"[ENTRAÎNEMENT] Début du fine-tuning de l'époque {start_epoch + 1} à {epochs}")
        print(f"[ENTRAÎNEMENT] Learning rate: {lr}")
        print(f"[ENTRAÎNEMENT] Device: {self.device}")

        for epoch in range(start_epoch, epochs):
            # Entraînement
            model.train()
            train_loss = 0.0
            train_total = 0
            train_correct = 0
            train_true = []
            train_pred = []

            progress_bar = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs}")
            for batch_idx, (mlo_images, cc_images, labels) in enumerate(progress_bar):
                mlo_images, cc_images, labels = mlo_images.to(self.device), cc_images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                # Forward à deux branches
                logits = model(mlo_images, cc_images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                train_true.extend(labels.cpu().numpy())
                train_pred.extend(predicted.cpu().numpy())

                # Accuracy batch
                batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
                progress_bar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'Acc': f'{batch_acc:.2f}%'
                })

            train_acc = 100. * train_correct / train_total
            from sklearn.metrics import recall_score
            train_recall = recall_score(train_true, train_pred, average='macro', zero_division=0)

            # Validation
            model.eval()
            val_loss = 0.0
            val_total = 0
            val_correct = 0
            all_true = []
            all_pred_final = []
            all_pred_stage1 = []
            all_pred_ab = []
            all_pred_cd = []

            with torch.no_grad():
                for mlo_images, cc_images, labels in val_loader:
                    mlo_images, cc_images, labels = mlo_images.to(self.device), cc_images.to(self.device), labels.to(self.device)
                    logits = model(mlo_images, cc_images)
                    pred_final = logits.argmax(dim=1)
                    val_loss += criterion(logits, labels).item()
                    val_total += labels.size(0)
                    val_correct += pred_final.eq(labels).sum().item()
                    all_true.extend(labels.cpu().numpy())
                    all_pred_final.extend(pred_final.cpu().numpy())


            val_acc = 100. * val_correct / val_total

            # Learning rate scheduling
            scheduler.step(val_acc)

            print(f"\n[ÉPOQUE {epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Train Recall: {100.*train_recall:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Écriture dans le fichier log à chaque époque
            with open("finetuning_epoch_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"[ÉPOQUE {epoch+1}/{epochs}]\n")
                log_file.write(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%\n")
                log_file.write(f"  Train Recall: {100.*train_recall:.2f}%\n")
                log_file.write(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%\n")
                log_file.write(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n\n")

            # Matrices de confusion
            import numpy as np
            from sklearn.metrics import confusion_matrix



            print("\nMatrice de confusion finale à l'époque (A/B/C/D):")
            print(confusion_matrix(all_true, all_pred_final, labels=[0, 1, 2, 3]))

            # Sauvegarde du meilleur modèle
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0

                # Sauvegarde du meilleur modèle
                best_model_path = os.path.join(save_dir, f'hybrid_model_best_{self.backbone}_2branches.pth')
                model.save_finetuned_weights(best_model_path)

                # Sauvegarde du checkpoint pour reprise
                checkpoint_path = os.path.join(save_dir, f'hybrid_model_checkpoint_{self.backbone}_2branches.pth')
                torch.save({
                    'backbone_name': self.backbone,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_val_acc,
                    'patience_counter': patience_counter
                }, checkpoint_path)

                print(f"   Nouveau meilleur modèle! Acc: {val_acc:.2f}%")
                print(f"   Modèle sauvegardé: {best_model_path}")
            else:
                patience_counter += 1
                print(f"   Pas d'amélioration ({patience_counter}/{patience})")

            # Early stopping
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Arrêt après {patience} époques sans amélioration")
                break

        # Sauvegarde finale
        final_save_path = os.path.join(save_dir, f'hybrid_model_final_{self.backbone}_2branches.pth')
        model.save_finetuned_weights(final_save_path)

        print(f"\n[FIN] Entraînement terminé du modèle hybride avec rexnet 150 sur CC ")
        print(f"  - Meilleure accuracy: {best_val_acc:.4f}")
        print(f"  - Modèle final: {final_save_path}")
        print(f"  - Meilleur modèle: {os.path.join(save_dir, f'hybrid_model_best_{self.backbone}_2branches.pth')}")

        return model


def main():
    parser = argparse.ArgumentParser(description='Entraînement du modèle hybride avec fine-tuning')
    parser.add_argument('--data_csv', default=CONFIG['ANNOTATIONS_CSV'], help='Chemin vers le CSV d\'annotations')
    parser.add_argument('--image_root_mlo', default=CONFIG['IMAGE_ROOT_MLO'], help='Chemin vers les images')
    parser.add_argument('--image_root_cc', default=CONFIG['IMAGE_ROOT_CC'], help='Chemin vers les images')
    parser.add_argument('--backbone', default=CONFIG['BACKBONE'], choices=['efficientnet_b0'], help='Backbone partagé (Siamese)')
    parser.add_argument('--epochs', type=int, default=CONFIG['EPOCHS'], help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=CONFIG['BATCH_SIZE'], help='Taille du batch')
    parser.add_argument('--lr', type=float, default=CONFIG['LEARNING_RATE'], help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda)')
    parser.add_argument('--use_augmentation', action='store_true', help='Utiliser la data augmentation')
    parser.add_argument('--resume_from', default=None, help='Chemin vers un checkpoint pour reprendre l\'entraînement')
   
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
            image_root_cc=args.image_root_cc, 
            image_root_mlo=args.image_root_mlo,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=CONFIG['FINETUNED_WEIGHTS_DIR'],
            lr=args.lr,
            use_augmentation=args.use_augmentation,
            resume_from=args.resume_from,
            shared_weights=None,
        )
        
        print(f"\n Entraînement terminé!")
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 