import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from tqdm import tqdm
import numpy as np
import argparse
import torchvision.transforms as transforms
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import time
import random

def set_seed(seed=42):
    """Fixe la seed globale pour garantir une reproductibilité stricte 100% déterministe."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

from models import FeatureExtractor, BinaryMLPClassifier, MLPClassifier, HierarchicalClassifier, FeatureDumper, FeatureMLPTrainer, FineTunedFeatureExtractor
from preprocessing import read_dicom, preprocess_image

# --- Configuration globale ---
VINDR_ROOT = '/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0'
CONFIG = {
    'ANNOTATIONS_CSV': os.path.join(VINDR_ROOT, 'breast-level_annotations.csv'),
    'IMAGE_ROOT': VINDR_ROOT,
    'BACKBONE': 'twins',
    'EPOCHS': 30,
    'BATCH_SIZE': 10,
    'LEARNING_RATE': 1e-4,
    'DENSITY_CLASSES': ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"],
    'CLASS_MAP': {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3},
    'FEATURES_DIR': 'featuresextracted',
    'MODELS_DIR': 'featuresmodels',
    'FINETUNED_WEIGHTS_DIR': 'featuresfinetuned_weights'
}

# --- Dataset unifié avec support data augmentation ---
class MammographyDataset(Dataset):
    """
    Dataset unifié pour charger les mammographies avec support pour classification binaire, multiclasse et data augmentation.
    """
    def __init__(self, annotations_df, image_dir, classes_to_use=None, label_map=None, 
                 use_augmentation=False, split='training'):
        self.image_dir = image_dir
        self.density_map = {
            "DENSITY A": "density_A", "DENSITY B": "density_B",
            "DENSITY C": "density_C", "DENSITY D": "density_D"
        }
        self.split_map = {"training": "train", "test": "test"}
        self.use_augmentation = use_augmentation
        self.split = split

        # Filtrage et remapping des labels
        if classes_to_use:
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
                raise ValueError("label_map doit être spécifié pour la classification multiclasse.")

        self.df = self.df.reset_index(drop=True)
        
        # Transforms pour l'augmentation
        if self.use_augmentation and split == 'training':
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=5),  # Rotation légère
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Ajustement luminosité/contraste
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),  # Crop aléatoire
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
            ])
        else:
            # Transforms de base sans augmentation
            self.augmentation_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Vérifier si c'est une image augmentée ou originale
        is_aug_val = str(row.get('is_augmented', '')).lower()
        is_augmented = is_aug_val == 'true' or '_aug' in str(row['image_id']) or '_aug' in str(row['study_id'])
        
        if is_augmented:
            # Pour les images augmentées, on utilise une transformation aléatoire
            study_id = row['study_id'].replace('_aug', '')
            image_id = row['image_id'].replace('_aug', '')
            density = self.density_map.get(row["breast_density"])
            split = self.split_map.get(row["split"])
            
            # Chemin vers l'image originale (chemin organisé par défaut)
            image_path = os.path.join(self.image_dir, split, density, study_id, f"{image_id}.dicom")
            
            # Fallback pour le format standard VinDr-Mammo (sans organized_layout)
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_dir, 'images', study_id, f"{image_id}.dicom")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.image_dir, study_id, f"{image_id}.dicom")

            # Lecture et prétraitement
            laterality = row['laterality']
            image = read_dicom(image_path)
            image = preprocess_image(image, laterality=laterality)
                
            # Conversion en PIL
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
                
            # Application des transformations d'augmentation
            image_tensor = self.augmentation_transforms(image_pil)
                
        else:
        
        
            # Pour les images originales
            split = self.split_map.get(row["split"])
            density = self.density_map.get(row["breast_density"])
            study_id = str(row["study_id"])
            image_id = str(row["image_id"])
            
            # Chemin vers l'image (chemin organisé par défaut)
            image_path = os.path.join(self.image_dir, split, density, study_id, f"{image_id}.dicom")
            
            # Fallback pour le format standard VinDr-Mammo (sans organized_layout)
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_dir, 'images', study_id, f"{image_id}.dicom")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.image_dir, study_id, f"{image_id}.dicom")

            # Lecture et prétraitement
            laterality = row['laterality']
            image = read_dicom(image_path)
            image = preprocess_image(image, laterality=laterality)
            
            # Conversion en PIL
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            
            # Application des transformations
            image_tensor = self.augmentation_transforms(image_pil)

        label = torch.tensor(row['label'], dtype=torch.long)
        return image_tensor, label

# --- Gestionnaire de features ---
class FeatureManager:
    """Gestionnaire centralisé pour l'extraction et la sauvegarde des features."""
    
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
        self.finetuned_weights_path = f'{CONFIG["FINETUNED_WEIGHTS_DIR"]}/{backbone}_finetuned_best.pth'
        
    def get_feature_extractor(self, use_finetuned=True):
        """Retourne le feature extractor approprié."""
        if use_finetuned and os.path.exists(self.finetuned_weights_path):
            print(f"[FeatureManager] Utilisation des poids fine-tunés: {self.finetuned_weights_path}")
            return FeatureExtractor(self.backbone, pretrained=True, finetuned_weights_path=self.finetuned_weights_path).to(self.device)
        else:
                print(f"[FeatureManager] Utilisation du modèle pré-entraîné sans les poids fine-tunés")
                return FeatureExtractor(self.backbone, pretrained=True).to(self.device)
        
    def get_feature_paths(self, use_finetuned=True):
        """Retourne les chemins des fichiers de features."""
        suffix = "finetuned" if use_finetuned else ""
        return {
            'train_features': f'{CONFIG["FEATURES_DIR"]}/{self.backbone}_{suffix}_features_train.npy',
            'train_labels': f'{CONFIG["FEATURES_DIR"]}/{self.backbone}_{suffix}_labels_train.npy',
            'test_features': f'{CONFIG["FEATURES_DIR"]}/{self.backbone}_{suffix}_features_test.npy',
            'test_labels': f'{CONFIG["FEATURES_DIR"]}/{self.backbone}_{suffix}_labels_test.npy'
        }
    
    def extract_features_if_needed(self, df, image_root, use_finetuned=True):
        """Extrait les features si elles n'existent pas déjà."""
        paths = self.get_feature_paths(use_finetuned)
        
        # Créer le dossier features s'il n'existe pas
        os.makedirs(CONFIG['FEATURES_DIR'], exist_ok=True)
        
        # Vérification et extraction pour le training
        if not (os.path.exists(paths['train_features']) and os.path.exists(paths['train_labels'])):
            print(f"[FeatureManager] Les fichiers {paths['train_features']} et {paths['train_labels']} n'existent pas, on fait l'extraction.")
            print(f"[FeatureManager] Extraction des features de training...")
            feature_extractor = self.get_feature_extractor(use_finetuned)
            dumper = FeatureDumper(feature_extractor, self.device)
            dumper.dump_features(df, image_root, paths['train_features'], paths['train_labels'], 
                               label_map=CONFIG['CLASS_MAP'], split='training')
        
        # Vérification et extraction pour le test
        if not (os.path.exists(paths['test_features']) and os.path.exists(paths['test_labels'])):
            print(f"[FeatureManager] Les fichiers {paths['test_features']} et {paths['test_labels']} n'existent pas, on fait l'extraction.")
            print(f"[FeatureManager] Extraction des features de test...")
            feature_extractor = self.get_feature_extractor(use_finetuned)
            dumper = FeatureDumper(feature_extractor, self.device)
            dumper.dump_features(df, image_root, paths['test_features'], paths['test_labels'], 
                               label_map=CONFIG['CLASS_MAP'], split='test')
        
        return paths

# --- Entraîneur unifié ---
class ModelTrainer:
    """Entraîneur unifié pour tous les types de modèles."""
    
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
        self.feature_manager = FeatureManager(backbone, device)
    
    def train_mlp4_classifier(self, data_csv, image_root, use_finetuned=True):
        """Entraîne le classifieur MLP 4 classes."""
        print(f"--- Entraînement du classifieur 4 classes ---")
        
        # Extraction des features
        df = pd.read_csv(data_csv)
        paths = self.feature_manager.extract_features_if_needed(df, image_root, use_finetuned)
        
        # Chargement des features
        features = np.load(paths['train_features'])
        labels = np.load(paths['train_labels'])
        print(f"[train_mlp4_classifier] Features shape: {features.shape}, Labels shape: {labels.shape}")
        
        # Création et entraînement du modèle
        in_dim = features.shape[1]
        mlp = MLPClassifier(in_dim=in_dim, out_dim=4)
        trainer = FeatureMLPTrainer(mlp)
        if use_finetuned:
            model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_finetuned_mlp4.pth'
        else:
            model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_mlp4.pth'
        if os.path.exists(model_path):
            print(f"[train_mlp4_classifier] Reprise de l'entraînement depuis {model_path}")
            trainer.load(model_path, device=self.device)
        
        trainer.train(features, labels, epochs=CONFIG['EPOCHS'], 
                     batch_size=CONFIG['BATCH_SIZE'], lr=CONFIG['LEARNING_RATE'], device=self.device)
        trainer.save(model_path)
        print(f"Modèle 4 classes sauvegardé : {model_path}")
    
    def train_binary_classifier(self, class_pair, data_csv, image_root, use_finetuned=True):
        """Entraîne un classifieur binaire pour une paire de classes."""
        print(f"--- Entraînement binaire {class_pair[0]} vs {class_pair[1]} ---")
        
        # Extraction des features
        df = pd.read_csv(data_csv)
        paths = self.feature_manager.extract_features_if_needed(df, image_root, use_finetuned)
        
        # Chargement et filtrage des features
        features = np.load(paths['train_features'])
        labels = np.load(paths['train_labels'])
        
        # Filtrage pour la paire de classes
        class_indices = [CONFIG['CLASS_MAP'][class_pair[0]], CONFIG['CLASS_MAP'][class_pair[1]]]
        mask = np.isin(labels, class_indices)
        features_bin = features[mask]
        labels_bin = labels[mask]
        
        # Remapping binaire
        labels_bin = (labels_bin == class_indices[1]).astype(int)
        
        if len(features_bin) == 0:
            print(f"Aucune image trouvée pour la paire {class_pair}. Entraînement ignoré.")
        return

        # Création et entraînement du modèle
        in_dim = features.shape[1]
        print(f"[train_binary_classifier] Features shape: {features_bin.shape}, Labels shape: {labels_bin.shape}")
    
        mlp_bin = BinaryMLPClassifier(in_dim=in_dim)
        trainer = FeatureMLPTrainer(mlp_bin)
        if use_finetuned:
            model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_finetuned_binary_{CONFIG["CLASS_MAP"][class_pair[0]]}_{CONFIG["CLASS_MAP"][class_pair[1]]}.pth'
        else:
            model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_binary_{CONFIG["CLASS_MAP"][class_pair[0]]}_{CONFIG["CLASS_MAP"][class_pair[1]]}.pth'
        if os.path.exists(model_path):
            print(f"[train_binary_classifier] Reprise de l'entraînement depuis {model_path}")
            trainer.load(model_path, device=self.device)
        
        trainer.train(features_bin, labels_bin, epochs=CONFIG['EPOCHS'], 
                     batch_size=CONFIG['BATCH_SIZE'], lr=CONFIG['LEARNING_RATE'], device=self.device)
        trainer.save(model_path)
        print(f"Modèle binaire sauvegardé : {model_path}")
    
    def train_all_binaries(self, data_csv, image_root, use_finetuned=True):
        """Entraîne tous les classifieurs binaires."""
        print("--- Entraînement de tous les binaires ---")
        
        # Extraction des features
        df = pd.read_csv(data_csv)
        paths = self.feature_manager.extract_features_if_needed(df, image_root, use_finetuned)
        
        # Chargement des features
        features = np.load(paths['train_features'])
        labels = np.load(paths['train_labels'])
        
        # Entraînement de tous les binaires
        for i, j in combinations(range(4), 2):
            class_pair = (CONFIG['DENSITY_CLASSES'][i], CONFIG['DENSITY_CLASSES'][j])
            print(f"Entraînement du binaire {i} vs {j} ({class_pair[0]} vs {class_pair[1]})...")
            
            # Filtrage pour la paire de classes
            mask = np.isin(labels, [i, j])
            features_bin = features[mask]
            labels_bin = labels[mask]
            labels_bin = (labels_bin == j).astype(int)
            
            # Création et entraînement du modèle
            in_dim = features.shape[1]
            mlp_bin = BinaryMLPClassifier(in_dim=in_dim)
            trainer = FeatureMLPTrainer(mlp_bin)
            
            if use_finetuned:
                model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_finetuned_binary_{i}_{j}.pth'
            else:
                model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_binary_{i}_{j}.pth'
            if os.path.exists(model_path):
                trainer.load(model_path, device=self.device)
            
            trainer.train(features_bin, labels_bin, epochs=CONFIG['EPOCHS'], 
                         batch_size=CONFIG['BATCH_SIZE'], lr=CONFIG['LEARNING_RATE'], device=self.device)
            trainer.save(model_path)
            print(f"Modèle binaire {i} vs {j} sauvegardé")

# --- Data Augmentation Manager ---
class DataAugmentationManager:
    """Gestionnaire pour la data augmentation"""
    
    def __init__(self, image_root, annotations_csv):
        self.image_root = image_root
        self.annotations_csv = annotations_csv
        self.df = pd.read_csv(annotations_csv)
        
        # Mapping des densités
        self.density_map = {
            "DENSITY A": "density_A", "DENSITY B": "density_B",
            "DENSITY C": "density_C", "DENSITY D": "density_D"
        }
        
        # Transforms pour l'augmentation
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=3),  # Rotation légère  # Ajustement luminosité/contraste
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),  # Crop aléatoire
        ])
    
    def load_and_preprocess_image(self, image_path, laterality):
        """Charge et prétraite une image DICOM"""
        image = read_dicom(image_path)
        image = preprocess_image(image, laterality=laterality)
        return image
    
    def apply_augmentation(self, image, num_augmentations=3):
        """Applique des augmentations à une image"""
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        augmented_images = []
        for _ in range(num_augmentations):
            aug_image = self.augmentation_transforms(image_pil)
            aug_array = np.array(aug_image) / 255.0
            augmented_images.append(aug_array)
        
        return augmented_images
    
    def balance_dataset(self, target_samples_per_class=1000):
        """Équilibre le dataset par augmentation"""
        print("=== ÉQUILIBRAGE DU DATASET PAR AUGMENTATION ===")
        
        # Analyse de la distribution actuelle
        class_counts = self.df['breast_density'].value_counts()
        print("Distribution actuelle:")
        for density, count in class_counts.items():
            print(f"  {density}: {count} images")
        
        # Calcul des augmentations nécessaires
        augmentations_needed = {}
        for density, count in class_counts.items():
            if count < target_samples_per_class:
                augmentations_needed[density] = target_samples_per_class - count
                print(f"  {density}: besoin de {augmentations_needed[density]} augmentations")
        
        augmented_data = []

        for density in augmentations_needed.keys():
            print(f"\nGénération d'augmentations pour {density}...")

            class_df = self.df[self.df['breast_density'] == density]
            num_augmentations_per_image = augmentations_needed[density] // len(class_df) + 1

            # NB: on ne charge/augmente PAS réellement les pixels ici. Cette étape ne
            # produit que les métadonnées des lignes "_aug" (le CSV rééquilibré) ;
            # l'augmentation pixel réelle est appliquée plus tard, à l'extraction des
            # features, par FeatureDumper.dump_features (voir models.py). Charger et
            # transformer chaque image ici pour jeter le résultat était pur gaspillage
            # de temps (lectures DICOM + transforms répétées inutilement).
            for idx, row in class_df.iterrows():
                split_dir = 'train' if row['split'] == 'training' else 'test'
                study_id = str(row['study_id'])
                image_id = str(row['image_id'])
                # Chemin vers l'image (chemin organisé par défaut)
                image_path = os.path.join(self.image_root, split_dir,
                                         self.density_map[row['breast_density']],
                                         study_id, f"{image_id}.dicom")

                # Fallback pour le format standard VinDr-Mammo (sans organized_layout)
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.image_root, 'images', study_id, f"{image_id}.dicom")
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.image_root, study_id, f"{image_id}.dicom")

                if not os.path.exists(image_path):
                    continue

                for _ in range(num_augmentations_per_image):
                    augmented_data.append({
                        'study_id': f"{row['study_id']}_aug",
                        'image_id': f"{row['image_id']}_aug",
                        'breast_density': row['breast_density'],
                        'split': row['split'],
                        'laterality': row['laterality'],
                    })

        print(f"Génération terminée: {len(augmented_data)} images augmentées créées")
        return augmented_data
    
    def create_augmented_dataset_for_finetuning(self, target_samples_per_class=2000):
        """Crée un dataset augmenté pour le fine-tuning des Transformers"""
        print("=== CRÉATION DU DATASET AUGMENTÉ POUR FINE-TUNING ===")
        
        # Génération des augmentations
        augmented_data = self.balance_dataset(target_samples_per_class)
        
        # Création du DataFrame augmenté
        augmented_rows = []
        
        # Ajouter les données originales
        for _, row in self.df.iterrows():
            augmented_rows.append({
                'study_id': row['study_id'],
                'image_id': row['image_id'],
                'breast_density': row['breast_density'],
                'split': row['split'],
                'laterality': row['laterality'],
                'is_augmented': False
            })
        
        # Ajouter les données augmentées
        for data in augmented_data:
            augmented_rows.append({
                'study_id': data['study_id'],
                'image_id': data['image_id'],
                'breast_density': data['breast_density'],
                'split': data['split'],
                'laterality': data['laterality'],
                'is_augmented': True
            })
        
        # Création du DataFrame
        augmented_df = pd.DataFrame(augmented_rows)
        
        # Sauvegarde du DataFrame augmenté
        output_path = 'DDSM/output_annotations_augmented_MLO.csv'
        augmented_df.to_csv(output_path, index=False)
        
        print(f"Dataset augmenté sauvegardé: {output_path}")
        print(f"Total d'échantillons: {len(augmented_df)}")
        
        # Statistiques
        class_counts = augmented_df['breast_density'].value_counts()
        print("\nNouvelle distribution:")
        for density, count in class_counts.items():
            print(f"  {density}: {count} échantillons")
        
        return augmented_df

# --- Fine-tuning Manager ---
class FineTuningManager:
    """Gestionnaire pour le fine-tuning des Transformers"""
    
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
    
    def finetune_transformer(self, data_csv, image_root, epochs=40, batch_size=10, lr=1e-4, 
                            save_dir=None, patience=5, min_delta=0.001,
                            use_augmentation=True, resume_from=None):
        """
        Fine-tune un Transformer sur les mammographies avec data augmentation
        
        Args:
            data_csv: Chemin vers le CSV d'annotations
            image_root: Chemin vers les images
            epochs: Nombre d'époques
            batch_size: Taille du batch
            lr: Learning rate
            save_dir: Dossier de sauvegarde (utilise CONFIG si None)
            device: Device
            patience: Patience pour early stopping
            min_delta: Delta minimum pour considérer une amélioration
            use_augmentation: Utiliser la data augmentation
            resume_from: Chemin vers les poids à reprendre (optionnel)
        """
        if save_dir is None:
            save_dir = CONFIG['FINETUNED_WEIGHTS_DIR']
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[FINETUNE] Début du fine-tuning pour {self.backbone}")
        print(f"[FINETUNE] Data augmentation: {'Activée' if use_augmentation else 'Désactivée'}")
        
        # Création du dataset et dataloader
        df = pd.read_csv(data_csv)
        train_dataset = MammographyDataset(df, image_root, split='training', use_augmentation=use_augmentation ,label_map=CONFIG['CLASS_MAP'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        print(f"[FINETUNE] Dataset créé: {len(train_dataset)} images d'entraînement")
        
        # Création du modèle
        model = FineTunedFeatureExtractor(self.backbone, num_classes=4, pretrained=True)
        
        # Reprise du fine-tuning si spécifié ou si un checkpoint existe
        start_epoch = 0
        best_accuracy = 0.0
        patience_counter = 0
        best_model_path = os.path.join(save_dir, f'{self.backbone}_finetuned_best.pth')
        checkpoint_path = os.path.join(save_dir, f'{self.backbone}_finetuned_checkpoint.pth')

        ckpt_to_load = None
        if resume_from and os.path.exists(resume_from):
            ckpt_to_load = resume_from
        elif os.path.exists(checkpoint_path):
            ckpt_to_load = checkpoint_path
        elif os.path.exists(best_model_path):
            ckpt_to_load = best_model_path

        if ckpt_to_load:
            print(f"[FINETUNE] Reprise automatique du fine-tuning depuis {ckpt_to_load}")
            model.load_finetuned_weights(ckpt_to_load, self.device)
            actual_ckpt = ckpt_to_load if '_checkpoint.pth' in ckpt_to_load else ckpt_to_load.replace('.pth', '_checkpoint.pth')
            if os.path.exists(actual_ckpt):
                checkpoint = torch.load(actual_ckpt, map_location=self.device)
                if isinstance(checkpoint, dict):
                    start_epoch = checkpoint.get('epoch', -1) + 1
                    best_accuracy = checkpoint.get('best_accuracy', 0.0)
                    patience_counter = checkpoint.get('patience_counter', 0)
                    print(f"[FINETUNE] ✅ Reprise réussie à l'époque {start_epoch + 1} (Meilleure Acc: {best_accuracy:.2f}%)")
        model.to(self.device)

        # Calcul des poids de classe (directement depuis le DataFrame, évite de charger toutes les images)
        labels = train_dataset.df['label'].tolist()
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        print(f"[FINETUNE] Poids de classe: {class_weights}")

        # Optimiseur et fonction de perte
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        # Entraînement
        model.train()
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            epoch_preds = []
            epoch_labels = []

            progress_bar = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs}")
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                epoch_preds.extend(predicted.detach().cpu().numpy().tolist())
                epoch_labels.extend(labels.detach().cpu().numpy().tolist())

                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

            # Calcul des métriques finales
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100. * correct / total

            print(f"\n[FINETUNE] Époque {epoch+1}/{epochs}:")
            print(f"  - Loss: {epoch_loss:.4f}")
            print(f"  - Accuracy: {epoch_accuracy:.2f}%")
            print("  - Rapport de classification (train, époque courante):")
            print(classification_report(
                epoch_labels, epoch_preds, labels=[0, 1, 2, 3],
                target_names=CONFIG['DENSITY_CLASSES'], zero_division=0
            ))
            print("  - Matrice de confusion (A/B/C/D):")
            print(confusion_matrix(epoch_labels, epoch_preds, labels=[0, 1, 2, 3]))
            
            # Learning rate scheduling
            scheduler.step(epoch_accuracy)
            
            # Sauvegarde du meilleur modèle
            if epoch_accuracy > best_accuracy + min_delta:
                best_accuracy = epoch_accuracy
                patience_counter = 0
                
                # Sauvegarde du meilleur modèle
                best_model_path = os.path.join(save_dir, f'{self.backbone}_finetuned_best.pth')
                os.makedirs(save_dir, exist_ok=True)
                model.save_finetuned_weights(best_model_path)
                
                # Sauvegarde du checkpoint pour reprise
                checkpoint_path = os.path.join(save_dir, f'{self.backbone}_finetuned_checkpoint.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'patience_counter': patience_counter
                }, checkpoint_path)
                
                print(f"[CHECKPOINT] Nouveau meilleur modèle! Acc: {epoch_accuracy:.4f}")
                print(f"[CHECKPOINT] Modèle sauvegardé: {best_model_path}")
                print(f"[CHECKPOINT] Checkpoint sauvegardé: {checkpoint_path}")
            else:
                patience_counter += 1
                print(f"[CHECKPOINT] Pas d'amélioration ({patience_counter}/{patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"[FINETUNE] Early stopping après {epoch+1} époques sans amélioration")
                break
        
        # Sauvegarde finale
        final_save_path = os.path.join(save_dir, f'{self.backbone}_finetuned_final.pth')
        model.save_finetuned_weights(final_save_path)
        
        print(f"[FINETUNE] Fine-tuning terminé.")
        print(f"  - Meilleure accuracy: {best_accuracy:.4f}")
        print(f"  - Modèle final: {final_save_path}")
        print(f"  - Meilleur modèle: {best_model_path}")
        
        return best_model_path

# --- Fonction d'inférence hiérarchique ---
def hierarchical_inference(image_tensor, hierarchical_model, device, class_map_inv):
    """Inférence hiérarchique sur une image."""
    hierarchical_model.eval()
    with torch.no_grad():
        logits4, logits_bin, top2 = hierarchical_model(image_tensor.unsqueeze(0).to(device))
        pred4 = logits4.argmax(dim=1).item()
        pred_bin = logits_bin.argmax(dim=1).item()
        
        # Récupération des deux classes candidates
        c1, c2 = top2[0].tolist()
        final_class = c1 if pred_bin == 0 else c2
        return pred4, final_class

# --- Fonction principale ---
def main():
    """Fonction principale avec gestion des modes d'exécution."""
    
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Système unifié d\'entraînement et fine-tuning pour mammographies')
    
    # Arguments principaux
    parser.add_argument('--mode', type=str, default='augment', 
                       choices=['dump_features', 'train_on_features', 'train', 'finetune', 'augment', 'finetune_augmented'],
                       help='Mode d\'exécution')
    
    parser.add_argument('--backbone', type=str, default='cvt-w24',
                       choices=['vit', 'deit', 'swin', 'swinv2', 'pvt', 't2t_vit', 'twins', 'resnet', 'resnet50', 'ResNet50', 'efficientnet', 'cvt', 'cvt-13', 'cvt-21', 'cvt-w24'],
                       help='Backbone à utiliser')
    
    parser.add_argument('--annotations_csv', type=str, default=CONFIG['ANNOTATIONS_CSV'],
                        help='Chemin vers le fichier CSV des annotations')
    
    parser.add_argument('--image_root', type=str, 
                       default=CONFIG['IMAGE_ROOT'],
                       help='Chemin vers le dossier racine des images')
    
    # Arguments d'entraînement
    parser.add_argument('--epochs', type=int, default=100,
                       help='Nombre d\'époques d\'entraînement')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Taille du batch')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Taux d\'apprentissage')
    
    # Arguments pour les features
    parser.add_argument('--use_finetuned', action='store_true', default=True,
                       help='Utiliser les features fine-tunées (défaut: True)')
    
    parser.add_argument('--no_finetuned', dest='use_finetuned', action='store_false',
                       help='Ne pas utiliser les features fine-tunées')
    
    # Arguments pour l'entraînement
    parser.add_argument('--all_binaries', action='store_true', default=False,
                       help='Entraîner tous les classifieurs binaires')
    
    parser.add_argument('--binary_pair', nargs=2, type=int, default=None,
                       help='Paire de classes pour entraîner un binaire spécifique (ex: 0 1)')
    
    # Arguments pour le MLP 4 classes
    parser.add_argument('--train_mlp4', action='store_true', default=False,
                       help='Entraîner le MLP 4 classes')
    
    # Arguments pour la data augmentation
    parser.add_argument('--target_samples', type=int, default=4000,
                       help='Nombre cible d\'échantillons par classe pour l\'augmentation')
    
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                       help='Utiliser la data augmentation')
    
    # Arguments pour le fine-tuning
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience pour early stopping')
    
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Delta minimum pour considérer une amélioration')
    
    # Arguments pour la reprise d'entraînement
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Chemin vers les poids à reprendre pour continuer l\'entraînement')
    
    # Parse des arguments
    args = parser.parse_args()
    
    # Mise à jour de la configuration avec les arguments
    CONFIG.update({
        'BACKBONE': args.backbone,
        'ANNOTATIONS_CSV': args.annotations_csv,
        'IMAGE_ROOT': args.image_root,
        'EPOCHS': args.epochs,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.learning_rate
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Utilisation du device : {device}")
    print(f"[INFO] Configuration:")
    print(f"  - Mode: {args.mode}")
    print(f"  - Backbone: {args.backbone}")
    print(f"  - Époques: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Features fine-tunées: {args.use_finetuned}")
    print(f"  - Data augmentation: {args.use_augmentation}")
    if args.resume_from:
        print(f"  - Reprise depuis: {args.resume_from}")
    
    if args.mode == 'augment':
        print("--- Mode: Data Augmentation ---")
        augmenter = DataAugmentationManager(args.image_root, args.annotations_csv)
        augmented_df = augmenter.create_augmented_dataset_for_finetuning(args.target_samples)
        print("Dataset augmenté créé pour le fine-tuning!")
        
    elif args.mode == 'finetune':
        print("--- Mode: Fine-tuning Transformer ---")
        finetuner = FineTuningManager(args.backbone, device)
        best_model_path = finetuner.finetune_transformer(
            data_csv=args.annotations_csv,
            image_root=args.image_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            patience=args.patience,
            min_delta=args.min_delta,
            use_augmentation=args.use_augmentation,
            resume_from=args.resume_from
        )
        print(f"Fine-tuning terminé. Meilleur modèle: {best_model_path}")
        
    elif args.mode == 'finetune_augmented':
        print("--- Mode: Fine-tuning avec Dataset Augmenté ---")
        # Vérifier si le dataset augmenté existe
        augmented_csv = 'DDSM/output_annotations_augmented.csv'
        if not os.path.exists(augmented_csv):
            print("Dataset augmenté non trouvé. Création en cours...")
            augmenter = DataAugmentationManager(args.image_root, args.annotations_csv)
            augmenter.create_augmented_dataset_for_finetuning(args.target_samples)
        
        finetuner = FineTuningManager(args.backbone, device)
        best_model_path = finetuner.finetune_transformer(
            data_csv=augmented_csv,
            image_root=args.image_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            patience=args.patience,
            min_delta=args.min_delta,
            use_augmentation=True,
            resume_from=args.resume_from
        )
        print(f"Fine-tuning avec dataset augmenté terminé. Meilleur modèle: {best_model_path}")
        
    else:  # Modes classiques (dump_features, train_on_features, train)
        trainer = ModelTrainer(CONFIG['BACKBONE'], device)
        
        if args.mode == 'dump_features':
            print("--- Mode: Extraction des features ---")
            df = pd.read_csv(CONFIG['ANNOTATIONS_CSV'])
            trainer.feature_manager.extract_features_if_needed(df, CONFIG['IMAGE_ROOT'], args.use_finetuned)
            
        elif args.mode == 'train_on_features':
            print("--- Mode: Entraînement sur features ---")
            df = pd.read_csv(CONFIG['ANNOTATIONS_CSV'])
            paths = trainer.feature_manager.extract_features_if_needed(df, CONFIG['IMAGE_ROOT'], args.use_finetuned)
            
            if args.all_binaries:
                trainer.train_all_binaries(CONFIG['ANNOTATIONS_CSV'], CONFIG['IMAGE_ROOT'], args.use_finetuned)
            elif args.binary_pair is not None:
                i, j = args.binary_pair
                class_pair = (CONFIG['DENSITY_CLASSES'][i], CONFIG['DENSITY_CLASSES'][j])
                trainer.train_binary_classifier(class_pair, CONFIG['ANNOTATIONS_CSV'], CONFIG['IMAGE_ROOT'], args.use_finetuned)
            elif args.train_mlp4:
                trainer.train_mlp4_classifier(CONFIG['ANNOTATIONS_CSV'], CONFIG['IMAGE_ROOT'], args.use_finetuned)
            else:
                print("[WARN] Aucun modèle spécifié. Utilisez --train_mlp4, --all_binaries ou --binary_pair")
                
        else:  # mode == 'train'
            print("--- Mode: Entraînement complet ---")
            
            # Entraînement de tous les binaires
            class_pairs = list(combinations(CONFIG['DENSITY_CLASSES'], 2))
            for pair in class_pairs:
                    trainer.train_binary_classifier(pair, CONFIG['ANNOTATIONS_CSV'], CONFIG['IMAGE_ROOT'])

        # Entraînement du MLP 4 classes
            trainer.train_mlp4_classifier(CONFIG['ANNOTATIONS_CSV'], CONFIG['IMAGE_ROOT'])

if __name__ == '__main__':
    main() 