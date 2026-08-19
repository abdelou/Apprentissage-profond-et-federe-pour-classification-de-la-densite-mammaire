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

class MammographyDataset(Dataset):
    # Dataset pour charger les mammographies
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

        if classes_to_use:
            self.df = annotations_df[annotations_df['breast_density'].isin(classes_to_use)].copy()
            self.label_remap = {classes_to_use[0]: 0, classes_to_use[1]: 1}
            self.df['label'] = self.df['breast_density'].map(self.label_remap)
        else:
            self.df = annotations_df.copy()
            if label_map:
                self.df['label'] = self.df['breast_density'].map(label_map)
            else:
                raise ValueError("Spécifier label_map")

        self.df = self.df.reset_index(drop=True)
        
        if self.use_augmentation and split == 'training':
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.augmentation_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        is_aug_val = str(row.get('is_augmented', '')).lower()
        is_augmented = is_aug_val == 'true' or '_aug' in str(row['image_id']) or '_aug' in str(row['study_id'])
        
        if is_augmented:
            study_id = row['study_id'].replace('_aug', '')
            image_id = row['image_id'].replace('_aug', '')
            density = self.density_map.get(row["breast_density"])
            split = self.split_map.get(row["split"])
            
            image_path = os.path.join(self.image_dir, split, density, study_id, f"{image_id}.dicom")
            
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_dir, 'images', study_id, f"{image_id}.dicom")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.image_dir, study_id, f"{image_id}.dicom")

            laterality = row['laterality']
            image = read_dicom(image_path)
            image = preprocess_image(image, laterality=laterality)
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image_tensor = self.augmentation_transforms(image_pil)
                
        else:
            split = self.split_map.get(row["split"])
            density = self.density_map.get(row["breast_density"])
            study_id = str(row["study_id"])
            image_id = str(row["image_id"])
            
            image_path = os.path.join(self.image_dir, split, density, study_id, f"{image_id}.dicom")
            
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_dir, 'images', study_id, f"{image_id}.dicom")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.image_dir, study_id, f"{image_id}.dicom")

            laterality = row['laterality']
            image = read_dicom(image_path)
            image = preprocess_image(image, laterality=laterality)
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image_tensor = self.augmentation_transforms(image_pil)

        label = torch.tensor(row['label'], dtype=torch.long)
        return image_tensor, label

class FeatureManager:
    # Gère l'extraction des features et les chemins d'accès
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
        self.finetuned_weights_path = f'{CONFIG["FINETUNED_WEIGHTS_DIR"]}/{backbone}_finetuned_best.pth'
        
    def get_feature_extractor(self, use_finetuned=True):
        if use_finetuned and os.path.exists(self.finetuned_weights_path):
            print(f"Chargement poids fine-tunés: {self.finetuned_weights_path}")
            return FeatureExtractor(self.backbone, pretrained=True, finetuned_weights_path=self.finetuned_weights_path).to(self.device)
        else:
            print("Utilisation modèle pré-entraîné simple")
            return FeatureExtractor(self.backbone, pretrained=True).to(self.device)
        
    def get_feature_paths(self, use_finetuned=True):
        suffix = "finetuned" if use_finetuned else ""
        return {
            'train_features': f'{CONFIG["FEATURES_DIR"]}/{self.backbone}_{suffix}_features_train.npy',
            'train_labels': f'{CONFIG["FEATURES_DIR"]}/{self.backbone}_{suffix}_labels_train.npy',
            'test_features': f'{CONFIG["FEATURES_DIR"]}/{self.backbone}_{suffix}_features_test.npy',
            'test_labels': f'{CONFIG["FEATURES_DIR"]}/{self.backbone}_{suffix}_labels_test.npy'
        }
    
    def extract_features_if_needed(self, df, image_root, use_finetuned=True):
        paths = self.get_feature_paths(use_finetuned)
        os.makedirs(CONFIG['FEATURES_DIR'], exist_ok=True)
        
        if not (os.path.exists(paths['train_features']) and os.path.exists(paths['train_labels'])):
            print("Extraction des features d'entraînement...")
            feature_extractor = self.get_feature_extractor(use_finetuned)
            dumper = FeatureDumper(feature_extractor, self.device)
            dumper.dump_features(df, image_root, paths['train_features'], paths['train_labels'], 
                               label_map=CONFIG['CLASS_MAP'], split='training')
        
        if not (os.path.exists(paths['test_features']) and os.path.exists(paths['test_labels'])):
            print("Extraction des features de test...")
            feature_extractor = self.get_feature_extractor(use_finetuned)
            dumper = FeatureDumper(feature_extractor, self.device)
            dumper.dump_features(df, image_root, paths['test_features'], paths['test_labels'], 
                               label_map=CONFIG['CLASS_MAP'], split='test')
        
        return paths

class ModelTrainer:
    # Entraîneur pour les différents modèles
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
        self.feature_manager = FeatureManager(backbone, device)
    
    def train_mlp4_classifier(self, data_csv, image_root, use_finetuned=True):
        print("--- MLP 4 classes ---")
        df = pd.read_csv(data_csv)
        paths = self.feature_manager.extract_features_if_needed(df, image_root, use_finetuned)
        
        features = np.load(paths['train_features'])
        labels = np.load(paths['train_labels'])
        
        in_dim = features.shape[1]
        mlp = MLPClassifier(in_dim=in_dim, out_dim=4)
        trainer = FeatureMLPTrainer(mlp)
        
        model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_finetuned_mlp4.pth' if use_finetuned else f'{CONFIG["MODELS_DIR"]}/{self.backbone}_mlp4.pth'
        if os.path.exists(model_path):
            print(f"Reprise depuis : {model_path}")
            trainer.load(model_path, device=self.device)
        
        trainer.train(features, labels, epochs=CONFIG['EPOCHS'], 
                     batch_size=CONFIG['BATCH_SIZE'], lr=CONFIG['LEARNING_RATE'], device=self.device)
        trainer.save(model_path)
        print(f"Modèle sauvegardé dans {model_path}")
    
    def train_binary_classifier(self, class_pair, data_csv, image_root, use_finetuned=True):
        print(f"--- Binaire {class_pair[0]} vs {class_pair[1]} ---")
        df = pd.read_csv(data_csv)
        paths = self.feature_manager.extract_features_if_needed(df, image_root, use_finetuned)
        
        features = np.load(paths['train_features'])
        labels = np.load(paths['train_labels'])
        
        class_indices = [CONFIG['CLASS_MAP'][class_pair[0]], CONFIG['CLASS_MAP'][class_pair[1]]]
        mask = np.isin(labels, class_indices)
        features_bin = features[mask]
        labels_bin = labels[mask]
        labels_bin = (labels_bin == class_indices[1]).astype(int)
        
        if len(features_bin) == 0:
            print(f"Pas d'images pour {class_pair}. Train ignoré.")
            return

        in_dim = features.shape[1]
        mlp_bin = BinaryMLPClassifier(in_dim=in_dim)
        trainer = FeatureMLPTrainer(mlp_bin)
        
        model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_finetuned_binary_{CONFIG["CLASS_MAP"][class_pair[0]]}_{CONFIG["CLASS_MAP"][class_pair[1]]}.pth' if use_finetuned else f'{CONFIG["MODELS_DIR"]}/{self.backbone}_binary_{CONFIG["CLASS_MAP"][class_pair[0]]}_{CONFIG["CLASS_MAP"][class_pair[1]]}.pth'
        if os.path.exists(model_path):
            print(f"Reprise depuis : {model_path}")
            trainer.load(model_path, device=self.device)
        
        trainer.train(features_bin, labels_bin, epochs=CONFIG['EPOCHS'], 
                     batch_size=CONFIG['BATCH_SIZE'], lr=CONFIG['LEARNING_RATE'], device=self.device)
        trainer.save(model_path)
        print(f"Sauvegardé dans {model_path}")
    
    def train_all_binaries(self, data_csv, image_root, use_finetuned=True):
        print("--- Tous les modèles binaires ---")
        df = pd.read_csv(data_csv)
        paths = self.feature_manager.extract_features_if_needed(df, image_root, use_finetuned)
        
        features = np.load(paths['train_features'])
        labels = np.load(paths['train_labels'])
        
        for i, j in combinations(range(4), 2):
            class_pair = (CONFIG['DENSITY_CLASSES'][i], CONFIG['DENSITY_CLASSES'][j])
            print(f"Train binaire {i} vs {j} ({class_pair[0]} vs {class_pair[1]})...")
            
            mask = np.isin(labels, [i, j])
            features_bin = features[mask]
            labels_bin = labels[mask]
            labels_bin = (labels_bin == j).astype(int)
            
            in_dim = features.shape[1]
            mlp_bin = BinaryMLPClassifier(in_dim=in_dim)
            trainer = FeatureMLPTrainer(mlp_bin)
            
            model_path = f'{CONFIG["MODELS_DIR"]}/{self.backbone}_finetuned_binary_{i}_{j}.pth' if use_finetuned else f'{CONFIG["MODELS_DIR"]}/{self.backbone}_binary_{i}_{j}.pth'
            if os.path.exists(model_path):
                trainer.load(model_path, device=self.device)
            
            trainer.train(features_bin, labels_bin, epochs=CONFIG['EPOCHS'], 
                         batch_size=CONFIG['BATCH_SIZE'], lr=CONFIG['LEARNING_RATE'], device=self.device)
            trainer.save(model_path)
            print(f"Sauvegardé {i} vs {j}")

class DataAugmentationManager:
    # Gère l'augmentation des données
    def __init__(self, image_root, annotations_csv):
        self.image_root = image_root
        self.annotations_csv = annotations_csv
        self.df = pd.read_csv(annotations_csv)
        self.density_map = {
            "DENSITY A": "density_A", "DENSITY B": "density_B",
            "DENSITY C": "density_C", "DENSITY D": "density_D"
        }
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=3),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        ])
    
    def load_and_preprocess_image(self, image_path, laterality):
        image = read_dicom(image_path)
        image = preprocess_image(image, laterality=laterality)
        return image
    
    def apply_augmentation(self, image, num_augmentations=3):
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        augmented_images = []
        for _ in range(num_augmentations):
            aug_image = self.augmentation_transforms(image_pil)
            aug_array = np.array(aug_image) / 255.0
            augmented_images.append(aug_array)
        return augmented_images
    
    def balance_dataset(self, target_samples_per_class=1000):
        # On genere juste les lignes d'augmentation dans le CSV
        class_counts = self.df['breast_density'].value_counts()
        print("Distribution actuelle :")
        for density, count in class_counts.items():
            print(f"  {density}: {count} images")
        
        augmentations_needed = {}
        for density, count in class_counts.items():
            if count < target_samples_per_class:
                augmentations_needed[density] = target_samples_per_class - count
                print(f"  {density}: besoin de {augmentations_needed[density]} augmentations")
        
        augmented_data = []

        for density in augmentations_needed.keys():
            print(f"Génération d'augmentations pour {density}...")
            class_df = self.df[self.df['breast_density'] == density]
            num_augmentations_per_image = augmentations_needed[density] // len(class_df) + 1

            for idx, row in class_df.iterrows():
                split_dir = 'train' if row['split'] == 'training' else 'test'
                study_id = str(row['study_id'])
                image_id = str(row['image_id'])
                image_path = os.path.join(self.image_root, split_dir,
                                         self.density_map[row['breast_density']],
                                         study_id, f"{image_id}.dicom")

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

        print(f"Génération terminée : {len(augmented_data)} lignes augmentées.")
        return augmented_data
    
    def create_augmented_dataset_for_finetuning(self, target_samples_per_class=2000):
        # Créer le DataFrame augmenté pour le fine-tuning
        augmented_data = self.balance_dataset(target_samples_per_class)
        augmented_rows = []
        
        for _, row in self.df.iterrows():
            augmented_rows.append({
                'study_id': row['study_id'],
                'image_id': row['image_id'],
                'breast_density': row['breast_density'],
                'split': row['split'],
                'laterality': row['laterality'],
                'is_augmented': False
            })
        
        for data in augmented_data:
            augmented_rows.append({
                'study_id': data['study_id'],
                'image_id': data['image_id'],
                'breast_density': data['breast_density'],
                'split': data['split'],
                'laterality': data['laterality'],
                'is_augmented': True
            })
        
        augmented_df = pd.DataFrame(augmented_rows)
        output_path = 'DDSM/output_annotations_augmented_MLO.csv'
        augmented_df.to_csv(output_path, index=False)
        
        print(f"Sauvegardé dans {output_path}")
        return augmented_df

class FineTuningManager:
    # Gère le fine-tuning des modeles
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
    
    def finetune_transformer(self, data_csv, image_root, epochs=40, batch_size=10, lr=1e-4, 
                            save_dir=None, patience=5, min_delta=0.001,
                            use_augmentation=True, resume_from=None):
        if save_dir is None:
            save_dir = CONFIG['FINETUNED_WEIGHTS_DIR']
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Fine-tuning {self.backbone}...")
        
        df = pd.read_csv(data_csv)
        train_dataset = MammographyDataset(df, image_root, split='training', use_augmentation=use_augmentation, label_map=CONFIG['CLASS_MAP'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        print(f"Dataset : {len(train_dataset)} images")
        
        model = FineTunedFeatureExtractor(self.backbone, num_classes=4, pretrained=True)
        
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
            print(f"Reprise depuis checkpoint : {ckpt_to_load}")
            model.load_finetuned_weights(ckpt_to_load, self.device)
            actual_ckpt = ckpt_to_load if '_checkpoint.pth' in ckpt_to_load else ckpt_to_load.replace('.pth', '_checkpoint.pth')
            if os.path.exists(actual_ckpt):
                checkpoint = torch.load(actual_ckpt, map_location=self.device)
                if isinstance(checkpoint, dict):
                    start_epoch = checkpoint.get('epoch', -1) + 1
                    best_accuracy = checkpoint.get('best_accuracy', 0.0)
                    patience_counter = checkpoint.get('patience_counter', 0)
                    print(f"Reprise époque {start_epoch + 1} (Meilleure Acc: {best_accuracy:.2f}%)")
        model.to(self.device)

        labels = train_dataset.df['label'].tolist()
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

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

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100. * correct / total

            print(f"\nÉpoque {epoch+1}/{epochs}:")
            print(f"  - Loss: {epoch_loss:.4f}")
            print(f"  - Accuracy: {epoch_accuracy:.2f}%")
            print(classification_report(
                epoch_labels, epoch_preds, labels=[0, 1, 2, 3],
                target_names=CONFIG['DENSITY_CLASSES'], zero_division=0
            ))
            print(confusion_matrix(epoch_labels, epoch_preds, labels=[0, 1, 2, 3]))
            
            scheduler.step(epoch_accuracy)
            
            if epoch_accuracy > best_accuracy + min_delta:
                best_accuracy = epoch_accuracy
                patience_counter = 0
                
                best_model_path = os.path.join(save_dir, f'{self.backbone}_finetuned_best.pth')
                os.makedirs(save_dir, exist_ok=True)
                model.save_finetuned_weights(best_model_path)
                
                checkpoint_path = os.path.join(save_dir, f'{self.backbone}_finetuned_checkpoint.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'patience_counter': patience_counter
                }, checkpoint_path)
                print(f"Meilleur modèle sauvegardé: {best_model_path} (Acc: {epoch_accuracy:.4f})")
            else:
                patience_counter += 1
                print(f"Pas d'amélioration ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print("Early stopping")
                break
        
        final_save_path = os.path.join(save_dir, f'{self.backbone}_finetuned_final.pth')
        model.save_finetuned_weights(final_save_path)
        return best_model_path

def hierarchical_inference(image_tensor, hierarchical_model, device, class_map_inv):
    hierarchical_model.eval()
    with torch.no_grad():
        logits4, logits_bin, top2 = hierarchical_model(image_tensor.unsqueeze(0).to(device))
        pred4 = logits4.argmax(dim=1).item()
        pred_bin = logits_bin.argmax(dim=1).item()
        
        c1, c2 = top2[0].tolist()
        final_class = c1 if pred_bin == 0 else c2
        return pred4, final_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='augment', 
                       choices=['dump_features', 'train_on_features', 'train', 'finetune', 'augment', 'finetune_augmented'])
    parser.add_argument('--backbone', type=str, default='cvt-w24',
                       choices=['vit', 'deit', 'swin', 'swinv2', 'pvt', 't2t_vit', 'twins', 'resnet', 'resnet50', 'ResNet50', 'efficientnet', 'cvt', 'cvt-13', 'cvt-21', 'cvt-w24'])
    parser.add_argument('--annotations_csv', type=str, default=CONFIG['ANNOTATIONS_CSV'])
    parser.add_argument('--image_root', type=str, default=CONFIG['IMAGE_ROOT'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--use_finetuned', action='store_true', default=True)
    parser.add_argument('--no_finetuned', dest='use_finetuned', action='store_false')
    parser.add_argument('--all_binaries', action='store_true', default=False)
    parser.add_argument('--binary_pair', nargs=2, type=int, default=None)
    parser.add_argument('--train_mlp4', action='store_true', default=False)
    parser.add_argument('--target_samples', type=int, default=4000)
    parser.add_argument('--use_augmentation', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--resume_from', type=str, default=None)
    
    args = parser.parse_args()
    
    CONFIG.update({
        'BACKBONE': args.backbone,
        'ANNOTATIONS_CSV': args.annotations_csv,
        'IMAGE_ROOT': args.image_root,
        'EPOCHS': args.epochs,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.learning_rate
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    
    if args.mode == 'augment':
        print("--- Mode: Data Augmentation ---")
        augmenter = DataAugmentationManager(args.image_root, args.annotations_csv)
        augmented_df = augmenter.create_augmented_dataset_for_finetuning(args.target_samples)
        print("Terminé.")
        
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
        print(f"Terminé. Meilleur modèle : {best_model_path}")
        
    elif args.mode == 'finetune_augmented':
        print("--- Mode: Fine-tuning avec Dataset Augmenté ---")
        augmented_csv = 'DDSM/output_annotations_augmented.csv'
        if not os.path.exists(augmented_csv):
            print("Création du dataset augmenté...")
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
        print(f"Terminé. Meilleur modèle : {best_model_path}")
        
    else:
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
                print("Aucun modèle spécifié.")
                
        else:
            print("--- Mode: Entraînement complet ---")
            class_pairs = list(combinations(CONFIG['DENSITY_CLASSES'], 2))
            for pair in class_pairs:
                trainer.train_binary_classifier(pair, CONFIG['ANNOTATIONS_CSV'], CONFIG['IMAGE_ROOT'])
            trainer.train_mlp4_classifier(CONFIG['ANNOTATIONS_CSV'], CONFIG['IMAGE_ROOT'])

if __name__ == '__main__':
    main()