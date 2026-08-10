import sys
import os

# Chemin vers le dossier contenant models.py
 # adapte ce chemin si besoin
sys.path.insert(0, "/kaggle/input/stages/Codes")
import torch
import torch.nn as nn
from timm import create_model
from itertools import combinations
import numpy as np
import os
import time
from PIL import Image
from preprocessing import read_dicom, preprocess_image
from sklearn.utils.class_weight import compute_class_weight

# Import Hugging Face Transformers for CVT
try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. CVT models will not be supported.")

BACKBONES = {
    'vit': 'vit_base_patch16_224',
    'deit': 'deit_base_patch16_224',
    'swin': 'swin_base_patch4_window7_224',
    'swinv2': 'swinv2_base_window12_192_22k',
    'pvt': 'pvt_v2_b2',
    't2t_vit': 't2t_vit_14',
    'twins': 'twins_pcpvt_large',
    'resnet': 'resnet50',
    'resnet50': 'resnet50',
    'ResNet50': 'resnet50',
    'efficientnet': 'efficientnet_b0',
    # CVT models from Hugging Face Transformers
    'cvt': 'microsoft/cvt-13',
    'cvt-13': 'microsoft/cvt-13',
    'cvt-21': 'microsoft/cvt-21',
    'cvt-w24': 'microsoft/cvt-w24-384-22k',
}

# CVT model configurations
CVT_CONFIGS = {
    'cvt': 'microsoft/cvt-13',
    'cvt-13': 'microsoft/cvt-13',
    'cvt-21': 'microsoft/cvt-21', 
    'cvt-w24': 'microsoft/cvt-w24-384-22k',
}

class HuggingFaceFeatureExtractor(nn.Module):
    """
    Feature extractor using Hugging Face Transformers models (CVT, etc.)
    """
    def __init__(self, model_name: str, pretrained=True):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for Hugging Face models")
        
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Convert tensor to PIL images for processor
        import torchvision.transforms as transforms
        to_pil = transforms.ToPILImage()
        
        batch_size = x.size(0)
        features = []
        
        for i in range(batch_size):
            # Convert single image tensor to PIL
            img_tensor = x[i]
            img_pil = to_pil(img_tensor)
            
            # Process with Hugging Face processor
            inputs = self.processor(img_pil, return_tensors="pt")
            
            # Move to same device as input
            device = x.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state as features
                feature = outputs.last_hidden_state.mean(dim=1).squeeze()  # [hidden_size]
                features.append(feature)
        
        return torch.stack(features)

class FineTunedFeatureExtractor(nn.Module):
    """
    Feature extractor avec fine-tuning sur les mammographies
    """
    def __init__(self, backbone_name: str, num_classes=4, pretrained=True):
        super().__init__()
        
        # Check if it's a Hugging Face model
        if backbone_name in CVT_CONFIGS:
            print(f"[DEBUG] FineTunedFeatureExtractor - Utilisation du modèle {backbone_name}")
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(f"transformers library required for {backbone_name}")
            
            model_name = CVT_CONFIGS[backbone_name]
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Add classification head - utiliser la dimension correcte
            if hasattr(self.model.config, 'hidden_sizes'):
                hidden_size = self.model.config.hidden_sizes[-1]
            elif hasattr(self.model.config, 'hidden_size'):
                hidden_size = self.model.config.hidden_size
            else:
                # Dimensions par défaut selon le modèle CVT
                if 'cvt-13' in backbone_name:
                    hidden_size = 384
                elif 'cvt-21' in backbone_name:
                    hidden_size = 512
                elif 'cvt-w24' in backbone_name:
                    hidden_size = 384
                else:
                    hidden_size = 384  # Par défaut
            
            # Créer une nouvelle head de classification avec le bon nombre de classes
            self.classifier = nn.Linear(hidden_size, num_classes)
            
        else:
            # Use timm models as before, but initialize them directly with the target number of classes
            print(f"[DEBUG] FineTunedFeatureExtractor - Utilisation du modèle {backbone_name}")
            self.model = create_model(BACKBONES[backbone_name], pretrained=pretrained, num_classes=num_classes)
            self.processor = None
            
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
    def forward(self, x):
        if self.backbone_name in CVT_CONFIGS:
            # Process with Hugging Face model
            import torchvision.transforms as transforms
            to_pil = transforms.ToPILImage()
            
            batch_size = x.size(0)
            features = []
            
            for i in range(batch_size):
                img_tensor = x[i]
                img_pil = to_pil(img_tensor)
                inputs = self.processor(img_pil, return_tensors="pt")
                
                device = x.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Extraire les features correctement
                    if hasattr(outputs, 'last_hidden_state'):
                        # Pour les modèles qui retournent last_hidden_state
                        # Aplatir les features spatiales
                        feature = outputs.last_hidden_state.mean(dim=1).squeeze()  # [hidden_size]
                    elif hasattr(outputs, 'pooler_output'):
                        # Pour les modèles qui ont un pooler
                        feature = outputs.pooler_output.squeeze()
                    else:
                        # Fallback: utiliser la première sortie et l'aplatir
                        feature = outputs[0].mean(dim=1).squeeze()  # [hidden_size]
                    
                    # S'assurer que la dimension est correcte
                    if feature.dim() == 0:
                        feature = feature.unsqueeze(0)
                    elif feature.dim() > 1:
                        # Si on a encore des dimensions spatiales, les aplatir
                        feature = feature.flatten()
                    
                    features.append(feature)
            
            features = torch.stack(features)
                        
            # Debug: afficher les dimensions
            print(f"[DEBUG] Features shape: {features.shape}")
            print(f"[DEBUG] Classifier input: {self.classifier.in_features}, output: {self.classifier.out_features}")
            
            # Vérifier que les dimensions correspondent
            if features.shape[1] != self.classifier.in_features:
                print(f"[WARN] Dimension mismatch! Features: {features.shape[1]}, Classifier expects: {self.classifier.in_features}")
                # Redimensionner si nécessaire
                if features.shape[1] < self.classifier.in_features:
                    # Padding avec des zéros
                    padding = torch.zeros(features.shape[0], self.classifier.in_features - features.shape[1], device=features.device)
                    features = torch.cat([features, padding], dim=1)
                else:
                    # Tronquer
                    features = features[:, :self.classifier.in_features]
            
            # Retourner les logits de classification (4 classes)
            return self.classifier(features)
            
        elif any(k in self.backbone_name for k in ['vit', 'deit', 'swin', 'pvt', 't2t','twins']):
            # For transformer backbones, return the classifier logits directly when the model is configured for classification.
            if hasattr(self.model, 'head') and self.model.head is not None:
                # Use the native classification head to produce logits for CrossEntropyLoss.
                return self.model(x)
            else:
                # Fallback: use forward_features if a head is absent.
                features = self.model.forward_features(x)
                return features
        else:
            # self.model was created via create_model(..., num_classes=num_classes), so
            # calling it directly returns classification logits (batch, num_classes) as
            # required by CrossEntropyLoss during fine-tuning. forward_features() instead
            # returns the pre-pooling spatial feature map (batch, C, H, W), which crashes
            # CrossEntropyLoss ("only batches of spatial targets supported").
            return self.model(x)
    
    def save_finetuned_weights(self, save_path):
        """Sauvegarde les poids fine-tunés"""
        if self.backbone_name in CVT_CONFIGS:
            # Save both model and classifier
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict()
            }, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        print(f"[FineTunedFeatureExtractor] Poids fine-tunés sauvegardés dans {save_path}")
    
    def load_finetuned_weights(self, load_path, device='cpu'):
        """Charge les poids fine-tunés avec gestion flexible des préfixes"""
        checkpoint = torch.load(load_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Tentative 1: Chargement direct dans FineTunedFeatureExtractor
        try:
            self.load_state_dict(state_dict)
        except Exception:
            # Tentative 2: Nettoyage des préfixes 'model.' pour charger dans self.model
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    clean_state_dict[k[6:]] = v
                else:
                    clean_state_dict[k] = v
            try:
                self.model.load_state_dict(clean_state_dict)
            except Exception:
                self.model.load_state_dict(clean_state_dict, strict=False)

        print(f"[FineTunedFeatureExtractor] Poids fine-tunés chargés depuis {load_path}")

class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name: str, pretrained=True, finetuned_weights_path=None):
        super().__init__()
        
        # Check if it's a Hugging Face model
        if backbone_name in CVT_CONFIGS:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(f"transformers library required for {backbone_name}")
            
            model_name = CVT_CONFIGS[backbone_name]
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.backbone_name = backbone_name
            
            # Load fine-tuned weights if specified
            if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                self.load_finetuned_weights(finetuned_weights_path)
                print(f"[FeatureExtractor] Poids fine-tunés chargés depuis {finetuned_weights_path}")
        else:
            # Use timm models as before
            self.model = create_model(BACKBONES[backbone_name], pretrained=pretrained, num_classes=0)
            self.processor = None
            self.backbone_name = backbone_name
            
            # Charge les poids fine-tunés si spécifié
            if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                self.load_finetuned_weights(finetuned_weights_path)
                print(f"[FeatureExtractor] Poids fine-tunés chargés depuis {finetuned_weights_path}")
    
    def forward(self, x):
        if self.backbone_name in CVT_CONFIGS:
            # Process with Hugging Face model
            import torchvision.transforms as transforms
            to_pil = transforms.ToPILImage()
            
            batch_size = x.size(0)
            features = []
            
            for i in range(batch_size):
                img_tensor = x[i]
                img_pil = to_pil(img_tensor)
                inputs = self.processor(img_pil, return_tensors="pt")
                
                device = x.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Extraire les features correctement
                    if hasattr(outputs, 'last_hidden_state'):
                        # Pour les modèles qui retournent last_hidden_state
                        # Aplatir les features spatiales
                        feature = outputs.last_hidden_state.mean(dim=1).squeeze()  # [hidden_size]
                    elif hasattr(outputs, 'pooler_output'):
                        # Pour les modèles qui ont un pooler
                        feature = outputs.pooler_output.squeeze()
                    else:
                        # Fallback: utiliser la première sortie et l'aplatir
                        feature = outputs[0].mean(dim=1).squeeze()  # [hidden_size]
                    
                    # S'assurer que la dimension est correcte
                    if feature.dim() == 0:
                        feature = feature.unsqueeze(0)
                    elif feature.dim() > 1:
                        # Si on a encore des dimensions spatiales, les aplatir
                        feature = feature.flatten()
                    
                    features.append(feature)
            
            features = torch.stack(features)
            
            # Debug: afficher les dimensions
            print(f"[DEBUG] FeatureExtractor - Features shape: {features.shape}")
            
            return features
            
        elif any(k in self.backbone_name for k in ['vit', 'deit', 'swin', 'pvt', 't2t']):
            feats = self.model(x)  # [B, 768] ou [B, 1024]
        else:
            feats = self.model(x)
        return feats
    
    def load_finetuned_weights(self, weights_path):
        """Charge les poids fine-tunés dans le backbone"""
        if self.backbone_name in CVT_CONFIGS:
            # For Hugging Face models, load the model weights
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            # Charge les poids
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Adapte les poids si nécessaire (par exemple, retirer la head de classification)
            adapted_state_dict = {}
            for key, value in state_dict.items():
                # Ignore les poids de la head de classification si elle existe
                if 'head' in key and 'classifier' in key:
                    continue
                adapted_state_dict[key] = value
            
            # Charge les poids adaptés
            self.model.load_state_dict(adapted_state_dict, strict=False)
        print(f"[FeatureExtractor] Poids fine-tunés adaptés chargés depuis {weights_path}")

#Classifieur de 4 classes 
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, out_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class BinaryMLPClassifier(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.mlp(x)
# Classifieur hiérarchique  qui utilise un MLP 4 classes et un MLP binaire pour chaque paire de classes et retourne 
# les 2 classes les plus probables et leur probabilités respectives
class HierarchicalClassifier(nn.Module):
    def __init__(self, backbone_name: str, in_dim=768, out_dim=4):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone_name)
        self.mlp4 = MLPClassifier(in_dim=in_dim, out_dim=out_dim) # 
        # Génère tous les classifieurs binaires pour chaque paire de classes
        self.class_pairs = list(combinations(range(out_dim), 2))
        self.binary_mlps = nn.ModuleDict({
            f"{i}_{j}": BinaryMLPClassifier(in_dim=in_dim) for i, j in self.class_pairs
        })
    def forward(self, feats):
        #feats = self.feature_extractor(x)
        logits4 = self.mlp4(feats) # permet de predire les probabilités de chaque classe parmi les 4 classes
        probs4 = torch.softmax(logits4, dim=1) # permet de normaliser les probabilités pour que la somme des probabilités soit égale à 1
        # Trouve les deux classes les plus probables
        top2 = torch.topk(probs4, 2, dim=1).indices  # [B, 2] permet de trouver les 2 classes les plus probables
        out_bin = [] # permet de stocker les sorties des classifieurs binaires
        for i in range(feats.size(0)): # pour chaque image
            c1, c2 = sorted(top2[i].tolist()) # permet de trier les 2 classes les plus probables
            key = f"{c1}_{c2}" # permet de creer la clé pour le classifieur binaire
            bin_logits = self.binary_mlps[key](feats[i].unsqueeze(0)) # permet de predire les probabilités de chaque classe parmi les 2 classes binaires
            out_bin.append(bin_logits)
        out_bin = torch.cat(out_bin, dim=0) # permet de concatener les sorties des classifieurs binaires
        return logits4, out_bin, top2 # permet de retourner les probabilités de chaque classe parmi les 4 classes, les probabilités de chaque classe parmi les 2 classes binaires et les 2 classes les plus probables
# Classe pour dumper les features et les labels en utilisant le modèle de feature extractor
class FeatureDumper:
    def __init__(self, feature_extractor, device):
        self.feature_extractor = feature_extractor
        self.device = device
        self.feature_extractor.eval()
        # Transformations appliquées aux échantillons "_aug" (mêmes réglages que
        # MammographyDataset.augmentation_transforms dans training.py).
        import torchvision.transforms as transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        ])
        # Normalisation ImageNet, cohérente avec MammographyDataset (training.py),
        # attendue par les backbones pré-entraînés (ViT/ResNet/etc.). Sans elle, le
        # backbone reçoit des pixels bruts 0-255 hors de la plage qu'il attend, ce
        # qui dégrade la qualité des features extraites.
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def dump_features(self, df, image_root, save_features_path, save_labels_path, label_map=None, split='training'):
        print(f"[FeatureDumper] Début de l'extraction des features pour le split : {split}")
        df = df[df['split'] == split]
        features = []
        labels = []
        skipped_missing = 0
        density_map = {
            "DENSITY A": "density_A", "DENSITY B": "density_B",
            "DENSITY C": "density_C", "DENSITY D": "density_D"
        }
        for idx, row in enumerate(df.itertuples(index=True)):
            if idx % 100 == 0 and idx > 0:
                print(f"[FeatureDumper] {idx}/{len(df)} images traitées...")
            split_dir = 'train' if row.split == 'training' else 'test'
            density = row.breast_density
            study_id = str(row.study_id)
            image_id = str(row.image_id)

            # Les lignes "_aug" (générées par DataAugmentationManager) n'ont pas de
            # fichier DICOM propre : on retrouve l'image réelle sous-jacente en
            # retirant le suffixe, puis on lui applique une augmentation aléatoire
            # après lecture (sinon la ligne est silencieusement ignorée plus bas).
            is_aug_val = str(getattr(row, 'is_augmented', '')).lower()
            is_augmented = is_aug_val == 'true' or '_aug' in study_id or '_aug' in image_id
            lookup_study_id = study_id.replace('_aug', '')
            lookup_image_id = image_id.replace('_aug', '')

            # Chemin par défaut organized_layout
            image_path = os.path.join(image_root, split_dir, density_map[density], lookup_study_id, f"{lookup_image_id}.dicom")

            # Fallback vers le format standard VinDr-Mammo
            if not os.path.exists(image_path):
                image_path = os.path.join(image_root, 'images', lookup_study_id, f"{lookup_image_id}.dicom")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_root, lookup_study_id, f"{lookup_image_id}.dicom")

            if not os.path.exists(image_path):
                skipped_missing += 1
                continue
            img = read_dicom(image_path)
            img = preprocess_image(img, laterality=row.laterality)

            if is_augmented:
                img_pil = Image.fromarray(img.astype(np.uint8))
                img = np.array(self.augment_transform(img_pil))

            img_tensor = self.normalize_transform(img.astype(np.uint8)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.feature_extractor(img_tensor)
            features.append(feat.cpu().numpy().squeeze())
            label = row.breast_density if label_map is None else label_map[row.breast_density]
            labels.append(label)    
        features = np.stack(features)
        np.save(save_features_path, features)
        np.save(save_labels_path, np.array(labels))
        print(f"[FeatureDumper] Extraction terminée. {len(features)} features gardées, {skipped_missing} lignes ignorées (fichier introuvable). Sauvegardés dans {save_features_path}, labels dans {save_labels_path}")

# Utilitaire pour charger les features et entraîner un MLP


class FeatureMLPTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, features, labels, epochs=10, batch_size=32, lr=1e-4, device='cpu', resume_path=None):
        print("[FeatureMLPTrainer] Début de l'entraînement du MLP sur les features...")

        if resume_path is not None and os.path.exists(resume_path):
            self.model.load_state_dict(torch.load(resume_path, map_location=device))
            print(f"[FeatureMLPTrainer] Poids chargés depuis {resume_path}, reprise de l'entraînement.")

        # -- Pondération automatique des classes --
        classes = np.unique(labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        # ----------------------------------------

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)  # pondération automatique native

        self.model.train()
        losses = []
        accuracies = []

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = self.model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            avg_loss = running_loss / len(loader)
            acc = correct / total if total > 0 else 0.0
            losses.append(avg_loss)
            accuracies.append(acc)

            if (epoch+1) % 2 == 0 or epoch == 0 or epoch == epochs-1:
                print(f"[FeatureMLPTrainer] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")

        print("[FeatureMLPTrainer] Entraînement terminé.")

        try:
            from pathlib import Path
            import matplotlib.pyplot as plt
            output_dir = Path("graphes")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / "mlp_training_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            pass

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, device='cpu'):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)

    def predict(self, features, device='cpu'):
        self.model.eval()
        X = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = self.model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds


# Exemple d'utilisation :
# Pour le MLP 4 classes :
#   mlp4 = MLPClassifier(in_dim=768, out_dim=4)
#   trainer4 = FeatureMLPTrainer(mlp4)
#   trainer4.train(features, labels, ...)
# Pour un binaire :
#   mlp_bin = BinaryMLPClassifier(in_dim=768)
#   trainer_bin = FeatureMLPTrainer(mlp_bin)
#   trainer_bin.train(features, labels, ...) 