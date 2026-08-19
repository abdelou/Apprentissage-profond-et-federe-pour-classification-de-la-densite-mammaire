import sys
import os
import torch
import torch.nn as nn
from timm import create_model
from itertools import combinations
import numpy as np
import time
from PIL import Image
from preprocessing import read_dicom, preprocess_image
from sklearn.utils.class_weight import compute_class_weight

try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers n'est pas disponible.")

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
    'cvt': 'microsoft/cvt-13',
    'cvt-13': 'microsoft/cvt-13',
    'cvt-21': 'microsoft/cvt-21',
    'cvt-w24': 'microsoft/cvt-w24-384-22k',
}

CVT_CONFIGS = {
    'cvt': 'microsoft/cvt-13',
    'cvt-13': 'microsoft/cvt-13',
    'cvt-21': 'microsoft/cvt-21', 
    'cvt-w24': 'microsoft/cvt-w24-384-22k',
}

class HuggingFaceFeatureExtractor(nn.Module):
    # Extracteur avec modeles huggingface
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("La bibliothèque transformers est requise")
        
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
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
                feature = outputs.last_hidden_state.mean(dim=1).squeeze()
                features.append(feature)
        
        return torch.stack(features)

class FineTunedFeatureExtractor(nn.Module):
    # Feature extractor fine-tune sur les mammographies
    def __init__(self, backbone_name, num_classes=4, pretrained=True):
        super().__init__()
        
        if backbone_name in CVT_CONFIGS:
            print(f"Modèle HF sélectionné : {backbone_name}")
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(f"transformers requis pour {backbone_name}")
            
            model_name = CVT_CONFIGS[backbone_name]
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            if hasattr(self.model.config, 'hidden_sizes'):
                hidden_size = self.model.config.hidden_sizes[-1]
            elif hasattr(self.model.config, 'hidden_size'):
                hidden_size = self.model.config.hidden_size
            else:
                if 'cvt-13' in backbone_name:
                    hidden_size = 384
                elif 'cvt-21' in backbone_name:
                    hidden_size = 512
                elif 'cvt-w24' in backbone_name:
                    hidden_size = 384
                else:
                    hidden_size = 384
            
            self.classifier = nn.Linear(hidden_size, num_classes)
            
        else:
            print(f"Modèle timm sélectionné : {backbone_name}")
            self.model = create_model(BACKBONES[backbone_name], pretrained=pretrained, num_classes=num_classes)
            self.processor = None
            
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
    def forward(self, x):
        if self.backbone_name in CVT_CONFIGS:
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
                    if hasattr(outputs, 'last_hidden_state'):
                        feature = outputs.last_hidden_state.mean(dim=1).squeeze()
                    elif hasattr(outputs, 'pooler_output'):
                        feature = outputs.pooler_output.squeeze()
                    else:
                        feature = outputs[0].mean(dim=1).squeeze()
                    
                    if feature.dim() == 0:
                        feature = feature.unsqueeze(0)
                    elif feature.dim() > 1:
                        feature = feature.flatten()
                    
                    features.append(feature)
            
            features = torch.stack(features)
            
            if features.shape[1] != self.classifier.in_features:
                print(f"Ajustement dimension : features={features.shape[1]}, attendu={self.classifier.in_features}")
                if features.shape[1] < self.classifier.in_features:
                    padding = torch.zeros(features.shape[0], self.classifier.in_features - features.shape[1], device=features.device)
                    features = torch.cat([features, padding], dim=1)
                else:
                    features = features[:, :self.classifier.in_features]
            
            return self.classifier(features)
            
        elif any(k in self.backbone_name for k in ['vit', 'deit', 'swin', 'pvt', 't2t','twins']):
            if hasattr(self.model, 'head') and self.model.head is not None:
                return self.model(x)
            else:
                features = self.model.forward_features(x)
                return features
        else:
            return self.model(x)
    
    def save_finetuned_weights(self, save_path):
        if self.backbone_name in CVT_CONFIGS:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict()
            }, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        print(f"Poids sauvegardés dans {save_path}")
    
    def load_finetuned_weights(self, load_path, device='cpu'):
        checkpoint = torch.load(load_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        try:
            self.load_state_dict(state_dict)
        except Exception:
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

        print(f"Poids chargés depuis {load_path}")

class FeatureExtractor(nn.Module):
    # Classe pour extraire les features
    def __init__(self, backbone_name, pretrained=True, finetuned_weights_path=None):
        super().__init__()
        
        if backbone_name in CVT_CONFIGS:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(f"transformers requis pour {backbone_name}")
            
            model_name = CVT_CONFIGS[backbone_name]
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.backbone_name = backbone_name
            
            if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                self.load_finetuned_weights(finetuned_weights_path)
        else:
            self.model = create_model(BACKBONES[backbone_name], pretrained=pretrained, num_classes=0)
            self.processor = None
            self.backbone_name = backbone_name
            
            if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                self.load_finetuned_weights(finetuned_weights_path)
    
    def forward(self, x):
        if self.backbone_name in CVT_CONFIGS:
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
                    if hasattr(outputs, 'last_hidden_state'):
                        feature = outputs.last_hidden_state.mean(dim=1).squeeze()
                    elif hasattr(outputs, 'pooler_output'):
                        feature = outputs.pooler_output.squeeze()
                    else:
                        feature = outputs[0].mean(dim=1).squeeze()
                    
                    if feature.dim() == 0:
                        feature = feature.unsqueeze(0)
                    elif feature.dim() > 1:
                        feature = feature.flatten()
                    
                    features.append(feature)
            
            features = torch.stack(features)
            return features
            
        elif any(k in self.backbone_name for k in ['vit', 'deit', 'swin', 'pvt', 't2t']):
            feats = self.model(x)
        else:
            feats = self.model(x)
        return feats
    
    def load_finetuned_weights(self, weights_path):
        if self.backbone_name in CVT_CONFIGS:
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            state_dict = torch.load(weights_path, map_location='cpu')
            adapted_state_dict = {}
            for key, value in state_dict.items():
                if 'head' in key and 'classifier' in key:
                    continue
                adapted_state_dict[key] = value
            self.model.load_state_dict(adapted_state_dict, strict=False)
        print(f"Chargement des poids fins-tunés depuis {weights_path}")

class MLPClassifier(nn.Module):
    # Classifieur MLP simple 4 classes
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
    # Classifieur MLP binaire
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

class HierarchicalClassifier(nn.Module):
    # Classifieur hierarchique
    def __init__(self, backbone_name, in_dim=768, out_dim=4):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone_name)
        self.mlp4 = MLPClassifier(in_dim=in_dim, out_dim=out_dim)
        self.class_pairs = list(combinations(range(out_dim), 2))
        self.binary_mlps = nn.ModuleDict({
            f"{i}_{j}": BinaryMLPClassifier(in_dim=in_dim) for i, j in self.class_pairs
        })
    def forward(self, feats):
        logits4 = self.mlp4(feats)
        probs4 = torch.softmax(logits4, dim=1)
        top2 = torch.topk(probs4, 2, dim=1).indices
        out_bin = []
        for i in range(feats.size(0)):
            c1, c2 = sorted(top2[i].tolist())
            key = f"{c1}_{c2}"
            bin_logits = self.binary_mlps[key](feats[i].unsqueeze(0))
            out_bin.append(bin_logits)
        out_bin = torch.cat(out_bin, dim=0)
        return logits4, out_bin, top2

class FeatureDumper:
    # Pour sauvegarder les features extraites
    def __init__(self, feature_extractor, device):
        self.feature_extractor = feature_extractor
        self.device = device
        self.feature_extractor.eval()
        
        import torchvision.transforms as transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        ])
        
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def dump_features(self, df, image_root, save_features_path, save_labels_path, label_map=None, split='training'):
        print(f"Extraction features split : {split}")
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
                print(f"Traitement : {idx}/{len(df)}...")
            split_dir = 'train' if row.split == 'training' else 'test'
            density = row.breast_density
            study_id = str(row.study_id)
            image_id = str(row.image_id)

            is_aug_val = str(getattr(row, 'is_augmented', '')).lower()
            is_augmented = is_aug_val == 'true' or '_aug' in study_id or '_aug' in image_id
            lookup_study_id = study_id.replace('_aug', '')
            lookup_image_id = image_id.replace('_aug', '')

            image_path = os.path.join(image_root, split_dir, density_map[density], lookup_study_id, f"{lookup_image_id}.dicom")

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
        print(f"Extraction terminee ({len(features)} Features). Sauvegarde dans {save_features_path}")

class FeatureMLPTrainer:
    # Trainer pour le MLP
    def __init__(self, model):
        self.model = model

    def train(self, features, labels, epochs=10, batch_size=32, lr=1e-4, device='cpu', resume_path=None):
        print("Debut entrainement du MLP...")

        if resume_path is not None and os.path.exists(resume_path):
            self.model.load_state_dict(torch.load(resume_path, map_location=device))

        classes = np.unique(labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

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
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")

        print("Entrainement MLP termine.")

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