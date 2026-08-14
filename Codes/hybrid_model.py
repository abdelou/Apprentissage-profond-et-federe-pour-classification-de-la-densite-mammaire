import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# Branche GLCM (Conservée intacte pour référence future)
class GLCMDescriptorMLP(nn.Module):
    """
    Branche MLP pour traiter les descripteurs GLCM (contraste, homogénéité, énergie, corrélation, entropie).
    """
    def __init__(self, hidden_dims: list = [32, 16], dropout: float = 0.2):
        super().__init__()
        self.input_dim = 5  # 5 descripteurs GLCM
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extrait les descripteurs GLCM d'un batch d'images et les passe dans le MLP.
        Args:
            image_tensor: (batch_size, 1, H, W)
        Returns:
            Features extraites (batch_size, output_dim)
        """
        batch_size = image_tensor.shape[0]
        glcm_features = []
        for i in range(batch_size):
            img = image_tensor[i, 0].cpu().numpy()
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            # GLCM sur distance 1, angle 0
            glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            # Entropie
            glcm_flat = glcm[:, :, 0, 0].flatten()
            entropy = -np.sum(glcm_flat * np.log2(glcm_flat + 1e-10))
            glcm_features.append([contrast, homogeneity, energy, correlation, entropy])
        glcm_features = torch.tensor(glcm_features, dtype=torch.float32, device=image_tensor.device)
        return self.mlp(glcm_features)


# Branche Histogramme (Conforme Section 6.2.3.2 du Mémoire)
class HistogramMLP(nn.Module):
    """
    Branche MLP pour traiter l'histogramme des niveaux de gris (entrée 256 bins -> MLP [128, 64] -> sortie 64D).
    Conforme à la section 6.2.3.2 du mémoire.
    """
    def __init__(self, input_dim: int = 256, hidden_dims: list = [128, 64], dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, hist_tensor: torch.Tensor) -> torch.Tensor:
        return self.mlp(hist_tensor)


class ImageBranch(nn.Module):
    """
    Branche pour traiter l'image de mammographie.
    Supporte CNN (ResNet50) ou ViT selon disponibilité.
    
    Args:
        backbone: Type de backbone ('cnn' ou 'vit')
        input_channels: Nombre de canaux d'entrée (1 pour mammographie)
        feature_dim: Dimension des features extraites
        pretrained: Utiliser des poids pré-entraînés
        finetuned_weights_path: Chemin vers des poids fine-tunés
    """
    
    def __init__(self, backbone: str = 'cnn', input_channels: int = 1, feature_dim: int = 512, 
                 pretrained: bool = True, finetuned_weights_path: Optional[str] = None):
        super(ImageBranch, self).__init__()
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        if backbone == 'cnn':
            print(f"[ImageBranch] Construction de la branche image avec CNN (ResNet50)")
            # Utilisation de ResNet50 pré-entraîné
            self.cnn = models.resnet50(pretrained=pretrained)
            
            # Modification de la première couche pour accepter 1 canal
            if input_channels == 1:
                self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Remplacer la dernière couche pour obtenir la dimension souhaitée
            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, feature_dim)
            
            # Charger des poids fine-tunés si spécifié
            if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                print(f"[ImageBranch] Chargement des poids fine-tunés CNN: {finetuned_weights_path}")
                self.load_finetuned_weights(finetuned_weights_path)
            
        elif backbone == 'vit':
            print(f"[ImageBranch] Construction de la branche image avec ViT")
            try:
                # Tentative d'import de ViT
                from transformers import ViTModel, ViTConfig
                
                # Configuration ViT qui correspond au modèle pré-entraîné
                config = ViTConfig(
                    image_size=224,      # Taille standard
                    patch_size=16,       # Même taille de patch que le modèle pré-entraîné
                    num_channels=1,       # Images en niveaux de gris
                    hidden_size=768,      # Même dimension que le modèle pré-entraîné
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_act='gelu',
                    attention_probs_dropout_prob=0.1,
                    hidden_dropout_prob=0.1,
                    initializer_range=0.02,
                    layer_norm_eps=1e-12,
                    qkv_bias=True
                )
                
                # Initialiser avec le modèle microsoft/biomedvlp-biovil-t
                self.vit = ViTModel.from_pretrained(
                    'microsoft/BiomedVLP-BioViL-T',
                    config=config,
                    ignore_mismatched_sizes=True  # Permet l'adaptation pour 1 canal
                )
                
                # Adapter la première couche pour les images en niveaux de gris
                if input_channels == 1:
                    old_embeddings = self.vit.embeddings.patch_embeddings.projection
                    new_embeddings = nn.Conv2d(1, config.hidden_size, 
                                             kernel_size=old_embeddings.kernel_size,
                                             stride=old_embeddings.stride,
                                             padding=old_embeddings.padding)
                    
                    # Initialiser les nouveaux poids en moyennant les canaux RGB
                    with torch.no_grad():
                        new_embeddings.weight.copy_(old_embeddings.weight.mean(dim=1, keepdim=True))
                        new_embeddings.bias.copy_(old_embeddings.bias)
                    
                    self.vit.embeddings.patch_embeddings.projection = new_embeddings
                
                self.vit_projection = nn.Linear(config.hidden_size, feature_dim)
                
                # Charger des poids fine-tunés si spécifié
                if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                    print(f"[ImageBranch] Chargement des poids fine-tunés ViT: {finetuned_weights_path}")
                    self.load_finetuned_weights(finetuned_weights_path)

            except ImportError:
                print("Warning: transformers non disponible, utilisation de ResNet50")
                self.backbone = 'cnn'
                self.cnn = models.resnet50(pretrained=pretrained)
                if input_channels == 1:
                    self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.cnn.fc = nn.Linear(self.cnn.fc.in_features, feature_dim)
                
                if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                    print(f"[ImageBranch] Chargement des poids fine-tunés: {finetuned_weights_path}")
                    self.load_finetuned_weights(finetuned_weights_path)
        else:
            raise ValueError(f"Backbone '{backbone}' non supporté. Utilisez 'cnn' ou 'vit'.")
    
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour l'image.
        """
        if self.backbone == 'cnn':
            return self.cnn(image_tensor)
        elif self.backbone == 'vit':
            outputs = self.vit(image_tensor)
            features = outputs.last_hidden_state[:, 0, :]  # Token [CLS] (batch_size, 768)
            return features
    
    def freeze_layers(self, freeze_last_layer: bool = False):
        """Gèle les couches du modèle."""
        if self.backbone == 'cnn':
            for name, param in self.cnn.named_parameters():
                if not freeze_last_layer and 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("[ImageBranch] Couches CNN gelées" + (" (incluant fc)" if freeze_last_layer else " (sauf fc)"))
        elif self.backbone == 'vit':
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.vit_projection.parameters():
                param.requires_grad = not freeze_last_layer
            print("[ImageBranch] ViT gelé" + (" (incluant projection)" if freeze_last_layer else " (sauf projection)"))
    
    def unfreeze_layers(self, last_n_layers: int = 0):
        """Dégèle les n dernières couches du modèle."""
        if self.backbone == 'cnn':
            if last_n_layers == 0:
                for param in self.cnn.parameters():
                    param.requires_grad = True
                print("[ImageBranch] Toutes les couches CNN dégelées")
            else:
                layers = list(self.cnn.children())
                for layer in layers[-last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"[ImageBranch] Dernières {last_n_layers} couches CNN dégelées")
        elif self.backbone == 'vit':
            if last_n_layers == 0:
                for param in self.vit.parameters():
                    param.requires_grad = True
                print("[ImageBranch] Toutes les couches ViT dégelées")
            else:
                for layer in self.vit.encoder.layer[-last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"[ImageBranch] Dernières {last_n_layers} couches ViT dégelées")
    
    def load_finetuned_weights(self, load_path: str, freeze_layers: bool = False, device: str = 'cpu'):
        """Charge les poids fine-tunés pour la branche image."""
        if os.path.exists(load_path):
            original_device = next(self.parameters()).device
            self.cpu()
            
            try:
                if self.backbone == 'cnn':
                    state_dict = torch.load(load_path, map_location='cpu')
                    keys_to_delete = [k for k in state_dict.keys() if 'fc.' in k]
                    for key in keys_to_delete:
                        del state_dict[key]
                    self.cnn.load_state_dict(state_dict, strict=False)
                    
                    if freeze_layers:
                        for name, param in self.cnn.named_parameters():
                            if 'fc.' not in name:
                                param.requires_grad = False
                        print("[ImageBranch] Couches convolutives gelées")
                    
                elif self.backbone == 'vit':
                    checkpoint = torch.load(load_path, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        vit_state = checkpoint.get('vit_state_dict', checkpoint)
                        vit_state = {k: v for k, v in vit_state.items() 
                                   if not k.startswith('classifier') and not k.startswith('pooler')}
                        self.vit.load_state_dict(vit_state, strict=False)
                        print("[ImageBranch] Poids ViT chargés avec succès")
                
                if device != 'cpu':
                    self.to(device)
                elif original_device.type == 'cuda':
                    self.to(original_device)
                
            except Exception as e:
                print(f"Erreur lors du chargement des poids: {str(e)}")
                if original_device.type == 'cuda':
                    self.to(original_device)
                raise e
        else:
            print(f"Aucun poids trouvé à: {load_path}")


class HybridMammographyClassifier(nn.Module):
    """
    Modèle hybride combinant CNN/ViT (Branche Image) et Histogramme 256 bins (Branche Histogramme)
    pour la classification de la densité mammaire. Conforme à la Section 6.2.3.2 du Mémoire.
    
    Args:
        backbone: Type de backbone pour l'image ('cnn' ou 'vit')
        input_channels: Nombre de canaux d'entrée (1 pour mammographie)
        image_feature_dim: Dimension des features d'image (512 par défaut)
        hist_hidden_dims: Dimensions cachées pour l'histogramme ([128, 64] -> sortie 64D)
        num_classes: Nombre de classes (4 pour densité A, B, C, D)
        dropout: Taux de dropout (0.3 par défaut)
        pretrained: Utiliser des poids pré-entraînés pour l'image
        finetuned_weights_path: Chemin vers des poids fine-tunés pour la branche image
    """
    
    def __init__(self, 
                 backbone: str = 'cnn',
                 input_channels: int = 1,
                 image_feature_dim: int = 512,
                 hist_hidden_dims: list = [128, 64],
                 num_classes: int = 4,
                 dropout: float = 0.3,
                 pretrained: bool = True,
                 finetuned_weights_path: Optional[str] = None):
        
        super(HybridMammographyClassifier, self).__init__()
        
        # Branche image (ResNet50 / ViT -> 512D)
        self.image_branch = ImageBranch(
            backbone=backbone,
            input_channels=input_channels,
            feature_dim=image_feature_dim,
            pretrained=pretrained,
            finetuned_weights_path=finetuned_weights_path
        )
        
        # Branche Histogramme (256 bins -> MLP [128, 64] -> sortie 64D conforme section 6.2.3.2)
        self.histogram_branch = HistogramMLP(input_dim=256, hidden_dims=hist_hidden_dims, dropout=dropout)

        # Déterminer la dimension des features d'image en fonction du backbone
        if backbone == 'vit':
            self.image_feature_dim = 768
        else:
            self.image_feature_dim = image_feature_dim

        # Fusion par concaténation (512D + 64D = 576D total)
        total_features = self.image_feature_dim + self.histogram_branch.output_dim
        print(f"[HybridModel] Dimensions des features:")
        print(f"  - Image features: {self.image_feature_dim}")
        print(f"  - Histogram features: {self.histogram_branch.output_dim}")
        print(f"  - Total features (concaténation): {total_features}")

        # Classifieur unique MLP 4 classes avec dropout 0.3
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation des poids pour les couches personnalisées."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, image_tensor: torch.Tensor, hist_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle hybride avec branche Histogramme (256 bins).
        Args:
            image_tensor: Tenseur d'image de forme (batch_size, 1, 224, 224)
            hist_tensor: Tenseur d'histogramme de forme (batch_size, 256)
        Returns:
            Logits de classification de forme (batch_size, num_classes)
        """
        image_features = self.image_branch(image_tensor)
        hist_features = self.histogram_branch(hist_tensor)
        combined_features = torch.cat([image_features, hist_features], dim=1)
        logits = self.classifier(combined_features)
        return logits
    
    def get_feature_dimensions(self) -> Tuple[int, int, int]:
        """
        Retourne les dimensions des features pour debugging.
        Returns:
            Tuple (image_features_dim, hist_features_dim, combined_features_dim)
        """
        return (
            self.image_branch.feature_dim,
            self.histogram_branch.output_dim,
            self.image_branch.feature_dim + self.histogram_branch.output_dim
        )
    
    def save_finetuned_weights(self, save_path: str):
        """Sauvegarde les poids du modèle hybride fine-tuné."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"Modèle hybride sauvegardé: {save_path}")
    
    def load_finetuned_weights(self, load_path: str, device: str = 'cpu'):
        """Charge les poids fine-tunés du modèle hybride avec gestion des incompatibilités."""
        try:
            self.load_state_dict(torch.load(load_path, map_location=device))
            print(f"Chargement réussi des poids depuis: {load_path}")
        except RuntimeError as e:
            print("\nAttention: Chargement du checkpoint avec adaptation des couches...")
            checkpoint = torch.load(load_path, map_location=device)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Paramètres chargés avec succès: {len(pretrained_dict)} / {len(checkpoint)}")
    
    def freeze_image_branch(self):
        """Gèle la branche image pour l'entraînement."""
        for param in self.image_branch.parameters():
            param.requires_grad = False
        print("Branche image gelée")
    
    def unfreeze_image_branch(self):
        """Dégèle la branche image pour l'entraînement."""
        for param in self.image_branch.parameters():
            param.requires_grad = True
        print("Branche image dégelée")


def create_histogram_from_image(image_tensor: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    """
    Crée un histogramme à partir d'un tenseur d'image.
    
    Args:
        image_tensor: Tenseur d'image de forme (batch_size, 1, H, W)
        num_bins: Nombre de bins pour l'histogramme (256 bins par défaut)
        
    Returns:
        Tenseur d'histogramme normalisé de forme (batch_size, num_bins)
    """
    batch_size = image_tensor.shape[0]
    
    # Normaliser les valeurs entre 0 et 1
    image_normalized = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-8)
    
    # Convertir en indices de bins (0 à num_bins-1)
    bin_indices = (image_normalized * (num_bins - 1)).long()
    
    # Créer les histogrammes
    histograms = torch.zeros(batch_size, num_bins, device=image_tensor.device)
    
    for i in range(batch_size):
        hist = torch.bincount(bin_indices[i].flatten(), minlength=num_bins)
        histograms[i] = hist.float()
    
    # Normaliser l'histogramme
    histograms = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-8)
    
    return histograms


# Exemple d'utilisation et test
if __name__ == "__main__":
    print("=== Test du modèle hybride (Section 6.2.3.2) ===")
    
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Création du modèle hybride (Image + Histogramme 256 bins)
    model = HybridMammographyClassifier(
        backbone='cnn',
        input_channels=1,
        image_feature_dim=512,
        hist_hidden_dims=[128, 64],
        num_classes=4,
        dropout=0.3
    ).to(device)
    
    # Tenseurs de test
    image_tensor = torch.randn(batch_size, 1, 224, 224).to(device)
    hist_tensor = torch.randn(batch_size, 256).to(device)
    
    print(f"Tenseur d'image: {image_tensor.shape}")
    print(f"Tenseur d'histogramme: {hist_tensor.shape}")
    
    # Test du forward pass
    model.eval()
    with torch.no_grad():
        output = model(image_tensor, hist_tensor)
        print(f"Sortie du modèle: {output.shape}")
        
        # Test avec histogramme auto-généré
        auto_hist = create_histogram_from_image(image_tensor)
        output_auto = model(image_tensor, auto_hist)
        print(f"Sortie avec histogramme auto-généré: {output_auto.shape}")
    
    # Dimensions des features
    img_dim, hist_dim, combined_dim = model.get_feature_dimensions()
    print(f"\nDimensions des features:")
    print(f"  - Branche Image (CNN): {img_dim}D")
    print(f"  - Branche Histogramme (MLP [128, 64]): {hist_dim}D")
    print(f"  - Fusion (Concaténation): {combined_dim}D")
    print("\n Test du modèle hybride terminé avec succès!")