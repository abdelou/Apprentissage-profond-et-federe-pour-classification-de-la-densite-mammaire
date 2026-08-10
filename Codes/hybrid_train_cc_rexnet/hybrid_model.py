import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import timm  # Pour RexNet et autres modèles



# Branche Histogramme
class HistogramMLP(nn.Module):
    """
    Branche MLP pour traiter l'histogramme des niveaux de gris (entrée 256 bins -> MLP [128, 64] -> sortie 64D).
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


# Branche GLCM
class GLCMDescriptorMLP(nn.Module):
    """
    Branche MLP pour traiter les descripteurs GLCM (contraste, homogénéité, énergie, corrélation, entropie).
    """

    def __init__(self, hidden_dims: list = [32, 16], dropout: float = 0.2, num_classes: int = None):
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
        # Tête de classification optionnelle
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Linear(self.output_dim, num_classes)

    def forward(self, image_tensor: torch.Tensor, classify: bool = False):
        batch_size = image_tensor.shape[0]
        glcm_features = []
        for i in range(batch_size):
            img = image_tensor[i, 0].cpu().numpy()
            # L'image reçue ici est déjà passée par transforms.Normalize(mean=[0.485], std=[0.229]) :
            # on annule cette normalisation pour retrouver les niveaux de gris réels avant de les
            # recalculer sur [0, 255], sinon les valeurs négatives font déborder le cast en uint8
            # et le GLCM est calculé sur du bruit numérique plutôt que sur l'image.
            img = img * 0.229 + 0.485
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            glcm_flat = glcm[:, :, 0, 0].flatten()
            entropy = -np.sum(glcm_flat * np.log2(glcm_flat + 1e-10))
            glcm_features.append([contrast, homogeneity, energy, correlation, entropy])
        glcm_features = torch.tensor(glcm_features, dtype=torch.float32, device=image_tensor.device)
        features = self.mlp(glcm_features)
        if classify and self.classifier is not None:
            return self.classifier(features)
        return features

    def save_finetuned_weights(self, save_path: str, include_classifier: bool = False):
        """Sauvegarde les poids de la branche GLCM (optionnellement avec la tête de classification)."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state = {'mlp': self.mlp.state_dict()}
        if include_classifier and self.classifier is not None:
            state['classifier'] = self.classifier.state_dict()
        torch.save(state, save_path)
        print(f"Poids de la branche GLCM sauvegardés: {save_path}")

    def load_finetuned_weights(self, load_path: str, load_classifier: bool = False):
        """Charge les poids de la branche GLCM (optionnellement avec la tête de classification)."""
        if os.path.exists(load_path):
            state = torch.load(load_path, map_location='cpu')
            self.mlp.load_state_dict(state['mlp'])
            if load_classifier and self.classifier is not None and 'classifier' in state:
                self.classifier.load_state_dict(state['classifier'])
            print(f"Poids GLCM chargés: {load_path}")
        else:
            print(f"Aucun poids GLCM trouvés à: {load_path}")


class ImageBranch(nn.Module):
    """
    Branche pour traiter l'image de mammographie.
    Supporte CNN (RexNet150) ou ViT selon disponibilité.
    
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
            print(f"[ImageBranch] construction de la branche image avec RexNet150")
            time.sleep(5)
            # Utilisation de RexNet150 pré-entraîné
            self.cnn = timm.create_model('rexnet_150', pretrained=pretrained)
            
            # Modification de la première couche pour accepter 1 canal
            if input_channels == 1:
                old_conv = self.cnn.stem.conv
                self.cnn.stem.conv = nn.Conv2d(
                    1, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
            
            # Remplacer la dernière couche pour obtenir la dimension souhaitée
            in_features = self.cnn.head.fc.in_features
            self.cnn.head.fc = nn.Linear(in_features, feature_dim)
            
            # Charger des poids fine-tunés si spécifié
            if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                print(f"[ImageBranch] Chargement des poids fine-tunés cnn: {finetuned_weights_path}")
                self.load_finetuned_weights(finetuned_weights_path)
                time.sleep(5)  # Pause pour permettre l'affichage du message
        
        elif backbone == 'vit':
            print(f"[ImageBranch] construction de la branche image avec ViT")
            time.sleep(5)
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
                
                # Initialiser avec le modèle google/vit-base-patch16-224
                self.vit = ViTModel.from_pretrained(
                    'google/vit-base-patch16-224',
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
                    print(f"[ImageBranch] Chargement des poids fine-tunés: {finetuned_weights_path}")
                    time.sleep(10)
                    self.load_finetuned_weights(finetuned_weights_path)
                    time.sleep(5)

            except ImportError:
                print("Warning: transformers non disponible, utilisation de RexNet150")
                self.backbone = 'cnn'
                self.cnn = timm.create_model('rexnet_150', pretrained=pretrained)
                if input_channels == 1:
                    old_conv = self.cnn.stem.conv
                    self.cnn.stem.conv = nn.Conv2d(
                        1, old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None
                    )
                in_features = self.cnn.head.fc.in_features
                self.cnn.head.fc = nn.Linear(in_features, feature_dim)
                if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                    print(f"[ImageBranch] Chargement des poids fine-tunés: {finetuned_weights_path}")
                    self.load_finetuned_weights(finetuned_weights_path)
        else:
            raise ValueError(f"Backbone '{backbone}' non supporté. Utilisez 'cnn' ou 'vit'.")
    
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour l'image.
        
        Args:
            image_tensor: Tenseur d'image de forme (batch_size, 1, 224, 224)
            
        Returns:
            Features extraites de l'image, de dimension 768 pour ViT ou feature_dim pour CNN
        """
        if self.backbone == 'cnn':
            return self.cnn(image_tensor)
        elif self.backbone == 'vit':
            outputs = self.vit(image_tensor)
            # Retourner directement le token [CLS] comme features
            features = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
            return features  # Pas de projection, retourner les features brutes
    
    def freeze_layers(self, freeze_last_layer: bool = False):
        """
        Gèle les couches du modèle pour économiser la mémoire GPU.
        
        Args:
            freeze_last_layer: Si True, gèle aussi la dernière couche (fc ou projection)
        """
        if self.backbone == 'cnn':
            for name, param in self.cnn.named_parameters():
                if not freeze_last_layer and 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("[ImageBranch] Couches CNN gelées" + (" (incluant fc)" if freeze_last_layer else " (sauf fc)"))
        elif self.backbone == 'vit':
            # Geler le transformer
            for param in self.vit.parameters():
                param.requires_grad = False
            # Contrôler le gel de la couche de projection
            for param in self.vit_projection.parameters():
                param.requires_grad = not freeze_last_layer
            print("[ImageBranch] ViT gelé" + (" (incluant projection)" if freeze_last_layer else " (sauf projection)"))
    
    def unfreeze_layers(self, last_n_layers: int = 0):
        """
        Dégèle les n dernières couches du modèle.
        
        Args:
            last_n_layers: Nombre de dernières couches à dégeler (0 pour tout dégeler)
        """
        if self.backbone == 'cnn':
            if last_n_layers == 0:
                # Dégeler toutes les couches
                for param in self.cnn.parameters():
                    param.requires_grad = True
                print("[ImageBranch] Toutes les couches CNN dégelées")
            else:
                # Dégeler seulement les n dernières couches
                layers_to_unfreeze = []
                if last_n_layers >= 1:
                    layers_to_unfreeze.append('fc')
                if last_n_layers >= 2:
                    layers_to_unfreeze.append('layer4')
                if last_n_layers >= 3:
                    layers_to_unfreeze.append('layer3')
                    
                for name, param in self.cnn.named_parameters():
                    param.requires_grad = any(layer in name for layer in layers_to_unfreeze)
                print(f"[ImageBranch] Dernier(s) {last_n_layers} bloc(s) CNN dégelé(s)")
        
        elif self.backbone == 'vit':
            if last_n_layers == 0:
                # Dégeler tout
                for param in self.vit.parameters():
                    param.requires_grad = True
                for param in self.vit_projection.parameters():
                    param.requires_grad = True
                print("[ImageBranch] ViT entièrement dégelé")
            else:
                # Toujours dégeler la couche de projection
                for param in self.vit_projection.parameters():
                    param.requires_grad = True
                # Dégeler les n derniers blocks du transformer
                blocks = list(self.vit.encoder.layer)[-last_n_layers:]
                for block in blocks:
                    for param in block.parameters():
                        param.requires_grad = True
                print(f"[ImageBranch] Dernier(s) {last_n_layers} bloc(s) ViT dégelé(s)")
    
    def save_finetuned_weights(self, save_path: str):
        """Sauvegarde les poids de la branche image fine-tunée."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.backbone == 'cnn':
            torch.save(self.cnn.state_dict(), save_path)
        elif self.backbone == 'vit':
            torch.save({
                'vit_state_dict': self.vit.state_dict(),
                'projection_state_dict': self.vit_projection.state_dict()
            }, save_path)
        print(f"Poids de la branche image sauvegardés: {save_path}")
    
    def load_finetuned_weights(self, load_path: str, device: str = 'cpu', freeze_layers: bool = True):
        """
        Charge les poids fine-tunés de la branche image avec une stratégie adaptée au type de backbone.
        
        Args:
            load_path: Chemin vers les poids fine-tunés
            device: Appareil sur lequel charger les poids ('cpu' ou 'cuda')
            freeze_layers: Si True, applique une stratégie de gel des couches optimisée
        """
        if os.path.exists(load_path):
            # Déplacer temporairement le modèle sur CPU pour le chargement
            original_device = next(self.parameters()).device
            self.cpu()
            
            try:
                if self.backbone == 'cnn':
                    # Stratégie pour CNN (ResNet)
                    state_dict = torch.load(load_path, map_location='cpu')
                    keys_to_delete = [k for k in state_dict.keys() if 'fc.' in k]
                    for key in keys_to_delete:
                        del state_dict[key]
                    self.cnn.load_state_dict(state_dict, strict=False)
                    
                    if freeze_layers:
                        for name, param in self.cnn.named_parameters():
                            if 'fc.' not in name:
                                param.requires_grad = False
                        print("[ImageBranch] Couches convolutives gelées pour économiser la mémoire")
                    
                elif self.backbone == 'vit':
                    print("[ImageBranch] Chargement des poids fine-tunés pour ViT avec stratégie adaptée")
                    checkpoint = torch.load(load_path, map_location='cpu')
                    
                    # Gestion du chargement des poids
                    if isinstance(checkpoint, dict):
                        vit_state = checkpoint.get('vit_state_dict', checkpoint)
                        # Ne garder que les poids du transformer (pas de classification/pooling)
                        vit_state = {k: v for k, v in vit_state.items() 
                                   if not k.startswith('classifier') and not k.startswith('pooler')}
                        self.vit.load_state_dict(vit_state, strict=False)
                        
                        if freeze_layers:
                            # Stratégie de fine-tuning optimisée pour ViT :
                            
                            # 1. Embeddings : toujours entraînables car spécifiques aux mammographies
                            for param in self.vit.embeddings.parameters():
                                param.requires_grad = True
                            print("[ImageBranch] Embeddings laissés entraînables pour adaptation aux mammographies")
                            
                            # 2. Encoder layers : gel progressif
                            for i, layer in enumerate(self.vit.encoder.layer):
                                # Geler les premières couches (features générales)
                                if i < 4:
                                    for param in layer.parameters():
                                        param.requires_grad = False
                                # Couches intermédiaires partiellement gelées
                                elif i < 8:
                                    # Garder attention entraînable, geler MLP
                                    for name, param in layer.named_parameters():
                                        if 'attention' in name:
                                            param.requires_grad = True
                                        else:
                                            param.requires_grad = False
                                # Dernières couches totalement entraînables
                                else:
                                    for param in layer.parameters():
                                        param.requires_grad = True
                            
                            print("[ImageBranch] Stratégie de gel progressif appliquée:")
                            print("- 4 premières couches gelées (features générales)")
                            print("- 4 couches intermédiaires: attention entraînable")
                            print("- 4 dernières couches entièrement entraînables")
                        
                        print("[ImageBranch] Poids ViT chargés avec succès")
                    else:
                        print("[ImageBranch] Format de checkpoint non reconnu, utilisation du modèle pré-entraîné")
                    
                    if freeze_layers:
                        # Geler toutes les couches sauf les 6 dernières du transformer
                        for i, layer in enumerate(self.vit.encoder.layer):
                            if i < 6:  # Geler les 6 premières couches
                                for param in layer.parameters():
                                    param.requires_grad = False
                        print("[ImageBranch] 6 premières couches du transformer gelées")
                        
                        # La couche de projection reste toujours entraînable
                        for param in self.vit_projection.parameters():
                            param.requires_grad = True
                        print("[ImageBranch] Couche de projection laissée entraînable")
                
                print(f"Poids fine-tunés chargés: {load_path}")
                
                # Déplacer le modèle vers le device demandé
                if device != 'cpu':
                    self.to(device)
                elif original_device.type == 'cuda':
                    self.to(original_device)
                
            except Exception as e:
                print(f"Erreur lors du chargement des poids: {str(e)}")
                # S'assurer que le modèle retourne à son device original en cas d'erreur
                if original_device.type == 'cuda':
                    self.to(original_device)
                raise e
        else:
            print(f"Aucun poids trouvé à: {load_path}")


class HybridMammographyClassifier(nn.Module):
    """
    Modèle hybride combinant CNN/ViT et histogramme pour la classification de densité mammaire.
    
    Args:
        backbone: Type de backbone pour l'image ('cnn' ou 'vit')
        input_channels: Nombre de canaux d'entrée (1 pour mammographie)
        image_feature_dim: Dimension des features d'image
        hist_hidden_dims: Dimensions cachées pour l'histogramme
        num_classes: Nombre de classes (4 pour densité A, B, C, D)
        dropout: Taux de dropout
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
        
        # Branche image
        self.image_branch = ImageBranch(
            backbone=backbone,
            input_channels=input_channels,
            feature_dim=image_feature_dim,
            pretrained=pretrained,
            finetuned_weights_path=finetuned_weights_path
        )
        
        self.histogram_branch = HistogramMLP(input_dim=256, hidden_dims=hist_hidden_dims, dropout=dropout)
        self.image_feature_dim = image_feature_dim
        total_features = self.image_feature_dim + self.histogram_branch.output_dim
        print(f"[HybridModel] Dimensions des features:")
        print(f"  - Image features: {self.image_feature_dim}")
        print(f"  - Histogram features: {self.histogram_branch.output_dim}")
        print(f"  - Total features: {total_features}")

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
    
    def forward(self, image_tensor: torch.Tensor, hist_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass du modèle hybride combinant ReXNet150 et Histogramme.
        """
        image_features = self.image_branch(image_tensor)
        if hist_tensor is not None and hasattr(self, 'histogram_branch'):
            hist_features = self.histogram_branch(hist_tensor)
            combined_features = torch.cat([image_features, hist_features], dim=1)
        elif hasattr(self, 'glcm_branch'):
            glcm_features = self.glcm_branch(image_tensor)
            combined_features = torch.cat([image_features, glcm_features], dim=1)
        else:
            combined_features = image_features
        logits = self.classifier(combined_features)
        return logits
    
    def get_feature_dimensions(self) -> Tuple[int, int, int]:
        """
        Retourne les dimensions des features pour debugging.
        Returns:
            Tuple (image_features_dim, glcm_features_dim, combined_features_dim)
        """
        return (
            self.image_branch.feature_dim,
            self.glcm_branch.output_dim,
            self.image_branch.feature_dim + self.glcm_branch.output_dim
        )
    
    def save_finetuned_weights(self, save_path: str):
        """Sauvegarde les poids du modèle hybride fine-tuné."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"Modèle hybride sauvegardé: {save_path}")
    
    def load_finetuned_weights(self, load_path: str, device: str = 'cpu'):
        """Charge les poids fine-tunés du modèle hybride avec gestion des incompatibilités."""
        try:
            # Tenter de charger le modèle directement
            self.load_state_dict(torch.load(load_path, map_location=device))
            print(f"Chargement réussi des poids depuis: {load_path}")
        except RuntimeError as e:
            print("\nAttention: Tentative de chargement du checkpoint avec gestion des incompatibilités...")
            
            # Charger le checkpoint
            checkpoint = torch.load(load_path, map_location=device)
            
            # Créer un nouveau dict d'état qui ne contient que les couches compatibles
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            # Mettre à jour le dict d'état avec les poids compatibles
            model_dict.update(pretrained_dict)
            
            # Charger le dict d'état mis à jour
            self.load_state_dict(model_dict)
            
            print("\nRésumé du chargement des poids:")
            print(f"Total des paramètres dans le checkpoint: {len(checkpoint)}")
            print(f"Paramètres chargés avec succès: {len(pretrained_dict)}")
            print(f"Paramètres initialisés aléatoirement: {len(checkpoint) - len(pretrained_dict)}")
            
            # Afficher les couches qui n'ont pas été chargées
            skipped_layers = set(checkpoint.keys()) - set(pretrained_dict.keys())
            if skipped_layers:
                print("\nCouches non chargées (dimensions incompatibles):")
                for layer in sorted(skipped_layers):
                    if layer in model_dict:
                        print(f"  {layer}:")
                        print(f"    - Forme attendue: {model_dict[layer].shape}")
                        print(f"    - Forme dans checkpoint: {checkpoint[layer].shape}")
    
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
        num_bins: Nombre de bins pour l'histogramme
        
    Returns:
        Tenseur d'histogramme de forme (batch_size, num_bins)
    """
    batch_size = image_tensor.shape[0]
    
    # Normaliser les valeurs entre 0 et 1
    image_normalized = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-8)
    
    # Convertir en indices de bins (0 à num_bins-1)
    bin_indices = (image_normalized * (num_bins - 1)).long()
    
    # Créer les histogrammes
    histograms = torch.zeros(batch_size, num_bins, device=image_tensor.device)
    
    for i in range(batch_size):
        # Compter les occurrences de chaque bin
        hist = torch.bincount(bin_indices[i].flatten(), minlength=num_bins)
        histograms[i] = hist.float()
    
    # Normaliser les histogrammes
    histograms = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-8)
    
    return histograms


# Exemple d'utilisation et test
if __name__ == "__main__":
    print("=== Test du modèle hybride ===")
    
    # Paramètres de test
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Création du modèle
    model = HybridMammographyClassifier(
        backbone='cnn',
        input_channels=1,
        image_feature_dim=512,
        hist_hidden_dims=[128, 64],
        num_classes=4,
        dropout=0.3
    ).to(device)
    
    # Création de tenseurs factices
    image_tensor = torch.randn(batch_size, 1, 224, 224).to(device)
    hist_tensor = torch.randn(batch_size, 256).to(device)
    
    print(f"Tenseur d'image: {image_tensor.shape}")
    print(f"Tenseur d'histogramme: {hist_tensor.shape}")
    
    # Test du forward pass
    model.eval()
    with torch.no_grad():
        output = model(image_tensor, hist_tensor)
        print(f"Sortie du modèle: {output.shape}")
        print(f"Logits: {output}")
        
        # Test avec histogramme généré automatiquement
        auto_hist = create_histogram_from_image(image_tensor)
        print(f"Histogramme auto-généré: {auto_hist.shape}")
        
        output_auto = model(image_tensor, auto_hist)
        print(f"Sortie avec histogramme auto: {output_auto.shape}")
    
    # Affichage des dimensions des features
    img_dim, hist_dim, combined_dim = model.get_feature_dimensions()
    print(f"\nDimensions des features:")
    print(f"  - Image: {img_dim}")
    print(f"  - Histogramme: {hist_dim}")
    print(f"  - Combinées: {combined_dim}")
    
    # Test avec ViT (si disponible)
    try:
        print("\n=== Test avec ViT ===")
        model_vit = HybridMammographyClassifier(
            backbone='vit',
            input_channels=1,
            image_feature_dim=512,
            num_classes=4
        ).to(device)
        
        with torch.no_grad():
            output_vit = model_vit(image_tensor, hist_tensor)
            print(f"Sortie ViT: {output_vit.shape}")
            
    except Exception as e:
        print(f"ViT non disponible: {e}")
    
    print("\n✅ Test du modèle hybride terminé!") 