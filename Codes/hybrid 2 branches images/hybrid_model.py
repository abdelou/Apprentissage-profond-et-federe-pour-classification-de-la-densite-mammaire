import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import timm # Pour RexNet et autres modèles

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
  
  def __init__(self, backbone = 'cnn', input_channels = 1, feature_dim = 512, 
         pretrained = True, finetuned_weights_path = None):
    super(ImageBranch, self).__init__()
    
    self.backbone = backbone
    self.feature_dim = feature_dim
    
    if backbone == 'rexnet_150':
      print(f"[ImageBranch] construction de la branche image avec RexNet150")
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
        print(f"[ImageBranch] Chargement des poids fine-tunés rexnet_150: {finetuned_weights_path}")
        self.load_finetuned_weights(finetuned_weights_path)
    elif backbone == 'resnet50':
      print(f"[ImageBranch] construction de la branche image avec ResNet50")
      import torchvision.models as models
      self.cnn = models.resnet50(pretrained=pretrained)
      if input_channels == 1:
        old_conv = self.cnn.conv1
        self.cnn.conv1 = nn.Conv2d(
          1, old_conv.out_channels,
          kernel_size=old_conv.kernel_size,
          stride=old_conv.stride,
          padding=old_conv.padding,
          bias=old_conv.bias is not None
        )
      in_features = self.cnn.fc.in_features
      self.cnn.fc = nn.Linear(in_features, feature_dim)
      if finetuned_weights_path and os.path.exists(finetuned_weights_path):
        print(f"[ImageBranch] Chargement des poids fine-tunés resnet50: {finetuned_weights_path}")
        self.load_finetuned_weights(finetuned_weights_path)
    
    
    else:
      raise ValueError(f"Backbone '{backbone}' non supporté. Utilisez 'cnn' ou 'vit'.")
  
  def forward(self, image_tensor) :
    """
    Forward pass pour l'image.
    
    Args:
      image_tensor: Tenseur d'image de forme (batch_size, 1, 224, 224)
      
    Returns:
      Features extraites de l'image, de dimension 768 pour ViT ou feature_dim pour CNN
    """
    if self.backbone in ['rexnet_150', 'resnet50']:
      return self.cnn(image_tensor)
    elif self.backbone == 'vit':
      outputs = self.vit(image_tensor)
      features = outputs.last_hidden_state[:, 0, :]
      return features
  
  def freeze_layers(self, freeze_last_layer = False):
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
  
  def unfreeze_layers(self, last_n_layers = 0):
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
  
  def save_finetuned_weights(self, save_path):
    """Sauvegarde les poids de la branche image fine-tunée."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if self.backbone == 'resnet50' or self.backbone == 'rexnet_150':
      torch.save(self.cnn.state_dict(), save_path)
    elif self.backbone == 'vit':
      torch.save({
        'vit_state_dict': self.vit.state_dict(),
        'projection_state_dict': self.vit_projection.state_dict()
      }, save_path)
    print(f"Poids de la branche image sauvegardés: {save_path}")
  
  def load_finetuned_weights(self, load_path, device = 'cpu', freeze_layers = False):
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
        if self.backbone == 'rexnet_150' or self.backbone == 'resnet50':
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
              if i < 6: # Geler les 6 premières couches
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
         input_channels = 1,
         image_feature_dim = 512,
         num_classes = 4,
         dropout = 0.3,
         pretrained = True,
         resnet50_weights = None,
         rexnet150_weights = None):
    super(HybridMammographyClassifier, self).__init__()

    # Branche image MLO (ResNet50)
    self.mlo_branch = ImageBranch(
      backbone='resnet50',
      input_channels=input_channels,
      feature_dim=image_feature_dim,
      pretrained=pretrained,
      finetuned_weights_path=resnet50_weights
    )
    # Branche image CC (RexNet150)
    self.cc_branch = ImageBranch(
      backbone='rexnet_150',
      input_channels=input_channels,
      feature_dim=image_feature_dim,
      pretrained=pretrained,
      finetuned_weights_path=rexnet150_weights
    )

    # Concaténation des features des deux branches
    total_features = 2 * image_feature_dim
    print(f"[HybridModel] Dimensions des features:")
    print(f" - MLO features: {image_feature_dim}")
    print(f" - CC features: {image_feature_dim}")
    print(f" - Total features: {total_features}")

    self.classifier = nn.Sequential(
      nn.Linear(total_features, 512),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(128, num_classes)
    )

    # Initialisation des poids
    self._initialize_weights()

  def forward(self, mlo_image, cc_image):
    # Extraire les features de chaque branche
    mlo_feat = self.mlo_branch(mlo_image)
    cc_feat = self.cc_branch(cc_image)
    # Concaténer
    features = torch.cat([mlo_feat, cc_feat], dim=1)
    # Classification finale
    out = self.classifier(features)
    return out

  def _initialize_weights(self):
    """Initialisation des poids pour les couches personnalisées."""
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
  
  
  
  def get_feature_dimensions(self) :
    """
    Retourne les dimensions des features pour debugging.
    Returns:
      Tuple (image_features_dim, glcm_features_dim, combined_features_dim)
    """
    return (
      self.mlo_branch.feature_dim,
      self.cc_branch.feature_dim,
      self.mlo_branch.feature_dim + self.cc_branch.feature_dim
    )
  
  def save_finetuned_weights(self, save_path):
    """Sauvegarde les poids du modèle hybride fine-tuné."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(self.state_dict(), save_path)
    print(f"Modèle hybride sauvegardé: {save_path}")
  
  def load_finetuned_weights(self, load_path, device = 'cpu'):
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
            print(f" {layer}:")
            print(f"  - Forme attendue: {model_dict[layer].shape}")
            print(f"  - Forme dans checkpoint: {checkpoint[layer].shape}")
  
  def freeze_image_branches(self):
    """Gèle les deux branches image pour l'entraînement."""
    for param in self.mlo_branch.parameters():
      param.requires_grad = False
    for param in self.cc_branch.parameters():
      param.requires_grad = False
    print("Branches image gelées (MLO et CC)")

  def unfreeze_image_branches(self):
    """Dégèle les deux branches image pour l'entraînement."""
    for param in self.mlo_branch.parameters():
      param.requires_grad = True
    for param in self.cc_branch.parameters():
      param.requires_grad = True
    print("Branches image dégelées (MLO et CC)")

