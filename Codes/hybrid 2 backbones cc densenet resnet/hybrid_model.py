import torch
import torch.nn as nn
from typing import Optional
import os
import timm
import torchvision.models as tvmodels


class ImageBranch(nn.Module):
  """Branche d'extraction pour un backbone donné (DenseNet-121 ou ResNet50)."""

  def __init__(self, backbone, input_channels = 1, feature_dim = 512,
         pretrained = True, finetuned_weights_path = None):
    super().__init__()
    self.backbone = backbone
    self.feature_dim = feature_dim

    if backbone == 'densenet121':
      # timm remplace directement la tête de classification par
      # Linear(in_features, feature_dim) via num_classes=feature_dim.
      self.cnn = timm.create_model('densenet121', pretrained=pretrained,
                     in_chans=input_channels, num_classes=feature_dim)
    elif backbone == 'resnet50':
      self.cnn = tvmodels.resnet50(pretrained=pretrained)
      if input_channels == 1:
        old_conv = self.cnn.conv1
        self.cnn.conv1 = nn.Conv2d(
          1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
          stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
        )
      in_features = self.cnn.fc.in_features
      self.cnn.fc = nn.Linear(in_features, feature_dim)
    else:
      raise ValueError(f"Backbone '{backbone}' non supporté ici (densenet121, resnet50).")

    if finetuned_weights_path and os.path.exists(finetuned_weights_path):
      print(f"[ImageBranch:{backbone}] Chargement des poids fine-tunés: {finetuned_weights_path}")
      self.load_finetuned_weights(finetuned_weights_path)

  def forward(self, x):
    return self.cnn(x)

  def save_finetuned_weights(self, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(self.cnn.state_dict(), save_path)

  def load_finetuned_weights(self, load_path):
    state_dict = torch.load(load_path, map_location='cpu')
    keys_to_delete = [k for k in state_dict.keys() if ('fc.' in k or 'classifier.' in k)]
    for k in keys_to_delete:
      del state_dict[k]
    self.cnn.load_state_dict(state_dict, strict=False)


class DualBackboneClassifier(nn.Module):
  """
  Fusion multi-backbones sur une SEULE vue (ici CC) : DenseNet-121 et ResNet50
  appliqués à la MÊME image, features concaténées + MLP. Même schéma FHDF que
  la variante MLO (RexNet150+ResNet50) — voir hybrid 2 backbones mlo rexnet
  resnet/hybrid_model.py pour la référence littérature.
  """

  def __init__(self,
         view = 'CC',
         backbone_a = 'densenet121',
         backbone_b = 'resnet50',
         input_channels = 1,
         image_feature_dim = 512,
         num_classes = 4,
         dropout = 0.3,
         pretrained = True,
         weights_a = None,
         weights_b = None):
    super().__init__()
    self.view = view

    self.branch_a = ImageBranch(backbone_a, input_channels, image_feature_dim, pretrained, weights_a)
    self.branch_b = ImageBranch(backbone_b, input_channels, image_feature_dim, pretrained, weights_b)

    total_features = 2 * image_feature_dim
    print(f"[DualBackboneModel] Vue: {view} | {backbone_a} + {backbone_b} | Total features: {total_features}")

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
    self._initialize_weights()

  def forward(self, image):
    feat_a = self.branch_a(image)
    feat_b = self.branch_b(image)
    features = torch.cat([feat_a, feat_b], dim=1)
    return self.classifier(features)

  def _initialize_weights(self):
    for module in self.classifier.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def save_finetuned_weights(self, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(self.state_dict(), save_path)
    print(f"Modèle double-backbone sauvegardé: {save_path}")

  def load_finetuned_weights(self, load_path, device = 'cpu'):
    self.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Chargement réussi des poids depuis: {load_path}")
