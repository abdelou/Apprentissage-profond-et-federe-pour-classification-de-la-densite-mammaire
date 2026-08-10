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
        """
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
        return self.mlp(glcm_features)


# Branche Histogramme (Conforme Section 6.2.3.2 & Section 6.2.4.1)
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


class ImageBranch(nn.Module):
    """
    Branche pour traiter l'image de mammographie avec ResNet50.
    """
    def __init__(self, backbone: str = 'cnn', input_channels: int = 1, feature_dim: int = 512, 
                 pretrained: bool = True, finetuned_weights_path: Optional[str] = None):
        super(ImageBranch, self).__init__()
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        print(f"[ImageBranch] Construction de la branche image avec CNN (ResNet50)")
        self.cnn = models.resnet50(pretrained=pretrained)
        
        if input_channels == 1:
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, feature_dim)
        
        if finetuned_weights_path and os.path.exists(finetuned_weights_path):
            print(f"[ImageBranch] Chargement des poids fine-tunés CNN: {finetuned_weights_path}")
            self.load_finetuned_weights(finetuned_weights_path)
    
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.cnn(image_tensor)

    def load_finetuned_weights(self, load_path: str, device: str = 'cpu'):
        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location='cpu')
            keys_to_delete = [k for k in state_dict.keys() if 'fc.' in k]
            for key in keys_to_delete:
                del state_dict[key]
            self.cnn.load_state_dict(state_dict, strict=False)

    def save_finetuned_weights(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.cnn.state_dict(), save_path)


class HybridMammographyClassifier(nn.Module):
    """
    Modèle hybride combinant ResNet50 (Branche Image MLO) et Histogramme 256 bins (Figure 15).
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
        print(f"  - Image features (ResNet50): {self.image_feature_dim}")
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
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, image_tensor: torch.Tensor, hist_tensor: torch.Tensor) -> torch.Tensor:
        image_features = self.image_branch(image_tensor)
        hist_features = self.histogram_branch(hist_tensor)
        combined_features = torch.cat([image_features, hist_features], dim=1)
        return self.classifier(combined_features)
    
    def save_finetuned_weights(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)


def create_histogram_from_image(image_tensor: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    batch_size = image_tensor.shape[0]
    image_normalized = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-8)
    bin_indices = (image_normalized * (num_bins - 1)).long()
    histograms = torch.zeros(batch_size, num_bins, device=image_tensor.device)
    for i in range(batch_size):
        hist = torch.bincount(bin_indices[i].flatten(), minlength=num_bins)
        histograms[i] = hist.float()
    histograms = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-8)
    return histograms