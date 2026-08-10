import torch
import torch.nn as nn
from typing import Optional, Tuple
import os
import timm


class SharedImageBranch(nn.Module):
    """
    Branche d'extraction d'image à poids partagés (schéma Siamese), utilisée pour
    les deux vues CC et MLO avec la MÊME instance de backbone. C'est le schéma
    standard de la littérature multi-vues en mammographie : les deux vues montrent
    la même anatomie (même sein), donc partager les poids améliore la généralisation
    avec moins de paramètres qu'une paire de backbones hétérogènes indépendants.

    Backbone: EfficientNet-B0 (timm), déjà utilisé ailleurs dans ce projet.
    """

    def __init__(self, backbone: str = 'efficientnet_b0', input_channels: int = 1,
                 feature_dim: int = 512, pretrained: bool = True,
                 finetuned_weights_path: Optional[str] = None):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        print(f"[SharedImageBranch] Construction du backbone partagé {backbone}")
        # num_classes=feature_dim: timm remplace directement la tête de classification
        # par une couche Linear(in_features, feature_dim), équivalent à ce que font les
        # autres branches du projet (nn.Linear(in_features, feature_dim) sur fc/head.fc).
        self.cnn = timm.create_model(
            backbone, pretrained=pretrained, in_chans=input_channels, num_classes=feature_dim
        )

        if finetuned_weights_path and os.path.exists(finetuned_weights_path):
            print(f"[SharedImageBranch] Chargement des poids fine-tunés: {finetuned_weights_path}")
            self.load_finetuned_weights(finetuned_weights_path)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.cnn(image_tensor)

    def save_finetuned_weights(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.cnn.state_dict(), save_path)
        print(f"Poids de la branche partagée sauvegardés: {save_path}")

    def load_finetuned_weights(self, load_path: str, device: str = 'cpu'):
        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location='cpu')
            keys_to_delete = [k for k in state_dict.keys() if 'classifier.' in k]
            for key in keys_to_delete:
                del state_dict[key]
            self.cnn.load_state_dict(state_dict, strict=False)
            print(f"Poids fine-tunés chargés: {load_path}")
        else:
            print(f"Aucun poids trouvé à: {load_path}")


class SiameseDoubleBranchClassifier(nn.Module):
    """
    Modèle double-branche Siamese : une seule branche EfficientNet-B0 à poids
    partagés, appliquée séparément aux vues CC et MLO, puis fusion par
    concaténation + MLP. À comparer à HybridMammographyClassifier (branches
    hétérogènes ResNet50/RexNet150) pour évaluer l'apport du partage de poids.
    """

    def __init__(self,
                 backbone: str = 'efficientnet_b0',
                 input_channels: int = 1,
                 image_feature_dim: int = 512,
                 num_classes: int = 4,
                 dropout: float = 0.3,
                 pretrained: bool = True,
                 shared_weights: Optional[str] = None):
        super().__init__()

        self.shared_branch = SharedImageBranch(
            backbone=backbone,
            input_channels=input_channels,
            feature_dim=image_feature_dim,
            pretrained=pretrained,
            finetuned_weights_path=shared_weights
        )

        total_features = 2 * image_feature_dim
        print(f"[SiameseModel] Dimensions des features:")
        print(f"  - MLO features (poids partagés): {image_feature_dim}")
        print(f"  - CC features (poids partagés): {image_feature_dim}")
        print(f"  - Total features: {total_features}")

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

    def forward(self, mlo_image, cc_image):
        # Même instance de branche appelée sur les deux vues -> poids partagés
        mlo_feat = self.shared_branch(mlo_image)
        cc_feat = self.shared_branch(cc_image)
        features = torch.cat([mlo_feat, cc_feat], dim=1)
        return self.classifier(features)

    def _initialize_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_feature_dimensions(self) -> Tuple[int, int, int]:
        return (
            self.shared_branch.feature_dim,
            self.shared_branch.feature_dim,
            2 * self.shared_branch.feature_dim
        )

    def save_finetuned_weights(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"Modèle Siamese sauvegardé: {save_path}")

    def load_finetuned_weights(self, load_path: str, device: str = 'cpu'):
        try:
            self.load_state_dict(torch.load(load_path, map_location=device))
            print(f"Chargement réussi des poids depuis: {load_path}")
        except RuntimeError:
            print("\nAttention: Tentative de chargement du checkpoint avec gestion des incompatibilités...")
            checkpoint = torch.load(load_path, map_location=device)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items()
                                if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Paramètres chargés avec succès: {len(pretrained_dict)}/{len(checkpoint)}")

    def freeze_shared_branch(self):
        for param in self.shared_branch.parameters():
            param.requires_grad = False
        print("Branche partagée gelée")

    def unfreeze_shared_branch(self):
        for param in self.shared_branch.parameters():
            param.requires_grad = True
        print("Branche partagée dégelée")
