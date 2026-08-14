import torch
import torch.nn as nn
import torchvision.models as tvmodels
import timm


class ResNet50DensityClassifier(nn.Module):
    """
    Classifieur image simple (ResNet50, vue CC, 1 canal) utilisé comme modèle de
    base pour l'apprentissage fédéré (FedAvg). Volontairement simple par rapport
    aux architectures hybrides du projet : le modèle fédéré doit être identique
    chez chaque client (VinDr-Mammo, DDSM/CBIS-DDSM), et DDSM ne fournit que des
    vues CC ici, donc pas de fusion multi-vues ni d'histogramme/GLCM possible.
    """

    def __init__(self, num_classes: int = 4, input_channels: int = 1, pretrained: bool = True):
        super().__init__()
        self.cnn = tvmodels.resnet50(pretrained=pretrained)
        if input_channels == 1:
            old_conv = self.cnn.conv1
            self.cnn.conv1 = nn.Conv2d(
                1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
            )
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)

    def forward(self, x):
        return self.cnn(x)


class SiameseFederatedClassifier(nn.Module):
    """
    Variante fédérée de l'architecture siamoise de l'Approche 6 (meilleur
    résultat du mémoire hors fédéré, 85% Test Acc) : un seul backbone
    EfficientNet-B0, poids partagés, appliqué séparément aux vues CC et MLO
    d'un même sein, puis fusion par concaténation + MLP. Une seule instance
    de backbone -> un seul jeu de poids à agréger par FedAvg, malgré les deux
    entrées image.
    """

    def __init__(self, num_classes: int = 4, input_channels: int = 1,
                 feature_dim: int = 512, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.shared_branch = timm.create_model(
            'efficientnet_b0', pretrained=pretrained, in_chans=input_channels, num_classes=feature_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * feature_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, cc_image, mlo_image):
        cc_feat = self.shared_branch(cc_image)
        mlo_feat = self.shared_branch(mlo_image)
        return self.classifier(torch.cat([cc_feat, mlo_feat], dim=1))
