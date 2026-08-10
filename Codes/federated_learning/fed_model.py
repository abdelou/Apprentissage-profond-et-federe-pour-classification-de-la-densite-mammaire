import torch
import torch.nn as nn
import torchvision.models as tvmodels


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
