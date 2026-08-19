import torch
import torch.nn as nn
import torchvision.models as tvmodels
import timm

class ResNet50DensityClassifier(nn.Module):
    # ResNet50 simple à 1 canal pour l'apprentissage fédéré (FedAvg)
    def __init__(self, num_classes=4, input_channels=1, pretrained=True):
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
    # Modèle siamois fédéré avec EfficientNet-B0 (poids partagés entre CC et MLO)
    def __init__(self, num_classes=4, input_channels=1, feature_dim=512, pretrained=True, dropout=0.3):
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
