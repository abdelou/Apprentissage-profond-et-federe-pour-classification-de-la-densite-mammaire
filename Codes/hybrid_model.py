import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class GLCMDescriptorMLP(nn.Module):
    # MLP pour les descripteurs GLCM
    def __init__(self, hidden_dims=[32, 16], dropout=0.2):
        super().__init__()
        self.input_dim = 5
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

    def forward(self, image_tensor):
        batch_size = image_tensor.shape[0]
        glcm_features = []
        for i in range(batch_size):
            img = image_tensor[i, 0].cpu().numpy()
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
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

class HistogramMLP(nn.Module):
    # MLP pour l'histogramme des niveaux de gris
    def __init__(self, input_dim=256, hidden_dims=[128, 64], dropout=0.3):
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

    def forward(self, hist_tensor):
        return self.mlp(hist_tensor)

class ImageBranch(nn.Module):
    # Branche image avec CNN ou ViT
    def __init__(self, backbone='cnn', input_channels=1, feature_dim=512, 
                 pretrained=True, finetuned_weights_path=None):
        super(ImageBranch, self).__init__()
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        if backbone == 'cnn':
            print("Branche image : CNN (ResNet50)")
            self.cnn = models.resnet50(pretrained=pretrained)
            if input_channels == 1:
                self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, feature_dim)
            
            if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                print(f"Chargement des poids fine-tunés CNN: {finetuned_weights_path}")
                self.load_finetuned_weights(finetuned_weights_path)
            
        elif backbone == 'vit':
            print("Branche image : ViT")
            try:
                from transformers import ViTModel, ViTConfig
                config = ViTConfig(
                    image_size=224,
                    patch_size=16,
                    num_channels=1,
                    hidden_size=768,
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
                
                self.vit = ViTModel.from_pretrained(
                    'microsoft/BiomedVLP-BioViL-T',
                    config=config,
                    ignore_mismatched_sizes=True
                )
                
                if input_channels == 1:
                    old_embeddings = self.vit.embeddings.patch_embeddings.projection
                    new_embeddings = nn.Conv2d(1, config.hidden_size, 
                                             kernel_size=old_embeddings.kernel_size,
                                             stride=old_embeddings.stride,
                                             padding=old_embeddings.padding)
                    with torch.no_grad():
                        new_embeddings.weight.copy_(old_embeddings.weight.mean(dim=1, keepdim=True))
                        new_embeddings.bias.copy_(old_embeddings.bias)
                    self.vit.embeddings.patch_embeddings.projection = new_embeddings
                
                self.vit_projection = nn.Linear(config.hidden_size, feature_dim)
                
                if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                    print(f"Chargement des poids fine-tunés ViT: {finetuned_weights_path}")
                    self.load_finetuned_weights(finetuned_weights_path)

            except ImportError:
                print("Transformers absent, utilisation ResNet50 par défaut")
                self.backbone = 'cnn'
                self.cnn = models.resnet50(pretrained=pretrained)
                if input_channels == 1:
                    self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.cnn.fc = nn.Linear(self.cnn.fc.in_features, feature_dim)
                
                if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                    self.load_finetuned_weights(finetuned_weights_path)
        else:
            raise ValueError("Backbone non supporté")
    
    def forward(self, image_tensor):
        if self.backbone == 'cnn':
            return self.cnn(image_tensor)
        elif self.backbone == 'vit':
            outputs = self.vit(image_tensor)
            features = outputs.last_hidden_state[:, 0, :]
            return features
    
    def freeze_layers(self, freeze_last_layer=False):
        if self.backbone == 'cnn':
            for name, param in self.cnn.named_parameters():
                if not freeze_last_layer and 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.backbone == 'vit':
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.vit_projection.parameters():
                param.requires_grad = not freeze_last_layer
    
    def unfreeze_layers(self, last_n_layers=0):
        if self.backbone == 'cnn':
            if last_n_layers == 0:
                for param in self.cnn.parameters():
                    param.requires_grad = True
            else:
                layers = list(self.cnn.children())
                for layer in layers[-last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        elif self.backbone == 'vit':
            if last_n_layers == 0:
                for param in self.vit.parameters():
                    param.requires_grad = True
            else:
                for layer in self.vit.encoder.layer[-last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
    
    def load_finetuned_weights(self, load_path, freeze_layers=False, device='cpu'):
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
                    
                elif self.backbone == 'vit':
                    checkpoint = torch.load(load_path, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        vit_state = checkpoint.get('vit_state_dict', checkpoint)
                        vit_state = {k: v for k, v in vit_state.items() 
                                   if not k.startswith('classifier') and not k.startswith('pooler')}
                        self.vit.load_state_dict(vit_state, strict=False)
                
                if device != 'cpu':
                    self.to(device)
                elif original_device.type == 'cuda':
                    self.to(original_device)
                
            except Exception as e:
                print(f"Erreur de chargement des poids: {str(e)}")
                if original_device.type == 'cuda':
                    self.to(original_device)
                raise e

class HybridMammographyClassifier(nn.Module):
    # Classifieur hybride
    def __init__(self, 
                 backbone='cnn',
                 input_channels=1,
                 image_feature_dim=512,
                 hist_hidden_dims=[128, 64],
                 num_classes=4,
                 dropout=0.3,
                 pretrained=True,
                 finetuned_weights_path=None):
        
        super(HybridMammographyClassifier, self).__init__()
        
        self.image_branch = ImageBranch(
            backbone=backbone,
            input_channels=input_channels,
            feature_dim=image_feature_dim,
            pretrained=pretrained,
            finetuned_weights_path=finetuned_weights_path
        )
        
        self.histogram_branch = HistogramMLP(input_dim=256, hidden_dims=hist_hidden_dims, dropout=dropout)

        if backbone == 'vit':
            self.image_feature_dim = 768
        else:
            self.image_feature_dim = image_feature_dim

        total_features = self.image_feature_dim + self.histogram_branch.output_dim

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
    
    def forward(self, image_tensor, hist_tensor):
        image_features = self.image_branch(image_tensor)
        hist_features = self.histogram_branch(hist_tensor)
        combined_features = torch.cat([image_features, hist_features], dim=1)
        logits = self.classifier(combined_features)
        return logits
    
    def get_feature_dimensions(self):
        return (
            self.image_branch.feature_dim,
            self.histogram_branch.output_dim,
            self.image_branch.feature_dim + self.histogram_branch.output_dim
        )
    
    def save_finetuned_weights(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"Modèle sauvegardé dans : {save_path}")
    
    def load_finetuned_weights(self, load_path, device='cpu'):
        try:
            self.load_state_dict(torch.load(load_path, map_location=device))
            print(f"Poids chargés : {load_path}")
        except RuntimeError:
            checkpoint = torch.load(load_path, map_location=device)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("Adaptation des couches effectuée lors du chargement")
    
    def freeze_image_branch(self):
        for param in self.image_branch.parameters():
            param.requires_grad = False
    
    def unfreeze_image_branch(self):
        for param in self.image_branch.parameters():
            param.requires_grad = True

def create_histogram_from_image(image_tensor, num_bins=256):
    batch_size = image_tensor.shape[0]
    
    image_normalized = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-8)
    bin_indices = (image_normalized * (num_bins - 1)).long()
    histograms = torch.zeros(batch_size, num_bins, device=image_tensor.device)
    
    for i in range(batch_size):
        hist = torch.bincount(bin_indices[i].flatten(), minlength=num_bins)
        histograms[i] = hist.float()
    
    histograms = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-8)
    return histograms

if __name__ == "__main__":
    print("Test du modèle...")
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridMammographyClassifier(
        backbone='cnn',
        input_channels=1,
        image_feature_dim=512,
        hist_hidden_dims=[128, 64],
        num_classes=4,
        dropout=0.3
    ).to(device)
    
    image_tensor = torch.randn(batch_size, 1, 224, 224).to(device)
    hist_tensor = torch.randn(batch_size, 256).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor, hist_tensor)
        print(f"Sortie modèle: {output.shape}")
        
        auto_hist = create_histogram_from_image(image_tensor)
        output_auto = model(image_tensor, auto_hist)
        print(f"Sortie avec auto-hist: {output_auto.shape}")
    
    img_dim, hist_dim, combined_dim = model.get_feature_dimensions()
    print("Dimensions :")
    print(f"  Image : {img_dim}D")
    print(f"  Hist : {hist_dim}D")
    print(f"  Fusion : {combined_dim}D")