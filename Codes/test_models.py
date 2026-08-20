import numpy as np
import torch
from models import MLPClassifier, BinaryMLPClassifier, HierarchicalClassifier, FeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import os
import argparse

def get_backbone_config(backbone_name):
  # Configuration par backbone
  configs = {
    'vit': {'in_dim': 768, 'model_name': 'vit_base_patch16_224'},
    'deit': {'in_dim': 768, 'model_name': 'deit_base_patch16_224'},
    'swin': {'in_dim': 1024, 'model_name': 'swin_base_patch4_window7_224'},
    'swinv2': {'in_dim': 1024, 'model_name': 'swinv2_base_window12_192_22k'},
    'pvt': {'in_dim': 512, 'model_name': 'pvt_v2_b2'},
    't2t_vit': {'in_dim': 640, 'model_name': 't2t_vit_14'},
    'twins': {'in_dim': 768, 'model_name': 'twins_pcpvt_large'},
    'resnet': {'in_dim': 2048, 'model_name': 'resnet50'},
    'efficientnet': {'in_dim': 1280, 'model_name': 'efficientnet_b0'},
    'cvt': {'in_dim': 384, 'model_name': 'microsoft/cvt-13'},
    'cvt-13': {'in_dim': 384, 'model_name': 'microsoft/cvt-13'},
    'cvt-21': {'in_dim': 512, 'model_name': 'microsoft/cvt-21'},
    'cvt-w24': {'in_dim': 384, 'model_name': 'microsoft/cvt-w24-384-22k'}
  }
  return configs.get(backbone_name, {'in_dim': 768, 'model_name': 'twins_pcpvt_large'})

def test_backbone_availability(backbone_name):
  try:
    if backbone_name.startswith('cvt'):
      from transformers import AutoImageProcessor, AutoModel
      print(f"Test disponibilité {backbone_name} (Hugging Face)...")
      return True
    else:
      from timm import create_model
      print(f"Test disponibilité {backbone_name} (timm)...")
      return True
  except ImportError as e:
    print(f"Backbone {backbone_name} non disponible : {e}")
    return False

def create_graphes_directory():
  graphes_dir = "graphes"
  if not os.path.exists(graphes_dir):
    os.makedirs(graphes_dir)
    print(f"Dossier '{graphes_dir}' créé")
  return graphes_dir

def save_plot(plt, filename, graphes_dir):
  filepath = os.path.join(graphes_dir, filename)
  plt.savefig(filepath, dpi=300, bbox_inches='tight')
  print(f"Graphique sauvegardé: {filepath}")
  plt.show()

BACKBONE = 'cvt-w24'
N_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['A', 'B', 'C', 'D']

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--backbone', type=str, default=BACKBONE,
            choices=['vit', 'deit', 'swin', 'swinv2', 'pvt', 't2t_vit', 'twins', 
                'resnet', 'efficientnet', 'cvt', 'cvt-13', 'cvt-21', 'cvt-w24'])
  parser.add_argument('--use_finetuned', action='store_true', default=True)
  parser.add_argument('--no_finetuned', dest='use_finetuned', action='store_false')
  parser.add_argument('--skip_availability_test', action='store_true')
  return parser.parse_args()

def main():
  args = parse_args()
  backbone = args.backbone
  graphes_dir = create_graphes_directory()
  
  print("Configuration :")
  print(f" - Backbone: {backbone}")
  print(f" - Device: {DEVICE}")
  print(f" - Features fine-tunées: {args.use_finetuned}")
  
  if not args.skip_availability_test:
    if not test_backbone_availability(backbone):
      print(f"Backbone {backbone} non disponible. Arrêt.")
      return
  
  config = get_backbone_config(backbone)
  IN_DIM = config['in_dim']
  print(f"Dimension des features: {IN_DIM}")
  
  suffix = "finetuned" if args.use_finetuned else ""
  FEATURES_PATH = f'{backbone}_{suffix}_features_test.npy'
  LABELS_PATH = f'{backbone}_{suffix}_labels_test.npy'
  
  print("Chargement des features et labels...")
  if not os.path.exists(FEATURES_PATH) or not os.path.exists(LABELS_PATH):
    print(f"Fichiers non trouvés: {FEATURES_PATH}, {LABELS_PATH}")
    return
  
  features = np.load(FEATURES_PATH)
  labels = np.load(LABELS_PATH)
  print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
  
  if features.shape[1] != IN_DIM:
    print(f"Ajustement dimension features : {features.shape[1]} au lieu de {IN_DIM}")
    IN_DIM = features.shape[1]
  
  # MLP 4 classes
  print("\nÉvaluation du MLP 4 classes...")
  mlp = MLPClassifier(in_dim=IN_DIM, out_dim=N_CLASSES)
  
  mlp_path = f'{backbone}_augmented_mlp4.pth'
  
  if not os.path.exists(mlp_path):
    print(f"Modèle MLP 4 classes non trouvé: {mlp_path}")
    return
  
  mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
  mlp.to(DEVICE)
  mlp.eval()
  
  with torch.no_grad():
    X = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    logits = mlp(X)
    preds = logits.argmax(dim=1).cpu().numpy()

  print(classification_report(labels, preds, target_names=CLASS_NAMES))
  
  cm = confusion_matrix(labels, preds)
  plt.figure(figsize=(6,5))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
  plt.xlabel('Prédit')
  plt.ylabel('Vrai')
  plt.title(f'Matrice de confusion - MLP 4 classes ({backbone})')
  save_plot(plt, f'{backbone}_mlp4_confusion_matrix.png', graphes_dir)

  # ROC One-vs-Rest
  from sklearn.preprocessing import label_binarize
  labels_bin = label_binarize(labels, classes=list(range(N_CLASSES)))
  probs = torch.softmax(logits, dim=1).cpu().numpy()
  plt.figure(figsize=(8,6))
  for i in range(N_CLASSES):
    fpr, tpr, _ = roc_curve(labels_bin[:,i], probs[:,i])
    auc = roc_auc_score(labels_bin[:,i], probs[:,i])
    plt.plot(fpr, tpr, label=f'Classe {CLASS_NAMES[i]} (AUC={auc:.2f})')
  plt.plot([0,1],[0,1],'k--')
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.title(f'ROC One-vs-Rest - MLP 4 classes ({backbone})')
  plt.legend()
  plt.grid()
  save_plot(plt, f'{backbone}_mlp4_roc_curves.png', graphes_dir)

  # Test des binaires
  from itertools import combinations
  print('\nÉvaluation des classifieurs binaires...')
  for i, j in combinations(range(N_CLASSES), 2):
    mask = np.isin(labels, [i, j])
    features_bin = features[mask]
    labels_bin = labels[mask]
    labels_bin = (labels_bin == j).astype(int)
    
    if len(features_bin) == 0:
      continue
    
    mlp_bin = BinaryMLPClassifier(in_dim=IN_DIM)
    
    bin_path = f'{backbone}_finetuned_binary_{i}_{j}.pth' if args.use_finetuned else f'{backbone}_binary_{i}_{j}.pth'
    
    if not os.path.exists(bin_path):
      continue
      
    mlp_bin.load_state_dict(torch.load(bin_path, map_location=DEVICE))
    mlp_bin.to(DEVICE)
    mlp_bin.eval()
    
    with torch.no_grad():
      Xb = torch.tensor(features_bin, dtype=torch.float32).to(DEVICE)
      logits_bin = mlp_bin(Xb)
      preds_bin = logits_bin.argmax(dim=1).cpu().numpy()
      
    print(f'\nBinaire {CLASS_NAMES[i]} vs {CLASS_NAMES[j]} :')
    print(classification_report(labels_bin, preds_bin, target_names=[CLASS_NAMES[i], CLASS_NAMES[j]]))
    
    cm_bin = confusion_matrix(labels_bin, preds_bin)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Oranges', 
          xticklabels=[CLASS_NAMES[i], CLASS_NAMES[j]], 
          yticklabels=[CLASS_NAMES[i], CLASS_NAMES[j]])
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.title(f'Matrice de confusion - Binaire {CLASS_NAMES[i]} vs {CLASS_NAMES[j]} ({backbone})')
    save_plot(plt, f'{backbone}_binary_{i}_{j}_confusion_matrix.png', graphes_dir)
    
    probs_bin = torch.softmax(logits_bin, dim=1).cpu().numpy()[:,1]
    fpr, tpr, _ = roc_curve(labels_bin, probs_bin)
    auc = roc_auc_score(labels_bin, probs_bin)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC - Binaire {CLASS_NAMES[i]} vs {CLASS_NAMES[j]} ({backbone})')
    plt.legend()
    plt.grid()
    save_plot(plt, f'{backbone}_binary_{i}_{j}_roc_curve.png', graphes_dir)

  # Test du pipeline hiérarchique
  print('\nÉvaluation du pipeline hiérarchique...')
  hierarchical_model = HierarchicalClassifier(backbone, in_dim=IN_DIM, out_dim=N_CLASSES).to(DEVICE)
  
  if os.path.exists(mlp_path):
    hierarchical_model.mlp4.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
  
  for (i, j) in hierarchical_model.class_pairs:
    key = f"{i}_{j}"
    path = f'{backbone}_finetuned_binary_{i}_{j}.pth' if args.use_finetuned else f'{backbone}_binary_{i}_{j}.pth'
    if os.path.exists(path):
      hierarchical_model.binary_mlps[key].load_state_dict(torch.load(path, map_location=DEVICE))
  
  hierarchical_model.eval()
  preds_hier = []
  
  with torch.no_grad():
    X = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    for k in range(0, len(X), 32):
      batch = X[k:k+32]
      logits4, logits_bin, top2 = hierarchical_model(batch)
      pred_bin = logits_bin.argmax(dim=1).cpu().numpy()
      for b in range(batch.size(0)):
        c1, c2 = top2[b].cpu().numpy()
        final_class = c1 if pred_bin[b]==0 else c2
        preds_hier.append(final_class)
  
  preds_hier = np.array(preds_hier)
  print(classification_report(labels, preds_hier, target_names=CLASS_NAMES))
  
  cm_hier = confusion_matrix(labels, preds_hier)
  plt.figure(figsize=(6,5))
  sns.heatmap(cm_hier, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
  plt.xlabel('Prédit')
  plt.ylabel('Vrai')
  plt.title(f'Matrice de confusion - Pipeline hiérarchique ({backbone})')
  save_plot(plt, f'{backbone}_hierarchical_confusion_matrix.png', graphes_dir)

if __name__ == '__main__':
  main()