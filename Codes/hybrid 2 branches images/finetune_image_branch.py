import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import torchvision.transforms as transforms
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from hybrid_model import ImageBranch
from preprocessing import read_dicom, preprocess_image

# --- Configuration globale ---
VINDR_ROOT = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
CONFIG = {
  # Pour fine-tuning explicite de la branche CC (rexnet_150) ou MLO (resnet50)
  'ANNOTATIONS_CSV_CC': os.path.join(VINDR_ROOT, 'breast-level_annotations.csv'),
  'IMAGE_ROOT_CC': VINDR_ROOT,
  'ANNOTATIONS_CSV_MLO': os.path.join(VINDR_ROOT, 'breast-level_annotations.csv'),
  'IMAGE_ROOT_MLO': VINDR_ROOT,
  'BACKBONE': 'rexnet_150', # ou 'resnet50' selon la branche à fine-tuner
  'EPOCHS': 3,
  'BATCH_SIZE': 10,
  'LEARNING_RATE': 1e-4,
  'DENSITY_CLASSES': ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"],
  'CLASS_MAP': {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3},
  'FEATURES_DIR': 'featuresextracted',
  'MODELS_DIR': 'featuresmodels',
  'FINETUNED_WEIGHTS_DIR': 'featuresfinetuned_weights',
  'RESUME_FROM_rexnet_150': "featuresfinetuned_weights/rexnet_150_image_branch_checkpoint.pth",
  'RESUME_FROM_resnet50': "featuresfinetuned_weights/resnet50_image_branch_checkpoint.pth"
}

class ImageOnlyDataset(Dataset):
  """
  Dataset unifié pour charger les mammographies avec support pour data augmentation.
  """
  def __init__(self, annotations_df, image_dir, classes_to_use=None, label_map=None, 
         use_augmentation=False, split='training', view_position=None):
    self.image_dir = image_dir
    self.density_map = {
      "DENSITY A": "density_A", "DENSITY B": "density_B",
      "DENSITY C": "density_C", "DENSITY D": "density_D"
    }
    self.split_map = {"training": "train", "test": "test"}
    self.use_augmentation = use_augmentation
    self.split = split
    self.view_position = view_position

    # Filtrage et remapping des labels
    if classes_to_use:
      # Mode classification binaire
      print(f"[DATASET] Utilisation des classes: {classes_to_use}")
      self.df = annotations_df[annotations_df['breast_density'].isin(classes_to_use)].copy()
      if self.view_position:
        self.df = self.df[self.df['view_position'] == self.view_position]
      self.label_remap = {classes_to_use[0]: 0, classes_to_use[1]: 1}
      self.df['label'] = self.df['breast_density'].map(self.label_remap)
      time.sleep(5) # Pause pour permettre la lecture du message
    else:
      # Mode classification multiclasse
      self.df = annotations_df.copy()
      if self.view_position:
        self.df = self.df[self.df['view_position'] == self.view_position]
      if label_map:
        self.df['label'] = self.df['breast_density'].map(label_map)
      else:
        self.df['label'] = self.df['breast_density'].map(CONFIG['CLASS_MAP'])

    self.df = self.df.reset_index(drop=True)
    
    # Transforms pour l'augmentation
    if self.use_augmentation and split == 'training':
      self.augmentation_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=1), # Rotation légère
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)), # Crop aléatoire
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]) # Normalisation pour 1 canal
      ])
    else:
      # Transforms de base sans augmentation
      self.augmentation_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]) # Normalisation pour 1 canal
      ])

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    
    # Vérifier si c'est une image augmentée ou originale
    is_augmented = row.get('is_augmented', False)
    
    if is_augmented:
      # Pour les images augmentées, on utilise une transformation aléatoire
      study_id = row['study_id'].replace('_aug', '')
      image_id = row['image_id'].replace('_aug', '')
      density = self.density_map.get(row["breast_density"])
      split = self.split_map.get(row["split"])
    else:
      # Pour les images originales
      study_id = row['study_id']
      image_id = row['image_id']
      density = self.density_map.get(row["breast_density"])
      split = self.split_map.get(row["split"])

    # Chemin par défaut organized_layout
    image_path = os.path.join(self.image_dir, split, density, study_id, f"{image_id}.dicom")
    # Fallback vers le format standard VinDr-Mammo (sans organized_layout)
    if not os.path.exists(image_path):
      image_path = os.path.join(self.image_dir, 'images', study_id, f"{image_id}.dicom")
      if not os.path.exists(image_path):
        image_path = os.path.join(self.image_dir, study_id, f"{image_id}.dicom")

    try:
      # Chargement et prétraitement de l'image
      image = read_dicom(image_path)
      image = preprocess_image(image, laterality=row["laterality"])
      
      # S'assurer que l'image est en 2D (niveaux de gris)
      if len(image.shape) == 3:
        # Si l'image a 3 dimensions, prendre le premier canal
        image = image[:, :, 0]
      elif len(image.shape) > 3:
        # Pour les images 4D, prendre la première dimension et le premier canal
        if len(image.shape) == 4:
          image = image[0, :, :, 0] # Prendre (0, :, :, 0)
        else:
          image = image.squeeze()
      
      # Conversion en PIL Image en niveaux de gris
      image = Image.fromarray(image.astype(np.uint8), mode='L')
      
      # Application des transformations
      image = self.augmentation_transforms(image)
      
      # Label
      label = row['label']
      return image, label
      
    except Exception as e:
      print(f"Erreur lors du chargement de {image_path}: {e}")
      # Image de secours (bruit) : pas de fichier de repli fixe à disposition,
      # on retourne un tenseur factice plutôt que de planter le worker du DataLoader.
      return torch.randn(1, 224, 224), row['label']


class ImageBranchManager:
  """
  Gestionnaire pour le fine-tuning de la branche image.
  """
  def __init__(self, backbone, device):
    self.backbone = backbone
    self.device = device
    self.model = None
    
  def create_model(self, input_channels=1, feature_dim=512, pretrained=True):
    """Crée le modèle de branche image (ResNet50 ou RexNet150)."""
    self.model = ImageBranch(
      backbone=self.backbone,
      input_channels=input_channels,
      feature_dim=feature_dim,
      pretrained=pretrained
    )
    return self.model
  
  def finetune_model(self, data_csv, image_root, epochs=30, batch_size=8, lr=1e-4,
           save_dir=None, patience= 4, min_delta=0.0005,
           use_augmentation=True, resume_from=None):
    """
    Fine-tune le modèle de branche image.
    """
    if save_dir is None:
      save_dir = CONFIG['FINETUNED_WEIGHTS_DIR']
    
    # Chargement des données
    df = pd.read_csv(data_csv)
    if self.backbone == 'rexnet_150':
      # Pour RexNet150, on utilise uniquement les vues CC
      df = df[df['view_position'] == 'CC'].reset_index(drop=True)
    elif self.backbone == 'resnet50':
      # Pour ResNet50, on utilise uniquement les vues MLO
      df = df[df['view_position'] == 'MLO'].reset_index(drop=True)
    
    # Séparer strictement les données d'entraînement et de test
    train_df = df[df['split'] == 'training'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Créer un ensemble de validation à partir des données d'entraînement
    train_indices = train_df.index.tolist()
    np.random.seed(42) # Pour la reproductibilité
    val_size = int(0.2 * len(train_indices))
    val_indices = np.random.choice(train_indices, size=val_size, replace=False)
    train_indices = list(set(train_indices) - set(val_indices))
    
    # Marquer les splits dans le DataFrame d'entraînement
    train_df.loc[train_indices, 'temp_split'] = 'train'
    train_df.loc[val_indices, 'temp_split'] = 'val'
    
    # Création des datasets avec une séparation stricte
    # Déterminer la vue à filtrer selon le backbone
    view_position = None
    if self.backbone == 'rexnet_150':
      view_position = 'CC'
    elif self.backbone == 'resnet50':
      view_position = 'MLO'

    train_dataset = ImageOnlyDataset(
      train_df[train_df['temp_split'] == 'train'],
      image_root,
      label_map=CONFIG['CLASS_MAP'],
      use_augmentation=use_augmentation,
      split='training',
      view_position=view_position
    )

    val_dataset = ImageOnlyDataset(
      train_df[train_df['temp_split'] == 'val'],
      image_root,
      label_map=CONFIG['CLASS_MAP'],
      use_augmentation=False,
      split='training',
      view_position=view_position
    )
    
    # Sauvegarder le DataFrame de test pour l'évaluation finale
    self.test_df = test_df
    
    # Création des dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
               shuffle=False, num_workers=2, drop_last=True)
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    
    # Création du modèle
    model = self.create_model()
    model.to(self.device)
    
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters()):,} paramètres")
    time.sleep(10) # Pause pour éviter les problèmes de timing
    # Reprise d'entraînement si spécifié
    if resume_from and os.path.exists(resume_from):
      print(f"Reprise depuis: {resume_from}")
      checkpoint = torch.load(resume_from, map_location=self.device)
      model.load_state_dict(checkpoint['model_state_dict'])
      print(f"Checkpoint chargé (epoch {checkpoint.get('epoch', 0)})")
      time.sleep(5) # Pause pour permettre la lecture du message
    
    # Entraînement
    trained_model = self._train_model(
      model, train_loader, val_loader,
      epochs=epochs, lr=lr, save_dir=save_dir,
      patience=patience, min_delta=min_delta
    )
    
    return trained_model
  
  def evaluate_on_test(self, model, image_root, batch_size=8):
    """
    Évalue le modèle sur l'ensemble de test qui n'a jamais été vu pendant l'entraînement.
    """
    # Création du dataset de test
    view_position = None
    if self.backbone == 'rexnet_150':
      view_position = 'CC'
    elif self.backbone == 'resnet50':
      view_position = 'MLO'
    test_dataset = ImageOnlyDataset(
      self.test_df,
      image_root,
      label_map=CONFIG['CLASS_MAP'],
      use_augmentation=False,
      split='test',
      view_position=view_position
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                shuffle=False, num_workers=2)
    
    print(f"Évaluation sur {len(test_dataset)} images de test")
    
    # Évaluation
    model.to(self.device)
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    os.makedirs('graphes', exist_ok=True)
    with torch.no_grad():
      for images, labels in tqdm(test_loader, desc="Test"):
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calcul des métriques
    test_acc = 100. * test_correct / test_total
    
    # Affichage du rapport de classification
    print("\nRapport de classification:")
    print(classification_report(all_labels, all_preds, 
                target_names=CONFIG['DENSITY_CLASSES']))
    
    # Création de la matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
          xticklabels=CONFIG['DENSITY_CLASSES'],
          yticklabels=CONFIG['DENSITY_CLASSES'])
    plt.title('Matrice de confusion sur l\'ensemble de test')
    plt.ylabel('Vraie classe')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig(f'graphes/{CONFIG["BACKBONE"]}_confusion_matrix.png')
    plt.close()
    
    return test_acc
    
  def _train_model(self, model, train_loader, val_loader, epochs=30, lr=1e-4,
          save_dir='featuresfinetuned_weights', patience=10, min_delta=0.001):
    """Entraînement du modèle."""
    model.to(self.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Créer le dossier de sauvegarde
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Début du fine-tuning de la branche image pour {epochs} époques")
    print(f"Learning rate: {lr}")
    print(f"Device: {self.device}")
    time.sleep(5) # Pause pour permettre la lecture du message
    for epoch in range(epochs):
      # Entraînement
      model.train()
      train_loss = 0.0
      train_correct = 0
      train_total = 0
      
      progress_bar = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs}")
      for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(self.device), labels.to(self.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        # Accuracy batch
        batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
        progress_bar.set_postfix({
          'Loss': f'{train_loss/(batch_idx+1):.4f}',
          'Acc': f'{batch_acc:.2f}%'
        })
      
      train_acc = 100. * train_correct / train_total
      # Calcul du recall général sur l'ensemble du train
      from sklearn.metrics import recall_score
      train_true = []
      train_pred = []
      for images, labels in train_loader:
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = model(images)
        pred = outputs.argmax(dim=1)
        train_true.extend(labels.cpu().numpy())
        train_pred.extend(pred.cpu().numpy())
      train_recall = recall_score(train_true, train_pred, average='macro')
      
      # Validation
      model.eval()
      val_loss = 0.0
      val_correct = 0
      val_total = 0
      
      with torch.no_grad():
        for images, labels in val_loader:
          images, labels = images.to(self.device), labels.to(self.device)
          outputs = model(images)
          loss = criterion(outputs, labels)
          
          val_loss += loss.item()
          _, predicted = outputs.max(1)
          val_total += labels.size(0)
          val_correct += predicted.eq(labels).sum().item()
      
      val_acc = 100. * val_correct / val_total
      
      # Learning rate scheduling
      scheduler.step(val_acc)
      
      print(f"\n[ÉPOQUE {epoch+1}/{epochs}]")
      print(f" Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
      print(f" Train Recall: {100.*train_recall:.2f}%")
      print(f" Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
      print(f" Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
      
      # Sauvegarde du meilleur modèle
      if val_acc > best_val_acc + min_delta:
        best_val_acc = val_acc
        patience_counter = 0
        # Sauvegarde du meilleur modèle
        backbone_name = model.backbone
        best_model_path = os.path.join(save_dir, f'{backbone_name}_image_branch_best.pth')
        model.save_finetuned_weights(best_model_path)
        # Sauvegarde du checkpoint pour reprise
        checkpoint_path = os.path.join(save_dir, f'{backbone_name}_image_branch_checkpoint.pth')
        torch.save({
          'backbone_name': backbone_name,
          'epoch': epoch+1,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'best_accuracy': best_val_acc,
          'patience_counter': patience_counter
        }, checkpoint_path)
        print(f"  Nouveau meilleur modèle! Acc: {val_acc:.2f}%")
        print(f" 💾 Modèle sauvegardé: {best_model_path}")
      else:
        patience_counter += 1
        print(f" ⏳ Pas d'amélioration ({patience_counter}/{patience})")
      
        # Early stopping
        if patience_counter >= patience:
          print(f"\n[EARLY STOPPING] Arrêt après {patience} époques sans amélioration")
          break

      # End of training loop
      # (do not dedent here)

      # After training loop: save final model and return
    backbone_name = model.backbone
    final_save_path = os.path.join(save_dir, f'{backbone_name}_image_branch_final.pth')
    model.save_finetuned_weights(final_save_path)
    print(f"\n[FIN] Fine-tuning de la branche image terminé.")
    print(f" - Meilleure accuracy: {best_val_acc:.4f}")
    print(f" - Modèle final: {final_save_path}")
    print(f" - Meilleur modèle: {os.path.join(save_dir, f'{backbone_name}_image_branch_best.pth')}")
    return model


def main():
  parser = argparse.ArgumentParser(description='Fine-tuning de la branche image')
  parser.add_argument('--backbone', default=CONFIG['BACKBONE'], choices=['rexnet_150', 'resnet50'], help='Backbone à fine-tuner (rexnet_150 pour CC, resnet50 pour MLO)')
  parser.add_argument('--epochs', type=int, default=CONFIG['EPOCHS'], help="Nombre d'époques")
  parser.add_argument('--batch_size', type=int, default=CONFIG['BATCH_SIZE'], help='Taille du batch')
  parser.add_argument('--lr', type=float, default=CONFIG['LEARNING_RATE'], help='Learning rate')
  parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda)')
  parser.add_argument('--use_augmentation', action='store_true', help='Utiliser la data augmentation')
  parser.add_argument('--resume_from', default=None, help="Chemin vers un checkpoint pour reprendre l'entraînement")
  parser.add_argument('--eval_only', action='store_true', help="Évaluer uniquement le modèle (pas d'entraînement)")
  parser.add_argument('--model_path', default=None, help='Chemin vers le modèle à évaluer')

  # Sélection explicite des chemins selon le backbone
  parser.add_argument('--data_csv', default=None, help="Chemin vers le CSV d'annotations (auto selon backbone si non spécifié)")
  parser.add_argument('--image_root', default=None, help="Chemin vers le dossier images (auto selon backbone si non spécifié)")

  args = parser.parse_args()

  # Configuration du device
  if args.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  else:
    device = torch.device(args.device)

  # Sélection automatique des chemins selon le backbone
  if args.backbone == 'rexnet_150':
    data_csv = args.data_csv if args.data_csv else CONFIG['ANNOTATIONS_CSV_CC']
    image_root = args.image_root if args.image_root else CONFIG['IMAGE_ROOT_CC']
    resume_from = args.resume_from if args.resume_from else CONFIG['RESUME_FROM_rexnet_150']
  elif args.backbone == 'resnet50':
    data_csv = args.data_csv if args.data_csv else CONFIG['ANNOTATIONS_CSV_MLO']
    image_root = args.image_root if args.image_root else CONFIG['IMAGE_ROOT_MLO']
    resume_from = args.resume_from if args.resume_from else CONFIG['RESUME_FROM_resnet50']
  else:
    raise ValueError(f"Backbone non supporté: {args.backbone}")

  print(f"=== Fine-tuning de la branche image ===")
  print(f"Device: {device}")
  print(f"Backbone: {args.backbone}")
  print(f"Data augmentation: {'Activée' if args.use_augmentation else 'Désactivée'}")
  print(f"Annotations CSV: {data_csv}")
  print(f"Image root: {image_root}")
  time.sleep(5) # Pause pour permettre la lecture du message

  try:
    manager = ImageBranchManager(args.backbone, device)

    if args.eval_only:
      # Évaluation uniquement
      if not args.model_path:
        print("Erreur: --model_path doit être spécifié en mode --eval_only")
        return
      print(f"\n[MODE ÉVALUATION SEULE] Chargement du modèle depuis {args.model_path}")
      # Créer le modèle
      model = manager.create_model()
      checkpoint = torch.load(args.model_path, map_location=device)
      if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
      else:
        model.load_state_dict(checkpoint)
      print("Modèle chargé. Évaluation sur l'ensemble de test...")
      # Charger le CSV pour récupérer le test_df
      df = pd.read_csv(data_csv)
      test_df = df[df['split'] == 'test'].copy()
      manager.test_df = test_df
      test_acc = manager.evaluate_on_test(
        model,
        image_root,
        batch_size=args.batch_size
      )
      print(f"\nPerformance finale sur le test: {test_acc:.2f}%")
      return

    # Mode entraînement classique
    trained_model = manager.finetune_model(
      data_csv=data_csv,
      image_root=image_root,
      epochs=args.epochs,
      batch_size=args.batch_size,
      lr=args.lr,
      use_augmentation=args.use_augmentation,
      resume_from=resume_from
    )

    print(f"\n Fine-tuning terminé!")
    
    # Évaluation sur l'ensemble de test
    print("\nÉvaluation sur l'ensemble de test jamais vu pendant l'entraînement...")
    test_acc = manager.evaluate_on_test(
      trained_model,
      image_root,
      batch_size=args.batch_size
    )
    print(f"\nPerformance finale sur le test: {test_acc:.2f}%")
    
  except Exception as e:
    print(f"Erreur lors du fine-tuning: {e}")
    import traceback
    traceback.print_exc()


if __name__ == "__main__":
  main() 