#!/usr/bin/env python3
# Script de vérification de l'environnement pour le projet de classification de densité mammaire

import sys
import importlib
import subprocess

def check_package(package_name, import_name=None):
  """Vérifie si un package est installé et retourne sa version"""
  if import_name is None:
    import_name = package_name
  
  try:
    module = importlib.import_module(import_name)
    version = getattr(module, '__version__', 'N/A')
    return True, version
  except ImportError:
    return False, None

def main():
  print("=" * 60)
  print("VÉRIFICATION DE L'ENVIRONNEMENT")
  print("Projet: Classification de densité mammaire par Vision Transformers")
  print("=" * 60)
  
  # Liste des packages requis avec leurs noms d'import
  required_packages = {
    'torch': 'torch',
    'torchvision': 'torchvision', 
    'timm': 'timm',
    'pydicom': 'pydicom',
    'opencv-python': 'cv2',
    'Pillow': 'PIL',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'sklearn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'tqdm': 'tqdm',
    'scipy': 'scipy',
    'shap': 'shap'
  }
  
  # Packages optionnels
  optional_packages = {
    'plotly': 'plotly',
    'jupyter': 'jupyter',
    'memory-profiler': 'memory_profiler'
  }
  
  print("\n VÉRIFICATION DES PACKAGES REQUIS")
  print("-" * 40)
  
  all_required_ok = True
  for package, import_name in required_packages.items():
    is_installed, version = check_package(package, import_name)
    if is_installed:
      print(f" {package}: {version}")
    else:
      print(f" {package}: NON INSTALLÉ")
      all_required_ok = False
  
  print("\n VÉRIFICATION DES PACKAGES OPTIONNELS")
  print("-" * 40)
  
  for package, import_name in optional_packages.items():
    is_installed, version = check_package(package, import_name)
    if is_installed:
      print(f" {package}: {version}")
    else:
      print(f" {package}: NON INSTALLÉ (optionnel)")
  
  print("\n VÉRIFICATION DE L'ENVIRONNEMENT PYTHON")
  print("-" * 40)
  print(f"Version Python: {sys.version}")
  print(f"Architecture: {sys.platform}")
  
  print("\n VÉRIFICATION CUDA ET GPU")
  print("-" * 40)
  
  try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
      print(f"Version CUDA: {torch.version.cuda}")
      print(f"Nombre de GPUs: {torch.cuda.device_count()}")
      for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
    else:
      print("  CUDA non disponible - utilisation CPU")
      
  except ImportError:
    print(" PyTorch non installé")
  
  print("\n TESTS DE FONCTIONNALITÉ")
  print("-" * 40)
  
  # Test de lecture DICOM
  try:
    import pydicom
    print(" Lecture DICOM: OK")
  except:
    print(" Lecture DICOM: ÉCHEC")
  
  # Test de traitement d'images
  try:
    import cv2
    import numpy as np
    # Test simple de création d'image
    test_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.resize(test_img, (50, 50))
    print(" Traitement d'images: OK")
  except:
    print(" Traitement d'images: ÉCHEC")
  
  # Test de deep learning
  try:
    import torch
    import timm
    # Test de création d'un modèle simple
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    print(" Deep Learning: OK")
  except:
    print(" Deep Learning: ÉCHEC")
  
  # Test de métriques
  try:
    from sklearn.metrics import accuracy_score, classification_report
    print(" Métriques d'évaluation: OK")
  except:
    print(" Métriques d'évaluation: ÉCHEC")
  
  print("\n" + "=" * 60)
  
  if all_required_ok:
    print(" ENVIRONNEMENT PRÊT POUR LE DÉVELOPPEMENT")
    print("Tous les packages requis sont installés.")
  else:
    print("  PROBLÈMES DÉTECTÉS")
    print("Certains packages requis ne sont pas installés.")
    print("Exécutez: pip install -r requirements.txt")
  
  print("=" * 60)
  
  # Suggestions d'amélioration
  print("\n SUGGESTIONS:")
  if 'torch' in sys.modules and not torch.cuda.is_available():
    print("- Installez CUDA pour accélérer l'entraînement")
  print("- Utilisez un environnement virtuel pour isoler les dépendances")
  print("- Vérifiez régulièrement les mises à jour des packages")

if __name__ == "__main__":
  main() 