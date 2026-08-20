import os
import numpy as np
import pydicom
import cv2
from glob import glob
from typing import List, Tuple
from PIL import Image

def read_dicom(path) :
  """Lit un fichier DICOM et retourne l'image sous forme de numpy array."""
  try:
    # Nettoyer le chemin des caractères nuls et le normaliser
    clean_path = os.path.normpath(path.replace('\x00', ''))
    print(f"Tentative de lecture de: {clean_path}")
    ds = pydicom.dcmread(clean_path)
    img = ds.pixel_array.astype(np.float32)
    
    # Gestion de la photométrie DICOM
    photometric_interpretation = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
  except Exception as e:
    print(f"Erreur lors de la lecture du fichier: {str(e)}")
    raise
  
  # Inversion si nécessaire pour avoir un fond noir cohérent
  if photometric_interpretation == 'MONOCHROME1':
    # Fond blanc -> Fond noir (inversion)
    img = ds.BitsAllocated - img - 1
    print(f"Image inversée (MONOCHROME1): {path}")
  elif photometric_interpretation == 'MONOCHROME2':
    # Déjà en fond noir, pas d'inversion nécessaire
    print(f"Image normale (MONOCHROME2): {path}")
  else:
    print(f"Photométrie inconnue: {photometric_interpretation}")
  
  return img

def merge_lr_images(left_img, right_img) :
  """Fusionne horizontalement les images gauche et droite."""
  h = min(left_img.shape[0], right_img.shape[0])
  left_img = cv2.resize(left_img, (left_img.shape[1], h))
  right_img = cv2.resize(right_img, (right_img.shape[1], h))
  merged = np.hstack([left_img, right_img])
  return merged

def remove_background(img, threshold=10) :
  """Supprime le fond par seuillage simple."""
  mask = img > threshold
  img_clean = img * mask
  return img_clean

def normalize_intensity_global(img, min_val=None, max_val=None) :
  """
  Normalise l'intensité avec des valeurs globales ou par image.
  Si min_val et max_val sont None, normalise par image.
  """
  if min_val is None or max_val is None:
    # Normalisation par image (comportement actuel)
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
  else:
    # Normalisation avec valeurs globales
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-8)
  
  img = (img * 255).astype(np.uint8)
  return img

def normalize_intensity(img) :
  """Normalise l'intensité entre 0 et 255 (compatibilité)."""
  return normalize_intensity_global(img)

def crop_lateral(img, laterality) :
  """Rogne 700 pixels à gauche si 'R', à droite si 'L'."""
  if laterality == 'R':
    return img[:, 700:]
  elif laterality == 'L':
    return img[:, :-700]
  else:
    return img

def preprocess_image(img, size=(224, 224), laterality = None, 
          normalize_global=False, global_min=None, global_max=None) :
  """
  Prétraite l'image avec gestion cohérente de la photométrie.
  
  Args:
    img: Image numpy array
    size: Taille de sortie
    laterality: Latéralité ('L' ou 'R')
    normalize_global: Si True, utilise une normalisation globale
    global_min: Valeur minimale globale pour la normalisation
    global_max: Valeur maximale globale pour la normalisation
  """
  if laterality is not None:
    img = crop_lateral(img, laterality)
    if laterality == 'L':
      img = np.fliplr(img) # symétrie verticale (gauche-droite) pour avoir la meme orientation que les images de la base de donnees
  
  # Redimensionnement
  img = cv2.resize(img, size)
  
  # Normalisation
  if normalize_global and global_min is not None and global_max is not None:
    img = normalize_intensity_global(img, global_min, global_max)
  else:
    img = normalize_intensity_global(img)
  
  # Conversion en RGB pour les backbones vision modernes (ViT, ResNet, etc.)
  if img.ndim == 2:
    img = np.repeat(img[..., None], 3, axis=2)
  elif img.ndim == 3 and img.shape[2] == 1:
    img = np.repeat(img, 3, axis=2)

  return img

def process_patient(left_path, right_path) :
  left_img = read_dicom(left_path)
  right_img = read_dicom(right_path)
  left_img = remove_background(left_img)
  right_img = remove_background(right_img)
  merged = merge_lr_images(left_img, right_img)
  merged = normalize_intensity(merged)
  merged = preprocess_image(merged)
  return merged

def analyze_dicom_metadata(dicom_path) :
  """
  Analyse les métadonnées DICOM pour diagnostiquer les problèmes de photométrie.
  """
  ds = pydicom.dcmread(dicom_path)
  
  metadata = {
    'PhotometricInterpretation': getattr(ds, 'PhotometricInterpretation', 'N/A'),
    'BitsAllocated': getattr(ds, 'BitsAllocated', 'N/A'),
    'BitsStored': getattr(ds, 'BitsStored', 'N/A'),
    'HighBit': getattr(ds, 'HighBit', 'N/A'),
    'PixelRepresentation': getattr(ds, 'PixelRepresentation', 'N/A'),
    'SamplesPerPixel': getattr(ds, 'SamplesPerPixel', 'N/A'),
    'Rows': getattr(ds, 'Rows', 'N/A'),
    'Columns': getattr(ds, 'Columns', 'N/A'),
    'PixelData': 'Present' if hasattr(ds, 'PixelData') else 'Missing'
  }
  
  return metadata

def batch_analyze_photometry(data_csv, image_root, sample_size=10):
  """
  Analyse la photométrie d'un échantillon d'images pour identifier les incohérences.
  """
  import pandas as pd
  import random
  
  df = pd.read_csv(data_csv)
  sample_df = df.sample(min(sample_size, len(df)))
  
  photometry_counts = {}
  
  for idx, row in sample_df.iterrows():
    split_dir = 'train' if row['split'] == 'training' else 'test'
    density_map = {
      "DENSITY A": "density_A", "DENSITY B": "density_B",
      "DENSITY C": "density_C", "DENSITY D": "density_D"
    }
    
    density = density_map.get(row['breast_density'])
    study_id = str(row['study_id'])
    image_id = str(row['image_id'])
    
    image_path = os.path.join(image_root, split_dir, density, study_id, f"{image_id}.dicom")
    
    if os.path.exists(image_path):
      metadata = analyze_dicom_metadata(image_path)
      photometry = metadata['PhotometricInterpretation']
      photometry_counts[photometry] = photometry_counts.get(photometry, 0) + 1
  
  print("=== ANALYSE DE PHOTOMÉTRIE ===")
  for photometry, count in photometry_counts.items():
    print(f"{photometry}: {count} images")
  
  return photometry_counts 