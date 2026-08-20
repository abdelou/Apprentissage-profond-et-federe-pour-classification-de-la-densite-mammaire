import os
import numpy as np
import pydicom
import cv2
from glob import glob
from PIL import Image

def read_dicom(path):
  # Lecture d'un fichier DICOM
  try:
    clean_path = os.path.normpath(path.replace('\x00', ''))
    print(f"Lecture DICOM : {clean_path}")
    ds = pydicom.dcmread(clean_path)
    img = ds.pixel_array.astype(np.float32)
    
    photometric_interpretation = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
  except Exception as e:
    print(f"Erreur lecture DICOM: {str(e)}")
    raise
  
  # Inversion si c'est monochrome1
  if photometric_interpretation == 'MONOCHROME1':
    img = ds.BitsAllocated - img - 1
    print(f"Image MONOCHROME1 inversée : {path}")
  elif photometric_interpretation == 'MONOCHROME2':
    print(f"Image MONOCHROME2 standard : {path}")
  else:
    print(f"Photométrie non reconnue : {photometric_interpretation}")
  
  return img

def merge_lr_images(left_img, right_img):
  # Fusionner les deux images (gauche et droite) horizontalement
  h = min(left_img.shape[0], right_img.shape[0])
  left_img = cv2.resize(left_img, (left_img.shape[1], h))
  right_img = cv2.resize(right_img, (right_img.shape[1], h))
  merged = np.hstack([left_img, right_img])
  return merged

def remove_background(img, threshold=10):
  # Enlever le fond avec un seuil simple
  mask = img > threshold
  img_clean = img * mask
  return img_clean

def normalize_intensity_global(img, min_val=None, max_val=None):
  # Normalisation de l'intensité
  if min_val is None or max_val is None:
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
  else:
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-8)
  
  img = (img * 255).astype(np.uint8)
  return img

def normalize_intensity(img):
  return normalize_intensity_global(img)

def crop_lateral(img, laterality):
  # Rogner les bords selon le cote gauche ou droit
  if laterality == 'R':
    return img[:, 700:]
  elif laterality == 'L':
    return img[:, :-700]
  else:
    return img

def preprocess_image(img, size=(224, 224), laterality=None, 
           normalize_global=False, global_min=None, global_max=None):
  # Pretraitement de l'image (rognage, redimensionnement et normalisation)
  if laterality is not None:
    img = crop_lateral(img, laterality)
    if laterality == 'L':
      img = np.fliplr(img) # retourner l'image pour avoir la meme orientation
  
  img = cv2.resize(img, size)
  
  if normalize_global and global_min is not None and global_max is not None:
    img = normalize_intensity_global(img, global_min, global_max)
  else:
    img = normalize_intensity_global(img)
  
  # Conversion en 3 canaux
  if img.ndim == 2:
    img = np.repeat(img[..., None], 3, axis=2)
  elif img.ndim == 3 and img.shape[2] == 1:
    img = np.repeat(img, 3, axis=2)

  return img

def process_patient(left_path, right_path):
  # Process complet pour un patient
  left_img = read_dicom(left_path)
  right_img = read_dicom(right_path)
  left_img = remove_background(left_img)
  right_img = remove_background(right_img)
  merged = merge_lr_images(left_img, right_img)
  merged = normalize_intensity(merged)
  merged = preprocess_image(merged)
  return merged

def analyze_dicom_metadata(dicom_path):
  # Récupérer les métadonnées principales du dicom
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
  # Analyse de la photometrie d'un lot d'images
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
  
  print("--- Analyse de la photometrie ---")
  for photometry, count in photometry_counts.items():
    print(f"{photometry} : {count}")
  
  return photometry_counts