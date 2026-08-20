#!/usr/bin/env python3
# Script de diagnostic pour analyser les problèmes de photométrie DICOM

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import analyze_dicom_metadata, batch_analyze_photometry, read_dicom
import pydicom

def visualize_dicom_comparison(image_path, save_path = None):
  """
  Visualise une image DICOM avec et sans correction de photométrie.
  """
  # Lecture DICOM brute
  ds = pydicom.dcmread(image_path)
  img_raw = ds.pixel_array
  
  # Lecture avec correction
  img_corrected = read_dicom(image_path)
  
  # Création de la figure
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  
  # Image brute
  axes[0].imshow(img_raw, cmap='gray')
  axes[0].set_title(f"Image brute\nPhotométrie: {getattr(ds, 'PhotometricInterpretation', 'N/A')}")
  axes[0].axis('off')
  
  # Image corrigée
  axes[1].imshow(img_corrected, cmap='gray')
  axes[1].set_title("Image corrigée\n(Fond noir cohérent)")
  axes[1].axis('off')
  
  # Histogramme comparatif
  axes[2].hist(img_raw.flatten(), bins=50, alpha=0.7, label='Brute', color='blue')
  axes[2].hist(img_corrected.flatten(), bins=50, alpha=0.7, label='Corrigée', color='red')
  axes[2].set_title("Histogramme comparatif")
  axes[2].set_xlabel("Intensité")
  axes[2].set_ylabel("Fréquence")
  axes[2].legend()
  
  plt.tight_layout()
  
  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure sauvegardée: {save_path}")
  
  plt.show()
  
  # Affichage des métadonnées
  metadata = analyze_dicom_metadata(image_path)
  print("\n=== MÉTADONNÉES DICOM ===")
  for key, value in metadata.items():
    print(f"{key}: {value}")

def analyze_multiple_images(data_csv, image_root, num_samples=5):
  """
  Analyse plusieurs images pour identifier les patterns de photométrie.
  """
  df = pd.read_csv(data_csv)
  sample_df = df.sample(min(num_samples, len(df)))
  
  print(f"=== ANALYSE DE {len(sample_df)} IMAGES ===")
  
  results = []
  
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
      
      # Lecture des images
      ds = pydicom.dcmread(image_path)
      img_raw = ds.pixel_array
      img_corrected = read_dicom(image_path)
      
      result = {
        'image_path': image_path,
        'photometry': metadata['PhotometricInterpretation'],
        'raw_min': np.min(img_raw),
        'raw_max': np.max(img_raw),
        'corrected_min': np.min(img_corrected),
        'corrected_max': np.max(img_corrected),
        'raw_mean': np.mean(img_raw),
        'corrected_mean': np.mean(img_corrected),
        'density': row['breast_density'],
        'laterality': row['laterality']
      }
      
      results.append(result)
      
      print(f"\nImage {idx+1}:")
      print(f" Photométrie: {result['photometry']}")
      print(f" Densité: {result['density']}")
      print(f" Latéralité: {result['laterality']}")
      print(f" Raw range: [{result['raw_min']:.1f}, {result['raw_max']:.1f}]")
      print(f" Corrected range: [{result['corrected_min']:.1f}, {result['corrected_max']:.1f}]")
  
  return results

def create_photometry_report(data_csv, image_root, sample_size=50):
  """
  Crée un rapport complet sur la photométrie des images.
  """
  print("=== RAPPORT DE PHOTOMÉTRIE DICOM ===")
  
  # Analyse par échantillon
  photometry_counts = batch_analyze_photometry(data_csv, image_root, sample_size)
  
  # Analyse détaillée
  results = analyze_multiple_images(data_csv, image_root, min(10, sample_size))
  
  # Statistiques
  if results:
    df_results = pd.DataFrame(results)
    
    print("\n=== STATISTIQUES ===")
    print(f"Nombre d'images analysées: {len(df_results)}")
    
    # Répartition par photométrie
    photometry_dist = df_results['photometry'].value_counts()
    print("\nRépartition par photométrie:")
    for photometry, count in photometry_dist.items():
      percentage = (count / len(df_results)) * 100
      print(f" {photometry}: {count} ({percentage:.1f}%)")
    
    # Statistiques des intensités
    print("\nStatistiques des intensités (brutes):")
    print(f" Min: {df_results['raw_min'].min():.1f}")
    print(f" Max: {df_results['raw_max'].max():.1f}")
    print(f" Moyenne: {df_results['raw_mean'].mean():.1f}")
    
    print("\nStatistiques des intensités (corrigées):")
    print(f" Min: {df_results['corrected_min'].min():.1f}")
    print(f" Max: {df_results['corrected_max'].max():.1f}")
    print(f" Moyenne: {df_results['corrected_mean'].mean():.1f}")
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    
    # Répartition par photométrie
    plt.subplot(2, 2, 1)
    photometry_dist.plot(kind='bar')
    plt.title('Répartition par photométrie')
    plt.ylabel('Nombre d\'images')
    plt.xticks(rotation=45)
    
    # Histogramme des intensités brutes
    plt.subplot(2, 2, 2)
    plt.hist(df_results['raw_mean'], bins=20, alpha=0.7, color='blue')
    plt.title('Distribution des intensités moyennes (brutes)')
    plt.xlabel('Intensité moyenne')
    plt.ylabel('Fréquence')
    
    # Histogramme des intensités corrigées
    plt.subplot(2, 2, 3)
    plt.hist(df_results['corrected_mean'], bins=20, alpha=0.7, color='red')
    plt.title('Distribution des intensités moyennes (corrigées)')
    plt.xlabel('Intensité moyenne')
    plt.ylabel('Fréquence')
    
    # Comparaison avant/après
    plt.subplot(2, 2, 4)
    plt.scatter(df_results['raw_mean'], df_results['corrected_mean'], alpha=0.6)
    plt.plot([df_results['raw_mean'].min(), df_results['raw_mean'].max()], 
        [df_results['corrected_mean'].min(), df_results['corrected_mean'].max()], 
        'r--', alpha=0.5)
    plt.title('Corrélation avant/après correction')
    plt.xlabel('Intensité moyenne (brute)')
    plt.ylabel('Intensité moyenne (corrigée)')
    
    plt.tight_layout()
    plt.savefig('photometry_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_results
  
  return None

if __name__ == '__main__':
  # Configuration
  ANNOTATIONS_CSV = '/Volumes/cimobile/Entrainement_cetic/vindr-mammo_dataset/physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations.csv'
  IMAGE_ROOT = '/Volumes/cimobile/Entrainement_cetic/vindr-mammo_dataset/organized_layout'
  
  # 1. Analyse générale
  print("=== DIAGNOSTIC DICOM ===")
  results = create_photometry_report(ANNOTATIONS_CSV, IMAGE_ROOT, sample_size= 0)
  
  # 2. Visualisation d'exemples
  if results is not None:
    print("\n=== VISUALISATION D'EXEMPLES ===")
    
    # Trouver des exemples de chaque type de photométrie
    df_results = results
    unique_photometries = df_results['photometry'].unique()
    
    for photometry in unique_photometries:
      example = df_results[df_results['photometry'] == photometry].iloc[0]
      print(f"\nExemple {photometry}: {example['image_path']}")
      
      # Créer le nom du fichier de sauvegarde
      save_path = f"dicom_example_{photometry.lower()}.png"
      visualize_dicom_comparison(example['image_path'], save_path)
  
 