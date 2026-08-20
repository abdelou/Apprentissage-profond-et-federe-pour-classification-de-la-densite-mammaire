#!/usr/bin/env python3
# Script simple pour tester la photométrie d'une image DICOM spécifique

import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from preprocessing import read_dicom, analyze_dicom_metadata

def test_single_dicom(image_path):
  """
  Teste une image DICOM spécifique pour diagnostiquer les problèmes de photométrie.
  """
  if not os.path.exists(image_path):
    print(f"Erreur: Le fichier {image_path} n'existe pas.")
    return
  
  print(f"=== TEST DICOM: {image_path} ===")
  
  # Lecture DICOM brute
  ds = pydicom.dcmread(image_path)
  img_raw = ds.pixel_array
  
  # Lecture avec correction
  img_corrected = read_dicom(image_path)
  
  # Métadonnées
  metadata = analyze_dicom_metadata(image_path)
  
  print("\n=== MÉTADONNÉES ===")
  for key, value in metadata.items():
    print(f"{key}: {value}")
  
  print(f"\n=== STATISTIQUES ===")
  print(f"Image brute:")
  print(f" Min: {np.min(img_raw):.1f}")
  print(f" Max: {np.max(img_raw):.1f}")
  print(f" Moyenne: {np.mean(img_raw):.1f}")
  print(f" Écart-type: {np.std(img_raw):.1f}")
  
  print(f"\nImage corrigée:")
  print(f" Min: {np.min(img_corrected):.1f}")
  print(f" Max: {np.max(img_corrected):.1f}")
  print(f" Moyenne: {np.mean(img_corrected):.1f}")
  print(f" Écart-type: {np.std(img_corrected):.1f}")
  
  # Visualisation
  fig, axes = plt.subplots(2, 3, figsize=(15, 10))
  
  # Image brute
  axes[0, 0].imshow(img_raw, cmap='gray')
  axes[0, 0].set_title(f"Image brute\nPhotométrie: {metadata['PhotometricInterpretation']}")
  axes[0, 0].axis('off')
  
  # Image corrigée
  axes[0, 1].imshow(img_corrected, cmap='gray')
  axes[0, 1].set_title("Image corrigée\n(Fond noir cohérent)")
  axes[0, 1].axis('off')
  
  # Différence
  diff = img_corrected - img_raw
  axes[0, 2].imshow(diff, cmap='RdBu', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
  axes[0, 2].set_title("Différence\n(Corrigée - Brute)")
  axes[0, 2].axis('off')
  
  # Histogrammes
  axes[1, 0].hist(img_raw.flatten(), bins=100, alpha=0.7, label='Brute', color='blue')
  axes[1, 0].set_title("Histogramme - Image brute")
  axes[1, 0].set_xlabel("Intensité")
  axes[1, 0].set_ylabel("Fréquence")
  
  axes[1, 1].hist(img_corrected.flatten(), bins=100, alpha=0.7, label='Corrigée', color='red')
  axes[1, 1].set_title("Histogramme - Image corrigée")
  axes[1, 1].set_xlabel("Intensité")
  axes[1, 1].set_ylabel("Fréquence")
  
  # Comparaison des histogrammes
  axes[1, 2].hist(img_raw.flatten(), bins=100, alpha=0.5, label='Brute', color='blue')
  axes[1, 2].hist(img_corrected.flatten(), bins=100, alpha=0.5, label='Corrigée', color='red')
  axes[1, 2].set_title("Comparaison des histogrammes")
  axes[1, 2].set_xlabel("Intensité")
  axes[1, 2].set_ylabel("Fréquence")
  axes[1, 2].legend()
  
  plt.tight_layout()
  plt.savefig('dicom_test_result.png', dpi=300, bbox_inches='tight')
  plt.show()
  
  # Analyse de la photométrie
  photometry = metadata['PhotometricInterpretation']
  if photometry == 'MONOCHROME1':
    print(f"\n=== DIAGNOSTIC ===")
    print(f"Cette image utilise MONOCHROME1 (fond blanc).")
    print(f"La correction a inversé l'image pour avoir un fond noir cohérent.")
    print(f"Différence moyenne: {np.mean(diff):.1f}")
  elif photometry == 'MONOCHROME2':
    print(f"\n=== DIAGNOSTIC ===")
    print(f"Cette image utilise MONOCHROME2 (fond noir).")
    print(f"Aucune correction nécessaire.")
    print(f"Différence moyenne: {np.mean(diff):.1f}")
  else:
    print(f"\n=== DIAGNOSTIC ===")
    print(f"Photométrie inconnue: {photometry}")
    print(f"Vérifiez les métadonnées DICOM.")

def generate_comparison_figure(image_path, output_filename):
  """
  Génère une figure unifiée à 4 panneaux (Anti-Plagiat) :
  1. Image brute avec métadonnées
  2. Image corrigée (fond noir harmonisé)
  3. Profil d'intensité transversal (coupe médiane 1D)
  4. Histogramme comparatif et distribution d'intensité (Emerald & Purple)
  """
  if not os.path.exists(image_path):
    return False
    
  ds = pydicom.dcmread(image_path)
  img_raw = ds.pixel_array
  img_corrected = read_dicom(image_path)
  photometry = str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2'))
  
  os.makedirs('graphes', exist_ok=True)
  out_path = os.path.join('graphes', output_filename)
  
  fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
  
  # Panel 1: Image brute
  axes[0].imshow(img_raw, cmap='gray')
  axes[0].set_title(f"1. DICOM Brut\n({photometry})", fontsize=11, fontweight='bold', color='#1a202c')
  axes[0].axis('off')
  
  # Panel 2: Image corrigée (fond noir)
  axes[1].imshow(img_corrected, cmap='gray')
  axes[1].set_title("2. Image Corrigée\n(Fond noir unifié)", fontsize=11, fontweight='bold', color='#1a202c')
  axes[1].axis('off')
  
  # Panel 3: Profil d'intensité transversal (Coupe médiane 1D) - ORIGINAL & ANTI-PLAGIAT
  mid_row_raw = img_raw[img_raw.shape[0] // 2, :]
  mid_row_corr = img_corrected[img_corrected.shape[0] // 2, :]
  axes[2].plot(mid_row_raw, label='Signal Brut', color='#7b2cbf', alpha=0.75, linewidth=1.5)
  axes[2].plot(mid_row_corr, label='Signal Corrigé', color='#2ca02c', alpha=0.85, linewidth=1.5, linestyle='--')
  axes[2].set_title("3. Profil Transversal\n(Ligne médiane 1D)", fontsize=11, fontweight='bold', color='#1a202c')
  axes[2].set_xlabel("Position X (Pixels)", fontsize=9)
  axes[2].set_ylabel("Niveau de gris", fontsize=9)
  axes[2].grid(True, linestyle=':', alpha=0.6)
  axes[2].legend(fontsize=8, loc='best')
  
  # Panel 4: Histogramme comparatif (Palette émeraude / violette)
  axes[3].hist(img_raw.flatten(), bins=80, alpha=0.5, label='Brut', color='#7b2cbf')
  axes[3].hist(img_corrected.flatten(), bins=80, alpha=0.5, label='Corrigé', color='#2ca02c')
  axes[3].set_title("4. Distribution Intensité\n(Histogramme)", fontsize=11, fontweight='bold', color='#1a202c')
  axes[3].set_xlabel("Valeur Pixel", fontsize=9)
  axes[3].set_ylabel("Fréquence", fontsize=9)
  axes[3].grid(True, linestyle=':', alpha=0.6)
  axes[3].legend(fontsize=8, loc='best')
  
  plt.tight_layout()
  plt.savefig(out_path, dpi=300, bbox_inches='tight')
  plt.close()
  print(f"[EXEMPLE ANTI-PLAGIAT] Graphique à 4 panneaux sauvegardé: {out_path}")
  return True


if __name__ == '__main__':
  base_dir = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
  
  mono1_path = None
  mono2_path = None
  
  if os.path.exists(base_dir):
    print("Recherche d'exemples MONOCHROME1 et MONOCHROME2 dans le dataset...")
    for root, dirs, files in os.walk(base_dir):
      for file in files:
        if file.endswith('.dicom'):
          full_path = os.path.join(root, file)
          try:
            ds = pydicom.dcmread(full_path, stop_before_pixels=True)
            photo = str(getattr(ds, 'PhotometricInterpretation', ''))
            if photo == 'MONOCHROME1' and mono1_path is None:
              mono1_path = full_path
            elif photo == 'MONOCHROME2' and mono2_path is None:
              mono2_path = full_path
          except Exception:
            pass
        if mono1_path and mono2_path:
          break
      if mono1_path and mono2_path:
        break
        
  if mono1_path:
    print(f"Image MONOCHROME1 trouvée: {mono1_path}")
    generate_comparison_figure(mono1_path, 'figure6_monochrome1.png')
    
  if mono2_path:
    print(f"Image MONOCHROME2 trouvée: {mono2_path}")
    generate_comparison_figure(mono2_path, 'figure7_monochrome2.png') 