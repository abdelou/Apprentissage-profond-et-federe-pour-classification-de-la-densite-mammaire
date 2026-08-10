#!/usr/bin/env python3
"""
Script pour parcourir aléatoirement la base de données et afficher des mammographies avec leurs densités.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse
from preprocessing import read_dicom, preprocess_image
import pydicom

class MammogramBrowser:
    """Classe pour parcourir et afficher des mammographies aléatoirement."""
    
    def __init__(self, annotations_csv, image_root):
        """
        Initialise le browser avec les chemins des données.
        
        Args:
            annotations_csv (str): Chemin vers le fichier CSV des annotations
            image_root (str): Chemin vers le dossier racine des images
        """
        self.annotations_csv = annotations_csv
        self.image_root = image_root
        self.df = pd.read_csv(annotations_csv)
        
        # Mapping des densités
        self.density_map = {
            "DENSITY A": "density_A", "DENSITY B": "density_B",
            "DENSITY C": "density_C", "DENSITY D": "density_D"
        }
        
        self.split_map = {"training": "train", "test": "test"}
        
        print(f"[INFO] Base de données chargée: {len(self.df)} images")
        print(f"[INFO] Répartition par densité:")
        density_counts = self.df['breast_density'].value_counts()
        for density, count in density_counts.items():
            print(f"  - {density}: {count} images")
    
    def get_random_sample(self, num_images=10, density_filter=None, split_filter=None):
        """
        Obtient un échantillon aléatoire d'images.
        
        Args:
            num_images (int): Nombre d'images à afficher
            density_filter (str): Filtrer par densité spécifique (ex: "DENSITY A")
            split_filter (str): Filtrer par split ("training" ou "test")
        
        Returns:
            pd.DataFrame: Échantillon d'images
        """
        filtered_df = self.df.copy()
        
        if density_filter:
            filtered_df = filtered_df[filtered_df['breast_density'] == density_filter]
            print(f"[INFO] Filtrage par densité: {density_filter}")
        
        if split_filter:
            filtered_df = filtered_df[filtered_df['split'] == split_filter]
            print(f"[INFO] Filtrage par split: {split_filter}")
        
        if len(filtered_df) == 0:
            print(f"[WARN] Aucune image trouvée avec les filtres spécifiés")
            return pd.DataFrame()
        
        # Échantillonnage aléatoire
        sample_size = min(num_images, len(filtered_df))
        sample_df = filtered_df.sample(sample_size, random_state=random.randint(1, 1000))
        
        print(f"[INFO] Échantillon de {len(sample_df)} images sélectionné")
        return sample_df
    
    def load_and_preprocess_image(self, row):
        """
        Charge et prétraite une image à partir d'une ligne du DataFrame.
        
        Args:
            row (pd.Series): Ligne du DataFrame avec les informations de l'image
        
        Returns:
            tuple: (image_originale, image_prétraitée, métadonnées)
        """
        # Construction du chemin
        split = self.split_map.get(row["split"])
        density = self.density_map.get(row["breast_density"])
        study_id = str(row["study_id"])
        image_id = str(row["image_id"])
        
        image_path = os.path.join(self.image_root, split, density, study_id, f"{image_id}.dicom")
        
        if not os.path.exists(image_path):
            print(f"[WARN] Image non trouvée: {image_path}")
            return None, None, None
        
        try:
            # Lecture DICOM brute
            ds = pydicom.dcmread(image_path)
            img_raw = ds.pixel_array
            
            # Lecture avec prétraitement
            img_processed = read_dicom(image_path)
            img_final = preprocess_image(img_processed, laterality=row['laterality'])
            
            # Métadonnées
            metadata = {
                'PhotometricInterpretation': getattr(ds, 'PhotometricInterpretation', 'N/A'),
                'BitsAllocated': getattr(ds, 'BitsAllocated', 'N/A'),
                'PixelRepresentation': getattr(ds, 'PixelRepresentation', 'N/A'),
                'WindowCenter': getattr(ds, 'WindowCenter', 'N/A'),
                'WindowWidth': getattr(ds, 'WindowWidth', 'N/A'),
                'ImageSize': f"{img_raw.shape[0]}x{img_raw.shape[1]}"
            }
            
            return img_raw, img_final, metadata
            
        except Exception as e:
            print(f"[ERROR] Erreur lors du chargement de {image_path}: {e}")
            return None, None, None
    
    def display_mammograms(self, num_images=5, density_filter=None, split_filter=None, 
                          save_path=None, show_metadata=True):
        """
        Affiche un échantillon aléatoire de mammographies.
        
        Args:
            num_images (int): Nombre d'images à afficher
            density_filter (str): Filtrer par densité spécifique
            split_filter (str): Filtrer par split
            save_path (str): Chemin pour sauvegarder la figure
            show_metadata (bool): Afficher les métadonnées
        """
        sample_df = self.get_random_sample(num_images, density_filter, split_filter)
        
        if len(sample_df) == 0:
            print("[ERROR] Aucune image à afficher")
            return
        
        # Calcul du layout
        cols = min(3, len(sample_df))
        rows = (len(sample_df) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Couleurs pour les densités
        density_colors = {
            "DENSITY A": "green",
            "DENSITY B": "blue", 
            "DENSITY C": "orange",
            "DENSITY D": "red"
        }
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            img_raw, img_final, metadata = self.load_and_preprocess_image(row)
            
            if img_raw is None:
                continue
            
            # Calcul de la position dans la grille
            row_idx = idx // cols
            col_idx = idx % cols
            ax = axes[row_idx, col_idx]
            
            # Affichage de l'image prétraitée
            ax.imshow(img_final, cmap='gray')
            
            # Titre avec informations
            density = row['breast_density']
            color = density_colors.get(density, 'black')
            title = f"{density}\n{row['split']} - {row['laterality']}"
            ax.set_title(title, color=color, fontweight='bold', fontsize=10)
            ax.axis('off')
            
            # Affichage des métadonnées si demandé
            if show_metadata and metadata:
                info_text = f"Size: {metadata['ImageSize']}\nPhotometry: {metadata['PhotometricInterpretation']}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Masquer les axes vides
        for idx in range(len(sample_df), rows * cols):
            row_idx = idx // cols
            col_idx = idx % cols
            axes[row_idx, col_idx].axis('off')
        
        plt.suptitle(f"Échantillon aléatoire de mammographies\n"
                    f"({len(sample_df)} images)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Figure sauvegardée: {save_path}")
        
        plt.show()
        
        # Affichage des statistiques
        self.print_sample_statistics(sample_df)
    
    def print_sample_statistics(self, sample_df):
        """Affiche les statistiques de l'échantillon."""
        print("\n=== STATISTIQUES DE L'ÉCHANTILLON ===")
        
        # Répartition par densité
        density_counts = sample_df['breast_density'].value_counts()
        print("Répartition par densité:")
        for density, count in density_counts.items():
            print(f"  - {density}: {count} images")
        
        # Répartition par split
        split_counts = sample_df['split'].value_counts()
        print("\nRépartition par split:")
        for split, count in split_counts.items():
            print(f"  - {split}: {count} images")
        
        # Répartition par latéralité
        laterality_counts = sample_df['laterality'].value_counts()
        print("\nRépartition par latéralité:")
        for laterality, count in laterality_counts.items():
            print(f"  - {laterality}: {count} images")
    
    def interactive_browse(self):
        """Mode de navigation interactive."""
        print("\n=== NAVIGATION INTERACTIVE ===")
        print("Commandes disponibles:")
        print("  - 'q': Quitter")
        print("  - 'n': Nouvel échantillon")
        print("  - 'a': Filtrer par DENSITY A")
        print("  - 'b': Filtrer par DENSITY B") 
        print("  - 'c': Filtrer par DENSITY C")
        print("  - 'd': Filtrer par DENSITY D")
        print("  - 't': Filtrer par training")
        print("  - 's': Filtrer par test")
        print("  - 'all': Afficher toutes les densités")
        
        current_filter = None
        current_split = None
        
        while True:
            try:
                command = input("\nCommande (ou 'help' pour l'aide): ").strip().lower()
                
                if command == 'q':
                    print("Au revoir!")
                    break
                elif command == 'n':
                    self.display_mammograms(num_images=5, density_filter=current_filter, split_filter=current_split)
                elif command == 'a':
                    current_filter = "DENSITY A"
                    print(f"Filtre activé: {current_filter}")
                    self.display_mammograms(num_images=5, density_filter=current_filter, split_filter=current_split)
                elif command == 'b':
                    current_filter = "DENSITY B"
                    print(f"Filtre activé: {current_filter}")
                    self.display_mammograms(num_images=5, density_filter=current_filter, split_filter=current_split)
                elif command == 'c':
                    current_filter = "DENSITY C"
                    print(f"Filtre activé: {current_filter}")
                    self.display_mammograms(num_images=5, density_filter=current_filter, split_filter=current_split)
                elif command == 'd':
                    current_filter = "DENSITY D"
                    print(f"Filtre activé: {current_filter}")
                    self.display_mammograms(num_images=5, density_filter=current_filter, split_filter=current_split)
                elif command == 't':
                    current_split = "training"
                    print(f"Split activé: {current_split}")
                    self.display_mammograms(num_images=5, density_filter=current_filter, split_filter=current_split)
                elif command == 's':
                    current_split = "test"
                    print(f"Split activé: {current_split}")
                    self.display_mammograms(num_images=5, density_filter=current_filter, split_filter=current_split)
                elif command == 'all':
                    current_filter = None
                    current_split = None
                    print("Filtres désactivés")
                    self.display_mammograms(num_images=5, density_filter=current_filter, split_filter=current_split)
                elif command == 'help':
                    print("Commandes disponibles:")
                    print("  - 'q': Quitter")
                    print("  - 'n': Nouvel échantillon")
                    print("  - 'a': Filtrer par DENSITY A")
                    print("  - 'b': Filtrer par DENSITY B") 
                    print("  - 'c': Filtrer par DENSITY C")
                    print("  - 'd': Filtrer par DENSITY D")
                    print("  - 't': Filtrer par training")
                    print("  - 's': Filtrer par test")
                    print("  - 'all': Afficher toutes les densités")
                else:
                    print("Commande non reconnue. Tapez 'help' pour l'aide.")
                    
            except KeyboardInterrupt:
                print("\nAu revoir!")
                break
            except Exception as e:
                print(f"Erreur: {e}")

def main():
    """Fonction principale avec arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description='Parcours aléatoire de mammographies')
    
    parser.add_argument('--annotations_csv', type=str, 
                       default='/Volumes/cimobile/Entrainement_cetic/vindr-mammo_dataset/physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations.csv',
                       help='Chemin vers le fichier CSV des annotations')
    
    parser.add_argument('--image_root', type=str,
                       default='/Volumes/cimobile/Entrainement_cetic/vindr-mammo_dataset/organized_layout',
                       help='Chemin vers le dossier racine des images')
    
    parser.add_argument('--num_images', type=int, default=10,
                       help='Nombre d\'images à afficher')
    
    parser.add_argument('--density', type=str, choices=['A', 'B', 'C', 'D'],
                       help='Filtrer par densité spécifique')
    
    parser.add_argument('--split', type=str, choices=['training', 'test'],
                       help='Filtrer par split')
    
    parser.add_argument('--save', type=str,
                       help='Chemin pour sauvegarder la figure')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Mode de navigation interactive')
    
    parser.add_argument('--no_metadata', action='store_true',
                       help='Ne pas afficher les métadonnées')
    
    args = parser.parse_args()
    
    # Conversion du filtre de densité
    density_filter = None
    if args.density:
        density_filter = f"DENSITY {args.density}"
    
    # Création du browser
    browser = MammogramBrowser(args.annotations_csv, args.image_root)
    
    if args.interactive:
        browser.interactive_browse()
    else:
        browser.display_mammograms(
            num_images=args.num_images,
            density_filter=density_filter,
            split_filter=args.split,
            save_path=args.save,
            show_metadata=not args.no_metadata
        )

if __name__ == '__main__':
    main() 