import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

def analyze_study_consistency(csv_path='output_annotations.csv'):
    """
    Analyse la cohérence des densités par study_id
    
    Args:
        csv_path: Chemin vers le fichier CSV des annotations
    """
    print("=== ANALYSE DE COHÉRENCE DES DENSITÉS PAR STUDY_ID ===")
    
    # Chargement des données
    df = pd.read_csv(csv_path)
    print(f"Total d'images dans le dataset: {len(df)}")
    print(f"Nombre de study_id uniques: {df['study_id'].nunique()}")
    
    time.sleep(5)
    # Groupement par study_id
    study_groups = df.groupby('study_id')
    
    # Statistiques par study_id
    study_stats = []
    inconsistent_studies = []
    consistent_studies = []
    
    for study_id, group in study_groups:
        num_images = len(group)
        unique_densities = group['breast_density'].unique()
        num_unique_densities = len(unique_densities)
        
        # Vérification de la cohérence
        is_consistent = num_unique_densities == 1
        
        study_info = {
            'study_id': study_id,
            'num_images': num_images,
            'unique_densities': list(unique_densities),
            'num_unique_densities': num_unique_densities,
            'is_consistent': is_consistent,
            'densities': list(group['breast_density']),
            'split': group['split'].iloc[0]  # Tous les images d'un study_id sont dans le même split
        }
        
        study_stats.append(study_info)
        
        if is_consistent:
            consistent_studies.append(study_info)
        else:
            inconsistent_studies.append(study_info)
    
    # Conversion en DataFrame pour analyse
    stats_df = pd.DataFrame(study_stats)
    
    # Affichage des résultats
    print(f"\n=== RÉSULTATS ===")
    print(f"Study_id cohérents: {len(consistent_studies)}")
    print(f"Study_id incohérents: {len(inconsistent_studies)}")
    print(f"Taux de cohérence: {len(consistent_studies)/len(stats_df)*100:.2f}%")
    
    # Statistiques sur le nombre d'images par study_id
    print(f"\n=== STATISTIQUES NOMBRE D'IMAGES PAR STUDY_ID ===")
    print(f"Nombre moyen d'images par study_id: {stats_df['num_images'].mean():.2f}")
    print(f"Nombre médian d'images par study_id: {stats_df['num_images'].median():.2f}")
    print(f"Nombre minimum d'images par study_id: {stats_df['num_images'].min()}")
    print(f"Nombre maximum d'images par study_id: {stats_df['num_images'].max()}")
    
    # Distribution du nombre d'images
    print(f"\n=== DISTRIBUTION DU NOMBRE D'IMAGES ===")
    image_counts = stats_df['num_images'].value_counts().sort_index()
    for count, freq in image_counts.items():
        print(f"  {count} image(s): {freq} study_id(s)")
    
    # Analyse des incohérences
    if inconsistent_studies:
        print(f"\n=== STUDY_ID INCOHÉRENTS ===")
        print("Voici les study_id avec des densités différentes:")
        
        for study in inconsistent_studies[:10]:  # Afficher les 10 premiers
            print(f"\nStudy ID: {study['study_id']}")
            print(f"  Nombre d'images: {study['num_images']}")
            print(f"  Densités trouvées: {study['unique_densities']}")
            print(f"  Densités détaillées: {study['densities']}")
            print(f"  Split: {study['split']}")
        
        if len(inconsistent_studies) > 10:
            print(f"\n... et {len(inconsistent_studies) - 10} autres study_id incohérents")
    
    # Analyse par split
    print(f"\n=== ANALYSE PAR SPLIT ===")
    split_analysis = stats_df.groupby('split').agg({
        'is_consistent': ['count', 'sum', 'mean'],
        'num_images': ['mean', 'median', 'min', 'max']
    }).round(3)
    
    print(split_analysis)
    
    # Création de visualisations
    create_visualizations(stats_df, consistent_studies, inconsistent_studies)
    
    return stats_df, consistent_studies, inconsistent_studies

def create_visualizations(stats_df, consistent_studies, inconsistent_studies):
    """Crée des visualisations pour l'analyse"""
    
    # Création du dossier pour les graphiques
    output_dir = Path("graphes")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Distribution du nombre d'images par study_id
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(stats_df['num_images'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Nombre d\'images par study_id')
    plt.ylabel('Fréquence')
    plt.title('Distribution du nombre d\'images par study_id')
    plt.grid(True, alpha=0.3)
    
    # 2. Cohérence vs Incohérence
    plt.subplot(2, 2, 2)
    consistency_counts = stats_df['is_consistent'].value_counts()
    colors = ['lightcoral' if not x else 'lightgreen' for x in consistency_counts.index]
    plt.pie(consistency_counts.values, labels=['cohérents', 'incohérents'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Répartition Cohérence/Incohérence')
    
    # 3. Nombre d'images par split
    plt.subplot(2, 2, 3)
    split_stats = stats_df.groupby('split')['num_images'].agg(['mean', 'median'])
    x = np.arange(len(split_stats))
    width = 0.35
    
    plt.bar(x - width/2, split_stats['mean'], width, label='Moyenne', alpha=0.8)
    plt.bar(x + width/2, split_stats['median'], width, label='Médiane', alpha=0.8)
    plt.xlabel('Split')
    plt.ylabel('Nombre d\'images')
    plt.title('Statistiques par split')
    plt.xticks(x, split_stats.index)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Taux de cohérence par split
    plt.subplot(2, 2, 4)
    consistency_by_split = stats_df.groupby('split')['is_consistent'].mean()
    plt.bar(consistency_by_split.index, consistency_by_split.values, 
            color=['lightblue', 'lightgreen'], alpha=0.8)
    plt.xlabel('Split')
    plt.ylabel('Taux de cohérence')
    plt.title('Taux de cohérence par split')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'study_consistency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Graphique détaillé des incohérences
    if inconsistent_studies:
        plt.figure(figsize=(15, 8))
        
        # Distribution des combinaisons de densités dans les study_id incohérents
        density_combinations = defaultdict(int)
        for study in inconsistent_studies:
            combo = tuple(sorted(study['unique_densities']))
            density_combinations[combo] += 1
        
        # Affichage des combinaisons les plus fréquentes
        sorted_combinations = sorted(density_combinations.items(), key=lambda x: x[1], reverse=True)
        
        combinations = [str(combo) for combo, _ in sorted_combinations[:10]]
        counts = [count for _, count in sorted_combinations[:10]]
        
        plt.barh(range(len(combinations)), counts, color='lightcoral', alpha=0.8)
        plt.yticks(range(len(combinations)), combinations)
        plt.xlabel('Nombre de study_id')
        plt.title('Combinaisons de densités les plus fréquentes dans les study_id incohérents')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'inconsistent_density_combinations.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_detailed_report(stats_df, consistent_studies, inconsistent_studies):
    """Génère un rapport détaillé en format texte"""
    
    report_lines = []
    report_lines.append("=== RAPPORT D'ANALYSE DE COHÉRENCE DES DENSITÉS ===")
    report_lines.append("")
    
    # Statistiques générales
    report_lines.append("STATISTIQUES GÉNÉRALES:")
    report_lines.append(f"- Total d'images: {len(stats_df) * stats_df['num_images'].sum()}")
    report_lines.append(f"- Nombre de study_id: {len(stats_df)}")
    report_lines.append(f"- Study_id cohérents: {len(consistent_studies)}")
    report_lines.append(f"- Study_id incohérents: {len(inconsistent_studies)}")
    report_lines.append(f"- Taux de cohérence: {len(consistent_studies)/len(stats_df)*100:.2f}%")
    report_lines.append("")
    
    # Statistiques par nombre d'images
    report_lines.append("DISTRIBUTION PAR NOMBRE D'IMAGES:")
    image_counts = stats_df['num_images'].value_counts().sort_index()
    for count, freq in image_counts.items():
        report_lines.append(f"- {count} image(s): {freq} study_id(s)")
    report_lines.append("")
    
    # Détails des incohérences
    if inconsistent_studies:
        report_lines.append("DÉTAILS DES INCOHÉRENCES:")
        for study in inconsistent_studies:
            report_lines.append(f"- Study ID {study['study_id']}: {study['num_images']} images, densités: {study['unique_densities']}")
        report_lines.append("")
    
    # Statistiques par split
    report_lines.append("STATISTIQUES PAR SPLIT:")
    split_analysis = stats_df.groupby('split').agg({
        'is_consistent': ['count', 'sum', 'mean'],
        'num_images': ['mean', 'median', 'min', 'max']
    }).round(3)
    
    for split in split_analysis.index:
        report_lines.append(f"- Split {split}:")
        report_lines.append(f"  * Nombre de study_id: {split_analysis.loc[split, ('is_consistent', 'count')]}")
        report_lines.append(f"  * Cohérents: {split_analysis.loc[split, ('is_consistent', 'sum')]}")
        report_lines.append(f"  * Taux de cohérence: {split_analysis.loc[split, ('is_consistent', 'mean')]*100:.1f}%")
        report_lines.append(f"  * Images moyennes par study_id: {split_analysis.loc[split, ('num_images', 'mean')]:.1f}")
    
    # Sauvegarde du rapport
    with open('graphes/study_consistency_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("Rapport détaillé sauvegardé dans 'graphes/study_consistency_report.txt'")
    
    return '\n'.join(report_lines)

def main():
    """Fonction principale"""
    
    # Analyse de cohérence
    stats_df, consistent_studies, inconsistent_studies = analyze_study_consistency()
    
    # Génération du rapport détaillé
    report = generate_detailed_report(stats_df, consistent_studies, inconsistent_studies)
    
    print("\n=== RÉSUMÉ ===")
    print(f"✅ {len(consistent_studies)} study_id avec densités cohérentes")
    print(f"❌ {len(inconsistent_studies)} study_id avec densités incohérentes")
    print(f"📊 Taux de cohérence: {len(consistent_studies)/len(stats_df)*100:.2f}%")
    
    if inconsistent_studies:
        print(f"\n⚠️  ATTENTION: {len(inconsistent_studies)} study_id ont des densités incohérentes!")
        print("   Cela peut indiquer des erreurs d'annotation ou des cas particuliers.")
        print("   Consultez le rapport détaillé pour plus d'informations.")

if __name__ == "__main__":
    main() 