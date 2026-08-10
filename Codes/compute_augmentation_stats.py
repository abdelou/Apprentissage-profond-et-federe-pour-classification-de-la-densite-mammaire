"""
Calcule les vrais chiffres pour la section "Data Augmentation" du memoire :
  1) repartition des images par densite et par split (train/test), AVANT
     augmentation (comptage brut du CSV VinDr-Mammo)
  2) repartition APRES augmentation, en reproduisant exactement la meme
     logique de suréchantillonnage que celle utilisee dans le code
     d'entrainement (target_count = 50% de la classe majoritaire, voir
     HybridGLCMDataset / HybridMammographyDataset)
  3) verification de la coherence gauche/droite : pour une meme patiente
     (meme study_id), est-ce que la densite du sein gauche et du sein droit
     est la meme ? (utile pour justifier que l'augmentation ne mélange pas
     des etiquettes incoherentes)

Ne fait aucune supposition sur les chiffres : tout est recalcule depuis le
CSV reel, pas de valeurs inventees.
"""
import pandas as pd
import os

VINDR_ROOT = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
ANNOTATIONS_CSV = os.path.join(VINDR_ROOT, "breast-level_annotations.csv")
CLASSES = ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]

df = pd.read_csv(ANNOTATIONS_CSV)

# ============================================================
# 1) Repartition brute par split et par classe (AVANT augmentation)
# ============================================================
print("=" * 70)
print("1) REPARTITION BRUTE (AVANT augmentation), par split et par classe")
print("=" * 70)
for split_name in ["training", "test"]:
    sub = df[df["split"] == split_name]
    counts = sub["breast_density"].value_counts().reindex(CLASSES, fill_value=0)
    print(f"\n--- split = {split_name} (total {len(sub)} images) ---")
    for c in CLASSES:
        print(f"  {c}: {counts[c]}")

# ============================================================
# 2) Repartition APRES augmentation (meme logique que le code d'entrainement)
#    Voir HybridGLCMDataset.__init__ : target_count = 50% de la classe
#    majoritaire, les classes en dessous sont dupliquees jusqu'a ce seuil.
# ============================================================
print("\n" + "=" * 70)
print("2) REPARTITION APRES suréchantillonnage (train uniquement, logique du code)")
print("=" * 70)
train_df = df[df["split"] == "training"]
class_counts = train_df["breast_density"].value_counts()
n_majority = class_counts.max()
target_count = int(n_majority * 0.5)
print(f"Classe majoritaire : {class_counts.idxmax()} ({n_majority} images)")
print(f"target_count (50% de la classe majoritaire) = {target_count}\n")

after_counts = {}
for c in CLASSES:
    n_avant = int(class_counts.get(c, 0))
    n_apres = max(n_avant, target_count) if n_avant < target_count else n_avant
    after_counts[c] = n_apres
    print(f"  {c}: {n_avant}  ->  {n_apres}")

ratio_avant = class_counts.max() / class_counts.min()
ratio_apres = max(after_counts.values()) / min(after_counts.values())
print(f"\nRatio classe majoritaire / classe minoritaire AVANT : 1:{ratio_avant:.1f}")
print(f"Ratio classe majoritaire / classe minoritaire APRES  : 1:{ratio_apres:.1f}")

# ============================================================
# 3) Coherence gauche/droite (meme patiente/etude, meme densite ?)
# ============================================================
print("\n" + "=" * 70)
print("3) Coherence de densite entre sein gauche et sein droit (meme study_id)")
print("=" * 70)
# on regroupe par study_id, on regarde s'il n'y a qu'UNE seule densite
# distincte pour toutes les images (CC+MLO, gauche+droite) de cette etude
grouped = df.groupby("study_id")["breast_density"].nunique()
n_coherent = (grouped == 1).sum()
n_total = len(grouped)
print(f"{n_coherent}/{n_total} études ont une densité cohérente sur toutes leurs images ({n_coherent/n_total*100:.1f}%)")
print("(NB : ceci regroupe par study_id, qui peut déjà être par sein selon la structure du CSV -- à vérifier avec les colonnes disponibles ci-dessous si besoin)")
print("\nColonnes disponibles dans le CSV :", list(df.columns))
