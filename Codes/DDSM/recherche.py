#%%
import pandas as pd

# Charger le CSV
csv_path = "output_annotations_augmented.csv"
df = pd.read_csv(csv_path)


study_id_recherche = "25e26de58e476ec0075b26f8d55898d3_aug"

# Filtrer les lignes correspondant à ce study_id
resultat = df[df["study_id"] == study_id_recherche]

# Afficher le résultat
print(resultat["image_id"])
print(resultat["laterality"])
print(resultat["split"])
print(len(resultat))
# Si tu veux sauvegarder le résultat dans un nouveau CSV :
# resultat.to_csv("lignes_pour_study_id.csv", index=False)

#%% Pour compter les images 
import pandas as pd

# Charger le CSV d'annotations
df = pd.read_csv('output_annotations.csv')

# Compter le nombre d'images par densité et par split
compte = df.groupby(['breast_density', 'split']).size().unstack(fill_value=0)

print(compte)
# %%
import os
from collections import defaultdict

# Chemin vers le dossier output
output_root = "/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/output"

# Dictionnaire pour stocker les comptes
compte = defaultdict(lambda: defaultdict(int))

for split in ["train", "test"]:
    for density in ["density_A", "density_B", "density_C", "density_D"]:
        split_density_dir = os.path.join(output_root, split, density)
        if not os.path.exists(split_density_dir):
            continue
        # Parcours tous les sous-dossiers (study_id)
        for study_id in os.listdir(split_density_dir):
            study_dir = os.path.join(split_density_dir, study_id)
            if not os.path.isdir(study_dir):
                continue
            # Compte les fichiers .dicom dans ce dossier
            for fname in os.listdir(study_dir):
                if fname.endswith(".dicom"):
                    compte[density][split] += 1

# Affichage des résultats
print("Densité    |  Train   |  Test")
print("-------------------------------")
for density in ["density_A", "density_B", "density_C", "density_D"]:
    n_train = compte[density]["train"]
    n_test = compte[density]["test"]
    print(f"{density:10} | {n_train:7} | {n_test:5}")
# %%
import timm
print(timm.list_models())
# %%
import timm
for el in timm.list_models():
    if 'pvt' in el:
        print(el)
# %%
from itertools import combinations
print(list(combinations(range(4), 2)))
# %%
# je veux creer un script qui va parcourir la liste des study_id et qui va compter le nombre d'images par study_id et me dire si les densités sont les meme pour les images de la meme study_id


