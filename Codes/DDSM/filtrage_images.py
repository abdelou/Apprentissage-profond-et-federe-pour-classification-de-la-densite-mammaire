import os
import shutil
import pandas as pd

# Chemins
csv_path = "breast-level_annotations.csv"
images_root ="/home_nfs/abdelouahada/images"
output_root = "/home_nfs/abdelouahada/BD"

# Lire le CSV
df = pd.read_csv(csv_path)

# Filtrage selon les critères
# (on garde split, density, laterality R ou L, sans distinction de vue)
df = df[
  (df["split"].isin(["training", "test"])) &
  (df["laterality"].isin(["R", "L"])) &
  (df["breast_density"].isin(["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]))
]

split_map = {"training": "train", "test": "test"}
density_map = {
  "DENSITY A": "density_A",
  "DENSITY B": "density_B",
  "DENSITY C": "density_C",
  "DENSITY D": "density_D"
}

# Nouvelle structure : output/{split}/{density}/{study_id}/image_id.dicom
moved_rows = []

notfound = []
for _, row in df.iterrows():
  study_id = str(row["study_id"])
  image_id = str(row["image_id"])
  split = split_map.get(row["split"])
  density = density_map.get(row["breast_density"])
  if not (split and density):
    continue
  moved_rows.append(row)

  # Chemin source de l'image
  old_path = os.path.join(images_root, study_id, f"{image_id}.dicom")

  try:
    if not os.path.exists(old_path):
      print(f"Fichier non trouvé: {old_path}")
      notfound.append({
        'study_id': study_id,
        'image_id': image_id,
        'path': old_path
      })
      continue

    # Création du répertoire de destination
    dest_dir = os.path.join(output_root, split, density, study_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_image = os.path.join(dest_dir, f"{image_id}.dicom")

    # Déplacement du fichier
    shutil.move(old_path, dest_image)

  except Exception as e:
    print(f"Erreur lors du traitement de {study_id}/{image_id}: {str(e)}")
    notfound.append({
      'study_id': study_id,
      'image_id': image_id,
      'path': old_path,
      'error': str(e)
    })
  

# Sauvegarde des informations sur les fichiers non trouvés
if notfound:
  print(f"\nNombre de fichiers non trouvés: {len(notfound)}")
  df_notfound = pd.DataFrame(notfound)
  df_notfound.to_csv("images_non_trouvees.csv", index=False)
  print("Liste des fichiers non trouvés sauvegardée dans 'images_non_trouvees.csv'")

# Sauvegarde des informations sur les fichiers déplacés
if moved_rows:
  print(f"\nNombre de fichiers déplacés: {len(moved_rows)}")
  df_moved = pd.DataFrame(moved_rows)
  df_moved.to_csv("images_deplacees.csv", index=False)
  print("Liste des fichiers déplacés sauvegardée dans 'images_deplacees.csv'")

# Supprimer les anciens dossiers BI-RADS s'ils sont vides
def remove_empty_dirs(path):
  if not os.path.isdir(path):
    return
  # Parcours récursif
  for entry in os.listdir(path):
    full_path = os.path.join(path, entry)
    if os.path.isdir(full_path):
      remove_empty_dirs(full_path)
  # Si le dossier est vide, on le supprime
  if not os.listdir(path):
    os.rmdir(path)
    print(f"Dossier supprimé: {path}")

#for birads_folder in ["breast_birads_1", "breast_birads_2"]:
  #birads_path = os.path.join(output_root, birads_folder)
  #remove_empty_dirs(birads_path)
# Générer le nouveau CSV de description
if moved_rows:
  new_df = pd.DataFrame(moved_rows)
  output_csv = os.path.join(output_root, "output_annotations.csv")
  new_df.to_csv(output_csv, index=False)
  print(f"CSV de description créé : {output_csv}")
else:
  print("Aucune image déplacée, pas de CSV généré.")

