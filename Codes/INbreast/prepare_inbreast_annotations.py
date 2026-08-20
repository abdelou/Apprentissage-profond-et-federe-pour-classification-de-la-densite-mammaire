# Script : prepare_inbreast_annotations.py
import argparse
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DENSITY_MAP = {1: "DENSITY A", 2: "DENSITY B", 3: "DENSITY C", 4: "DENSITY D"}


def find_dicom(images_root, file_id):
  """Cherche le fichier .dcm dont le nom commence par l'identifiant de fichier."""
  pattern = os.path.join(images_root, f"{file_id}_*.dcm")
  matches = glob.glob(pattern)
  return matches[0] if matches else None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--xls_path", required=True, help="Chemin vers INbreast.xls")
  parser.add_argument("--images_root", required=True, help="Dossier ALL-IMGS contenant les DICOM")
  parser.add_argument("--out_csv", required=True, help="Chemin du CSV d'annotations a generer")
  parser.add_argument("--test_size", type=float, default=0.15, help="Fraction du split test (defaut 0.15)")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  df = pd.read_excel(args.xls_path)

  # Densite : coercion numerique -> vire la chaine vide, le NaN, et la
  # valeur texte "28" (hors plage 1-4) en un seul passage.
  df["ACR"] = pd.to_numeric(df["ACR"], errors="coerce")
  df = df.dropna(subset=["ACR"])
  df["ACR"] = df["ACR"].astype(int)
  df = df[df["ACR"].isin(DENSITY_MAP.keys())]
  df["breast_density"] = df["ACR"].map(DENSITY_MAP)

  # Vue et laterality : on ne garde que CC/MLO valides (ecarte "FB" et NaN)
  df = df[df["View"].isin(["CC", "MLO"])]
  df = df.dropna(subset=["Laterality"])
  df["view_position"] = df["View"]
  df["laterality"] = df["Laterality"]

  # File Name arrive en flottant depuis Excel (22678622.0) -> entier -> str
  df = df.dropna(subset=["File Name"])
  df["file_id"] = df["File Name"].astype(int).astype(str)

  rows = []
  not_found = 0
  for _, row in df.iterrows():
    dcm_path = find_dicom(args.images_root, row["file_id"])
    if dcm_path is None:
      not_found += 1
      continue
    rows.append({
      "study_id": row["file_id"],
      "image_id": row["file_id"],
      "laterality": row["laterality"],
      "view_position": row["view_position"],
      "breast_density": row["breast_density"],
      "image_path": dcm_path,
    })

  out_df = pd.DataFrame(rows)
  print(f"{len(df)} lignes annotees valides -> {len(out_df)} images resolues sur disque, {not_found} non trouvees")

  # Pas de split officiel dans INbreast -> split stratifie par densite, seed fixe
  train_df, test_df = train_test_split(
    out_df, test_size=args.test_size, random_state=args.seed,
    stratify=out_df["breast_density"],
  )
  train_df = train_df.copy(); train_df["split"] = "training"
  test_df = test_df.copy(); test_df["split"] = "test"
  out_df = pd.concat([train_df, test_df], ignore_index=True)

  out_df.to_csv(args.out_csv, index=False)
  print(f"Repartition split:\n{out_df['split'].value_counts()}")
  print(f"Repartition densite (total):\n{out_df['breast_density'].value_counts()}")
  print(f"Repartition densite (train):\n{train_df['breast_density'].value_counts()}")
  print(f"Repartition densite (test):\n{test_df['breast_density'].value_counts()}")
  print(f"CSV sauvegarde: {args.out_csv}")


if __name__ == "__main__":
  main()
