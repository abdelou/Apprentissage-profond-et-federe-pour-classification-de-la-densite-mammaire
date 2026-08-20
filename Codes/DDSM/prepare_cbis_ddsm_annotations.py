import argparse
import glob
import os
import pandas as pd

DENSITY_MAP = {1: "DENSITY A", 2: "DENSITY B", 3: "DENSITY C", 4: "DENSITY D"}
SIDE_MAP = {"LEFT": "L", "RIGHT": "R"}

def load_all_cases(csv_dir):
  frames = []
  for fname, split, abnormality in [
    ("calc_case_description_train_set.csv", "training", "Calc"),
    ("calc_case_description_test_set.csv", "test", "Calc"),
    ("mass_case_description_train_set.csv", "training", "Mass"),
    ("mass_case_description_test_set.csv", "test", "Mass"),
  ]:
    path = os.path.join(csv_dir, fname)
    if not os.path.exists(path):
      print(f"Fichier manquant ignoré: {path}")
      continue
    df = pd.read_csv(path)
    df["split"] = split
    df["abnormality_prefix"] = abnormality
    frames.append(df)
  return pd.concat(frames, ignore_index=True)

def resolve_image_path(root, abnormality_prefix, split_label, patient_id, side, view):
  # Résolution des chemins d'image
  split_folder_tag = "Training" if split_label == "training" else "Test"
  pattern = os.path.join(root, f"{abnormality_prefix}-{split_folder_tag}_{patient_id}_{side}_{view}*")
  case_dirs = glob.glob(pattern)
  if not case_dirs:
    return None
  best_path, best_size = None, -1
  for case_dir in case_dirs:
    for dcm_path in glob.glob(os.path.join(case_dir, "**", "*.dcm"), recursive=True):
      size = os.path.getsize(dcm_path)
      if size > best_size:
        best_size, best_path = size, dcm_path
  return best_path

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv_dir", required=True)
  parser.add_argument("--images_root", required=True)
  parser.add_argument("--out_csv", required=True)
  args = parser.parse_args()

  df = load_all_cases(args.csv_dir)
  df = df.rename(columns={
    "breast density": "breast_density_raw",
    "left or right breast": "side",
    "image view": "view_position",
  })
  df["laterality"] = df["side"].map(SIDE_MAP)
  df["breast_density_raw"] = pd.to_numeric(df["breast_density_raw"], errors="coerce")
  df = df.dropna(subset=["breast_density_raw"])
  df["breast_density_raw"] = df["breast_density_raw"].astype(int)
  df = df[df["breast_density_raw"].isin(DENSITY_MAP.keys())]
  df["breast_density"] = df["breast_density_raw"].map(DENSITY_MAP)

  # Déduplication
  dedup = df.drop_duplicates(subset=["patient_id", "side", "view_position", "split"]).copy()
  print(f"Brut : {len(df)} -> Unique : {len(dedup)}")

  rows = []
  not_found = 0
  for _, row in dedup.iterrows():
    img_path = resolve_image_path(
      args.images_root, row["abnormality_prefix"], row["split"],
      row["patient_id"], row["side"], row["view_position"]
    )
    if img_path is None:
      not_found += 1
      continue
    study_id = f"{row['patient_id']}_{row['side']}"
    image_id = f"{row['patient_id']}_{row['side']}_{row['view_position']}"
    rows.append({
      "study_id": study_id,
      "image_id": image_id,
      "laterality": row["laterality"],
      "view_position": row["view_position"],
      "breast_density": row["breast_density"],
      "split": row["split"],
      "image_path": img_path,
    })

  out_df = pd.DataFrame(rows)
  out_df.to_csv(args.out_csv, index=False)
  print(f"Résolues : {len(out_df)} - Non trouvées : {not_found}")
  print(f"CSV sauvegardé dans : {args.out_csv}")

if __name__ == "__main__":
  main()
