"""
Construit un CSV d'annotations propre pour CBIS-DDSM, compatible avec le pipeline
du projet (mêmes colonnes clés que breast-level_annotations.csv de VinDr-Mammo :
study_id, image_id, laterality, view_position, breast_density, split).

CBIS-DDSM ne fournit pas directement de fichier "breast-level" comme VinDr — il faut
le reconstruire à partir des 4 CSV officiels (calc/mass x train/test), qui décrivent
des ANOMALIES (une image peut apparaître plusieurs fois si elle a plusieurs anomalies).
On déduplique donc par (patient_id, side, view, split) puisque la densité mammaire
est une propriété du sein, pas de l'anomalie.

ici : les noms de fichiers réels sur disque (UUID, ex.
"9587fd8e-e6d7-4de0-83c8-d945c6f5330c.dcm") ne correspondent PAS aux noms indiqués
dans la colonne "image file path" des CSV (ex. "000000.dcm") — TCIA/NBIA a renommé
les fichiers au téléchargement. De plus, le dossier "sans suffixe" attendu pour
l'image complète (ex. "Calc-Test_P_00038_LEFT_CC") n'existe souvent pas : l'image
complète ET le crop ROI se retrouvent tous les deux dans le dossier AVEC suffixe
(ex. "Calc-Test_P_00038_LEFT_CC_1"). On résout donc chaque image en cherchant tous
les dossiers de cas correspondant à (patient_id, side, view) quel que soit le
suffixe, et en prenant le plus gros fichier .dcm trouvé (l'image complète pèse
plusieurs Mo, le crop/masque ROI ne pèse que quelques centaines de Ko).
"""
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
            print(f"[WARN] Fichier manquant, ignoré: {path}")
            continue
        df = pd.read_csv(path)
        df["split"] = split
        df["abnormality_prefix"] = abnormality
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def resolve_image_path(root, abnormality_prefix, split_label, patient_id, side, view):
    """Cherche tous les dossiers de cas (n'importe quel suffixe d'anomalie) pour
    (patient_id, side, view) et renvoie le plus gros fichier .dcm trouvé."""
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
    parser.add_argument("--csv_dir", required=True, help="Dossier contenant les 4 CSV calc/mass_case_description_*")
    parser.add_argument("--images_root", required=True, help="Dossier cbis_ddsm contenant les dossiers de cas")
    parser.add_argument("--out_csv", required=True, help="Chemin du CSV d'annotations à générer")
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

    # Déduplication : une image (patient, side, view, split) peut apparaître
    # plusieurs fois si le sein a plusieurs anomalies annotées.
    dedup = df.drop_duplicates(subset=["patient_id", "side", "view_position", "split"]).copy()
    print(f"[INFO] {len(df)} lignes brutes -> {len(dedup)} images uniques (patient/côté/vue/split)")

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
    print(f"[INFO] {len(out_df)} images résolues, {not_found} non trouvées sur disque")
    print(f"[INFO] Répartition split:\n{out_df['split'].value_counts()}")
    print(f"[INFO] Répartition densité:\n{out_df['breast_density'].value_counts()}")
    print(f"[INFO] CSV sauvegardé: {args.out_csv}")


if __name__ == "__main__":
    main()
