"""
Combine les probabilites d'Approche 2 (ResNet50, fine-tuning direct, vues
MLO) et d'Approche 6 - variante siamoise (EfficientNet-B0, CC+MLO), par
fusion tardive (moyenne des probabilites), en alignant les deux jeux par
cle (study_id + laterality) plutot que par position -- les deux scripts de
dump construisent leurs jeux de donnees differemment (un par image, un par
paire), donc l'ordre des lignes n'est pas garanti identique.
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]

probs_a2 = np.load("ensemble_probs_a2_keyed.npy")
labels_a2 = np.load("ensemble_labels_a2_keyed.npy")
keys_a2 = np.load("ensemble_keys_a2_keyed.npy", allow_pickle=True)

probs_a6 = np.load("hybrid 2 branches siamese efficientnet/ensemble_probs_a6siam.npy")
labels_a6 = np.load("hybrid 2 branches siamese efficientnet/ensemble_labels_a6siam.npy")
keys_a6 = np.load("hybrid 2 branches siamese efficientnet/ensemble_keys_a6siam.npy", allow_pickle=True)

print(f"[INFO] Approche 2 : {len(keys_a2)} images -- Approche 6 siamoise : {len(keys_a6)} paires")

# Alignement par cle (study_id_laterality), commune aux deux
index_a2 = {k: i for i, k in enumerate(keys_a2)}
index_a6 = {k: i for i, k in enumerate(keys_a6)}
common_keys = sorted(set(index_a2.keys()) & set(index_a6.keys()))
print(f"[INFO] {len(common_keys)} seins communs aux deux jeux (intersection des cles)")

idx_a2 = [index_a2[k] for k in common_keys]
idx_a6 = [index_a6[k] for k in common_keys]

probs_a2_aligned = probs_a2[idx_a2]
probs_a6_aligned = probs_a6[idx_a6]
labels_a2_aligned = labels_a2[idx_a2]
labels_a6_aligned = labels_a6[idx_a6]

assert np.array_equal(labels_a2_aligned, labels_a6_aligned), "Labels incoherents apres alignement par cle !"
labels_true = labels_a2_aligned
print(f"[VERIF] {len(labels_true)} labels compares apres alignement, coherence OK")

for name, probs in [("Approche 2 seule", probs_a2_aligned), ("Approche 6 siamoise seule", probs_a6_aligned)]:
    preds = probs.argmax(axis=1)
    print(f"\n=== {name} (sur l'intersection, verification) ===")
    print(classification_report(labels_true, preds, target_names=CLASS_NAMES, zero_division=0))

probs_ensemble = (probs_a2_aligned + probs_a6_aligned) / 2.0
preds_ensemble = probs_ensemble.argmax(axis=1)

print("\n=======================================================")
print("=== ENSEMBLE Approche 2 + Approche 6 siamoise (moyenne) ===")
print("=======================================================")
print(classification_report(labels_true, preds_ensemble, target_names=CLASS_NAMES, zero_division=0))
print("Matrice de confusion (A/B/C/D):")
print(confusion_matrix(labels_true, preds_ensemble, labels=[0, 1, 2, 3]))
print("\nTermine.")
