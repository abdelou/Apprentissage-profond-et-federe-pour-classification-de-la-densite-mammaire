# Script : combine_ensemble.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]

probs_a2 = np.load("ensemble_probs_a2.npy")
labels_a2 = np.load("ensemble_labels_a2.npy")
probs_a3 = np.load("ensemble_probs_a3.npy")
labels_a3 = np.load("ensemble_labels_a3.npy")

assert probs_a2.shape[0] == probs_a3.shape[0], "Nombre d'images different entre les deux !"
assert np.array_equal(labels_a2, labels_a3), "Les labels ne correspondent pas -- desalignement entre les deux jeux !"
print(f"[VERIF] {len(labels_a2)} labels compares, alignement OK")

labels_true = labels_a2

for name, probs in [("Approche 3 seule", probs_a3), ("Approche 2 seule", probs_a2)]:
  preds = probs.argmax(axis=1)
  print(f"\n=== {name} (verification) ===")
  print(classification_report(labels_true, preds, target_names=CLASS_NAMES, zero_division=0))

probs_ensemble = (probs_a2 + probs_a3) / 2.0
preds_ensemble = probs_ensemble.argmax(axis=1)

print("\n=======================================================")
print("=== ENSEMBLE (moyenne simple des probabilites) ===")
print("=======================================================")
print(classification_report(labels_true, preds_ensemble, target_names=CLASS_NAMES, zero_division=0))
print("Matrice de confusion (A/B/C/D):")
print(confusion_matrix(labels_true, preds_ensemble, labels=[0, 1, 2, 3]))
print("\nTermine.")
