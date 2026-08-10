"""
Test demande par le prof (rdv d'il y a 4 semaines) : si le modele ne detecte
pas B, C ou D avec une bonne confiance, on pourrait dire automatiquement
que c'est la classe A (vu que A est tres minoritaire et souvent ratee).

Avant de mettre en place ce genre de regle dans le code, il faut d'abord
verifier ce que le modele repond VRAIMENT quand on lui donne une vraie image
de classe A. Est-ce qu'il hesite (aucune classe a plus de 40% par exemple),
ou est-ce qu'il se trompe carrement en etant sur de lui pour une autre
classe (ex: 80% de confiance sur D alors que c'est du A) ?

On prend donc uniquement les images du test set qui sont vraiment classe A,
on fait tourner le modele dessus, et pour chaque image on affiche les 4
probabilites (softmax) au lieu de juste la prediction finale.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hybrid_finetuning_glcm import HybridGLCMClassifier, HybridGLCMDataset, VINDR_ROOT, ANNOTATIONS_CSV

VIEW = "CC"
BACKBONE = "cnn"
WEIGHTS_PATH = f"featuresfinetuned_weights/hybrid_model_best_{VIEW.lower()}_{BACKBONE}_GLCM.pth"
CLASS_NAMES = ["A", "B", "C", "D"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# on charge le meme jeu de test que celui utilise pour l'evaluation finale
# du modele (split == 'test', vue CC uniquement)
df = pd.read_csv(ANNOTATIONS_CSV)
test_df = df[df["split"] == "test"].copy()

test_dataset = HybridGLCMDataset(test_df, VINDR_ROOT, view_position=VIEW, use_augmentation=False, split="test")

# on ne garde que les images dont le vrai label est A (label 0 dans CLASS_MAP)
indices_classe_a = [i for i in range(len(test_dataset)) if test_dataset.df.iloc[i]["label"] == 0]
print(f"[INFO] {len(indices_classe_a)} images de classe A trouvees dans le test set ({VIEW})")

model = HybridGLCMClassifier(backbone=BACKBONE, num_classes=4, dropout=0.3)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

# compteurs pour le resume final
nb_confiant_faux = 0       # le modele est sur de lui (>40%) mais se trompe
nb_incertain = 0           # aucune classe B/C/D ne depasse 40%
nb_correct = 0             # le modele trouve A directement

print("\n" + "=" * 70)
print(f"Detail des {len(indices_classe_a)} images de classe A (probabilites par classe)")
print("=" * 70)

with torch.no_grad():
    for idx in indices_classe_a:
        image, label = test_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

        # on regarde seulement les probas B, C, D (indices 1,2,3) pour voir
        # si le modele est "confiant" sur une classe qui n'est pas A
        max_bcd = max(probs[1], probs[2], probs[3])

        ligne = " | ".join(f"{CLASS_NAMES[c]}: {probs[c]*100:5.1f}%" for c in range(4))
        marqueur = ""
        if pred == 0:
            nb_correct += 1
            marqueur = "  <-- correct"
        elif max_bcd >= 0.40:
            nb_confiant_faux += 1
            marqueur = f"  <-- confiant mais FAUX (predit {CLASS_NAMES[pred]})"
        else:
            nb_incertain += 1
            marqueur = "  <-- incertain (aucune classe B/C/D > 40%)"

        print(f"image {idx:4d} | {ligne}{marqueur}")

print("\n" + "=" * 70)
print("RESUME")
print("=" * 70)
print(f"Correctement predit A directement       : {nb_correct}/{len(indices_classe_a)}")
print(f"Confiant (>40%) mais faux sur B/C/D     : {nb_confiant_faux}/{len(indices_classe_a)}")
print(f"Incertain (aucune classe B/C/D > 40%)   : {nb_incertain}/{len(indices_classe_a)}")
print("\nSi 'incertain' est majoritaire, la regle du prof (si pas B/C/D avec")
print("confiance alors A) a de bonnes chances de bien fonctionner. Si c'est")
print("'confiant mais faux' qui domine, la regle ne suffira pas telle quelle.")
