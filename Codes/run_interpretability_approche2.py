"""
Lance l'analyse d'interpretabilite (t-SNE + Grad-CAM) sur l'Approche 2
(ResNet50, fine-tuning direct, classification a 4 sorties sans branche
auxiliaire). Prepare en parallele de l'analyse deja faite sur l'Approche 3,
en attendant la reponse du prof sur quel modele garder comme resultat final
du memoire -- si Approche 2 est choisie, ce chapitre est deja pret.

Contrairement a l'Approche 3, ce modele prend une seule entree (l'image),
donc pas besoin de wrapper : il presente deja l'interface generique
attendue par interpretability_analysis.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch

from models import FineTunedFeatureExtractor
from training import MammographyDataset, CONFIG
from interpretability_analysis import run_gradcam_grid, run_tsne, run_tsne_train_test, get_module_by_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "featuresfinetuned_weights/resnet50_finetuned_best.pth"
    out_dir = "graphes_interpretabilite"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Chargement du modele depuis {weights_path}")
    model = FineTunedFeatureExtractor("resnet", num_classes=4, pretrained=False)
    model.load_finetuned_weights(weights_path, device)
    model.to(device)
    model.eval()

    target_layer = get_module_by_name(model, "model.layer4")

    df = pd.read_csv(CONFIG["ANNOTATIONS_CSV"])
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_train = df[df["split"] == "training"].reset_index(drop=True)

    test_dataset = MammographyDataset(
        df_test, CONFIG["IMAGE_ROOT"], split="test",
        use_augmentation=False, label_map=CONFIG["CLASS_MAP"],
    )
    train_dataset = MammographyDataset(
        df_train, CONFIG["IMAGE_ROOT"], split="training",
        use_augmentation=False, label_map=CONFIG["CLASS_MAP"],
    )
    print(f"[INFO] {len(test_dataset)} images de test, {len(train_dataset)} images de train chargees")

    print("\n[1/2] Generation de la grille Grad-CAM (4 classes x 3 exemples)...")
    run_gradcam_grid(
        model, target_layer, test_dataset, device,
        out_path=f"{out_dir}/gradcam_approche2_resnet50.png",
        n_per_class=3,
    )

    print("\n[2/3] Generation de la projection t-SNE (test seul)...")
    run_tsne(
        model, target_layer, test_dataset, device,
        out_path=f"{out_dir}/tsne_approche2_resnet50.png",
        n_samples=300,
        backbone_name="ResNet50 (fine-tuning direct)",
    )

    print("\n[3/3] Generation de la projection t-SNE train vs test...")
    run_tsne_train_test(
        model, target_layer, train_dataset, test_dataset, device,
        out_path=f"{out_dir}/tsne_approche2_resnet50_traintest.png",
        n_samples_train=300, n_samples_test=300,
        backbone_name="ResNet50 (fine-tuning direct)",
    )

    print("\nTermine.")


if __name__ == "__main__":
    main()
