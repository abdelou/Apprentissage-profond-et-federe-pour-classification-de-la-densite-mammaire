"""
Script d'analyse d'interpretabilite.

Ici on regroupe les deux techniques demandees par le prof pour "ouvrir la
boite noire" du modele :
    1) t-SNE : on prend les features juste avant la couche de classification
       (donc un vecteur de plusieurs centaines de dimensions par image) et on
       les projette en 2D pour voir si les 4 classes de densite sont bien
       separees dans l'espace appris par le reseau.
    2) Grad-CAM : pour une image donnee, on regarde quelles zones de l'image
       ont le plus fait "reagir" le modele pour sa prediction. Ca genere une
       carte de chaleur qu'on superpose a la mammographie.

Ce script est volontairement generique (on ne code pas en dur le nom du
modele) parce qu'au moment ou j'ecris ca, plusieurs entrainements tournent
encore sur le cluster et je ne sais pas encore lequel sera le meilleur.
Des que j'ai le modele final, il suffit de remplir les arguments en ligne de
commande (voir tout en bas, section if __name__ == "__main__") avec le bon
chemin de checkpoint et le bon nom de couche.

Pour trouver le nom de la derniere couche convolutive d'un modele, le plus
simple est de faire un print(model) dans un shell python et de regarder la
derniere couche Conv2d avant le pooling/la couche lineaire finale.
"""
import argparse
import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif sur le cluster (pas d'ecran)
import matplotlib.pyplot as plt

# Les 4 classes de densite mammaire du projet, dans l'ordre utilise partout
# ailleurs dans le code (CLASS_MAP = {"DENSITY A": 0, ... "DENSITY D": 3})
CLASS_NAMES = ["Density A", "Density B", "Density C", "Density D"]
# Couleurs choisies a la main pour que ce soit lisible et pas les couleurs
# par defaut de matplotlib qui se ressemblent trop entre elles
CLASS_COLORS = ["#1c5cab", "#e07b28", "#3a9e4a", "#c0392b"]


# ==========================================================================
# PARTIE 1 : GRAD-CAM
# ==========================================================================
# Grad-CAM a ete propose par Selvaraju et al. (2017), voir la formule dans le
# rapport (section Grad-CAM). En resume :
#   - on recupere les activations A^k de la derniere couche convolutive
#     (une carte par filtre k)
#   - on calcule le gradient du score de la classe predite par rapport a ces
#     activations, et on fait une moyenne spatiale (Global Average Pooling)
#     de ce gradient -> ca donne un poids alpha_k pour chaque filtre k
#   - on fait la somme ponderee des cartes d'activation par ces poids, puis
#     un ReLU pour ne garder que ce qui "pousse" vers la classe (pas ce qui
#     la freine)
#   - enfin on remet la carte a la taille de l'image d'origine
#
# Pour recuperer les activations ET les gradients pendant le forward/backward
# pass normal du modele, on utilise les "hooks" de pytorch. C'est juste une
# fonction que pytorch appelle automatiquement au bon moment, sans qu'on ait
# besoin de modifier le code du modele lui-meme.
class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None  # sera rempli par le hook forward
        self.gradients = None    # sera rempli par le hook backward

        # on "accroche" nos deux fonctions sur la couche cible
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # appelee automatiquement quand la couche cible fait son forward
        # .detach() pour ne pas garder le graphe de calcul inutilement
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # appelee automatiquement pendant le backward() ; grad_output[0]
        # correspond au gradient de la sortie de la couche
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, target_class=None):
        # on remet les gradients a zero avant chaque nouvelle image, sinon
        # pytorch les accumule d'un appel a l'autre (comportement par defaut)
        self.model.zero_grad()

        output = self.model(image_tensor)

        # si on ne precise pas de classe cible, on prend celle predite par
        # le modele (utile quand on veut juste "expliquer" la decision)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # on ne backward que sur le score de LA classe qui nous interesse,
        # pas sur toute la sortie -> c'est ce qui rend Grad-CAM specifique
        # a une classe donnee
        score = output[0, target_class]
        score.backward()

        # alpha_k^c = moyenne spatiale du gradient (equation du rapport)
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)

        # combinaison ponderee des cartes d'activation + ReLU
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # la carte issue de la derniere conv est petite (genre 7x7), on la
        # remet a la taille de l'image d'entree pour pouvoir la superposer
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)

        # on met les valeurs entre 0 et 1 pour l'affichage (sinon le colormap
        # jet ne veut rien dire)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        return cam, target_class, probs


def get_module_by_name(model, dotted_name):
    """Petit utilitaire pour recuperer un sous-module a partir d'un nom du
    style 'cnn.layer4' (utile parce que le nom exact de la derniere couche
    conv change d'un modele a l'autre)."""
    module = model
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def load_model(model_module, model_class, weights_path, device, **model_kwargs):
    # importlib permet de charger dynamiquement le bon fichier .py selon
    # l'argument --model_module passe en ligne de commande, plutot que de
    # devoir faire un import different pour chaque modele possible
    mod = importlib.import_module(model_module)
    cls = getattr(mod, model_class)
    model = cls(**model_kwargs)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # tres important, sinon dropout/batchnorm faussent tout
    return model


def run_gradcam_grid(model, target_layer, dataset, device, out_path, n_per_class=2):
    """Genere la figure principale demandee par le prof : format classique
    (colonnes = les 4 classes, comme dans la version d'origine), mais avec
    4 lignes au lieu de 2, et DEUX images DIFFERENTES par classe (pas la
    meme repetee) pour montrer les deux facons de calculer Grad-CAM :
      ligne 1 : originale, exemple 1
      ligne 2 : Grad-CAM de l'exemple 1 sur sa VRAIE classe -- ou se trouve
                la preuve de la bonne reponse, meme si le modele se trompe.
      ligne 3 : originale, exemple 2 (image differente de l'exemple 1)
      ligne 4 : Grad-CAM de l'exemple 2 sur la classe REELLEMENT PREDITE
                (argmax du modele) -- ce qui a vraiment fait pencher le
                modele vers sa reponse, correcte ou non.
    4 classes x 4 lignes = 16 images. Une seule colorbar verticale, partagee
    par toute la figure."""
    gradcam = GradCAM(model, target_layer)

    # deux images reelles et differentes par classe : la premiere pour le
    # calcul "vraie classe", la seconde pour le calcul "classe predite"
    by_class = {0: [], 1: [], 2: [], 3: []}
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        label = int(label)
        if len(by_class[label]) < 2:
            by_class[label].append(image)
        if all(len(v) >= 2 for v in by_class.values()):
            break

    fig, axes = plt.subplots(4, 4, figsize=(4.2 * 4, 5.0 * 4))

    def _to_display(image):
        img_np = image.cpu().numpy().squeeze()
        # certains backbones (Approche 2) recoivent une image repliquee
        # sur 3 canaux (C,H,W) -- comme c'est un niveau de gris duplique,
        # un seul canal suffit pour l'affichage
        if img_np.ndim == 3:
            img_np = img_np[0]
        return (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    last_im = None
    for c in range(4):
        image1, image2 = by_class[c][0], by_class[c][1]

        # 1) exemple 1, carte sur la vraie classe (forcee)
        cam_true, _, probs_true = gradcam(image1.unsqueeze(0).to(device), target_class=c)
        # 2) exemple 2, carte sur la classe que le modele predit reellement (argmax)
        cam_pred, pred_class, probs_pred = gradcam(image2.unsqueeze(0).to(device), target_class=None)

        img1_np = _to_display(image1)
        img2_np = _to_display(image2)

        ax_orig1 = axes[0, c]
        ax_orig1.imshow(img1_np, cmap="gray")
        ax_orig1.set_title(f"{CLASS_NAMES[c]} -- ex. 1 : Originale", fontsize=10)
        ax_orig1.axis("off")

        ax_true = axes[1, c]
        ax_true.imshow(img1_np, cmap="gray")
        last_im = ax_true.imshow(cam_true, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax_true.set_title(
            f"Grad-CAM (vraie classe, {probs_true[c]*100:.1f}%)", fontsize=10,
        )
        ax_true.axis("off")

        ax_orig2 = axes[2, c]
        ax_orig2.imshow(img2_np, cmap="gray")
        ax_orig2.set_title(f"{CLASS_NAMES[c]} -- ex. 2 : Originale", fontsize=10)
        ax_orig2.axis("off")

        ax_pred = axes[3, c]
        ax_pred.imshow(img2_np, cmap="gray")
        ax_pred.imshow(cam_pred, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        correct = " = vraie classe" if pred_class == c else " -- erreur"
        ax_pred.set_title(
            f"Grad-CAM (predit {CLASS_NAMES[pred_class]}, {probs_pred[pred_class]*100:.1f}%{correct})",
            fontsize=10,
        )
        ax_pred.axis("off")

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    # une seule colorbar verticale a droite de toute la grille, plutot que
    # d'en avoir une par sous-image (illisible) ou une horizontale en bas
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Intensite d'activation Grad-CAM", fontsize=10)

    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Figure Grad-CAM enregistree :", out_path)


# ==========================================================================
# PARTIE 2 : t-SNE
# ==========================================================================
def run_tsne(model, feature_layer, dataset, device, out_path, n_samples=300, backbone_name=""):
    # sklearn est importe seulement ici pour eviter de le charger si on
    # n'utilise que la partie Grad-CAM du script
    from sklearn.manifold import TSNE

    features_list, labels_list = [], []
    activations = {}

    # meme principe de hook que pour Grad-CAM, mais cette fois on veut juste
    # recuperer la sortie de la couche, pas de backward necessaire
    def hook(module, input, output):
        activations["feat"] = output.detach()

    handle = feature_layer.register_forward_hook(hook)

    # on ne prend pas TOUT le jeu de test (ca serait trop lent et le nuage
    # de points deviendrait illisible), juste un echantillon aleatoire.
    # random_state=42 pour pouvoir reproduire exactement le meme graphe
    # si on relance le script plus tard.
    #
    # ATTENTION : un tirage purement aleatoire peut manquer entierement la
    # classe A (environ 10 images sur 2000 en test, ~0.5%) -- avec
    # n_samples=300, la probabilite de n'en tirer aucune est d'environ 22%,
    # ce qui s'est reellement produit une fois sur ce projet. Comme la
    # classe A est justement celle qui interesse le plus ce memoire, on
    # force son inclusion : on prend TOUTES ses images, puis on complete le
    # reste du quota au hasard parmi les autres classes.
    rng = np.random.RandomState(42)
    if hasattr(dataset, "df") and "label" in dataset.df.columns:
        all_labels = dataset.df["label"].values
        forced_idx = np.where(all_labels == 0)[0]  # classe A garantie
        remaining_idx = np.where(all_labels != 0)[0]
        n_fill = max(0, min(n_samples, len(dataset)) - len(forced_idx))
        filled_idx = rng.choice(remaining_idx, size=min(n_fill, len(remaining_idx)), replace=False)
        indices = np.concatenate([forced_idx, filled_idx])
    else:
        indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():  # pas besoin de gradient ici, juste de l'inference
        for idx in indices:
            image, label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            model(image_tensor)
            feat = activations["feat"]
            # si la couche cible sort encore une carte spatiale (C,H,W) et
            # pas un vecteur, on fait un average pooling pour obtenir un
            # vecteur de taille fixe par image
            if feat.dim() > 2:
                feat = feat.mean(dim=(2, 3))
            features_list.append(feat.cpu().numpy().squeeze())
            labels_list.append(int(label))

    handle.remove()  # on enleve le hook, sinon il reste actif pour rien

    features = np.stack(features_list)
    labels = np.array(labels_list)

    print(f"[INFO] {features.shape[0]} images, {features.shape[1]} dimensions par image -> reduction t-SNE en 2D")

    # perplexity doit etre plus petit que le nombre d'echantillons, sinon
    # sklearn plante ; 30 est la valeur par defaut recommandee dans la doc
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    proj = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    for c in range(4):
        mask = labels == c
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=CLASS_COLORS[c], label=f"ACR{c+1} ({CLASS_NAMES[c][-1]})",
            s=40, alpha=0.8, edgecolors="white", linewidths=0.5,
        )
    ax.set_xlabel("Composante t-SNE 1")
    ax.set_ylabel("Composante t-SNE 2")
    ax.set_title(f"Visualisation t-SNE de l'espace latent ({backbone_name})", fontsize=13, fontweight="bold")
    ax.legend(title="Classe reelle (BI-RADS)", loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print("Figure t-SNE enregistree :", out_path)


def run_tsne_train_test(model, feature_layer, train_dataset, test_dataset, device, out_path,
                         n_samples_train=200, n_samples_test=200, backbone_name=""):
    """Variante de run_tsne() qui projette train ET test dans le meme espace
    2D (le t-SNE est bien ajuste sur les deux ensembles combines, sinon les
    deux projections ne seraient pas comparables). Utile pour verifier
    visuellement si le train et le test occupent les memes regions de
    l'espace latent (signe de bonne generalisation) ou si le test forme des
    zones separees (signe de distribution shift / possible surapprentissage)."""
    from sklearn.manifold import TSNE

    features_list, labels_list, split_list = [], [], []
    activations = {}

    def hook(module, input, output):
        activations["feat"] = output.detach()

    handle = feature_layer.register_forward_hook(hook)

    def extract(dataset, n_samples, split_name):
        # meme correction que dans run_tsne() : on force l'inclusion de la
        # classe A, sinon un tirage purement aleatoire peut la manquer
        # entierement (~22% de chances sur ce projet, deja arrive une fois)
        rng = np.random.RandomState(42)
        if hasattr(dataset, "df") and "label" in dataset.df.columns:
            all_labels = dataset.df["label"].values
            forced_idx = np.where(all_labels == 0)[0]
            remaining_idx = np.where(all_labels != 0)[0]
            n_fill = max(0, min(n_samples, len(dataset)) - len(forced_idx))
            filled_idx = rng.choice(remaining_idx, size=min(n_fill, len(remaining_idx)), replace=False)
            indices = np.concatenate([forced_idx, filled_idx])
        else:
            indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
        with torch.no_grad():
            for idx in indices:
                image, label = dataset[idx]
                image_tensor = image.unsqueeze(0).to(device)
                model(image_tensor)
                feat = activations["feat"]
                if feat.dim() > 2:
                    feat = feat.mean(dim=(2, 3))
                features_list.append(feat.cpu().numpy().squeeze())
                labels_list.append(int(label))
                split_list.append(split_name)

    extract(train_dataset, n_samples_train, "train")
    extract(test_dataset, n_samples_test, "test")
    handle.remove()

    features = np.stack(features_list)
    labels = np.array(labels_list)
    splits = np.array(split_list)

    print(f"[INFO] {features.shape[0]} images ({(splits=='train').sum()} train + "
          f"{(splits=='test').sum()} test), {features.shape[1]} dimensions -> reduction t-SNE en 2D")

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    proj = tsne.fit_transform(features)

    # points ronds uniformes pour les deux splits (comme la version test
    # seul), on ne distingue train/test que par la transparence -- pas par
    # la forme du marqueur, pour garder le meme rendu visuel que l'ancienne
    # figure
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    alphas = {"train": 0.35, "test": 0.85}
    for c in range(4):
        for split_name in ("train", "test"):
            mask = (labels == c) & (splits == split_name)
            if mask.sum() == 0:
                continue
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=CLASS_COLORS[c], marker="o",
                s=45, alpha=alphas[split_name], edgecolors="white", linewidths=0.5,
            )

    # deux legendes separees : une pour la classe (couleur), une pour le
    # split train/test (transparence) -- sinon impossible de combiner
    # proprement les deux dans une seule legende
    from matplotlib.lines import Line2D
    class_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c],
               markersize=9, label=f"ACR{c+1} ({CLASS_NAMES[c][-1]})")
        for c in range(4)
    ]
    split_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=9,
               alpha=alphas["train"], label="Train"),
        Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=9,
               alpha=alphas["test"], label="Test"),
    ]
    legend1 = ax.legend(handles=class_handles, title="Classe reelle (BI-RADS)", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=split_handles, title="Split", loc="upper right")

    ax.set_xlabel("Composante t-SNE 1")
    ax.set_ylabel("Composante t-SNE 2")
    ax.set_title(f"Visualisation t-SNE, train vs test ({backbone_name})", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print("Figure t-SNE train/test enregistree :", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_module", required=True, help="module python contenant la classe du modele, ex: models ou fed_model")
    parser.add_argument("--model_class", required=True, help="nom de la classe, ex: FineTunedFeatureExtractor")
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--target_layer", required=True, help="nom pointe de la derniere couche conv, ex: cnn.layer4")
    parser.add_argument("--feature_layer", default=None, help="couche a utiliser pour le t-SNE (par defaut = target_layer)")
    parser.add_argument("--backbone_name", default="")
    parser.add_argument("--n_tsne_samples", type=int, default=300)
    parser.add_argument("--out_dir", default="graphes_interpretabilite")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO (a faire des qu'on a choisi le modele final) :
    #   1) charger le modele avec load_model(...) et les bons model_kwargs
    #   2) construire le dataset de test correspondant (MammographyDataset,
    #      HybridGLCMDataset, etc. selon le modele choisi)
    #   3) appeler run_gradcam_grid(...) et run_tsne(...)
    # Je laisse volontairement une erreur explicite ici plutot que du code
    # a moitie ecrit qui planterait sans dire pourquoi.
    raise NotImplementedError(
        "A completer une fois le modele final choisi : chargement du "
        "modele + du dataset de test, puis appel de run_gradcam_grid() et "
        "run_tsne()."
    )
