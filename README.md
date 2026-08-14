# Classification de la densité mammaire — Apprentissage profond et fédéré

Mémoire de Master (UMONS, Faculté des Sciences, Département d'Informatique IA) : *Apprentissage profond et
fédéré pour la classification de la densité mammaire pour la prédiction du
cancer du sein*.

Le projet classe les mammographies DICOM selon 4 classes de densité BI-RADS
(DENSITY A/B/C/D), entraîné sur VinDr-Mammo et évalué sur DDSM/CBIS-DDSM. Huit
approches de classification sont comparées, puis la meilleure architecture
compatible est reprise dans une simulation d'apprentissage fédéré (FedAvg)
entre deux centres (clients).

## Environnement

```bash
# venv Python 3.9 à la racine du dépôt (../.venv depuis Codes/)
pip install -r requirements.txt
```

Dépendances principales : PyTorch/torchvision, timm, Transformers (Hugging
Face), Pydicom, OpenCV, Scikit-image, Scikit-learn, Matplotlib/Seaborn.
Entraînements lancés sur le cluster GPU de l'UMONS (SLURM ou `nohup` direct),
jamais en local.

## Structure du projet

```
Codes/
├── training.py                    # CLI unifiée (augment, finetune, dump_features, train_on_features...)
├── models.py                      # Backbones, MLP, HierarchicalClassifier, FeatureDumper
├── preprocessing.py               # Lecture DICOM, crop/resize/normalisation
├── three_stage_pipeline.py        # Approche 2 : cascade (A,B)/(C,D) -> A/B -> C/D (features gelées)
├── cascade_finetuned_pipeline.py  # Approche 2 : même cascade, backbones fine-tunés de bout en bout
├── hybrid_model.py / hybrid_finetuning.py   # Modèle hybride image + branche statistique
├── eval_finetuned_report.py       # Réévaluation d'un backbone déjà fine-tuné (filtrage vue/split)
├── interpretability_analysis.py   # Grad-CAM (grille 3 ex./classe) + t-SNE (test seul, train/test)
├── dump_probs_approche2*.py, combine_ensemble*.py   # Approche 8 : fusion tardive
├── federated_learning/
│   ├── federated_training.py      # FedAvg (agrégation, resume, min_weight, poids pré-entraînés)
│   ├── fed_model.py               # ResNet50DensityClassifier (1 canal, une branche)
│   └── fed_datasets.py            # Clients H1 (VinDr, CC) / H2 (DDSM/CBIS-DDSM)
├── DDSM/
│   ├── prepare_cbis_ddsm_annotations.py   # Résolution des annotations CBIS-DDSM (image_path, densité)
│   └── filtrage_images.py, analyze_study_consistency.py
└── variantes de l'Approche 7 (double branche), chacune avec sa propre copie
    hybrid_model.py / hybrid_finetuning.py / test_hybrid_model.py :
    ├── hybrid 2 branches images/                 # RexNet150/CC + ResNet50/MLO (originale)
    ├── hybrid 2 branches siamese efficientnet/    # EfficientNet-B0, poids partagés CC+MLO
    ├── hybrid 2 backbones mlo rexnet resnet/      # RexNet150+ResNet50, fusion même-vue MLO
    ├── hybrid 2 backbones cc densenet resnet/     # DenseNet121+ResNet50, fusion même-vue CC
    ├── hybrid train mlo resnet/                   # Approche 4 : ResNet50+Histogramme, MLO
    ├── hybrid train mlo rexnet/                    # RexNet150+Histogramme, MLO
    └── hybrid_train_cc_rexnet/                     # Approche 5 : RexNet150+Histogramme, CC
```

Ce ne sont pas des doublons accidentels : chaque dossier de variante est une
copie indépendante, réglée pour un couple backbone/vue précis. Un bug corrigé
dans l'une doit être vérifié dans les autres.

## Les 8 approches comparées

1. Architecture unimodale basée sur les Vision Transformers
2. Classification hiérarchique en cascade (AB vs CD → A vs B & C vs D)
3. Fine-tuning de backbones (ViT et ResNet50), classification directe
4. ResNet50 + Histogramme (vues MLO)
5. RexNet150 + Histogramme (vues CC)
6. Intégration d'une branche GLCM (analyse texturale)
7. Architecture à double branche (CC + MLO, 4 variantes)
8. Ensemble par fusion tardive (Approche 3 + Approche 4)

Détail des architectures, résultats chiffrés et analyse comparative dans le
rapport complet du mémoire.

## Apprentissage fédéré (FedAvg)

Simulation sur une seule machine du cluster (pas de déploiement réellement
distribué), deux clients :

- **H1** = VinDr-Mammo, vues CC, 6801 images d'entraînement.
- **H2** = DDSM/CBIS-DDSM, 531 images, substitut technique au réseau
  hospitalier privé HELORA (cible réelle du sujet, inaccessible pour cette
  réplication).

Modèle fédéré : ResNet50 une seule branche (seule architecture directement
réutilisable sans modification parmi les 8 approches, `fed_model.py`).

```bash
cd federated_learning
python3 federated_training.py --mode federated --rounds 15 --local_epochs 1
python3 federated_training.py --mode centralized --epochs 5   # baseline non fédérée
```

Options ajoutées pendant ce mémoire : `--resume` (reprise après interruption),
`--min_weight` (poids plancher pour le client minoritaire dans l'agrégation),
`--pretrained_checkpoint` (initialiser $W_0$ depuis un backbone déjà fine-tuné,
ex. l'Approche 3, plutôt que depuis ImageNet).

## Interprétabilité

`interpretability_analysis.py` produit, par approche : une grille Grad-CAM
(3 exemples par classe, original + heatmap) et une projection t-SNE de
l'espace latent (test seul, puis train vs. test).

```bash
python3 run_interpretability_approche2.py
cd "hybrid train mlo resnet" && python3 run_interpretability.py
```

## Commandes principales

```bash
# Pipeline par extraction de caractéristiques (Approche 1 / training.py)
python training.py --mode augment --target_samples 2000
python training.py --mode finetune --backbone cvt-w24 --epochs 100 --batch_size 10 --use_augmentation
python training.py --mode dump_features --backbone cvt-w24 --use_finetuned
python training.py --mode train_on_features --backbone cvt-w24 --train_mlp4 --all_binaries --use_finetuned

# Cascade fine-tunée de bout en bout (Approche 2)
python cascade_finetuned_pipeline.py --mode train --stage 1 --backbone resnet50 --epochs 10
python cascade_finetuned_pipeline.py --mode evaluate --backbone resnet50

# Modèle hybride à deux branches (depuis un dossier de variante)
python hybrid_finetuning.py --epochs 50 --batch_size 8 --lr 1e-4 --use_augmentation
python test_hybrid_model.py

# Réévaluation filtrée (ex. comparaison à jeu de test identique au fédéré)
python eval_finetuned_report.py resnet50 featuresfinetuned_weights/resnet50_finetuned_best.pth --split test --view CC
```

Voir `UNIFIED_TRAINING_GUIDE.md` pour le détail des modes de `training.py` et
les hyperparamètres recommandés par backbone.

## Classes de densité

| Index | Classe | Description |
|---|---|---|
| 0 | DENSITY A | Tissu principalement adipeux |
| 1 | DENSITY B | Densités fibroglandulaires éparses |
| 2 | DENSITY C | Tissu hétérogène dense |
| 3 | DENSITY D | Tissu extrêmement dense |

## Auteur

ALLA Abdelouahad — UMONS, Faculté des Sciences, Département d'Informatique IA.
