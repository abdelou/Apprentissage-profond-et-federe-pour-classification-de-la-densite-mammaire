# Classification de la densité mammaire — Apprentissage profond et fédéré

Mémoire de Master (UMONS, Faculté sciences ) : *Apprentissage profond et
fédéré pour la classification de la densité mammaire pour la prédiction du
cancer du sein*.

Le projet classe les mammographies DICOM selon 4 classes de densité BI-RADS
(DENSITY A/B/C/D), entraîné sur VinDr-Mammo et évalué sur DDSM/CBIS-DDSM. Huit
approches de classification sont comparées, puis la meilleure architecture
compatible est reprise dans une simulation d'apprentissage fédéré (FedAvg)
entre deux clients.

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
├── three_stage_pipeline.py        # Approche 7 : cascade (A,B)/(C,D) -> A/B -> C/D
├── hybrid_model.py / hybrid_finetuning.py   # Modèle hybride image + branche statistique
├── eval_finetuned_report.py       # Réévaluation d'un backbone déjà fine-tuné (filtrage vue/split)
├── interpretability_analysis.py   # Grad-CAM (grille 3 ex./classe) + t-SNE (test seul, train/test)
├── dump_probs_approche2*.py, combine_ensemble*.py   # Approche 8 : fusion tardive
├── federated_learning/
│   ├── federated_training.py      # FedAvg (agrégation, resume, min_weight, poids pré-entraînés)
│   ├── fed_model.py                # ResNet50DensityClassifier (1 canal, une branche)
│   └── fed_datasets.py             # Clients H1 (VinDr, CC) / H2 (DDSM/CBIS-DDSM)
├── DDSM/
│   ├── prepare_cbis_ddsm_annotations.py   # Résolution des annotations CBIS-DDSM (image_path, densité)
│   └── filtrage_images.py, analyze_study_consistency.py
└── variantes de l'Approche 6 (double branche), chacune avec sa propre copie
    hybrid_model.py / hybrid_finetuning.py / test_hybrid_model.py :
    ├── hybrid 2 branches images/                 # RexNet150/CC + ResNet50/MLO (originale)
    ├── hybrid 2 branches siamese efficientnet/    # EfficientNet-B0, poids partagés CC+MLO
    ├── hybrid 2 backbones mlo rexnet resnet/      # RexNet150+ResNet50, fusion même-vue MLO
    ├── hybrid 2 backbones cc densenet resnet/     # DenseNet121+ResNet50, fusion même-vue CC
    ├── hybrid train mlo resnet/                   # Approche 3 : ResNet50+Histogramme, MLO
    ├── hybrid train mlo rexnet/                   # RexNet150+Histogramme, MLO (Focal Loss)
    └── hybrid_train_cc_rexnet/                    # Approche 4 : RexNet150+Histogramme, CC
```

Ce ne sont pas des doublons accidentels : chaque dossier de variante est une
copie indépendante, réglée pour un couple backbone/vue précis. Un bug corrigé
dans l'une doit être vérifié dans les autres.

## Les 8 approches

| Approche | Architecture | Test Acc | Remarque |
|---|---|---|---|
| 1 | ViT/Twins (feature-based) + arbitrage binaire ciblé | 76.66\% (MLP4 direct) / 60.57\% (hiérarchique) | classe A jamais prédite en direct (rappel 0\%) |
| 2 | Fine-tuning direct ResNet50 (ou ViT) | 74.00\% (4000 img) | seule architecture qui détecte bien la classe A (rappel 0.95) |
| 3 | ResNet50 + Histogramme, vues MLO | **85.00\%** | meilleur score du mémoire, mais rappel classe A = 0\% |
| 4 | RexNet150 + Histogramme, vues CC | 80.00\% | |
| 5 | RexNet150/ResNet50 + GLCM (CC/MLO) | 74.85\% / 75.95\% | Modèle 3 (Focal Loss) : 84.00\%, classe A toujours à 0\% |
| 6 | Double branche CC+MLO (4 variantes) | 83.00\% à **85.00\%** (siamoise, meilleure variante) | |
| 7 | Cascade hiérarchique (A,B)/(C,D) puis A/B, C/D | 42.53\% (ViT) / 48.50\% (ResNet50) | erreurs de routage à l'étape 1 non récupérables |
| 8 | Ensemble par fusion tardive (Approche 2 + 3) | 80.55\% | meilleur macro F1 (0.62), rappel classe A = 0.30 |

Aucune architecture ne domine sur tous les critères à la fois : accuracy
globale, détection de la classe A, et équilibre entre classes pointent vers
des modèles différents (voir la synthèse comparative dans le rapport complet).

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
ex. l'Approche 2, plutôt que depuis ImageNet).

Résultats de test (2000 images VinDr CC, mêmes images pour toutes les lignes) :

| Configuration | VinDr Test Acc | DDSM Test Acc |
|---|---|---|
| Approche 2 seule (sans fédération) | 75.40\% | -- |
| Baseline centralisée (poolée, non fédérée) | 82.30\% | 31.17\% |
| FedAvg, $W_0$ = ImageNet | 81.60\% | 38.31\% |
| FedAvg, $W_0$ = Approche 2 pré-entraînée | 81.80\% | 32.47\% |

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

ALLA Abdelouahad — UMONS, Faculté Sciences.
Encadrement : Xavier Lessage (CETIC), Prof. Saïd Mahmoudi (UMONS).
