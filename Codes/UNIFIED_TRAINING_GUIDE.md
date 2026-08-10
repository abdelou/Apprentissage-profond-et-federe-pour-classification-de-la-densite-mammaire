# 🚀 Guide d'utilisation - Système unifié d'entraînement

## 📋 Vue d'ensemble

Le fichier `training.py` est maintenant un système unifié qui intègre toutes les fonctionnalités :
- **Extraction de features** (dump_features)
- **Entraînement sur features** (train_on_features)
- **Entraînement complet** (train)
- **Fine-tuning de Transformers** (finetune)
- **Data augmentation** (augment)
- **Fine-tuning avec dataset augmenté** (finetune_augmented)

## 🔧 Modes disponibles

### 1. **`augment`** - Data Augmentation
Crée un dataset augmenté pour équilibrer les classes.

```bash
# Créer un dataset augmenté avec 2000 échantillons par classe
python training.py --mode augment --target_samples 2000

# Avec paramètres personnalisés
python training.py \
    --mode augment \
    --target_samples 2500 \
    --image_root /path/to/images \
    --annotations_csv DDSM/output_annotations.csv
```

**Résultat :** Crée `DDSM/output_annotations_augmented.csv`

### 2. **`finetune`** - Fine-tuning Transformer
Fine-tune un Transformer avec augmentation en temps réel.

```bash
# Fine-tuning avec augmentation
python training.py \
    --mode finetune \
    --backbone cvt-w24 \
    --epochs 100 \
    --batch_size 10 \
    --learning_rate 1e-4 \
    --use_augmentation \
    --patience 5

# Fine-tuning sans augmentation
python training.py \
    --mode finetune \
    --backbone twins \
    --epochs 80 \
    --batch_size 15 \
    --learning_rate 1e-4
```

### 3. **`finetune_augmented`** - Fine-tuning avec Dataset Augmenté
Utilise le dataset pré-augmenté pour le fine-tuning.

```bash
# Fine-tuning avec dataset augmenté
python training.py \
    --mode finetune_augmented \
    --backbone cvt-w24 \
    --epochs 100 \
    --batch_size 10 \
    --learning_rate 1e-4 \
    --patience 5
```

### 4. **`dump_features`** - Extraction de Features
Extrait les features des images pour l'entraînement des MLPs.

```bash
# Extraction avec features fine-tunées
python training.py \
    --mode dump_features \
    --backbone cvt-w24 \
    --use_finetuned

# Extraction avec features pré-entraînées
python training.py \
    --mode dump_features \
    --backbone twins \
    --no_finetuned
```

### 5. **`train_on_features`** - Entraînement sur Features
Entraîne les MLPs sur les features extraites.

```bash
# Entraînement du MLP 4 classes
python training.py \
    --mode train_on_features \
    --backbone cvt-w24 \
    --train_mlp4 \
    --use_finetuned

# Entraînement de tous les binaires
python training.py \
    --mode train_on_features \
    --backbone twins \
    --all_binaries \
    --use_finetuned

# Entraînement d'un binaire spécifique
python training.py \
    --mode train_on_features \
    --backbone cvt-w24 \
    --binary_pair 0 1 \
    --use_finetuned
```

### 6. **`train`** - Entraînement Complet
Entraîne tous les modèles (binaires + MLP 4 classes).

```bash
python training.py \
    --mode train \
    --backbone twins \
    --epochs 30 \
    --batch_size 20
```

## 🎯 Workflows recommandés

### **Workflow 1 : Fine-tuning avec augmentation**
```bash
# 1. Créer le dataset augmenté
python training.py --mode augment --target_samples 2000

# 2. Fine-tuning avec dataset augmenté
python training.py \
    --mode finetune_augmented \
    --backbone cvt-w24 \
    --epochs 100 \
    --batch_size 10

# 3. Extraction des features fine-tunées
python training.py \
    --mode dump_features \
    --backbone cvt-w24 \
    --use_finetuned

# 4. Entraînement des MLPs
python training.py \
    --mode train_on_features \
    --backbone cvt-w24 \
    --train_mlp4 \
    --all_binaries \
    --use_finetuned
```

### **Workflow 2 : Fine-tuning simple**
```bash
# 1. Fine-tuning avec augmentation en temps réel
python training.py \
    --mode finetune \
    --backbone cvt-w24 \
    --use_augmentation \
    --epochs 100

# 2. Extraction et entraînement
python training.py \
    --mode dump_features \
    --backbone cvt-w24 \
    --use_finetuned

python training.py \
    --mode train_on_features \
    --backbone cvt-w24 \
    --train_mlp4 \
    --use_finetuned
```

### **Workflow 3 : Entraînement classique**
```bash
# 1. Extraction des features pré-entraînées
python training.py \
    --mode dump_features \
    --backbone twins \
    --no_finetuned

# 2. Entraînement des modèles
python training.py \
    --mode train_on_features \
    --backbone twins \
    --train_mlp4 \
    --all_binaries \
    --no_finetuned
```

## 🔍 Paramètres détaillés

### **Arguments principaux :**
- `--mode` : Mode d'exécution (augment, finetune, finetune_augmented, dump_features, train_on_features, train)
- `--backbone` : Backbone à utiliser (cvt-w24, twins, vit, etc.)
- `--annotations_csv` : Chemin vers le CSV des annotations
- `--image_root` : Chemin vers les images

### **Arguments d'entraînement :**
- `--epochs` : Nombre d'époques (défaut: 100)
- `--batch_size` : Taille du batch (défaut: 10)
- `--learning_rate` : Taux d'apprentissage (défaut: 1e-4)
- `--patience` : Patience pour early stopping (défaut: 5)
- `--min_delta` : Delta minimum pour amélioration (défaut: 0.001)

### **Arguments pour features :**
- `--use_finetuned` : Utiliser features fine-tunées (défaut: True)
- `--no_finetuned` : Ne pas utiliser features fine-tunées

### **Arguments pour data augmentation :**
- `--target_samples` : Nombre cible d'échantillons par classe (défaut: 2000)
- `--use_augmentation` : Activer data augmentation

### **Arguments pour entraînement :**
- `--train_mlp4` : Entraîner MLP 4 classes
- `--all_binaries` : Entraîner tous les binaires
- `--binary_pair` : Paire spécifique pour binaire (ex: 0 1)

## 📊 Comparaison des modes

| Mode | Description | Utilisation |
|------|-------------|-------------|
| **augment** | Crée dataset augmenté | Préparation des données |
| **finetune** | Fine-tuning avec augmentation temps réel | Tests rapides |
| **finetune_augmented** | Fine-tuning avec dataset pré-augmenté | Production |
| **dump_features** | Extraction de features | Préparation pour MLPs |
| **train_on_features** | Entraînement MLPs | Classification finale |
| **train** | Entraînement complet | Workflow classique |

## 🎯 Paramètres recommandés par backbone

### **CVT-W24 :**
```bash
python training.py \
    --mode finetune_augmented \
    --backbone cvt-w24 \
    --epochs 100 \
    --batch_size 10 \
    --learning_rate 1e-4 \
    --patience 5
```

### **Twins :**
```bash
python training.py \
    --mode finetune \
    --backbone twins \
    --epochs 80 \
    --batch_size 15 \
    --learning_rate 1e-4 \
    --patience 3
```

### **PVT :**
```bash
python training.py \
    --mode finetune \
    --backbone pvt \
    --epochs 90 \
    --batch_size 12 \
    --learning_rate 1e-4 \
    --patience 4
```

## 🔧 Fonctionnalités avancées

### **Learning Rate Scheduling :**
- Réduction automatique du LR si pas d'amélioration
- Facteur de réduction : 0.5
- Patience : 3 époques

### **Early Stopping :**
- Arrêt automatique si pas d'amélioration
- Patience configurable
- Sauvegarde du meilleur modèle

### **Data Augmentation :**
- Rotation légère (±3°)
- Flip horizontal (p=0.5)
- Ajustement luminosité/contraste (±10%)
- Crop aléatoire (échelle 0.9-1.0)
- Normalisation ImageNet

### **Pondération des classes :**
- Calcul automatique des poids
- Gestion du déséquilibre
- Compatible avec tous les modes

## 🧠 Modèle Hybride (Nouveau)

Le modèle hybride combine une image de mammographie et son histogramme pour une classification plus robuste.

### **Fichiers :**
- `hybrid_model.py` : Définition du modèle hybride
- `hybrid_finetuning.py` : Script d'entraînement complet avec fine-tuning
- `hybrid_training_example.py` : Exemple d'entraînement basique
- `HYBRID_MODEL_README.md` : Documentation détaillée
- `HYBRID_USAGE_GUIDE.md` : Guide d'utilisation complet

### **Architecture :**
```
Image (1, 224, 224) → CNN/ViT → (512)
                                    ↓
Histogramme (256) → MLP → (64) → Concat → (576) → Final MLP → (4)
```

### **Utilisation :**
```bash
# Entraînement complet avec fine-tuning
python hybrid_finetuning.py --backbone cnn --use_augmentation

# Avec paramètres personnalisés
python hybrid_finetuning.py \
    --backbone cnn \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --use_augmentation

# Reprise d'entraînement
python hybrid_finetuning.py \
    --resume_from features/finetuned_weights/hybrid_model_checkpoint.pth
```

### **Avantages :**
- **Information complémentaire** : Image + histogramme
- **Robustesse** : Moins sensible aux variations d'éclairage
- **Interprétabilité** : Analyse des contributions
- **Flexibilité** : Supporte CNN et ViT
- **Fine-tuning complet** : Entraînement end-to-end
- **Checkpointing** : Reprise d'entraînement possible

### **Fonctionnalités avancées :**
- ✅ **Data augmentation** : Transforms pour améliorer la généralisation
- ✅ **Early stopping** : Arrêt automatique si pas d'amélioration
- ✅ **Learning rate scheduling** : Adaptation automatique du LR
- ✅ **Checkpointing** : Sauvegarde et reprise d'entraînement
- ✅ **Monitoring** : Affichage en temps réel des métriques
- ✅ **Évaluation automatique** : Matrice de confusion et métriques

---

## 📁 Structure des fichiers générés

```
Codes/
├── training.py                           # Système unifié
├── models.py                             # Définitions des modèles
├── preprocessing.py                      # Prétraitement des images
├── UNIFIED_TRAINING_GUIDE.md            # Ce guide
├── DDSM/
│   ├── output_annotations.csv           # Dataset original
│   └── output_annotations_augmented.csv # Dataset augmenté
└── features/                            # Dossier organisé des résultats
    ├── README.md                        # Documentation de la structure
    ├── extracted/                       # Features extraites
    │   ├── twins_features_train.npy
    │   ├── twins_labels_train.npy
    │   ├── cvt-w24_finetuned_features_train.npy
    │   └── cvt-w24_finetuned_labels_train.npy
    ├── models/                          # Modèles MLP entraînés
    │   ├── twins_mlp4.pth
    │   ├── twins_finetuned_mlp4.pth
    │   ├── twins_binary_0_1.pth
    │   └── cvt-w24_finetuned_mlp4.pth
    └── finetuned_weights/              # Poids fine-tunés
        ├── cvt-w24_finetuned_best.pth
        ├── cvt-w24_finetuned_final.pth
        └── cvt-w24_finetuned_checkpoint.pth
```

## 🚨 Points d'attention

### **Ordre des opérations :**
1. **Data augmentation** (si nécessaire)
2. **Fine-tuning** du Transformer
3. **Extraction** des features
4. **Entraînement** des MLPs

### **Gestion des fichiers :**
- Vérification automatique de l'existence des fichiers
- Création des dossiers si nécessaire
- Messages informatifs sur l'état des opérations

### **Optimisation mémoire :**
- Batch size adapté selon le backbone
- Extraction par batches pour les grandes datasets
- Gestion automatique du device (CPU/GPU)

## 📞 Support et dépannage

### **Erreurs courantes :**
1. **Dataset non trouvé** : Vérifiez les chemins vers les images
2. **GPU out of memory** : Réduisez la batch size
3. **Features manquantes** : Lancez d'abord `dump_features`
4. **Modèle non trouvé** : Vérifiez que le fine-tuning est terminé

### **Conseils d'optimisation :**
- Utilisez `finetune_augmented` pour de meilleures performances
- Ajustez `patience` selon la complexité du modèle
- Surveillez l'utilisation GPU pendant l'entraînement
- Sauvegardez régulièrement les meilleurs modèles

---

**Note :** Ce système unifié simplifie grandement l'utilisation tout en conservant toutes les fonctionnalités avancées. Choisissez le mode approprié selon vos besoins spécifiques. 