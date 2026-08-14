"""
Cascade hierarchique (Approche 2) avec fine-tuning propre a chaque etage,
au lieu de features gelees + petits MLP (three_stage_pipeline.py).

Les 3 etages :
  1. (A,B) vs (C,D) -- backbone fine-tune sur les 4 classes, labels groupes
  2. A vs B          -- backbone fine-tune uniquement sur les images A/B
  3. C vs D          -- backbone fine-tune uniquement sur les images C/D

Chaque etage a son propre backbone independant (pas de poids partages),
entraine de bout en bout (pas de features pre-extraites). C'est le test
direct de l'hypothese avancee dans le memoire : est-ce que le probleme de
la cascade vient des features gelees, ou du routage lui-meme (deja
identifie comme la vraie cause structurelle) ?
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from training import MammographyDataset, CONFIG
from models import FineTunedFeatureExtractor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def labels_of(dataset):
    # random_split() renvoie un Subset (dataset original + indices), pas le
    # dataset lui-meme -- il faut indexer .dataset.df par .indices (meme
    # correction que federated_training.py).
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset.df.iloc[dataset.indices]['label'].values
    return dataset.df['label'].values


def weighted_sampler_for(dataset, num_classes):
    """Sur-echantillonne les classes minoritaires a chaque batch. Plus efficace
    qu'une simple ponderation de la loss quand une classe est tres rare (ex.
    DENSITY A ~5% du sous-ensemble A/B) : avec une ponderation de loss seule et
    un petit batch_size, beaucoup de batches ne contiennent aucun exemple de la
    classe rare, donc le signal de gradient pour cette classe reste trop rare
    sur quelques epoques -- le modele peut collapser sur la classe majoritaire
    malgre la ponderation (observe sur l'etage 2 : 98.76% val mais 0% recall
    sur DENSITY A en test)."""
    labels = labels_of(dataset)
    counts = np.bincount(labels, minlength=num_classes)
    class_weights = np.array([1.0 / c if c > 0 else 0.0 for c in counts], dtype=np.float64)
    sample_weights = class_weights[labels]
    return torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def finetune_stage(backbone_name, train_dataset, val_dataset, num_classes, epochs, batch_size, lr, save_path):
    """Fine-tuning de bout en bout d'un backbone pour un etage de la cascade."""
    sampler = weighted_sampler_for(train_dataset, num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FineTunedFeatureExtractor(backbone_name, num_classes=num_classes, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_balanced_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total if total else 0.0

        # Le choix du meilleur modele se fait sur la recall macro (moyenne des
        # recalls par classe), pas sur l'accuracy globale : avec un val set
        # deja tres desequilibre, l'accuracy globale peut rester elevee meme
        # si le modele ne predit jamais la classe rare (c'est exactement ce
        # qui s'est produit avant ce correctif : 98.76% val mais 0% recall sur
        # DENSITY A en test).
        model.eval()
        val_correct, val_total = 0, 0
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                for c in range(num_classes):
                    mask = labels == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (preds[mask] == c).sum().item()
        val_acc = val_correct / val_total if val_total else 0.0
        per_class_recall = np.array([class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
                                      for c in range(num_classes)])
        balanced_acc = per_class_recall.mean()

        print(f"  [Epoque {epoch+1}/{epochs}] Train Loss: {running_loss/total:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | "
              f"Val Recall/classe: {[f'{r*100:.1f}%' for r in per_class_recall]} | "
              f"Val Balanced Acc: {balanced_acc*100:.2f}%", flush=True)

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(best_state, save_path)
    print(f"  Meilleur modele sauvegarde ({best_balanced_acc*100:.2f}% balanced acc val) : {save_path}", flush=True)
    return model


@torch.no_grad()
def infer_probs(model, dataset, batch_size=16):
    """Renvoie les probabilites softmax (classe 1) pour tout le dataset, dans l'ordre."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def build_datasets(df, image_root, use_augmentation):
    """Construit les 3 jeux d'entrainement (un par etage) + leurs splits de validation interne (85/15, seed=42)."""
    group_map = {"DENSITY A": 0, "DENSITY B": 0, "DENSITY C": 1, "DENSITY D": 1}

    stage1_train = MammographyDataset(df, image_root, label_map=group_map, use_augmentation=use_augmentation, split='training')
    stage2_train = MammographyDataset(df, image_root, classes_to_use=["DENSITY A", "DENSITY B"], use_augmentation=use_augmentation, split='training')
    stage3_train = MammographyDataset(df, image_root, classes_to_use=["DENSITY C", "DENSITY D"], use_augmentation=use_augmentation, split='training')

    def split_train_val(dataset):
        n_val = max(1, int(0.15 * len(dataset)))
        n_train = len(dataset) - n_val
        return torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    return {
        1: split_train_val(stage1_train),
        2: split_train_val(stage2_train),
        3: split_train_val(stage3_train),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet50 ou vit (BACKBONES dans models.py)')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train',
                         help='train : fine-tune un seul etage (--stage). '
                              'evaluate : combine les 3 checkpoints deja entraines et evalue la cascade sur le test.')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=1,
                         help='Etage a entrainer en mode train (1=AB vs CD, 2=A vs B, 3=C vs D). '
                              'Lancer les 3 en parallele, un par GPU, plutot que sequentiellement.')
    parser.add_argument('--epochs', type=int, default=15, help='Epoques pour cet etage (backbone complet, donc moins que les 30 des MLP)')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_augmentation', action='store_true', default=True)
    parser.add_argument('--out_dir', default='cascade_finetuned_weights')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(CONFIG['ANNOTATIONS_CSV'])
    df_train = df[df['split'] == 'training'].reset_index(drop=True)
    df_test = df[df['split'] == 'test'].reset_index(drop=True)

    splits = build_datasets(df_train, CONFIG['IMAGE_ROOT'], args.use_augmentation)

    stage_names = {1: "(A,B) vs (C,D)", 2: "A vs B", 3: "C vs D"}
    stage_files = {1: "stage1_abcd", 2: "stage2_avsb", 3: "stage3_cvsd"}

    if args.mode == 'train':
        print(f"\n=== ETAGE {args.stage}/3 : {stage_names[args.stage]}, backbone fine-tune ===", flush=True)
        train_s, val_s = splits[args.stage]
        finetune_stage(args.backbone, train_s, val_s, num_classes=2, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr,
                        save_path=f"{args.out_dir}/{args.backbone}_{stage_files[args.stage]}.pth")
        print(f"\nEtage {args.stage} termine. Une fois les 3 etages faits, relancer avec --mode evaluate.", flush=True)
        return

    # --- Mode evaluate : recharge les 3 checkpoints deja entraines separement ---
    model1 = FineTunedFeatureExtractor(args.backbone, num_classes=2, pretrained=False).to(DEVICE)
    model1.load_state_dict(torch.load(f"{args.out_dir}/{args.backbone}_stage1_abcd.pth", map_location=DEVICE))
    model2 = FineTunedFeatureExtractor(args.backbone, num_classes=2, pretrained=False).to(DEVICE)
    model2.load_state_dict(torch.load(f"{args.out_dir}/{args.backbone}_stage2_avsb.pth", map_location=DEVICE))
    model3 = FineTunedFeatureExtractor(args.backbone, num_classes=2, pretrained=False).to(DEVICE)
    model3.load_state_dict(torch.load(f"{args.out_dir}/{args.backbone}_stage3_cvsd.pth", map_location=DEVICE))

    _, val1 = splits[1]

    # Seuil de routage de l'etage 1, choisi sur la validation interne (meme
    # methode -- critere de Youden -- que la version a features gelees, pour
    # rester comparable).
    print("\n=== Choix du seuil de routage (validation, critere de Youden) ===", flush=True)
    val1_probs, val1_labels = infer_probs(model1, val1)
    fpr, tpr, thresholds = roc_curve(val1_labels, val1_probs[:, 1])
    optimal_idx = np.argmax(tpr - fpr)
    threshold = float(thresholds[optimal_idx])
    print(f"Seuil optimal : {threshold:.4f} (TPR={tpr[optimal_idx]:.2%}, FPR={fpr[optimal_idx]:.2%})", flush=True)

    # --- Evaluation finale de la cascade sur le test, jamais vu avant ---
    print("\n=== EVALUATION FINALE DE LA CASCADE FINE-TUNEE (TEST) ===", flush=True)
    test_dataset = MammographyDataset(df_test, CONFIG['IMAGE_ROOT'], label_map=CONFIG['CLASS_MAP'],
                                       use_augmentation=False, split='test')
    test_probs_stage1, true_labels_4c = infer_probs(model1, test_dataset)
    routed_to_ab = test_probs_stage1[:, 1] <= threshold  # proba classe "CD" <= seuil -> route vers AB

    final_preds = np.full(len(true_labels_4c), -1)

    ab_idx = np.where(routed_to_ab)[0]
    cd_idx = np.where(~routed_to_ab)[0]
    print(f"Routage etape 1 : {len(ab_idx)} images vers (A,B), {len(cd_idx)} images vers (C,D)", flush=True)

    if len(ab_idx) > 0:
        ab_subset = torch.utils.data.Subset(test_dataset, ab_idx.tolist())
        probs_ab, _ = infer_probs(model2, ab_subset)
        final_preds[ab_idx] = np.where(probs_ab[:, 1] > 0.5, 1, 0)  # 0=A, 1=B

    if len(cd_idx) > 0:
        cd_subset = torch.utils.data.Subset(test_dataset, cd_idx.tolist())
        probs_cd, _ = infer_probs(model3, cd_subset)
        final_preds[cd_idx] = np.where(probs_cd[:, 1] > 0.5, 3, 2)  # 2=C, 3=D

    accuracy = np.mean(final_preds == true_labels_4c)
    print(f"\nAccuracy globale de la cascade fine-tunee : {accuracy*100:.2f}%", flush=True)
    print("\nRapport de classification :", flush=True)
    print(classification_report(true_labels_4c, final_preds, target_names=CONFIG['DENSITY_CLASSES'], zero_division=0))
    print("Matrice de confusion (A/B/C/D) :", flush=True)
    print(confusion_matrix(true_labels_4c, final_preds, labels=[0, 1, 2, 3]))

    # Diagnostic : rappel de routage de l'etape 1 sur les vrais (A,B), pour
    # comparer directement au chiffre deja documente pour la version a
    # features gelees.
    true_ab = np.isin(true_labels_4c, [0, 1])
    routing_recall_ab = np.mean(routed_to_ab[true_ab])
    print(f"\n[DIAGNOSTIC] Rappel de routage etape 1 sur les vrais (A,B) : "
          f"{routing_recall_ab*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
