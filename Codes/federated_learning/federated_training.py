# Apprentissage fédéré (FedAvg) pour la classification de densité mammaire.
# Deux clients : H1 (VinDr-Mammo) et H2 (DDSM).

import argparse
import copy
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from fed_model import ResNet50DensityClassifier, SiameseFederatedClassifier
from fed_datasets import VinDrCCDataset, DDSMDataset, VinDrPairedDataset, DDSMPairedDataset, INbreastDataset, DENSITY_CLASSES

VINDR_ROOT = "/home_nfs/abdelouahada/dataset_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
VINDR_ANNOTATIONS = os.path.join(VINDR_ROOT, "breast-level_annotations.csv")
DDSM_ANNOTATIONS = "/home_nfs/abdelouahada/Entrainement_cetic/CBIS-DDSM-manifest/cbis_ddsm_annotations.csv"
INBREAST_ANNOTATIONS = "/home_nfs/abdelouahada/dataset_extracted/INbreast_raw/inbreast_annotations.csv"

def class_weights_for(dataset, device):
    # Calcul des poids de classe pour le déséquilibre
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    indices = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else range(len(base))
    if hasattr(base, 'df'):
        labels = base.df.iloc[list(indices)]['label'].values
    else:
        labels = np.array([base.pairs[i]['label'] for i in indices])
    counts = np.bincount(labels, minlength=4)
    n = counts.sum()
    weights = np.array([n / (4 * c) if c > 0 else 0.0 for c in counts], dtype=np.float32)
    return torch.tensor(weights, device=device)

def local_train(model, loader, device, epochs, lr):
    # Entraînement local
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights_for(loader.dataset, device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    last_train_acc, last_train_loss = 0.0, 0.0
    for _ in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        last_train_acc = correct / total if total else 0.0
        last_train_loss = running_loss / total if total else 0.0
    return model.state_dict(), last_train_acc, last_train_loss

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    running_loss, total = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / total if total else 0.0
    return acc, avg_loss, all_labels, all_preds

def local_train_paired(model, loader, device, epochs, lr):
    # Entraînement pour le modèle siamois
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights_for(loader.dataset, device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    last_train_acc, last_train_loss = 0.0, 0.0
    for _ in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for cc_images, mlo_images, labels in loader:
            cc_images, mlo_images, labels = cc_images.to(device), mlo_images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(cc_images, mlo_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * cc_images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        last_train_acc = correct / total if total else 0.0
        last_train_loss = running_loss / total if total else 0.0
    return model.state_dict(), last_train_acc, last_train_loss

@torch.no_grad()
def evaluate_paired(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    running_loss, total = 0.0, 0
    for cc_images, mlo_images, labels in loader:
        cc_images, mlo_images, labels = cc_images.to(device), mlo_images.to(device), labels.to(device)
        outputs = model(cc_images, mlo_images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * cc_images.size(0)
        total += labels.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / total if total else 0.0
    return acc, avg_loss, all_labels, all_preds

def fedavg_aggregate(state_dicts, sample_counts, min_weight=None):
    # Agrégation des poids FedAvg
    total = sum(sample_counts)
    raw_weights = [n / total for n in sample_counts]

    if min_weight is not None and len(sample_counts) > 1:
        weights = [max(w, min_weight) for w in raw_weights]
        excess = sum(weights) - 1.0
        if excess > 1e-9:
            above_floor_idx = [i for i, w in enumerate(weights) if w > min_weight]
            above_floor_total = sum(weights[i] for i in above_floor_idx)
            for i in above_floor_idx:
                weights[i] -= excess * (weights[i] / above_floor_total)
    else:
        weights = raw_weights

    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        avg_state[key] = sum(
            state_dicts[i][key].float() * weights[i] for i in range(len(state_dicts))
        )
        if state_dicts[0][key].dtype != torch.float32:
            avg_state[key] = avg_state[key].to(state_dicts[0][key].dtype)
    return avg_state

def load_pretrained_backbone(model, checkpoint_path, device):
    # Charger des poids pré-entraînés pour initialiser W0
    raw_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw_state, dict) and 'model_state_dict' in raw_state:
        raw_state = raw_state['model_state_dict']

    target_state = model.state_dict()
    loaded, skipped = 0, []
    for key, value in raw_state.items():
        target_key = f"cnn.{key}"
        if target_key not in target_state:
            skipped.append(key)
            continue
        if target_key == "cnn.conv1.weight" and value.shape[1] == 3 and target_state[target_key].shape[1] == 1:
            value = value.mean(dim=1, keepdim=True)
        if value.shape != target_state[target_key].shape:
            skipped.append(key)
            continue
        target_state[target_key] = value
        loaded += 1

    model.load_state_dict(target_state)
    print(f"Chargement des poids : {loaded} chargés, {len(skipped)} ignorés.")

def build_client_data(name, batch_size, architecture='resnet50'):
    # Charger les datasets des clients
    if architecture == 'siamese':
        if name == 'vindr':
            train = VinDrPairedDataset(VINDR_ANNOTATIONS, VINDR_ROOT, split='training', use_augmentation=True)
            test = VinDrPairedDataset(VINDR_ANNOTATIONS, VINDR_ROOT, split='test', use_augmentation=False)
        elif name == 'ddsm':
            train = DDSMPairedDataset(DDSM_ANNOTATIONS, split='training', use_augmentation=True)
            test = DDSMPairedDataset(DDSM_ANNOTATIONS, split='test', use_augmentation=False)
        else:
            raise ValueError(name)
    elif name == 'vindr':
        train = VinDrCCDataset(VINDR_ANNOTATIONS, VINDR_ROOT, split='training', use_augmentation=True)
        test = VinDrCCDataset(VINDR_ANNOTATIONS, VINDR_ROOT, split='test', use_augmentation=False)
    elif name == 'ddsm':
        train = DDSMDataset(DDSM_ANNOTATIONS, split='training', use_augmentation=True)
        test = DDSMDataset(DDSM_ANNOTATIONS, split='test', use_augmentation=False)
    elif name == 'inbreast':
        train = INbreastDataset(INBREAST_ANNOTATIONS, split='training', use_augmentation=True)
        test = INbreastDataset(INBREAST_ANNOTATIONS, split='test', use_augmentation=False)
    else:
        raise ValueError(name)

    n_val = max(1, int(0.15 * len(train)))
    n_train = len(train) - n_val
    train_split, val_split = torch.utils.data.random_split(
        train, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    return (
        DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
        DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=4),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4),
        len(train_split),
    )

def run_federated(rounds, local_epochs, batch_size, lr, out_dir, resume=False, min_weight=None,
                   pretrained_checkpoint=None, architecture='resnet50', h2_dataset='ddsm'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    is_siamese = architecture == 'siamese'
    train_fn = local_train_paired if is_siamese else local_train
    eval_fn = evaluate_paired if is_siamese else evaluate
    unit = "paires" if is_siamese else "images"
    h2_label = {"ddsm": "DDSM", "inbreast": "INbreast"}[h2_dataset]

    print("Préparation des données...")
    vindr_train, vindr_val, vindr_test, n_vindr = build_client_data('vindr', batch_size, architecture)
    ddsm_train, ddsm_val, ddsm_test, n_ddsm = build_client_data(h2_dataset, batch_size, architecture)
    print(f"H1 VinDr : {n_vindr} {unit} | H2 {h2_label} : {n_ddsm} {unit}")

    if is_siamese:
        global_model = SiameseFederatedClassifier(num_classes=4).to(device)
        pretrained_checkpoint = None
    else:
        global_model = ResNet50DensityClassifier(num_classes=4).to(device)
        
    history = {"round": [], "vindr_val_acc": [], "ddsm_val_acc": [], "global_val_acc": [],
               "vindr_train_acc": [], "ddsm_train_acc": [],
               "vindr_val_loss": [], "ddsm_val_loss": [], "global_val_loss": [],
               "vindr_train_loss": [], "ddsm_train_loss": []}
    best_global_acc = 0.0
    start_round = 0

    latest_ckpt = os.path.join(out_dir, "federated_global_latest.pth")
    history_path = os.path.join(out_dir, "federated_history.json")
    if resume and os.path.exists(latest_ckpt) and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        start_round = len(history["round"])
        global_model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        best_global_acc = max(history["global_val_acc"], default=0.0) / 100.0
        print(f"Reprise du round {start_round + 1}...")
    
    if pretrained_checkpoint and start_round == 0:
        load_pretrained_backbone(global_model, pretrained_checkpoint, device)

    for t in range(start_round, rounds):
        print(f"\n=== ROUND {t+1}/{rounds} ===")
        local_states, local_ns, local_train_accs, local_train_losses = [], [], [], []

        for name, loader, n in [("H1-VinDr", vindr_train, n_vindr), (f"H2-{h2_label}", ddsm_train, n_ddsm)]:
            local_model = (SiameseFederatedClassifier(num_classes=4) if is_siamese else ResNet50DensityClassifier(num_classes=4)).to(device)
            local_model.load_state_dict(global_model.state_dict())
            print(f"Entraînement local client {name} ({local_epochs} époques)...")
            state_dict, train_acc, train_loss = train_fn(local_model, loader, device, local_epochs, lr)
            local_states.append(state_dict)
            local_ns.append(n)
            local_train_accs.append(train_acc)
            local_train_losses.append(train_loss)
            print(f"  Client {name} Train Loss: {train_loss:.4f}, Acc locale: {train_acc*100:.2f}%")

        global_model.load_state_dict(fedavg_aggregate(local_states, local_ns, min_weight=min_weight))

        vindr_val_acc, vindr_val_loss, _, _ = eval_fn(global_model, vindr_val, device)
        ddsm_val_acc, ddsm_val_loss, _, _ = eval_fn(global_model, ddsm_val, device)
        n_total = n_vindr + n_ddsm
        global_val_acc = (vindr_val_acc * n_vindr + ddsm_val_acc * n_ddsm) / n_total
        global_val_loss = (vindr_val_loss * n_vindr + ddsm_val_loss * n_ddsm) / n_total

        print(f"Val Loss - VinDr : {vindr_val_loss:.4f} | {h2_label} : {ddsm_val_loss:.4f} | Global : {global_val_loss:.4f}")
        print(f"Val Acc - VinDr : {vindr_val_acc*100:.2f}% | {h2_label} : {ddsm_val_acc*100:.2f}% | Global : {global_val_acc*100:.2f}%")

        history["round"].append(t + 1)
        history["vindr_val_acc"].append(vindr_val_acc * 100)
        history["ddsm_val_acc"].append(ddsm_val_acc * 100)
        history["global_val_acc"].append(global_val_acc * 100)
        history["vindr_train_acc"].append(local_train_accs[0] * 100)
        history["ddsm_train_acc"].append(local_train_accs[1] * 100)
        history["vindr_val_loss"].append(vindr_val_loss)
        history["ddsm_val_loss"].append(ddsm_val_loss)
        history["global_val_loss"].append(global_val_loss)
        history["vindr_train_loss"].append(local_train_losses[0])
        history["ddsm_train_loss"].append(local_train_losses[1])
        with open(os.path.join(out_dir, "federated_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        torch.save(global_model.state_dict(), os.path.join(out_dir, "federated_global_latest.pth"))

        if global_val_acc > best_global_acc:
            best_global_acc = global_val_acc
            torch.save(global_model.state_dict(), os.path.join(out_dir, "federated_global_best.pth"))
            print(f"Nouveau meilleur modèle global fédéré: {global_val_acc*100:.2f}%")

    print("\n=== ÉVALUATION FINALE ===")
    global_model.load_state_dict(torch.load(os.path.join(out_dir, "federated_global_best.pth"), map_location=device))

    for name, loader in [("VinDr-Mammo (H1)", vindr_test), (f"{h2_label} (H2)", ddsm_test)]:
        acc, loss, labels, preds = eval_fn(global_model, loader, device)
        print(f"\nTest {name} : {acc*100:.2f}% (loss {loss:.4f})")
        print(confusion_matrix(labels, preds, labels=[0, 1, 2, 3]))
        print(classification_report(labels, preds, labels=[0, 1, 2, 3], target_names=DENSITY_CLASSES, zero_division=0))

def run_centralized(epochs, batch_size, lr, out_dir):
    # Centralisé (baseline)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    vindr_train, vindr_val, vindr_test, n_vindr = build_client_data('vindr', batch_size)
    ddsm_train, ddsm_val, ddsm_test, n_ddsm = build_client_data('ddsm', batch_size)

    pooled_train = ConcatDataset([vindr_train.dataset, ddsm_train.dataset])
    pooled_loader = DataLoader(pooled_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    pooled_val = ConcatDataset([vindr_val.dataset, ddsm_val.dataset])
    pooled_val_loader = DataLoader(pooled_val, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ResNet50DensityClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in pooled_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_loss = running_loss / total

        val_acc, val_loss, _, _ = evaluate(model, pooled_val_loader, device)
        print(f"Époque {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        history["epoch"].append(epoch + 1)
        history["train_acc"].append(train_acc * 100)
        history["val_acc"].append(val_acc * 100)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        with open(os.path.join(out_dir, "centralized_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "centralized_best.pth"))

    print("\n=== ÉVALUATION FINALE ===")
    model.load_state_dict(torch.load(os.path.join(out_dir, "centralized_best.pth"), map_location=device))
    for name, loader in [("VinDr-Mammo", vindr_test), ("DDSM/CBIS-DDSM", ddsm_test)]:
        acc, loss, labels, preds = evaluate(model, loader, device)
        print(f"\nTest {name} : {acc*100:.2f}% (loss {loss:.4f})")
        print(confusion_matrix(labels, preds, labels=[0, 1, 2, 3]))
        print(classification_report(labels, preds, labels=[0, 1, 2, 3], target_names=DENSITY_CLASSES, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["federated", "centralized"], default="federated")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="featuresfinetuned_weights")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min_weight", type=float, default=None)
    parser.add_argument("--pretrained_checkpoint", default=None)
    parser.add_argument("--architecture", choices=["resnet50", "siamese"], default="resnet50")
    parser.add_argument("--h2_dataset", choices=["ddsm", "inbreast"], default="ddsm")
    args = parser.parse_args()

    if args.mode == "federated":
        run_federated(args.rounds, args.local_epochs, args.batch_size, args.lr, args.out_dir,
                       resume=args.resume, min_weight=args.min_weight,
                       pretrained_checkpoint=args.pretrained_checkpoint, architecture=args.architecture,
                       h2_dataset=args.h2_dataset)
    else:
        run_centralized(args.epochs, args.batch_size, args.lr, args.out_dir)
