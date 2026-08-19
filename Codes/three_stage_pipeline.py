import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os
import json
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from models import FeatureExtractor, BinaryMLPClassifier, FeatureMLPTrainer
from preprocessing import read_dicom, preprocess_image

class ThreeStagePipeline:
    # Classification en 3 étapes :
    # 1. Modèle 1: (A,B) vs (C,D)
    # 2. Modèle 2: A vs B  
    # 3. Modèle 3: C vs D
    def __init__(self, backbone, device, feature_paths):
        self.backbone = backbone
        self.device = device
        self.feature_paths = feature_paths
        
        self.class_map = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
        self.class_names = ["A", "B", "C", "D"]
        
        self.model1 = None
        self.model2 = None
        self.model3 = None
        
    def load_features(self, split='train'):
        features = np.load(self.feature_paths[f'{split}_features'])
        labels = np.load(self.feature_paths[f'{split}_labels'])
        return features, labels
    
    def prepare_binary_data(self, features, labels, class_pair):
        mask = np.isin(labels, class_pair)
        features_bin = features[mask]
        labels_bin = labels[mask]
        labels_bin = (labels_bin == class_pair[1]).astype(int)
        return features_bin, labels_bin
    
    def train_model1(self, features, labels, seed=42, force_retrain=True):
        print(f"Entraînement Modèle 1 (Seed={seed}) : (A,B) vs (C,D)")
        labels_ab_cd = np.isin(labels, [2, 3]).astype(int)
        features_ab_cd = features
        
        in_dim = features.shape[1]
        self.model1 = BinaryMLPClassifier(in_dim=in_dim)
        trainer = FeatureMLPTrainer(self.model1)
        
        model_path = f'./{self.backbone}_three_stage_model1.pth'
        if force_retrain and os.path.exists(model_path):
            os.remove(model_path)
            
        if os.path.exists(model_path):
            trainer.load(model_path, device=self.device)
            print(f"Poids du Modèle 1 chargés : {model_path}")
        else:
            set_seed(seed)
            trainer.train(features_ab_cd, labels_ab_cd, epochs=30, 
                         batch_size=32, lr=1e-4, device=self.device)
            trainer.save(model_path)
            print(f"Modèle 1 sauvegardé : {model_path}")
        
    def train_model2(self, features, labels):
        print("Entraînement Modèle 2 : A vs B")
        features_bin, labels_bin = self.prepare_binary_data(features, labels, (0, 1))
        
        in_dim = features.shape[1]
        self.model2 = BinaryMLPClassifier(in_dim=in_dim)
        trainer = FeatureMLPTrainer(self.model2)
        
        model_path = f'./{self.backbone}_three_stage_model2.pth'
        if os.path.exists(model_path):
            trainer.load(model_path, device=self.device)
            print(f"Poids du Modèle 2 chargés : {model_path}")
        else:
            trainer.train(features_bin, labels_bin, epochs=30, 
                         batch_size=32, lr=1e-4, device=self.device)
            trainer.save(model_path)
            print(f"Modèle 2 sauvegardé : {model_path}")
        
    def train_model3(self, features, labels):
        print("Entraînement Modèle 3 : C vs D")
        features_bin, labels_bin = self.prepare_binary_data(features, labels, (2, 3))
        
        in_dim = features.shape[1]
        self.model3 = BinaryMLPClassifier(in_dim=in_dim)
        trainer = FeatureMLPTrainer(self.model3)
        
        model_path = f'./{self.backbone}_three_stage_model3.pth'
        if os.path.exists(model_path):
            trainer.load(model_path, device=self.device)
            print(f"Poids du Modèle 3 chargés : {model_path}")
        else:
            trainer.train(features_bin, labels_bin, epochs=30, 
                         batch_size=32, lr=1e-4, device=self.device)
            trainer.save(model_path)
            print(f"Modèle 3 sauvegardé : {model_path}")
    
    def find_optimal_stage1_threshold(self, features_val, labels_val):
        # Recherche du seuil avec Youden
        from sklearn.metrics import roc_curve
        if self.model1 is not None:
            self.model1.eval()
            
        features_tensor = torch.tensor(features_val, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits1 = self.model1(features_tensor)
            probs1 = torch.softmax(logits1, dim=1)
            probs1_ab = probs1[:, 0].cpu().numpy()
            
        true_ab_val = np.isin(labels_val, [0, 1]).astype(int)
        
        fpr, tpr, thresholds = roc_curve(true_ab_val, probs1_ab)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Seuil optimal Youden : {optimal_threshold:.4f}")
        print(f"TPR : {tpr[optimal_idx]:.4f} - FPR : {fpr[optimal_idx]:.4f}")
        
        return float(optimal_threshold)

    def predict_three_stage(self, features, stage1_threshold=0.5):
        if self.model1 is not None: self.model1.eval()
        if self.model2 is not None: self.model2.eval()
        if self.model3 is not None: self.model3.eval()
        
        predictions = []
        probabilities = np.zeros((len(features), 4))
        confidence_scores = np.zeros((len(features), 3))
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Etape 1
        with torch.no_grad():
            logits1 = self.model1(features_tensor)
            probs1 = torch.softmax(logits1, dim=1)
            stage1_conf = torch.max(probs1, dim=1)[0]
            is_ab = probs1[:, 0] > stage1_threshold
            
            ab_indices = torch.where(is_ab)[0]
            cd_indices = torch.where(~is_ab)[0]
            
            confidence_scores[:, 0] = stage1_conf.cpu().numpy()
        
        ab_is_a_idx = np.array([], dtype=int)
        ab_not_a_idx = np.array([], dtype=int)
        cd_is_c_idx = np.array([], dtype=int)
        cd_not_c_idx = np.array([], dtype=int)

        # Etape 2
        if len(ab_indices) > 0:
            with torch.no_grad():
                logits2 = self.model2(features_tensor[ab_indices])
                probs2 = torch.softmax(logits2, dim=1)
                stage2_conf = torch.max(probs2, dim=1)[0]
                is_a = probs2[:, 0] > 0.5
                
                ab_is_a_idx = ab_indices[is_a].cpu().numpy()
                ab_not_a_idx = ab_indices[~is_a].cpu().numpy()
                ab_idx_all = ab_indices.cpu().numpy()
                
                probabilities[ab_is_a_idx, 0] = probs2[is_a, 0].cpu().numpy()
                probabilities[ab_not_a_idx, 1] = probs2[~is_a, 1].cpu().numpy()
                
                confidence_scores[ab_idx_all, 1] = stage2_conf.cpu().numpy()
        
        # Etape 3
        if len(cd_indices) > 0:
            with torch.no_grad():
                logits3 = self.model3(features_tensor[cd_indices])
                probs3 = torch.softmax(logits3, dim=1)
                stage3_conf = torch.max(probs3, dim=1)[0]
                is_c = probs3[:, 0] > 0.5
                
                cd_is_c_idx = cd_indices[is_c].cpu().numpy()
                cd_not_c_idx = cd_indices[~is_c].cpu().numpy()
                cd_idx_all = cd_indices.cpu().numpy()
                
                probabilities[cd_is_c_idx, 2] = probs3[is_c, 0].cpu().numpy()
                probabilities[cd_not_c_idx, 3] = probs3[~is_c, 1].cpu().numpy()
                
                confidence_scores[cd_idx_all, 2] = stage3_conf.cpu().numpy()
        
        final_preds = np.zeros(len(features))
        final_preds[ab_is_a_idx] = 0
        final_preds[ab_not_a_idx] = 1
        final_preds[cd_is_c_idx] = 2
        final_preds[cd_not_c_idx] = 3
        
        return final_preds.astype(int), probabilities, confidence_scores
    
    def evaluate_pipeline(self, features_test, labels_test, stage1_threshold=0.5):
        print(f"Evaluation (seuil={stage1_threshold:.4f})")
        
        predictions, probabilities, confidence_scores = self.predict_three_stage(features_test, stage1_threshold=stage1_threshold)
        
        accuracy = np.mean(predictions == labels_test)
        print(f"Accuracy globale : {accuracy:.4f}")
        
        # Analyse des etapes
        true_ab = np.isin(labels_test, [0, 1])
        pred_ab = np.isin(predictions, [0, 1])
        level1_acc = np.mean(true_ab == pred_ab)
        
        features_tensor = torch.tensor(features_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits1 = self.model1(features_tensor)
            probs1 = torch.softmax(logits1, dim=1)
            routed_to_ab = (probs1[:, 0] > stage1_threshold).cpu().numpy()
        routing_recall_ab = np.mean(routed_to_ab[true_ab])
        
        print(f"Niveau 1 Accuracy : {level1_acc:.4f}")
        print(f"Rappel de routage AB : {routing_recall_ab:.4f}")
        
        ab_mask = np.isin(labels_test, [0, 1])
        if np.any(ab_mask):
            level2_acc = np.mean(predictions[ab_mask] == labels_test[ab_mask])
            print(f"Niveau 2 Accuracy : {level2_acc:.4f}")
        
        cd_mask = np.isin(labels_test, [2, 3])
        if np.any(cd_mask):
            level3_acc = np.mean(predictions[cd_mask] == labels_test[cd_mask])
            print(f"Niveau 3 Accuracy : {level3_acc:.4f}")
        
        print("\nRapport de classification :")
        print(classification_report(labels_test, predictions, target_names=self.class_names))
        
        cm1 = confusion_matrix(true_ab, pred_ab)
        self.plot_confusion_matrix(cm1, "Niveau 1", ["A,B", "C,D"])

        cm = confusion_matrix(labels_test, predictions)
        self.plot_confusion_matrix(cm, "Classification finale", self.class_names)
        
        self.evaluate_model2_isolated(features_test, labels_test)
        
        results = {'accuracy': accuracy, 'predictions': predictions}
        return results
    
    def evaluate_model2_isolated(self, features_test, labels_test):
        # Evaluation isolée de M2
        features_ab, labels_ab = self.prepare_binary_data(features_test, labels_test, (0, 1))
        
        if self.model2 is not None:
            self.model2.eval()
            
        features_tensor = torch.tensor(features_ab, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits2 = self.model2(features_tensor)
            probs2 = torch.softmax(logits2, dim=1)
            preds_m2 = (probs2[:, 1] > 0.5).cpu().numpy().astype(int)
            
        acc_m2 = np.mean(preds_m2 == labels_ab)
        print(f"Accuracy Modèle 2 seul (A vs B) : {acc_m2:.4f}")
        return acc_m2
    
    def plot_confusion_matrix(self, cm, title, class_names=None):
        if class_names is None:
            class_names = self.class_names
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'Matrice de confusion - {title}')
        plt.ylabel('Vrai')
        plt.xlabel('Prédit')
        
        output_dir = Path("graphes")
        output_dir.mkdir(exist_ok=True)
        safe_title = title.replace(" ", "_").lower()
        plt.savefig(output_dir / f'{self.backbone}_three_stage_{safe_title}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_with_cross_validation(self, features, labels, n_folds=5):
        # Validation croisee et reglage des hyperparametres
        from sklearn.model_selection import StratifiedKFold
        import itertools
        
        print("Validation croisée...")
        
        lr_values = [1e-4, 1e-3, 5e-4]
        batch_sizes = [16, 32, 64]
        architectures = [(512, 256), (1024, 512), (256, 128, 64)]
        
        best_params = {
            'model1': {'lr': None, 'batch_size': None, 'architecture': None, 'score': 0},
            'model2': {'lr': None, 'batch_size': None, 'architecture': None, 'score': 0},
            'model3': {'lr': None, 'batch_size': None, 'architecture': None, 'score': 0}
        }
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        param_combinations = list(itertools.product(lr_values, batch_sizes, architectures))
        
        for model_name, class_pair, is_first_level in [
            ('model1 (AB vs CD)', (0, 1), True),
            ('model2 (A vs B)', (0, 1), False),
            ('model3 (C vs D)', (2, 3), False)
        ]:
            print(f"Recherche hyperparamètres pour {model_name}...")
            best_score = 0
            best_params_model = None
            
            for combo_idx, (lr, batch_size, architecture) in enumerate(param_combinations, 1):
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
                    X_train, X_val = features[train_idx], features[val_idx]
                    y_train, y_val = labels[train_idx], labels[val_idx]
                    
                    if is_first_level:
                        y_train_bin = np.isin(y_train, [0, 1]).astype(int)
                        y_val_bin = np.isin(y_val, [0, 1]).astype(int)
                    else:
                        mask_train = np.isin(y_train, class_pair)
                        mask_val = np.isin(y_val, class_pair)
                        
                        X_train = X_train[mask_train]
                        X_val = X_val[mask_val]
                        y_train_bin = (y_train[mask_train] == class_pair[1]).astype(int)
                        y_val_bin = (y_val[mask_val] == class_pair[1]).astype(int)
                    
                    model = BinaryMLPClassifier(in_dim=features.shape[1], hidden_dims=architecture)
                    trainer = FeatureMLPTrainer(model)
                    
                    trainer.train(X_train, y_train_bin, epochs=10,
                                batch_size=batch_size, lr=lr,
                                device=self.device,
                                validation_data=(X_val, y_val_bin))
                    
                    val_score = trainer.evaluate(X_val, y_val_bin)
                    fold_scores.append(val_score)
                
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params_model = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'architecture': architecture,
                        'score': mean_score
                    }
            
            best_params[model_name] = best_params_model
        
        # Entraînement final
        self.model1 = BinaryMLPClassifier(in_dim=features.shape[1], hidden_dims=best_params['model1']['architecture'])
        trainer1 = FeatureMLPTrainer(self.model1)
        trainer1.train(features, np.isin(labels, [0, 1]).astype(int),
                      epochs=10, batch_size=best_params['model1']['batch_size'],
                      lr=best_params['model1']['lr'], device=self.device)
        
        features_ab = features[np.isin(labels, [0, 1])]
        labels_ab = labels[np.isin(labels, [0, 1])]
        self.model2 = BinaryMLPClassifier(in_dim=features.shape[1], hidden_dims=best_params['model2']['architecture'])
        trainer2 = FeatureMLPTrainer(self.model2)
        trainer2.train(features_ab, (labels_ab == 1).astype(int),
                      epochs=10, batch_size=best_params['model2']['batch_size'],
                      lr=best_params['model2']['lr'], device=self.device)
        
        features_cd = features[np.isin(labels, [2, 3])]
        labels_cd = labels[np.isin(labels, [2, 3])]
        self.model3 = BinaryMLPClassifier(in_dim=features.shape[1], hidden_dims=best_params['model3']['architecture'])
        trainer3 = FeatureMLPTrainer(self.model3)
        trainer3.train(features_cd, (labels_cd == 3).astype(int),
                      epochs=10, batch_size=best_params['model3']['batch_size'],
                      lr=best_params['model3']['lr'], device=self.device)
        
        return best_params
    
    def run_multi_seed_experiment(self, seeds=[42, 43, 44, 45, 46]):
        # Test sur plusieurs seeds différentes
        features_train, labels_train = self.load_features('train')
        features_test, labels_test = self.load_features('test')
        
        self.train_model2(features_train, labels_train)
        self.train_model3(features_train, labels_train)
        
        results_by_seed = []
        
        print("\nExpériences multi-seeds...")
        
        for seed in seeds:
            self.train_model1(features_train, labels_train, seed=seed, force_retrain=True)
            
            if self.model1 is not None: self.model1.eval()
            if self.model2 is not None: self.model2.eval()
            if self.model3 is not None: self.model3.eval()
            
            predictions, _, _ = self.predict_three_stage(features_test, stage1_threshold=0.5)
            overall_acc = np.mean(predictions == labels_test)
            
            true_ab = np.isin(labels_test, [0, 1])
            pred_ab = np.isin(predictions, [0, 1])
            level1_acc = np.mean(true_ab == pred_ab)
            
            features_tensor = torch.tensor(features_test, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits1 = self.model1(features_tensor)
                probs1 = torch.softmax(logits1, dim=1)
                routed_to_ab = (probs1[:, 0] > 0.5).cpu().numpy()
            routing_recall_ab = np.mean(routed_to_ab[true_ab])
            
            results_by_seed.append({
                'seed': seed,
                'level1_acc': level1_acc,
                'routing_recall_ab': routing_recall_ab,
                'overall_acc': overall_acc
            })
            print(f"Seed {seed} - Niveau 1 Acc : {level1_acc:.4f} | Rappel AB : {routing_recall_ab:.4f} | Acc globale : {overall_acc:.4f}")
            
        level1_accs = [r['level1_acc'] for r in results_by_seed]
        routing_recalls = [r['routing_recall_ab'] for r in results_by_seed]
        overall_accs = [r['overall_acc'] for r in results_by_seed]
        
        print("\nSynthèse statistiques multi-seeds :")
        print(f"Accuracy Niveau 1 : {np.mean(level1_accs):.4f} +/- {np.std(level1_accs):.4f}")
        print(f"Rappel Routage AB : {np.mean(routing_recalls):.4f} +/- {np.std(routing_recalls):.4f}")
        print(f"Accuracy Globale : {np.mean(overall_accs):.4f} +/- {np.std(overall_accs):.4f}")

    def train_all_models(self):
        print(f"Entraînement complet de la pipeline avec {self.backbone}")

        features_train_full, labels_train_full = self.load_features('train')
        features_test, labels_test = self.load_features('test')

        indices = np.arange(len(labels_train_full))
        np.random.seed(42)
        val_size = int(0.2 * len(indices))
        val_indices = np.random.choice(indices, size=val_size, replace=False)
        train_indices = np.setdiff1d(indices, val_indices)

        features_train = features_train_full[train_indices]
        labels_train = labels_train_full[train_indices]
        features_val = features_train_full[val_indices]
        labels_val = labels_train_full[val_indices]

        # Supprimer les anciens checkpoints
        for i in (1, 2, 3):
            stale_path = f'./{self.backbone}_three_stage_model{i}.pth'
            if os.path.exists(stale_path):
                os.remove(stale_path)

        self.train_model1(features_train, labels_train, seed=42, force_retrain=True)
        self.train_model2(features_train, labels_train)
        self.train_model3(features_train, labels_train)

        optimal_threshold = self.find_optimal_stage1_threshold(features_val, labels_val)

        print("\nEvaluation finale :")
        self.evaluate_pipeline(features_test, labels_test, stage1_threshold=optimal_threshold)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='cvt-w24')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'])
    parser.add_argument('--use_finetuned', action='store_true', default=True)
    parser.add_argument('--use_augmentation', action='store_true', default=False)
    parser.add_argument('--augmented_csv', type=str, default='')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    finetuned_suffix = "finetuned" if args.use_finetuned else ""
    paths_to_try = [
        f'featuresextracted/{args.backbone}_{finetuned_suffix}_features_train.npy' if finetuned_suffix else f'featuresextracted/{args.backbone}_features_train.npy',
        f'featuresextracted/{args.backbone}_finetuned_features_train.npy',
        f'featuresextracted/{args.backbone}_features_train.npy'
    ]
    
    selected_train = None
    for p in paths_to_try:
        if os.path.exists(p):
            selected_train = p
            break
            
    if selected_train is None:
        selected_train = paths_to_try[0]
        
    prefix = selected_train.replace('_features_train.npy', '')
    feature_paths = {
        'train_features': f'{prefix}_features_train.npy',
        'train_labels': f'{prefix}_labels_train.npy',
        'test_features': f'{prefix}_features_test.npy',
        'test_labels': f'{prefix}_labels_test.npy'
    }

    pipeline = ThreeStagePipeline(args.backbone, device, feature_paths)

    if args.mode == 'train':
        if args.use_augmentation and args.augmented_csv:
            print(f"Utilisation du dataset augmenté : {args.augmented_csv}")
        pipeline.train_all_models()
    else:
        features_test, labels_test = pipeline.load_features('test')
        results = pipeline.evaluate_pipeline(features_test, labels_test)
        print(f"\nAccuracy finale : {results['accuracy']:.4f}")

if __name__ == '__main__':
    main()