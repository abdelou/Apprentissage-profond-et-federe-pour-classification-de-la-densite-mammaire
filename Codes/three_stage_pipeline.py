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
    """Bascule le système en mode 100% déterministe pour une reproductibilité exacte."""
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
    """
    Pipeline en 3 étapes pour la classification de densité mammographique:
    1. Modèle 1: (A,B) vs (C,D)
    2. Modèle 2: A vs B  
    3. Modèle 3: C vs D
    """
    
    def __init__(self, backbone, device, feature_paths):
        self.backbone = backbone
        self.device = device
        self.feature_paths = feature_paths
        
        # Mapping des classes
        self.class_map = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
        self.class_names = ["A", "B", "C", "D"]
        
        # Modèles de la pipeline
        self.model1 = None  # (A,B) vs (C,D)
        self.model2 = None  # A vs B
        self.model3 = None  # C vs D
        
    def load_features(self, split='train'):
        """Charge les features et labels pour un split donné"""
        features = np.load(self.feature_paths[f'{split}_features'])
        labels = np.load(self.feature_paths[f'{split}_labels'])
        return features, labels
    
    def prepare_binary_data(self, features, labels, class_pair):
        """
        Prépare les données pour un problème binaire
        """
        mask = np.isin(labels, class_pair)
        features_bin = features[mask]
        labels_bin = labels[mask]
        labels_bin = (labels_bin == class_pair[1]).astype(int)
        return features_bin, labels_bin
    
    def train_model1(self, features, labels, seed=42, force_retrain=True):
        """Entraîne le modèle 1: (A,B) vs (C,D) avec 30 époques, lr=1e-4 et seed paramétrable"""
        print(f"=== Entraînement Modèle 1 (Seed={seed}): Groupe (A,B) vs (C,D) ===")
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
            print(f"Poids du Modèle 1 chargés depuis: {model_path}")
        else:
            set_seed(seed)
            trainer.train(features_ab_cd, labels_ab_cd, epochs=30, 
                         batch_size=32, lr=1e-4, device=self.device)
            trainer.save(model_path)
            print(f"Modèle 1 (Seed={seed}) sauvegardé: {model_path}")
        
    def train_model2(self, features, labels):
        """Entraîne ou charge le modèle 2: Classe A vs Classe B"""
        print("=== Entraînement / Chargement Modèle 2: Classe A vs Classe B ===")
        features_bin, labels_bin = self.prepare_binary_data(features, labels, (0, 1))
        
        in_dim = features.shape[1]
        self.model2 = BinaryMLPClassifier(in_dim=in_dim)
        trainer = FeatureMLPTrainer(self.model2)
        
        model_path = f'./{self.backbone}_three_stage_model2.pth'
        if os.path.exists(model_path):
            trainer.load(model_path, device=self.device)
            print(f"Poids du Modèle 2 chargés depuis: {model_path}")
        else:
            trainer.train(features_bin, labels_bin, epochs=30, 
                         batch_size=32, lr=1e-4, device=self.device)
            trainer.save(model_path)
            print(f"Modèle 2 sauvegardé: {model_path}")
        
    def train_model3(self, features, labels):
        """Entraîne ou charge le modèle 3: Classe C vs Classe D"""
        print("=== Entraînement / Chargement Modèle 3: Classe C vs Classe D ===")
        features_bin, labels_bin = self.prepare_binary_data(features, labels, (2, 3))
        
        in_dim = features.shape[1]
        self.model3 = BinaryMLPClassifier(in_dim=in_dim)
        trainer = FeatureMLPTrainer(self.model3)
        
        model_path = f'./{self.backbone}_three_stage_model3.pth'
        if os.path.exists(model_path):
            trainer.load(model_path, device=self.device)
            print(f"Poids du Modèle 3 chargés depuis: {model_path}")
        else:
            trainer.train(features_bin, labels_bin, epochs=30, 
                         batch_size=32, lr=1e-4, device=self.device)
            trainer.save(model_path)
            print(f"Modèle 3 sauvegardé: {model_path}")
    
    def find_optimal_stage1_threshold(self, features_val, labels_val):
        """
        Calcule le seuil optimal pour le routage de l'Étape 1 (A,B vs C,D)
        en utilisant la courbe ROC et le critère de Youden (J = TPR - FPR) sur un jeu de validation.
        """
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
        # Critère de Youden : maximise (tpr - fpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print("\n=======================================================", flush=True)
        print("=== OPTIMISATION SEUIL ÉTAPE 1 (ROC & YOUDEN) ===", flush=True)
        print("=======================================================", flush=True)
        print(f"  -> Seuil optimal trouvé (sur validation): {optimal_threshold:.4f}", flush=True)
        print(f"  -> TPR (Rappel A/B sur validation): {tpr[optimal_idx]:.4f} ({tpr[optimal_idx]:.2%})", flush=True)
        print(f"  -> FPR (Faux positifs C/D sur validation): {fpr[optimal_idx]:.4f} ({fpr[optimal_idx]:.2%})", flush=True)
        
        return float(optimal_threshold)

    def predict_three_stage(self, features, stage1_threshold=0.5):
        """
        Prédiction avec la pipeline en 3 étapes avec seuil de routage configurable pour l'Étape 1
        """
        if self.model1 is not None: self.model1.eval()
        if self.model2 is not None: self.model2.eval()
        if self.model3 is not None: self.model3.eval()
        
        predictions = []
        probabilities = np.zeros((len(features), 4))
        confidence_scores = np.zeros((len(features), 3))  # Confiance pour chaque niveau
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Étape 1: (A,B) vs (C,D)
        with torch.no_grad():
            logits1 = self.model1(features_tensor)
            probs1 = torch.softmax(logits1, dim=1)
            stage1_conf = torch.max(probs1, dim=1)[0]  # Confiance du niveau 1
            is_ab = probs1[:, 0] > stage1_threshold  # Seuil de routage réglable
            
            # Préparation pour les étapes suivantes
            ab_indices = torch.where(is_ab)[0]
            cd_indices = torch.where(~is_ab)[0]
            
            # Stockage des scores de confiance niveau 1
            confidence_scores[:, 0] = stage1_conf.cpu().numpy()
        
        # Initialisation sécurisée des index de prédiction
        ab_is_a_idx = np.array([], dtype=int)
        ab_not_a_idx = np.array([], dtype=int)
        cd_is_c_idx = np.array([], dtype=int)
        cd_not_c_idx = np.array([], dtype=int)

        # Étape 2: A vs B pour les cas (A,B)
        if len(ab_indices) > 0:
            with torch.no_grad():
                logits2 = self.model2(features_tensor[ab_indices])
                probs2 = torch.softmax(logits2, dim=1)
                stage2_conf = torch.max(probs2, dim=1)[0]
                is_a = probs2[:, 0] > 0.5
                
                # Stockage des probabilités A et B
                ab_is_a_idx = ab_indices[is_a].cpu().numpy()
                ab_not_a_idx = ab_indices[~is_a].cpu().numpy()
                ab_idx_all = ab_indices.cpu().numpy()
                
                probabilities[ab_is_a_idx, 0] = probs2[is_a, 0].cpu().numpy()
                probabilities[ab_not_a_idx, 1] = probs2[~is_a, 1].cpu().numpy()
                
                # Stockage des scores de confiance niveau 2
                confidence_scores[ab_idx_all, 1] = stage2_conf.cpu().numpy()
        
        # Étape 3: C vs D pour les cas (C,D)
        if len(cd_indices) > 0:
            with torch.no_grad():
                logits3 = self.model3(features_tensor[cd_indices])
                probs3 = torch.softmax(logits3, dim=1)
                stage3_conf = torch.max(probs3, dim=1)[0]
                is_c = probs3[:, 0] > 0.5
                
                # Stockage des probabilités C et D
                cd_is_c_idx = cd_indices[is_c].cpu().numpy()
                cd_not_c_idx = cd_indices[~is_c].cpu().numpy()
                cd_idx_all = cd_indices.cpu().numpy()
                
                probabilities[cd_is_c_idx, 2] = probs3[is_c, 0].cpu().numpy()
                probabilities[cd_not_c_idx, 3] = probs3[~is_c, 1].cpu().numpy()
                
                # Stockage des scores de confiance niveau 3
                confidence_scores[cd_idx_all, 2] = stage3_conf.cpu().numpy()
        
        # Génération des prédictions finales
        final_preds = np.zeros(len(features))
        # Cas A/B
        final_preds[ab_is_a_idx] = 0  # A
        final_preds[ab_not_a_idx] = 1  # B
        # Cas C/D
        final_preds[cd_is_c_idx] = 2  # C
        final_preds[cd_not_c_idx] = 3  # D
        
        return final_preds.astype(int), probabilities, confidence_scores
    
    def evaluate_pipeline(self, features_test, labels_test, stage1_threshold=0.5):
        """Évalue la pipeline complète avec métriques détaillées et seuil de routage configurable"""
        print(f"=== Évaluation de la Pipeline (stage1_threshold={stage1_threshold:.4f}) ===")
        
        # Prédictions avec scores de confiance
        predictions, probabilities, confidence_scores = self.predict_three_stage(features_test, stage1_threshold=stage1_threshold)
        
        # Métriques globales
        accuracy = np.mean(predictions == labels_test)
        print(f"Accuracy globale: {accuracy:.4f}")
        
        # Analyse par niveau de décision
        print("\n=== Analyse par niveau de décision ===")
        
        # Niveau 1: (A,B) vs (C,D)
        true_ab = np.isin(labels_test, [0, 1])
        pred_ab = np.isin(predictions, [0, 1])
        level1_acc = np.mean(true_ab == pred_ab)
        
        # Diagnostic: Rappel du routage de l'étage 1 sur les vrais échantillons A/B
        features_tensor = torch.tensor(features_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits1 = self.model1(features_tensor)
            probs1 = torch.softmax(logits1, dim=1)
            routed_to_ab = (probs1[:, 0] > stage1_threshold).cpu().numpy()
        routing_recall_ab = np.mean(routed_to_ab[true_ab])
        
        print(f"\nNiveau 1 (A,B) vs (C,D):")
        print(f"Accuracy: {level1_acc:.4f}")
        print(f"Confiance moyenne: {np.mean(confidence_scores[:, 0]):.4f}")
        print(f"[DIAGNOSTIC ROUTAGE] Rappel de routage Étape 1 sur les vrais (A,B): {routing_recall_ab:.4f} ({routing_recall_ab:.2%})", flush=True)
        
        # Niveau 2: A vs B
        ab_mask = np.isin(labels_test, [0, 1])
        if np.any(ab_mask):
            level2_acc = np.mean(predictions[ab_mask] == labels_test[ab_mask])
            level2_conf = np.mean(confidence_scores[ab_mask, 1])
            print(f"\nNiveau 2 (A vs B):")
            print(f"Accuracy: {level2_acc:.4f}")
            print(f"Confiance moyenne: {level2_conf:.4f}")
        
        # Niveau 3: C vs D
        cd_mask = np.isin(labels_test, [2, 3])
        if np.any(cd_mask):
            level3_acc = np.mean(predictions[cd_mask] == labels_test[cd_mask])
            level3_conf = np.mean(confidence_scores[cd_mask, 2])
            print(f"\nNiveau 3 (C vs D):")
            print(f"Accuracy: {level3_acc:.4f}")
            print(f"Confiance moyenne: {level3_conf:.4f}")
        
        # Analyse des erreurs par niveau de confiance
        print("\n=== Analyse des erreurs par niveau de confiance ===")
        confidence_thresholds = [0.7, 0.8, 0.9]
        for threshold in confidence_thresholds:
            # Niveau 1
            high_conf_mask = confidence_scores[:, 0] > threshold
            if np.any(high_conf_mask):
                high_conf_acc = np.mean(true_ab[high_conf_mask] == pred_ab[high_conf_mask])
                print(f"\nConfiance > {threshold}:")
                print(f"Niveau 1 accuracy: {high_conf_acc:.4f}")
                print(f"Proportion des cas: {np.mean(high_conf_mask):.2%}")
        
        # Rapport de classification détaillé
        print("\n=== Rapport de classification détaillé ===")
        print(classification_report(labels_test, predictions, 
                                  target_names=self.class_names))
        
        # Matrices de confusion par niveau
        print("\n=== Matrices de confusion par niveau ===")
        
        # Matrice niveau 1
        cm1 = confusion_matrix(true_ab, pred_ab)
        print("Matrice niveau 1 (A,B) vs (C,D):")
        print(cm1)
        self.plot_confusion_matrix(cm1, "Niveau 1 (A,B) vs (C,D)", ["A,B", "C,D"])

        # Matrice globale
        cm = confusion_matrix(labels_test, predictions)
        print("Matrice globale (A/B/C/D):")
        print(cm)
        self.plot_confusion_matrix(cm, "Classification finale", self.class_names)
        
        # Exécution automatique du diagnostic isolé sur le Modèle 2
        self.evaluate_model2_isolated(features_test, labels_test)
        
        results = {'accuracy': accuracy, 'predictions': predictions}
        return results
    
    def evaluate_model2_isolated(self, features_test, labels_test):
        """
        Étape 1 — Diagnostic: Évalue le Modèle 2 (A vs B) seul, indépendamment du routage de l'étage 1
        """
        print("\n=======================================================", flush=True)
        print("=== ÉTAPE 1 - DIAGNOSTIC ISOLÉ : MODÈLE 2 (A vs B) SEUL ===", flush=True)
        print("=======================================================", flush=True)
        
        # Filtre directement les vrais échantillons A/B du test
        features_ab, labels_ab = self.prepare_binary_data(features_test, labels_test, (0, 1))
        
        if self.model2 is not None:
            self.model2.eval()
            
        features_tensor = torch.tensor(features_ab, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits2 = self.model2(features_tensor)
            probs2 = torch.softmax(logits2, dim=1)
            preds_m2 = (probs2[:, 1] > 0.5).cpu().numpy().astype(int)  # 0=A, 1=B
            
        acc_m2 = np.mean(preds_m2 == labels_ab)
        print(f"Accuracy Modèle 2 isolé (A vs B): {acc_m2:.4f} ({acc_m2:.2%})", flush=True)
        
        print("\nRapport de classification Modèle 2 isolé (A vs B):", flush=True)
        print(classification_report(labels_ab, preds_m2, target_names=["A", "B"]), flush=True)
        
        cm_m2 = confusion_matrix(labels_ab, preds_m2)
        print("Matrice de confusion Modèle 2 isolé (A vs B):", flush=True)
        print(cm_m2, flush=True)
        
        return acc_m2
    
    def plot_confusion_matrix(self, cm, title, class_names=None):
        """
        Affiche et sauvegarde la matrice de confusion
        
        Args:
            cm: Matrice de confusion
            title: Titre du graphique
            class_names: Noms des classes à utiliser (utilise self.class_names par défaut)
        """
        if class_names is None:
            class_names = self.class_names
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'Matrice de confusion - {title}')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        
        # Sauvegarde avec un nom spécifique pour chaque niveau
        output_dir = Path("graphes")
        output_dir.mkdir(exist_ok=True)
        safe_title = title.replace(" ", "_").lower()
        plt.savefig(output_dir / f'{self.backbone}_three_stage_{safe_title}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_with_cross_validation(self, features, labels, n_folds=5):
        """
        Entraîne les modèles avec validation croisée et optimisation des hyperparamètres
        """
        from sklearn.model_selection import KFold
        from sklearn.model_selection import StratifiedKFold
        import itertools
        
        print("=== Entraînement avec validation croisée ===")
        
        # Hyperparamètres à tester
        lr_values = [1e-4, 1e-3, 5e-4]
        batch_sizes = [16, 32, 64]
        architectures = [(512, 256), (1024, 512), (256, 128, 64)]
        
        # Meilleurs hyperparamètres pour chaque niveau
        best_params = {
            'model1': {'lr': None, 'batch_size': None, 'architecture': None, 'score': 0},
            'model2': {'lr': None, 'batch_size': None, 'architecture': None, 'score': 0},
            'model3': {'lr': None, 'batch_size': None, 'architecture': None, 'score': 0}
        }
        
        # Validation croisée stratifiée
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Pour chaque combinaison d'hyperparamètres
        param_combinations = list(itertools.product(lr_values, batch_sizes, architectures))
        
        for model_name, class_pair, is_first_level in [
            ('model1 (AB vs CD)', (0, 1), True),
            ('model2 (A vs B)', (0, 1), False),
            ('model3 (C vs D)', (2, 3), False)
        ]:
            print(f"\n[HIERARCHICAL] Optimisation des hyperparamètres pour {model_name}...", flush=True)
            
            best_score = 0
            best_params_model = None
            
            for combo_idx, (lr, batch_size, architecture) in enumerate(param_combinations, 1):
                fold_scores = []
                print(f"  -> Combinaison {combo_idx}/{len(param_combinations)} (lr={lr}, batch={batch_size}, arch={architecture})...", flush=True)
                
                for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
                    X_train, X_val = features[train_idx], features[val_idx]
                    y_train, y_val = labels[train_idx], labels[val_idx]
                    
                    # Préparation des données binaires pour le modèle actuel
                    if is_first_level:
                        # Pour model1: regrouper (A,B) vs (C,D)
                        y_train_bin = np.isin(y_train, [0, 1]).astype(int)
                        y_val_bin = np.isin(y_val, [0, 1]).astype(int)
                    else:
                        # Pour model2 et model3: filtrer les classes pertinentes
                        mask_train = np.isin(y_train, class_pair)
                        mask_val = np.isin(y_val, class_pair)
                        
                        X_train = X_train[mask_train]
                        X_val = X_val[mask_val]
                        y_train_bin = (y_train[mask_train] == class_pair[1]).astype(int)
                        y_val_bin = (y_val[mask_val] == class_pair[1]).astype(int)
                    
                    # Création et entraînement du modèle
                    model = BinaryMLPClassifier(in_dim=features.shape[1], 
                                              hidden_dims=architecture)
                    trainer = FeatureMLPTrainer(model)
                    
                    # Entraînement
                    trainer.train(X_train, y_train_bin, epochs=10,
                                batch_size=batch_size, lr=lr,
                                device=self.device,
                                validation_data=(X_val, y_val_bin))
                    
                    # Évaluation
                    val_score = trainer.evaluate(X_val, y_val_bin)
                    fold_scores.append(val_score)
                
                # Score moyen sur les folds
                mean_score = np.mean(fold_scores)
                print(f"lr={lr}, batch_size={batch_size}, architecture={architecture}")
                print(f"Score moyen: {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params_model = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'architecture': architecture,
                        'score': mean_score
                    }
            
            best_params[model_name] = best_params_model
            print(f"\nMeilleurs paramètres pour {model_name}:")
            print(best_params[model_name])
        
        # Entraînement final avec les meilleurs paramètres
        print("\n=== Entraînement final avec les meilleurs paramètres ===")
        
        # Model 1: (A,B) vs (C,D)
        self.model1 = BinaryMLPClassifier(
            in_dim=features.shape[1],
            hidden_dims=best_params['model1']['architecture']
        )
        trainer1 = FeatureMLPTrainer(self.model1)
        trainer1.train(features, np.isin(labels, [0, 1]).astype(int),
                      epochs=10,
                      batch_size=best_params['model1']['batch_size'],
                      lr=best_params['model1']['lr'],
                      device=self.device)
        
        # Model 2: A vs B
        features_ab = features[np.isin(labels, [0, 1])]
        labels_ab = labels[np.isin(labels, [0, 1])]
        self.model2 = BinaryMLPClassifier(
            in_dim=features.shape[1],
            hidden_dims=best_params['model2']['architecture']
        )
        trainer2 = FeatureMLPTrainer(self.model2)
        trainer2.train(features_ab, (labels_ab == 1).astype(int),
                      epochs=10,
                      batch_size=best_params['model2']['batch_size'],
                      lr=best_params['model2']['lr'],
                      device=self.device)
        
        # Model 3: C vs D
        features_cd = features[np.isin(labels, [2, 3])]
        labels_cd = labels[np.isin(labels, [2, 3])]
        self.model3 = BinaryMLPClassifier(
            in_dim=features.shape[1],
            hidden_dims=best_params['model3']['architecture']
        )
        trainer3 = FeatureMLPTrainer(self.model3)
        trainer3.train(features_cd, (labels_cd == 3).astype(int),
                      epochs=10,
                      batch_size=best_params['model3']['batch_size'],
                      lr=best_params['model3']['lr'],
                      device=self.device)
        
        return best_params
    
    def run_multi_seed_experiment(self, seeds=[42, 43, 44, 45, 46]):
        """
        Exécute l'entraînement du Modèle 1 sur 5 seeds différentes (42, 43, 44, 45, 46)
        et calcule la moyenne et l'écart-type des 3 métriques clés.
        """
        features_train, labels_train = self.load_features('train')
        features_test, labels_test = self.load_features('test')
        
        # S'assurer que le Modèle 2 et le Modèle 3 sont chargés (sans ré-entraînement)
        self.train_model2(features_train, labels_train)
        self.train_model3(features_train, labels_train)
        
        results_by_seed = []
        
        print("\n=======================================================", flush=True)
        print("=== EXPÉRIMENTATION MULTI-SEEDS DU MODÈLE 1 (5 RUNS) ===", flush=True)
        print("=======================================================", flush=True)
        
        for seed in seeds:
            # 1. Entraînement du Modèle 1 avec la seed spécifique
            self.train_model1(features_train, labels_train, seed=seed, force_retrain=True)
            
            # 2. Activation déterministe d'inférence (.eval())
            if self.model1 is not None: self.model1.eval()
            if self.model2 is not None: self.model2.eval()
            if self.model3 is not None: self.model3.eval()
            
            # 3. Prédictions avec le seuil par défaut 0.5
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
            
            print(f"[Seed {seed:2d}] Niveau 1 Acc: {level1_acc:.4f} | Rappel AB Routage: {routing_recall_ab:.4f} ({routing_recall_ab:.2%}) | Accuracy Globale: {overall_acc:.4f}", flush=True)
            
        # Calcul Moyennes et Écarts-types
        level1_accs = [r['level1_acc'] for r in results_by_seed]
        routing_recalls = [r['routing_recall_ab'] for r in results_by_seed]
        overall_accs = [r['overall_acc'] for r in results_by_seed]
        
        mean_l1, std_l1 = np.mean(level1_accs), np.std(level1_accs)
        mean_rec, std_rec = np.mean(routing_recalls), np.std(routing_recalls)
        mean_ov, std_ov = np.mean(overall_accs), np.std(overall_accs)
        
        print("\n=======================================================", flush=True)
        print("=== SYNTHÈSE STATISTIQUE SUR LES 5 SEEDS ===", flush=True)
        print("=======================================================", flush=True)
        print(f"  -> Accuracy Niveau 1 (AB vs CD) : {mean_l1:.4f} ± {std_l1:.4f} ({mean_l1:.2%} ± {std_l1:.2%})", flush=True)
        print(f"  -> Rappel Routage Étape 1 (AB)   : {mean_rec:.4f} ± {std_rec:.4f} ({mean_rec:.2%} ± {std_rec:.2%})", flush=True)
        print(f"  -> Accuracy Globale Pipeline     : {mean_ov:.4f} ± {std_ov:.4f} ({mean_ov:.2%} ± {std_ov:.2%})", flush=True)
        print("=======================================================\n", flush=True)

    def train_all_models(self):
        """Entraîne directement les 3 modèles de la pipeline hiérarchique (AB vs CD / A vs B / C vs D)"""
        print(f"=== Entraînement Direct de la Pipeline Hiérarchique avec Backbone: {self.backbone} ===", flush=True)

        # Chargement des données d'entraînement et de test
        features_train_full, labels_train_full = self.load_features('train')
        features_test, labels_test = self.load_features('test')

        # Découpage d'un split de validation interne au train (80/20, seed=42, même
        # convention que le reste du projet) : sert uniquement à choisir le seuil de
        # routage de l'Étape 1, jamais le jeu de test.
        indices = np.arange(len(labels_train_full))
        np.random.seed(42)
        val_size = int(0.2 * len(indices))
        val_indices = np.random.choice(indices, size=val_size, replace=False)
        train_indices = np.setdiff1d(indices, val_indices)

        features_train = features_train_full[train_indices]
        labels_train = labels_train_full[train_indices]
        features_val = features_train_full[val_indices]
        labels_val = labels_train_full[val_indices]

        print(f"[SPLIT] Train interne: {len(train_indices)} · Validation interne: {len(val_indices)} · Test (jamais utilisé avant l'évaluation finale): {len(labels_test)}", flush=True)

        # Repartir d'entraînements propres pour ce split (les anciens checkpoints
        # correspondaient à un entraînement sur 100% du train, désormais incohérent)
        for i in (1, 2, 3):
            stale_path = f'./{self.backbone}_three_stage_model{i}.pth'
            if os.path.exists(stale_path):
                os.remove(stale_path)

        # 1. Modèle 1: (A,B) vs (C,D)
        print("\n--- 1. Entraînement Étape 1: Groupe (A,B) vs (C,D) ---", flush=True)
        self.train_model1(features_train, labels_train, seed=42, force_retrain=True)

        # 2. Modèle 2: A vs B
        print("\n--- 2. Entraînement Étape 2: Classe A vs Classe B ---", flush=True)
        self.train_model2(features_train, labels_train)

        # 3. Modèle 3: C vs D
        print("\n--- 3. Entraînement Étape 3: Classe C vs Classe D ---", flush=True)
        self.train_model3(features_train, labels_train)

        # 4. Choix du seuil de routage de l'Étape 1 sur la validation interne (ROC + Youden),
        # jamais sur le test.
        optimal_threshold = self.find_optimal_stage1_threshold(features_val, labels_val)

        # 5. Évaluation finale unique sur le test, avec le seuil déjà fixé.
        print("\n=======================================================", flush=True)
        print(f"=== ÉVALUATION PIPELINE HIÉRARCHIQUE ({self.backbone.upper()}) ===", flush=True)
        print("=======================================================", flush=True)
        self.evaluate_pipeline(features_test, labels_test, stage1_threshold=optimal_threshold)

        print("\nEntraînement et Évaluation terminés avec succès !", flush=True)

def main():
    """Fonction principale"""
    set_seed(42)
    parser = argparse.ArgumentParser(description='Pipeline en 3 étapes pour classification mammographique')
    parser.add_argument('--backbone', type=str, default='cvt-w24',
                        help='Backbone à utiliser')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Mode: train ou evaluate')
    parser.add_argument('--use_finetuned', action='store_true', default=True,
                        help='Utiliser les features fine-tunées')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                        help='Utiliser les features extraites à partir du dataset augmenté')
    parser.add_argument('--augmented_csv', type=str, default='',
                        help='Chemin vers le CSV d\'annotations augmenté (optionnel)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Construction des chemins de features en vérifiant les fichiers existants
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

    # Création de la pipeline
    pipeline = ThreeStagePipeline(args.backbone, device, feature_paths)

    if args.mode == 'train':
        # Si l'utilisateur veut utiliser un CSV augmenté pour l'extraction des features,
        # il faut générer les features augmentées avant d'appeler train_all_models.
        if args.use_augmentation and args.augmented_csv:
            print(f"[INFO] Vous avez demandé l'utilisation du dataset augmenté: {args.augmented_csv}")
            print("[INFO] Assurez-vous d'avoir généré les fichiers .npy de features correspondants avant de lancer l'entraînement.")
        pipeline.train_all_models()
    else:  # evaluate
        # Chargement des données de test
        features_test, labels_test = pipeline.load_features('test')

        # Évaluation
        results = pipeline.evaluate_pipeline(features_test, labels_test)
        print(f"\nRésultats finaux:")
        print(f"Accuracy: {results['accuracy']:.4f}")

if __name__ == '__main__':
    main() 