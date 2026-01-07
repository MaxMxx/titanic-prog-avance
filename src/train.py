"""
Script d'entraînement des modèles pour le projet Titanic
=========================================================
Auteur: Personne 3 - Équipe Titanic
Date: Janvier 2026

Ce script permet d'entraîner et d'évaluer plusieurs modèles de classification
sur les données prétraitées du Titanic.

Usage:
    python train.py
    python train.py --model random_forest
    python train.py --model all --cv 10
"""

import argparse
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Import du module de prétraitement
from preprocessing import load_data, preprocess_train

# Configuration globale
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Chemins par défaut
DATA_RAW_PATH = "../data/raw/train.csv"
MODELS_PATH = "../models"
RESULTS_PATH = "../models"


def get_models():
    """
    Retourne un dictionnaire de modèles à entraîner.
    
    Returns
    -------
    dict : Dictionnaire {nom: modèle sklearn}
    """
    return {
        'logistic_regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE
        ),
        'svm': SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=RANDOM_STATE
        )
    }


def evaluate_model_cv(model, X, y, cv, model_name="Model"):
    """
    Évalue un modèle avec validation croisée sur plusieurs métriques.
    
    Parameters
    ----------
    model : estimator
        Modèle sklearn à évaluer
    X : array-like
        Features d'entraînement
    y : array-like
        Variable cible
    cv : cross-validation strategy
        Stratégie de validation croisée
    model_name : str
        Nom du modèle pour l'affichage
    
    Returns
    -------
    results : dict
        Dictionnaire contenant les scores moyens et écarts-types
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )
    
    results = {
        'model': model_name,
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'precision_mean': cv_results['test_precision'].mean(),
        'precision_std': cv_results['test_precision'].std(),
        'recall_mean': cv_results['test_recall'].mean(),
        'recall_std': cv_results['test_recall'].std(),
        'f1_mean': cv_results['test_f1'].mean(),
        'f1_std': cv_results['test_f1'].std(),
        'roc_auc_mean': cv_results['test_roc_auc'].mean(),
        'roc_auc_std': cv_results['test_roc_auc'].std(),
    }
    
    return results


def evaluate_on_test(model, X_test, y_test, model_name="Model"):
    """
    Évalue un modèle sur le test set.
    
    Parameters
    ----------
    model : estimator
        Modèle sklearn entraîné
    X_test : array-like
        Features de test
    y_test : array-like
        Variable cible de test
    model_name : str
        Nom du modèle
    
    Returns
    -------
    results : dict
        Dictionnaire des métriques
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }


def print_results(results, title="Résultats"):
    """Affiche les résultats de manière formatée."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(f"{'Métrique':<15} {'Score':>15}")
    print('-'*35)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        if f'{metric}_mean' in results:
            print(f"{metric.capitalize():<15} {results[f'{metric}_mean']:>10.4f} ± {results[f'{metric}_std']:.4f}")
        elif metric in results:
            print(f"{metric.capitalize():<15} {results[metric]:>15.4f}")


def train_and_evaluate(
    data_path=DATA_RAW_PATH,
    model_names=None,
    n_splits=5,
    test_size=0.2,
    save_models=True
):
    """
    Pipeline complet d'entraînement et d'évaluation.
    
    Parameters
    ----------
    data_path : str
        Chemin vers les données brutes
    model_names : list or None
        Liste des modèles à entraîner. Si None, tous les modèles.
    n_splits : int
        Nombre de folds pour la validation croisée
    test_size : float
        Proportion du test set
    save_models : bool
        Sauvegarder les modèles entraînés
    
    Returns
    -------
    results_df : DataFrame
        Résultats de tous les modèles
    """
    print("="*70)
    print("SCRIPT D'ENTRAÎNEMENT - PROJET TITANIC")
    print("="*70)
    
    # 1. Chargement et prétraitement des données
    print("\n[1/5] Chargement des données...")
    train_raw, _ = load_data(data_path)
    
    print("\n[2/5] Prétraitement des données...")
    train_processed, encoders = preprocess_train(train_raw)
    
    # 2. Séparation features / cible
    X = train_processed.drop('Survived', axis=1)
    y = train_processed['Survived']
    
    # 3. Split train/test stratifié
    print(f"\n[3/5] Séparation train/test ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"  - Train: {X_train.shape[0]} échantillons")
    print(f"  - Test:  {X_test.shape[0]} échantillons")
    
    # 4. Configuration validation croisée
    cv_strategy = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    # 5. Sélection des modèles
    all_models = get_models()
    
    if model_names is None or 'all' in model_names:
        models_to_train = all_models
    else:
        models_to_train = {k: v for k, v in all_models.items() if k in model_names}
    
    if not models_to_train:
        print(f"Erreur: Aucun modèle trouvé. Modèles disponibles: {list(all_models.keys())}")
        return None
    
    # 6. Entraînement et évaluation
    print(f"\n[4/5] Entraînement de {len(models_to_train)} modèle(s)...")
    
    cv_results = []
    test_results = []
    trained_models = {}
    
    for name, model in models_to_train.items():
        print(f"\n  → {name}...")
        
        # Validation croisée
        cv_result = evaluate_model_cv(model, X_train, y_train, cv_strategy, name)
        cv_results.append(cv_result)
        
        # Entraînement final sur tout le train set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Évaluation sur test set
        test_result = evaluate_on_test(model, X_test, y_test, name)
        test_results.append(test_result)
        
        print_results(test_result, f"Résultats sur Test Set - {name}")
    
    # 7. Sauvegarde
    print(f"\n[5/5] Sauvegarde des résultats...")
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    if save_models:
        for name, model in trained_models.items():
            filepath = os.path.join(MODELS_PATH, f"{name}.joblib")
            joblib.dump(model, filepath)
            print(f"  ✓ Modèle sauvegardé: {filepath}")
    
    # Sauvegarder les résultats
    results_df = pd.DataFrame(test_results)
    results_path = os.path.join(RESULTS_PATH, "model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  ✓ Résultats sauvegardés: {results_path}")
    
    # Sauvegarder les données train/test
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X_train.columns)
    }
    data_path = os.path.join(MODELS_PATH, "train_test_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"  ✓ Données sauvegardées: {data_path}")
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    
    best_idx = results_df['f1'].idxmax()
    best_model = results_df.loc[best_idx, 'model']
    best_f1 = results_df.loc[best_idx, 'f1']
    
    print(f"\n🏆 Meilleur modèle: {best_model}")
    print(f"   F1-Score: {best_f1:.4f}")
    
    print("\nClassement par F1-Score:")
    ranking = results_df.sort_values('f1', ascending=False)
    for i, (_, row) in enumerate(ranking.iterrows(), 1):
        print(f"  {i}. {row['model']}: {row['f1']:.4f}")
    
    print("\n✓ Entraînement terminé avec succès!")
    
    return results_df


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Entraînement des modèles pour le projet Titanic"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'logistic_regression', 'random_forest', 
                 'gradient_boosting', 'svm'],
        help="Modèle(s) à entraîner (default: all)"
    )
    
    parser.add_argument(
        '--cv', '-c',
        type=int,
        default=5,
        help="Nombre de folds pour la validation croisée (default: 5)"
    )
    
    parser.add_argument(
        '--test-size', '-t',
        type=float,
        default=0.2,
        help="Proportion du test set (default: 0.2)"
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=DATA_RAW_PATH,
        help=f"Chemin vers les données (default: {DATA_RAW_PATH})"
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Ne pas sauvegarder les modèles"
    )
    
    args = parser.parse_args()
    
    train_and_evaluate(
        data_path=args.data,
        model_names=args.model,
        n_splits=args.cv,
        test_size=args.test_size,
        save_models=not args.no_save
    )


if __name__ == "__main__":
    main()
