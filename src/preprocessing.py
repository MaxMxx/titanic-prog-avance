"""
Module de prétraitement pour le dataset Titanic
================================================
Auteur: Personne 2 - Équipe Titanic
Date: Janvier 2025

Ce module contient toutes les fonctions et classes nécessaires pour 
nettoyer et transformer les données du Titanic en vue de la modélisation.

Stratégies appliquées:
- Gestion des valeurs manquantes (Age, Cabin, Embarked)
- Encodage des variables catégorielles
- Feature engineering (extraction du titre, taille de famille, etc.)
- Mise à l'échelle des variables numériques
- Pipeline sklearn réutilisable
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import re
import warnings
warnings.filterwarnings('ignore')

# Configuration globale
RANDOM_STATE = 42


# =============================================================================
# 1. CLASSES DE TRANSFORMATEURS PERSONNALISÉS
# =============================================================================

class TitleExtractor(BaseEstimator, TransformerMixin):
    """
    Extracteur de titre à partir du nom des passagers.
    
    Les titres sont regroupés en catégories:
    - Mr, Mrs, Miss, Master (titres courants)
    - Rare (tous les autres titres nobles ou rares)
    
    Justification: Le titre reflète le statut social et l'âge,
    deux facteurs fortement corrélés à la survie (EDA insight).
    """
    
    # Mapping des titres vers des catégories simplifiées
    TITLE_MAPPING = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',      # Mademoiselle -> Miss
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',        # Madame -> Mrs
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Title'] = X['Name'].apply(self._extract_title)
        return X
    
    def _extract_title(self, name):
        """Extrait et mappe le titre d'un nom."""
        # Pattern: "Lastname, Title. Firstname"
        title_search = re.search(r' ([A-Za-z]+)\.', name)
        if title_search:
            title = title_search.group(1)
            return self.TITLE_MAPPING.get(title, 'Rare')
        return 'Rare'


class FamilySizeCreator(BaseEstimator, TransformerMixin):
    """
    Crée des features liées à la taille de la famille.
    
    Features créées:
    - FamilySize: SibSp + Parch + 1 (la personne elle-même)
    - IsAlone: 1 si FamilySize == 1, sinon 0
    - FamilySizeGroup: catégorisation (Alone, Small, Medium, Large)
    
    Justification: L'EDA montre que voyager seul ou en famille
    impacte les chances de survie.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Taille totale de la famille
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        
        # Indicateur "seul"
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        
        # Catégorisation de la taille
        X['FamilySizeGroup'] = X['FamilySize'].apply(self._categorize_family)
        
        return X
    
    def _categorize_family(self, size):
        """Catégorise la taille de famille."""
        if size == 1:
            return 'Alone'
        elif size <= 3:
            return 'Small'
        elif size <= 5:
            return 'Medium'
        else:
            return 'Large'


class CabinProcessor(BaseEstimator, TransformerMixin):
    """
    Traite la variable Cabin (77% de valeurs manquantes).
    
    Stratégie:
    - Extraire la lettre du pont (deck) quand disponible
    - Créer un indicateur HasCabin (avoir une cabine = statut élevé)
    - Remplacer les NA par 'Unknown'
    
    Justification: Bien que très incomplète, la présence d'une cabine
    est un proxy du statut social (1ère classe avait plus de cabines documentées).
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Indicateur de présence de cabine
        X['HasCabin'] = X['Cabin'].notna().astype(int)
        
        # Extraction du pont (première lettre)
        X['Deck'] = X['Cabin'].apply(self._extract_deck)
        
        return X
    
    def _extract_deck(self, cabin):
        """Extrait la lettre du pont."""
        if pd.isna(cabin):
            return 'Unknown'
        return cabin[0]


class AgeGroupCreator(BaseEstimator, TransformerMixin):
    """
    Crée des groupes d'âge après imputation.
    
    Catégories:
    - Child: 0-12 ans (priorité d'évacuation)
    - Teen: 13-19 ans
    - Adult: 20-59 ans
    - Senior: 60+ ans
    
    Justification: "Women and children first" - l'âge catégorisé
    capture mieux cette politique d'évacuation.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['AgeGroup'] = X['Age'].apply(self._categorize_age)
        return X
    
    def _categorize_age(self, age):
        """Catégorise l'âge."""
        if pd.isna(age):
            return 'Unknown'
        elif age < 13:
            return 'Child'
        elif age < 20:
            return 'Teen'
        elif age < 60:
            return 'Adult'
        else:
            return 'Senior'


class FareBinner(BaseEstimator, TransformerMixin):
    """
    Crée des catégories de tarif basées sur les quartiles.
    
    Justification: Le tarif a une distribution très asymétrique
    avec des outliers. La catégorisation réduit leur impact.
    """
    
    def __init__(self):
        self.quartiles = None
    
    def fit(self, X, y=None):
        # Calculer les quartiles sur les données d'entraînement
        self.quartiles = X['Fare'].quantile([0.25, 0.5, 0.75]).values
        return self
    
    def transform(self, X):
        X = X.copy()
        X['FareGroup'] = X['Fare'].apply(self._categorize_fare)
        return X
    
    def _categorize_fare(self, fare):
        """Catégorise le tarif en quartiles."""
        if pd.isna(fare):
            return 'Unknown'
        elif fare <= self.quartiles[0]:
            return 'Low'
        elif fare <= self.quartiles[1]:
            return 'Medium'
        elif fare <= self.quartiles[2]:
            return 'High'
        else:
            return 'VeryHigh'


# =============================================================================
# 2. FONCTIONS DE PRÉTRAITEMENT
# =============================================================================

def load_data(train_path, test_path=None):
    """
    Charge les données brutes.
    
    Parameters
    ----------
    train_path : str
        Chemin vers le fichier train.csv
    test_path : str, optional
        Chemin vers le fichier test.csv
    
    Returns
    -------
    train_df : DataFrame
    test_df : DataFrame or None
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    
    print(f"Données chargées:")
    print(f"  - Train: {train_df.shape[0]} lignes, {train_df.shape[1]} colonnes")
    if test_df is not None:
        print(f"  - Test: {test_df.shape[0]} lignes, {test_df.shape[1]} colonnes")
    
    return train_df, test_df


def analyze_missing_values(df, name="Dataset"):
    """
    Analyse et affiche les valeurs manquantes.
    
    Parameters
    ----------
    df : DataFrame
    name : str
        Nom du dataset pour l'affichage
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Manquantes': missing,
        'Pourcentage': missing_pct
    })
    missing_df = missing_df[missing_df['Manquantes'] > 0].sort_values(
        'Pourcentage', ascending=False
    )
    
    print(f"\n{'='*50}")
    print(f"Valeurs manquantes - {name}")
    print('='*50)
    
    if len(missing_df) == 0:
        print("Aucune valeur manquante!")
    else:
        for col, row in missing_df.iterrows():
            print(f"  {col}: {int(row['Manquantes'])} ({row['Pourcentage']:.1f}%)")
    
    return missing_df


def impute_age(df, strategy='median_by_title'):
    """
    Impute les valeurs manquantes de Age.
    
    Stratégies disponibles:
    - 'median': médiane globale
    - 'median_by_title': médiane par titre (plus précis)
    - 'median_by_class': médiane par classe
    
    Justification du choix median_by_title:
    - Un "Master" (jeune garçon) n'a pas le même âge qu'un "Mr"
    - Cette stratégie préserve la relation titre-âge
    
    Parameters
    ----------
    df : DataFrame
    strategy : str
    
    Returns
    -------
    df : DataFrame avec Age imputé
    """
    df = df.copy()
    
    # S'assurer que le titre existe
    if 'Title' not in df.columns:
        title_extractor = TitleExtractor()
        df = title_extractor.transform(df)
    
    if strategy == 'median':
        median_age = df['Age'].median()
        df['Age'].fillna(median_age, inplace=True)
        print(f"  Age imputé avec médiane globale: {median_age:.1f}")
        
    elif strategy == 'median_by_title':
        age_by_title = df.groupby('Title')['Age'].median()
        
        for title in df['Title'].unique():
            mask = (df['Age'].isnull()) & (df['Title'] == title)
            median_age = age_by_title.get(title, df['Age'].median())
            df.loc[mask, 'Age'] = median_age
        
        print("  Age imputé par médiane selon le titre:")
        for title, age in age_by_title.items():
            print(f"    - {title}: {age:.1f} ans")
            
    elif strategy == 'median_by_class':
        age_by_class = df.groupby('Pclass')['Age'].median()
        
        for pclass in df['Pclass'].unique():
            mask = (df['Age'].isnull()) & (df['Pclass'] == pclass)
            df.loc[mask, 'Age'] = age_by_class[pclass]
        
        print("  Age imputé par médiane selon la classe")
    
    return df


def impute_embarked(df):
    """
    Impute les valeurs manquantes de Embarked.
    
    Stratégie: Mode (valeur la plus fréquente = 'S' Southampton)
    
    Justification: Seulement 2 valeurs manquantes, le mode est 
    la stratégie la plus simple et raisonnable.
    
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    df : DataFrame avec Embarked imputé
    """
    df = df.copy()
    
    mode_embarked = df['Embarked'].mode()[0]
    n_missing = df['Embarked'].isnull().sum()
    
    df['Embarked'].fillna(mode_embarked, inplace=True)
    
    print(f"  Embarked: {n_missing} valeurs imputées avec '{mode_embarked}'")
    
    return df


def impute_fare(df):
    """
    Impute les valeurs manquantes de Fare.
    
    Stratégie: Médiane par classe (Fare dépend fortement de Pclass)
    
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    df : DataFrame avec Fare imputé
    """
    df = df.copy()
    
    n_missing = df['Fare'].isnull().sum()
    
    if n_missing > 0:
        fare_by_class = df.groupby('Pclass')['Fare'].median()
        
        for pclass in df['Pclass'].unique():
            mask = (df['Fare'].isnull()) & (df['Pclass'] == pclass)
            df.loc[mask, 'Fare'] = fare_by_class[pclass]
        
        print(f"  Fare: {n_missing} valeurs imputées par médiane de classe")
    else:
        print("  Fare: aucune valeur manquante")
    
    return df


def create_features(df):
    """
    Applique tout le feature engineering.
    
    Features créées:
    1. Title (depuis Name)
    2. FamilySize, IsAlone, FamilySizeGroup (depuis SibSp, Parch)
    3. HasCabin, Deck (depuis Cabin)
    4. AgeGroup (depuis Age)
    
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    df : DataFrame avec nouvelles features
    """
    df = df.copy()
    
    print("\nCréation des features...")
    
    # 1. Extraire le titre
    title_extractor = TitleExtractor()
    df = title_extractor.transform(df)
    print("  ✓ Title extrait du nom")
    
    # 2. Features de famille
    family_creator = FamilySizeCreator()
    df = family_creator.transform(df)
    print("  ✓ FamilySize, IsAlone, FamilySizeGroup créés")
    
    # 3. Features de cabine
    cabin_processor = CabinProcessor()
    df = cabin_processor.transform(df)
    print("  ✓ HasCabin, Deck créés")
    
    # 4. Groupes d'âge
    age_creator = AgeGroupCreator()
    df = age_creator.transform(df)
    print("  ✓ AgeGroup créé")
    
    return df


def encode_categorical(df, columns=None):
    """
    Encode les variables catégorielles.
    
    Stratégie:
    - Sex: Label Encoding (binaire)
    - Autres: One-Hot Encoding
    
    Parameters
    ----------
    df : DataFrame
    columns : list, optional
        Colonnes à encoder. Si None, détection automatique.
    
    Returns
    -------
    df : DataFrame encodé
    encoders : dict de LabelEncoders utilisés
    """
    df = df.copy()
    encoders = {}
    
    print("\nEncodage des variables catégorielles...")
    
    # Sex -> Label Encoding (0/1)
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    encoders['Sex'] = le_sex
    print(f"  Sex: {dict(zip(le_sex.classes_, range(len(le_sex.classes_))))}")
    
    # Colonnes à encoder en One-Hot
    onehot_columns = ['Embarked', 'Title', 'FamilySizeGroup', 'Deck', 'AgeGroup']
    onehot_columns = [c for c in onehot_columns if c in df.columns]
    
    for col in onehot_columns:
        # One-Hot encoding
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        print(f"  {col}: {len(dummies.columns)} nouvelles colonnes")
    
    return df, encoders


def select_features(df, drop_columns=None):
    """
    Sélectionne les features finales pour la modélisation.
    
    Colonnes supprimées par défaut:
    - PassengerId: identifiant
    - Name: texte brut (titre extrait)
    - Ticket: trop de valeurs uniques, peu informatif
    - Cabin: remplacé par HasCabin et Deck
    - Colonnes catégorielles originales (encodées)
    
    Parameters
    ----------
    df : DataFrame
    drop_columns : list, optional
        Colonnes supplémentaires à supprimer
    
    Returns
    -------
    df : DataFrame avec features sélectionnées
    """
    df = df.copy()
    
    # Colonnes à supprimer
    cols_to_drop = [
        'PassengerId', 'Name', 'Ticket', 'Cabin',  # Non utiles
        'Embarked', 'Title', 'FamilySizeGroup', 'Deck', 'AgeGroup'  # Encodées
    ]
    
    if drop_columns:
        cols_to_drop.extend(drop_columns)
    
    # Ne supprimer que les colonnes présentes
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    df.drop(columns=cols_to_drop, inplace=True)
    
    print(f"\nColonnes supprimées: {cols_to_drop}")
    print(f"Features finales: {list(df.columns)}")
    
    return df


def preprocess_train(df, save_path=None):
    """
    Pipeline complet de prétraitement pour les données d'entraînement.
    
    Étapes:
    1. Analyse des valeurs manquantes
    2. Feature engineering (crée Title d'abord)
    3. Imputation des valeurs manquantes
    4. Encodage des variables catégorielles
    5. Sélection des features
    
    Parameters
    ----------
    df : DataFrame
        Données brutes d'entraînement
    save_path : str, optional
        Chemin pour sauvegarder les données prétraitées
    
    Returns
    -------
    df_processed : DataFrame
        Données prêtes pour la modélisation
    encoders : dict
        Encodeurs pour réutilisation sur le test set
    """
    print("="*60)
    print("PRÉTRAITEMENT DES DONNÉES D'ENTRAÎNEMENT")
    print("="*60)
    
    # 1. Analyse initiale
    analyze_missing_values(df, "Avant prétraitement")
    
    # 2. Feature engineering (AVANT imputation pour utiliser Title)
    df = create_features(df)
    
    # 3. Imputation
    print("\nImputation des valeurs manquantes...")
    df = impute_age(df, strategy='median_by_title')
    df = impute_embarked(df)
    df = impute_fare(df)
    
    # 4. Encodage
    df, encoders = encode_categorical(df)
    
    # 5. Sélection des features
    df = select_features(df)
    
    # Vérification finale
    analyze_missing_values(df, "Après prétraitement")
    
    # Sauvegarde optionnelle
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nDonnées sauvegardées: {save_path}")
    
    print(f"\nDimensions finales: {df.shape}")
    
    return df, encoders


def preprocess_test(df, encoders, train_columns=None, save_path=None):
    """
    Pipeline de prétraitement pour les données de test.
    
    Utilise les mêmes transformations que l'entraînement
    pour éviter le data leakage.
    
    Parameters
    ----------
    df : DataFrame
        Données brutes de test
    encoders : dict
        Encodeurs fittés sur les données d'entraînement
    train_columns : list, optional
        Liste des colonnes du train set pour alignement
    save_path : str, optional
    
    Returns
    -------
    df_processed : DataFrame
    """
    print("="*60)
    print("PRÉTRAITEMENT DES DONNÉES DE TEST")
    print("="*60)
    
    # Sauvegarder PassengerId pour les soumissions
    passenger_ids = df['PassengerId'].copy()
    
    # Même pipeline que train
    df = create_features(df)
    df = impute_age(df, strategy='median_by_title')
    df = impute_embarked(df)
    df = impute_fare(df)
    df, _ = encode_categorical(df)
    df = select_features(df)
    
    # Aligner les colonnes avec le train set
    if train_columns is not None:
        # Ajouter les colonnes manquantes (avec des 0)
        for col in train_columns:
            if col not in df.columns and col != 'Survived':
                df[col] = 0
        
        # Supprimer les colonnes en trop
        extra_cols = [c for c in df.columns if c not in train_columns]
        if extra_cols:
            df.drop(columns=extra_cols, inplace=True)
        
        # Réordonner selon train (sans Survived)
        cols_order = [c for c in train_columns if c != 'Survived' and c in df.columns]
        df = df[cols_order]
        
        print(f"\n✓ Colonnes alignées avec le train set")
    
    # Vérification
    analyze_missing_values(df, "Test après prétraitement")
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nDonnées sauvegardées: {save_path}")
    
    return df, passenger_ids


# =============================================================================
# 3. PIPELINE SKLEARN (BONUS)
# =============================================================================

def create_sklearn_pipeline():
    """
    Crée un pipeline sklearn complet pour le prétraitement.
    
    Ce pipeline peut être intégré directement avec un modèle
    pour éviter le data leakage lors de la validation croisée.
    
    Returns
    -------
    preprocessor : ColumnTransformer
    """
    # Colonnes par type
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title']
    
    # Pipeline pour variables numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline pour variables catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinaison
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Supprime les autres colonnes
    )
    
    return preprocessor


# =============================================================================
# 4. POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation du module.
    
    Usage:
        python preprocessing.py
    """
    import os
    
    # Chemins
    DATA_RAW = "../data/raw"
    DATA_PROCESSED = "../data/processed"
    
    # Vérifier que les données existent
    train_path = os.path.join(DATA_RAW, "train.csv")
    test_path = os.path.join(DATA_RAW, "test.csv")
    
    if not os.path.exists(train_path):
        print("ERREUR: train.csv non trouvé!")
        print("Téléchargez les données depuis Kaggle:")
        print("  kaggle competitions download -c titanic")
        exit(1)
    
    # Charger les données
    train_df, test_df = load_data(train_path, test_path)
    
    # Prétraiter
    train_processed, encoders = preprocess_train(
        train_df, 
        save_path=os.path.join(DATA_PROCESSED, "train_processed.csv")
    )
    
    if test_df is not None:
        test_processed, passenger_ids = preprocess_test(
            test_df,
            encoders,
            save_path=os.path.join(DATA_PROCESSED, "test_processed.csv")
        )
    
    print("\n" + "="*60)
    print("PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*60)
