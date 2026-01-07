# Titanic Survival Prediction

> Projet d'examen - Mini pipeline de bout en bout  
> Master IA 1 - HEXAGONE

## Description du projet

Ce projet vise à prédire la survie des passagers du Titanic en utilisant des techniques de Machine Learning. Il s'agit d'un problème de **classification binaire** où la variable cible est `Survived` (0 = décédé, 1 = survécu).

## Dataset

- **Source** : [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Taille** : 891 passagers (train) / 418 passagers (test)
- **Features** : 11 variables (âge, sexe, classe, tarif, etc.)

## Objectif

Prédire si un passager a survécu au naufrage du Titanic en fonction de ses caractéristiques personnelles et de voyage.

---

## Installation et environnement

### Prérequis
- Python 3.13.2
- pip ou conda

### Installation

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/titanic-ml-project.git
cd titanic-ml-project

# Créer un environnement virtuel
python -m venv venv

# Installer les dépendances
pip install -r requirements.txt
```

### Télécharger les données

```bash
# Option 1 : Via Kaggle API
kaggle competitions download -c titanic -p data/raw/

# Option 2 : Télécharger manuellement depuis Kaggle et placer dans data/raw/
```

---

## Structure du projet

```
titanic-ml-project/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
├── data/
│   ├── raw/                     # Données brutes (train.csv, test.csv)
│   └── processed/               # Données nettoyées
├── notebooks/
│   ├── 010_data_exploration.ipynb    # Compréhension des données
│   ├── 011_eda.ipynb                 # Analyse exploratoire complète
│   ├── 02_preprocessing.ipynb        # Nettoyage et prétraitement
│   ├── 03_modeling.ipynb            # Modélisation et évaluation
│   └── 04_tuning.ipynb               # Tuning des hyperparamètres
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Fonctions de prétraitement
│   └── train.py                      # Script d'entraînement
├── reports/
│   └── figures/                 # Graphiques exportés
└── models/                      # Modèles sauvegardés (.pkl)
```

---

## Reproduire les résultats

### Ordre d'exécution des notebooks

Exécuter les notebooks dans l'ordre suivant :

1. `notebooks/010_data_exploration.ipynb` - Comprendre les données
2. `notebooks/011_eda.ipynb` - Analyse exploratoire complète
3. `notebooks/02_preprocessing.ipynb` - Nettoyage et prétraitement
4. `notebooks/03_modeling.ipynb` - Entraînement et évaluation des modèles
5. `notebooks/04_tuning.ipynb` - Optimisation des hyperparamètres

### Commandes rapides

```bash
# Exécuter le pipeline complet
jupyter notebook notebooks/

# Ou utiliser les scripts Python
python src/preprocessing.py
python src/train.py
```

---

## Résumé de l'EDA

### Observations clés

- **Taux de survie global** : ~38%
- **Survie par sexe** : Les femmes ont un taux de survie beaucoup plus élevé (~74%) que les hommes (~19%)
- **Survie par classe** : La 1ère classe a le meilleur taux de survie (~63%)
- **Âge** : Les enfants ont un taux de survie plus élevé
- **Valeurs manquantes** : Age (~20%), Cabin (~77%), Embarked (~0.2%)

### Visualisations clés

| Figure | Description |
|--------|-------------|
| `reports/figures/target_distribution.png` | Distribution de la variable cible (Survie) |
| `reports/figures/survival_by_sex.png` | Taux de survie par Sexe |
| `reports/figures/survival_by_class.png` | Taux de survie par Classe |
| `reports/figures/survival_by_age_group.png` | Survie par groupe d'âge |
| `reports/figures/correlation_heatmap.png` | Matrice de corrélation |

### Insights pour la modélisation

*   **Facteurs principaux** : Le modèle devra fortement pondérer le **Sexe** et la **Classe** (Features les plus corrélées).
*   **Age** : Variable critique mais incomplète (20% NA). Une imputation intelligente est requise pour ne pas biaiser le modèle (ex: Médiane par Titre).
*   **Nouvelles Features** :
    *   `Title` : Extrait du nom, capture le statut social mieux que la classe seule.
    *   `FamilySize` : La survie n'est pas linéaire (les personnes seules et les grandes familles meurent plus).
*   **Encodage** : `Sex` doit être binaire. `Embarked` et `Title` nécessitent du OneHotEncoding.

---

## 🤖 Modélisation

### Modèles testés

Quatre modèles de classification ont été entraînés et comparés :

1. **Logistic Regression** : Modèle linéaire simple et interprétable
2. **Random Forest** : Ensemble d'arbres de décision, robuste aux outliers
3. **Gradient Boosting** : Boosting séquentiel, bon pour les relations complexes
4. **SVM (SVC)** : Machine à vecteurs de support avec noyau RBF

### Stratégie d'évaluation

- **Split train/test** : 80/20 avec stratification pour préserver la distribution de la cible
- **Validation croisée** : 5-fold stratifiée pour évaluer la robustesse
- **Métriques** : Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Résultats sur le test set

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.8436** | **0.8060** | **0.7826** | **0.7941** | **0.8705** |
| Random Forest | 0.8101 | 0.7966 | 0.6812 | 0.7344 | 0.8542 |
| Gradient Boosting | 0.8156 | 0.8103 | 0.6812 | 0.7402 | 0.8444 |
| SVM (SVC) | 0.6257 | 0.5333 | 0.2319 | 0.3232 | 0.7119 |

### Explication des modèles

#### Logistic Regression
- **Principe** : Modèle linéaire qui utilise la fonction logistique pour prédire la probabilité d'appartenir à une classe
- **Avantages** : Simple, rapide, interprétable (coefficients), peu de risque d'overfitting
- **Hypothèses** : Relation linéaire entre features et log-odds de la survie
- **Pourquoi choisi** : Baseline solide pour problèmes de classification binaire, excellent compromis performance/interprétabilité

#### Random Forest
- **Principe** : Ensemble de nombreux arbres de décision entraînés sur des sous-échantillons différents
- **Avantages** : Capture les interactions non-linéaires, importance des features, robuste aux outliers
- **Hypothèses** : Pas d'hypothèse forte sur la distribution des données
- **Pourquoi choisi** : Modèle puissant qui peut capturer des patterns complexes sans overfitting excessif

#### Gradient Boosting
- **Principe** : Entraîne séquentiellement des arbres faibles, chaque arbre corrige les erreurs du précédent
- **Avantages** : Très performant, peut capturer des relations complexes
- **Hypothèses** : Les erreurs sont corrigeables de manière additive
- **Pourquoi choisi** : Souvent parmi les meilleurs modèles pour problèmes tabulaires

#### SVM
- **Principe** : Trouve l'hyperplan optimal qui sépare les classes avec la marge maximale
- **Avantages** : Efficace en haute dimension, bon avec noyau RBF pour non-linéarité
- **Hypothèses** : Les classes sont séparables (ou presque) dans l'espace transformé
- **Pourquoi choisi** : Modèle classique pour comparaison, mais s'est révélé moins performant ici

### Sélection du meilleur modèle

Le **Logistic Regression** a été sélectionné comme meilleur modèle car :
- **F1-Score le plus élevé** (0.7941) - meilleur équilibre précision/rappel
- **ROC-AUC élevé** (0.8705) - bonne capacité de discrimination
- **Interprétabilité** - coefficients explicables
- **Simplicité** - moins de risque d'overfitting, plus facile à déployer

---

## 🔧 Tuning des hyperparamètres

### Méthode utilisée

**GridSearchCV** avec validation croisée 5-fold stratifiée sur le modèle Logistic Regression (meilleur modèle baseline).

### Espace de recherche

Les hyperparamètres optimisés sont :

- **C** : Force de régularisation `[0.01, 0.1, 1.0, 10.0, 100.0]`
  - Plus C est élevé, moins la régularisation est forte
  - Justification : Trouver le bon équilibre entre biais et variance

- **penalty** : Type de régularisation `['l1', 'l2']`
  - L1 (Lasso) : peut mettre certains coefficients à zéro (sélection de features)
  - L2 (Ridge) : réduit l'amplitude des coefficients sans les éliminer
  - Justification : Tester les deux approches pour voir laquelle fonctionne le mieux

- **solver** : Algorithme d'optimisation `['liblinear', 'lbfgs']`
  - liblinear : efficace pour petits datasets, supporte L1 et L2
  - lbfgs : plus rapide pour datasets moyens, supporte seulement L2
  - Justification : Adapter l'algorithme selon le type de régularisation

**Total de combinaisons testées** : 5 × 2 × 2 = 20 combinaisons

### Résultats du tuning

Les meilleurs hyperparamètres trouvés sont sauvegardés dans `models/best_hyperparameters.csv`.

### Meilleurs hyperparamètres trouvés

Après optimisation avec GridSearchCV (20 combinaisons testées, 5-fold CV) :

- **C** : `1.0` (régularisation modérée)
- **penalty** : `l2` (Ridge - régularisation L2)
- **solver** : `lbfgs` (algorithme d'optimisation)
- **Meilleur score CV (F1)** : `0.7683`

### Comparaison Baseline vs Tuned

| Métrique | Baseline | Tuned | Amélioration |
|----------|----------|-------|--------------|
| Accuracy | 0.8436 | 0.8436 | 0.0000 |
| Precision | 0.8060 | 0.8060 | 0.0000 |
| Recall | 0.7826 | 0.7826 | 0.0000 |
| F1-Score | 0.7941 | 0.7941 | 0.0000 |
| ROC-AUC | 0.8705 | 0.8705 | 0.0000 |

**Observation** : Les hyperparamètres optimaux correspondent aux valeurs par défaut de scikit-learn pour ce dataset. Le modèle baseline était déjà bien calibré, ce qui explique l'absence d'amélioration significative après tuning.

### Justification de la méthode

- **GridSearchCV** : Recherche exhaustive dans l'espace défini, garantit de trouver le meilleur point
- **CV 5-fold** : Évalue la robustesse sur différents splits, évite l'overfitting
- **Scoring F1** : Métrique choisie car elle équilibre précision et rappel (important pour dataset déséquilibré)

---

## Modèle final

### Modèle sélectionné

**Logistic Regression optimisée** (après tuning des hyperparamètres)

### Performance finale

Le modèle final atteint les performances suivantes sur le test set :

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Accuracy** | 0.8436 | 84.36% de prédictions correctes |
| **Precision** | 0.8060 | Parmi les prédits survivants, 80.6% ont réellement survécu |
| **Recall** | 0.7826 | Le modèle détecte 78.26% des vrais survivants |
| **F1-Score** | 0.7941 | Bon équilibre précision/rappel |
| **ROC-AUC** | 0.8705 | Excellente capacité de discrimination entre classes |

**Hyperparamètres finaux** :
- C = 1.0
- penalty = 'l2' (Ridge)
- solver = 'lbfgs'

### Importance des variables

Les **top 10 features** les plus importantes (par coefficient absolu) sont :

| Rang | Feature | Coefficient | Interprétation |
|------|---------|-------------|----------------|
| 1 | **Title_Mr** | -1.7933 | Avoir le titre "Mr" réduit fortement les chances de survie |
| 2 | **Sex** | -0.9449 | Le sexe est très prédictif (femmes > hommes) |
| 3 | **Deck_E** | 0.8818 | Être dans le pont E augmente les chances de survie |
| 4 | **Deck_G** | -0.8177 | Être dans le pont G réduit les chances de survie |
| 5 | **Deck_D** | 0.8052 | Être dans le pont D augmente les chances de survie |
| 6 | **Title_Mrs** | 0.7822 | Avoir le titre "Mrs" augmente les chances de survie |
| 7 | **Title_Rare** | -0.6933 | Titres rares (ex: Sir, Lady) réduisent les chances |
| 8 | **Pclass** | -0.6828 | La classe est importante (1ère > 2ème > 3ème) |
| 9 | **FamilySizeGroup_Large** | -0.5473 | Familles nombreuses réduisent les chances |
| 10 | **AgeGroup_Child** | 0.4609 | Être un enfant augmente les chances de survie |

**Visualisation** : `reports/figures/feature_coefficients_tuned.png`

**Insights** :
- Les coefficients négatifs indiquent une réduction des chances de survie
- Les coefficients positifs indiquent une augmentation des chances
- Cette importance confirme les insights de l'EDA : **"Women and children first"** était la règle d'évacuation
- Le pont (Deck) est un facteur important, probablement lié à la proximité des canots de sauvetage

### Sauvegarde

Le modèle final est sauvegardé dans :
- `models/logistic_regression_tuned.joblib` : Modèle optimisé prêt à l'emploi

---

## Analyse d'erreurs

### Matrice de confusion

La matrice de confusion (sauvegardée dans `reports/figures/confusion_matrix_tuned.png`) révèle :

| | Prédit Décédé | Prédit Survivant |
|---|---|---|
| **Réel Décédé** | TN = 97 | FP = 13 |
| **Réel Survivant** | FN = 15 | TP = 54 |

- **Vrais Positifs (TP)** : 54 passagers correctement prédits comme survivants
- **Vrais Négatifs (TN)** : 97 passagers correctement prédits comme décédés
- **Faux Positifs (FP)** : 13 passagers prédits survivants mais décédés
- **Faux Négatifs (FN)** : 15 passagers prédits décédés mais survivants

**Taux d'erreur global** : 15.6% (28 erreurs sur 179 prédictions)

### Analyse quantitative des erreurs

**Répartition des erreurs** :
- **Faux Positifs (FP)** : 13 erreurs (7.3% du test set)
  - Le modèle a prédit la survie pour 13 passagers qui sont décédés
  - Impact : Surestimation des chances de survie
  
- **Faux Négatifs (FN)** : 15 erreurs (8.4% du test set)
  - Le modèle a prédit le décès pour 15 passagers qui ont survécu
  - Impact : Sous-estimation des chances de survie

**Bilan** : Le modèle fait légèrement plus d'erreurs en sous-estimant la survie (FN > FP), ce qui est cohérent avec un dataset où la classe majoritaire est "décédé" (62%).

### Caractéristiques des erreurs

#### Faux Positifs (FP = 13)
Passagers prédits comme survivants mais décédés. Ces erreurs peuvent concerner :
- **Femmes de 3ème classe** : Le modèle surestime l'effet du sexe sans tenir assez compte de la classe sociale
- **Hommes de classe élevée** : Le modèle surestime leur chance de survie basée sur la classe
- **Passagers avec caractéristiques mixtes** : Combinaisons de features qui brouillent la prédiction

#### Faux Négatifs (FN = 15)
Passagers prédits comme décédés mais survivants. Ces erreurs peuvent concerner :
- **Hommes jeunes de classe élevée** : Le modèle sous-estime leur chance de survie
- **Cas exceptionnels** : Passagers ayant survécu malgré des caractéristiques défavorables (ex: hommes adultes de 3ème classe)
- **Interactions complexes** : Relations non-linéaires non capturées par le modèle linéaire

### Limites du modèle

1. **Dataset historique** : Données de 1912, contexte social différent
2. **Variables manquantes** : 20% d'âges manquants, 77% de cabines manquantes
3. **Relations complexes** : Certaines interactions (ex: sexe × classe × âge) peuvent être mal capturées par un modèle linéaire
4. **Taille du dataset** : 891 échantillons seulement, limite la complexité du modèle

### Pistes d'amélioration

1. **Feature engineering** : Créer plus d'interactions (ex: Sex × Pclass, Age × FamilySize)
2. **Modèles non-linéaires** : Tester XGBoost ou LightGBM pour capturer des patterns plus complexes
3. **Ensemble** : Combiner plusieurs modèles (voting, stacking)
4. **Imputation avancée** : Utiliser des modèles prédictifs pour imputer l'âge plutôt que la médiane
5. **Traitement du déséquilibre** : Utiliser SMOTE ou ajuster les poids de classe

---

## Équipe

*   Zaccharie BERNARD
*   Maxence BESSERVE
*   Eliott CHIFFRE
*   Mathias KOWALSKI

---

## Limites et perspectives

### Limites actuelles

- **Taille du dataset** : 891 échantillons, limite la complexité des modèles
- **Données historiques** : Contexte social de 1912, difficilement généralisable
- **Variables manquantes** : Impact de l'imputation sur les performances
- **Modèle linéaire** : Peut manquer certaines interactions complexes

### Perspectives d'amélioration

- **Modèles avancés** : XGBoost, LightGBM, réseaux de neurones
- **Feature engineering** : Interactions, transformations non-linéaires
- **Validation externe** : Tester sur d'autres datasets similaires
- **Déploiement** : Interface Streamlit pour prédictions en temps réel

---

## Références

1. [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
4. [Logistic Regression - Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
