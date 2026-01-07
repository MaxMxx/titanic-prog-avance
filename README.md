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
- Python 3.10 ou supérieur
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
│   └── 02_preprocessing.ipynb       # Nettoyage et prétraitement
├── src/
│   ├── __init__.py
│   └── preprocessing.py         # Fonctions de prétraitement
├── reports/
│   └── figures/                 # Graphiques exportés
└── models/                      # Modèles sauvegardés (.pkl)
```

---

## Reproduire les résultats

Exécuter les notebooks dans l'ordre suivant :

1. `notebooks/01_data_exploration.ipynb` - Comprendre les données
2. `notebooks/02_eda.ipynb` - Analyse exploratoire
3. `notebooks/03_preprocessing.ipynb` - Nettoyer et préparer

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

<!-- TODO: Compléter -->

---

## 🔧 Tuning des hyperparamètres

<!-- TODO: Compléter -->

---

## 🎯 Modèle final

<!-- TODO: Compléter -->

---

## ❌ Analyse d'erreurs

<!-- TODO: Compléter -->

---

## Équipe

*   Zaccharie BERNARD
*   Maxence BESSERVE
*   Eliott CHIFFRE
*   Mathias KOWALSKI

---

## Références

1.  [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
2.  [Scikit-learn Documentation](https://scikit-learn.org/stable/)
