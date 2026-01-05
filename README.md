# Titanic Survival Prediction

> Projet d'examen - Mini pipeline de bout en bout  
> Master IA 1 - HEXAGONE

## 📋 Description du projet

Ce projet vise à prédire la survie des passagers du Titanic en utilisant des techniques de Machine Learning.

## 🔗 Dataset

- **Source** : [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Taille** : 891 passagers (train) / 418 passagers (test)
- **Features** : 11 variables (âge, sexe, classe, tarif, etc.)

## 🎯 Objectif

Prédire si un passager a survécu au naufrage du Titanic en fonction de ses caractéristiques personnelles et de voyage.

---

## 🛠️ Installation et environnement

### Prérequis
- Python 3.10 ou supérieur
- pip

### Télécharger les données

```bash
# Option 1 : Via Kaggle API
kaggle competitions download -c titanic -p data/raw/

# Option 2 : Télécharger manuellement depuis Kaggle et placer dans data/raw/
```

---

## 📁 Structure du projet

```
titanic-ml-project/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
├── data/
│   ├── raw/                     # Données brutes (train.csv, test.csv)
│   └── processed/               # Données nettoyées
├── notebooks/
├── src/
├── reports/
│   └── figures/                 # Graphiques exportés
├── models/                      # Modèles sauvegardés (.pkl)
```

---

## 👥 Équipe

| Membre |
|--------|
| Zaccharie BERNARD |
| Maxence BESSERVE |
| Eliott CHIFFRE |
| Mathias KOWALSKI |

---

## 📚 Références

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 📜 License

Ce projet est réalisé dans le cadre d'un examen.
