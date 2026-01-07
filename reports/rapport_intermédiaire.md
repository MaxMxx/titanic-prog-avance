# 📊 Rapport Intermédiaire - Personne 3 → Personne 4

## 1. Résumé Exécutif

Quatre modèles de classification ont été entraînés et évalués sur le dataset Titanic pour prédire la survie des passagers. L'évaluation a été réalisée avec :
- **Validation croisée stratifiée** (5-fold) sur le train set
- **Évaluation finale** sur un test set séparé (20% des données)

### 🏆 Meilleur modèle : **Logistic Regression**

| Métrique | Score |
|----------|-------|
| **F1-Score** | **0.7941** |
| Accuracy | 0.8436 |
| Precision | 0.8060 |
| Recall | 0.7826 |
| ROC-AUC | 0.8705 |

---

## 2. Tableau Comparatif des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.8436 | 0.8060 | 0.7826 | **0.7941** | **0.8705** |
| Gradient Boosting | 0.8156 | 0.8103 | 0.6812 | 0.7402 | 0.8444 |
| Random Forest | 0.8101 | 0.7966 | 0.6812 | 0.7344 | 0.8542 |
| SVM (SVC) | 0.6257 | 0.5333 | 0.2319 | 0.3232 | 0.7119 |

### Classement par F1-Score
1. 🥇 Logistic Regression (0.7941)
2. 🥈 Gradient Boosting (0.7402)
3. 🥉 Random Forest (0.7344)
4. ❌ SVM (0.3232) - Performance insuffisante

---

## 3. Observations Clés

### ✅ Points positifs
- **Logistic Regression** offre le meilleur équilibre precision/recall
- Les modèles d'ensemble (RF, GB) ont de bonnes performances en ROC-AUC
- Faible variance entre validation croisée et test set → pas d'overfitting majeur

### ⚠️ Points d'attention
- **SVM** : Performances très faibles, probablement dû à un problème de scaling ou d'hyperparamètres
- **Recall** : Tous les modèles ont un recall inférieur à 80% (on rate ~20-30% des survivants)
- **Random Forest** et **Gradient Boosting** : Potentiel d'amélioration avec tuning

---

## 4. Recommandations pour le Tuning (Personne 4)

### Modèles prioritaires à tuner

#### 1️⃣ Gradient Boosting (Priorité haute)
```python
param_grid_gb = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}
```
**Justification** : ROC-AUC élevé (0.844), potentiel d'amélioration du recall

#### 2️⃣ Random Forest (Priorité moyenne)
```python
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```
**Justification** : Fournit l'importance des features, bon pour l'interprétation

#### 3️⃣ Logistic Regression (Priorité basse)
```python
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```
**Justification** : Déjà performant, mais peut être affiné

#### ❌ SVM (Non recommandé)
Le SVM nécessiterait un travail important de preprocessing (scaling) et de tuning. Compte tenu du temps limité, il est préférable de se concentrer sur les autres modèles.

---

## 5. Importance des Features (Random Forest)

Les 5 features les plus importantes pour la prédiction :

| Rang | Feature | Importance |
|------|---------|------------|
| 1 | **Sex** | ~0.25 |
| 2 | **Fare** | ~0.20 |
| 3 | **Age** | ~0.18 |
| 4 | **Pclass** | ~0.15 |
| 5 | **FamilySize** | ~0.08 |

> 💡 Le sexe et le tarif payé sont les meilleurs prédicteurs de survie.

---

## 6. Fichiers Disponibles

### Modèles sauvegardés (`models/`)
```
models/
├── logistic_regression.joblib    # Meilleur modèle
├── random_forest.joblib
├── gradient_boosting.joblib
├── svm_svc.joblib
├── model_results.csv             # Tableau des résultats
└── train_test_data.pkl           # Données train/test
```

### Comment charger les modèles
```python
import joblib
import pickle

# Charger un modèle
model = joblib.load('models/logistic_regression.joblib')

# Charger les données
with open('models/train_test_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
```

### Figures générées (`reports/figures/`)
- `model_comparison.png` - Comparaison des métriques
- `roc_curves.png` - Courbes ROC
- `confusion_matrices.png` - Matrices de confusion
- `feature_importance.png` - Importance des features

---

## 7. Prochaines Étapes (Personne 4)

- [ ] Effectuer le tuning avec `GridSearchCV` ou `RandomizedSearchCV`
- [ ] Comparer les scores avant/après tuning
- [ ] Analyser les erreurs de prédiction
- [ ] Utiliser SHAP pour l'interprétation (optionnel)
- [ ] Rédiger le README final
- [ ] Préparer la présentation

---

## 8. Configuration Utilisée

```python
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
```

### Hyperparamètres par défaut utilisés

| Modèle | Paramètres |
|--------|------------|
| Logistic Regression | `max_iter=1000` |
| Random Forest | `n_estimators=100, max_depth=10, min_samples_split=5` |
| Gradient Boosting | `n_estimators=100, learning_rate=0.1, max_depth=3` |
| SVM | `kernel='rbf', C=1.0, probability=True` |

---

## 9. Métriques Détaillées

### Pourquoi ces métriques ?

- **Accuracy** : Proportion de prédictions correctes (peut être trompeuse sur jeu déséquilibré)
- **Precision** : Proportion de vrais positifs parmi les prédictions positives (utile si coût des faux positifs élevé)
- **Recall** : Capacité à identifier tous les survivants (important pour ne pas "oublier" de survivants)
- **F1-Score** : Moyenne harmonique precision/recall (bon compromis pour jeu déséquilibré)
- **ROC-AUC** : Capacité à discriminer entre les classes (0.5 = hasard, 1.0 = parfait)

### Distribution de la cible

- **Non survivants** : ~62%
- **Survivants** : ~38%

Le jeu est modérément déséquilibré, ce qui justifie l'utilisation du F1-Score comme métrique principale.

---

## 10. Méthodologie

### Pipeline de modélisation

1. **Prétraitement** (Personne 2)
   - Imputation des valeurs manquantes
   - Feature engineering (Title, FamilySize, etc.)
   - Encodage des variables catégorielles
   - Standardisation

2. **Séparation des données**
   - Train/Test split stratifié (80%/20%)
   - Random state fixé à 42

3. **Validation croisée**
   - StratifiedKFold 5-fold
   - Préserve la distribution de la cible

4. **Entraînement**
   - 4 modèles entraînés
   - Évaluation sur 5 métriques

5. **Évaluation finale**
   - Test set maintenu intact
   - Métriques calculées sur test set

---

**Contact** : Pour toute question, voir le notebook `notebooks/03_modeling.ipynb`
