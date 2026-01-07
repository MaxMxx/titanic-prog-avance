"""
Module src - Titanic Pipeline
=============================

Ce module contient le code Python réutilisable pour le projet Titanic.

Modules disponibles:
- preprocessing: Nettoyage, imputation, feature engineering
"""

from .preprocessing import (
    load_data,
    preprocess_train,
    preprocess_test,
    create_sklearn_pipeline,
    TitleExtractor,
    FamilySizeCreator,
    CabinProcessor,
    AgeGroupCreator,
    FareBinner
)

__all__ = [
    'load_data',
    'preprocess_train',
    'preprocess_test',
    'create_sklearn_pipeline',
    'TitleExtractor',
    'FamilySizeCreator',
    'CabinProcessor',
    'AgeGroupCreator',
    'FareBinner'
]

__version__ = '1.0.0'
