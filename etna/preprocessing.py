# Preprocessing (Scaling, Encoding)
# Scaling : Min-Max,RobustScaler and so on.
# Encoding : One-Hot,Categorical..

"""
ETNA Preprocessing Module
------------------------
This module provides production-grade preprocessing utilities for the ETNA neural network framework.
All heavy numerical operations can be delegated to the Rust core (if available) for optimal performance.

Features:
- Scalers: MinMaxScaler, StandardScaler, RobustScaler, L2Normalizer
- Encoders: OneHotEncoder (with Rust fallback), one_hot_encode function
- Imputation: SimpleImputer
- Feature Engineering: LogTransformer, Binarizer, PolynomialFeatures
- Pipeline: PreprocessingPipeline for chaining steps

Example usage:
    from etna.preprocessing import MinMaxScaler, OneHotEncoder, PreprocessingPipeline
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(labels)
    pipeline = PreprocessingPipeline([
        ("scale", MinMaxScaler()),
        ("encode", OneHotEncoder())
    ])
    X_proc = pipeline.fit_transform(X)
"""
# etna/preprocessing.py

import numpy as np

RUST_ONE_HOT_AVAILABLE = False
rust_one_hot_encode = None
# Rust-based one-hot encoding is not available; using pure Python implementation.
__all__ = [
    'MinMaxScaler', 'StandardScaler', 'RobustScaler', 'L2Normalizer',
    'OneHotEncoder', 'LabelEncoder', 'SimpleImputer',
    'LogTransformer', 'Binarizer', 'PolynomialFeatures',
    'PreprocessingPipeline', 'one_hot_encode'
]


# ======================= SCALERS ============================

class MinMaxScaler:
    """
    Scales features to a given range [min, max] (default [0, 1]).
    """
    def __init__(self, feature_range=(0, 1)):
        self.min, self.max = feature_range

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("MinMaxScaler only supports 2D arrays")
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.scale_ = (self.max - self.min) / (self.data_max_ - self.data_min_ + 1e-8)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("MinMaxScaler only supports 2D arrays")
        return (X - self.data_min_) * self.scale_ + self.min

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    """
    Scales features using statistics that are robust to outliers (median and IQR).
    """
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        self.iqr_ = (np.percentile(X, 75, axis=0) -
                     np.percentile(X, 25, axis=0)) + 1e-8
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class L2Normalizer:
    """
    Scales input vectors individually to unit norm (L2).
    """
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / norm


# ======================= ENCODERS ============================


class OneHotEncoder:
    """
    Encode categorical integer features as a one-hot numeric array.
    Uses Rust backend if available, otherwise falls back to NumPy implementation.
    """
    def __init__(self):
        self.categories_ = None
        self.num_classes_ = None

    def fit(self, X):
        X = np.asarray(X).flatten()
        self.categories_ = np.unique(X)
        self.num_classes_ = len(self.categories_)
        return self

    def transform(self, X):
        X = np.asarray(X).flatten()
        if RUST_ONE_HOT_AVAILABLE:
            return np.array(rust_one_hot_encode(X.tolist(), self.num_classes_))
        out = np.zeros((len(X), self.num_classes_), dtype=int)
        for i, val in enumerate(X):
            idx = int(np.where(self.categories_ == val)[0][0])
            out[i, idx] = 1
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def one_hot_encode(X, num_classes=None):
    """
    One-hot encode a 1D array using Rust backend if available, else NumPy.
    Args:
        X: 1D array-like of integer labels
        num_classes: Number of classes (optional)
    Returns:
        2D numpy array (n_samples, num_classes)
    """
    X = np.asarray(X).flatten()
    if num_classes is None:
        num_classes = int(X.max()) + 1
    if RUST_ONE_HOT_AVAILABLE:
        return np.array(rust_one_hot_encode(X.tolist(), num_classes))
    out = np.zeros((X.shape[0], num_classes), dtype=int)
    out[np.arange(X.shape[0]), X.astype(int)] = 1
    return out


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    """
    def fit(self, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        y = np.asarray(y)
        return np.array([np.where(self.classes_ == val)[0][0] for val in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)


# ======================= IMPUTATION ============================

class SimpleImputer:
    """
    Impute missing values using mean, median, most frequent, or constant value.
    """
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = [
                np.bincount(X[:, i][~np.isnan(X[:, i])].astype(int)).argmax()
                for i in range(X.shape[1])
            ]
        elif self.strategy == 'constant':
            self.statistics_ = self.fill_value
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mask = np.isnan(X)
        X_filled = X.copy()
        if np.isscalar(self.statistics_):
            X_filled[mask] = self.statistics_
        else:
            # Fill each column's missing values with the corresponding statistic
            for col in range(X.shape[1]):
                X_filled[mask[:, col], col] = self.statistics_[col]
        return X_filled

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ======================= FEATURE ENGINEERING ============================

class LogTransformer:
    """
    Apply log(1 + x) transformation to features.
    """
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log1p(X)


class Binarizer:
    """
    Binarize features based on a threshold.
    """
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X > self.threshold).astype(int)


class PolynomialFeatures:
    """
    Generate polynomial and interaction features up to a given degree.
    """
    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, X):
        from itertools import combinations_with_replacement
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        combs = list(combinations_with_replacement(range(n_features), self.degree))
        X_poly = np.empty((n_samples, len(combs)), dtype=float)
        for i, comb in enumerate(combs):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)
        return X_poly


# ======================= PIPELINE ============================

class PreprocessingPipeline:
    """
    Chains multiple preprocessing steps (fit/transform) in sequence.
    """
    def __init__(self, steps):
        self.steps = steps  # List of (name, transformer)

    def fit(self, X):
        for _, step in self.steps:
            if hasattr(step, 'fit'):
                step.fit(X)
                X = step.transform(X)
        return self

    def transform(self, X):
        for _, step in self.steps:
            if hasattr(step, 'transform'):
                X = step.transform(X)
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
