"""
Utility functions for the ETNA framework.

Includes:
- Random seed control (with Rust fallback)
- Array validation
- Safe math operations
- Matrix helpers (transpose, argmax, accuracy) with Rust fallback if available
- File loading/saving helpers (CSV, JSON, NPY), file existence, and directory creation
"""
import numpy as np
import random
import os
import json
import pandas as pd
#from etsi import etna_core
# Rust seed initialization will be attempted only if needed
_RUST_SEED_ENABLED = None

# Try to import Rust-backed helpers (if available)
# Rust-backed helpers are not available; set _RUST_UTILS to None
_RUST_UTILS = None

def set_random_seed(seed: int):
    """
    Set the global random seed for Python, NumPy, and the Rust core (if available).

    Parameters:
        seed (int): The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    global _RUST_SEED_ENABLED
    if _RUST_SEED_ENABLED is None:
        # Optional: etna_rust is not required unless Rust-backed seed is needed
        try:
            # Rust-backed seed initialization is not available; skip import
            _RUST_SEED_ENABLED = False
        except Exception:
            _RUST_SEED_ENABLED = False
    if callable(_RUST_SEED_ENABLED):
        _RUST_SEED_ENABLED(seed)


def check_array(X, dtype: str = "float32", ensure_2d: bool = False):
    """
    Validate input and return a NumPy array with the requested dtype (and shape).

    Parameters:
        X: array‐like input
        dtype (str): NumPy dtype to cast to (e.g. "float32", "int32")
        ensure_2d (bool): if True, reshapes 1D input to (n_samples, 1) and 
                          raises on ndim != 2

    Returns:
        np.ndarray: Cast array.

    Raises:
        ValueError: If ensure_2d is True and result is not 2D.
    """
    arr = np.asarray(X, dtype=dtype)
    if ensure_2d:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Input must be 1D or 2D when ensure_2d=True, got shape {arr.shape}")
    return arr


def safe_divide(a, b, fill_value: float = 0.0):
    """
    Elementwise division of `a / b`, replacing any divide‐by‐zero results
    with `fill_value`.

    Parameters:
        a, b: array‐like inputs
        fill_value (float): value to use when b == 0

    Returns:
        np.ndarray: Result of division.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a_arr / b_arr
    # replace infinities and NaNs
    mask = ~np.isfinite(result)
    if np.any(mask):
        result[mask] = fill_value
    return result


def transpose(matrix):
    """
    Transpose a matrix using Rust implementation if available, otherwise fall back to NumPy.

    Parameters:
        matrix: 2D array‐like input

    Returns:
        np.ndarray: Transposed matrix.
    """
    if _RUST_UTILS is not None:
        try:
            return np.array(_RUST_UTILS.transpose(matrix))
        except Exception:
            pass
    return np.asarray(matrix).T


def argmax_rows(matrix):
    """
    Get the indices of the maximum values along the rows of a matrix.
    Uses Rust implementation if available, otherwise falls back to NumPy.

    Parameters:
        matrix: 2D array‐like input

    Returns:
        np.ndarray: Indices of the maximum values along the rows.
    """
    if _RUST_UTILS is not None:
        try:
            return np.array(_RUST_UTILS.argmax_rows(matrix))
        except Exception:
            pass
    return np.argmax(matrix, axis=1)


def accuracy(preds, labels):
    """
    Compute the accuracy of predictions compared to labels.
    Uses Rust implementation if available, otherwise falls back to NumPy.

    Parameters:
        preds: array‐like of predictions
        labels: array‐like of true labels

    Returns:
        float: Accuracy as a fraction of correctly predicted samples.
    """
    if _RUST_UTILS is not None:
        try:
            return _RUST_UTILS.accuracy(list(preds), list(labels))
        except Exception:
            pass
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return (preds == labels).mean()

# ========== File Loading/Saving ==========

def load_csv(path, **kwargs):
    """Load a CSV file into a numpy array."""
    return pd.read_csv(path, **kwargs).values

def save_csv(path, arr, **kwargs):
    """Save a numpy array to CSV."""
    pd.DataFrame(arr).to_csv(path, index=False, **kwargs)

def load_json(path):
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, obj):
    """Save an object as JSON."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_npy(path):
    """Load a .npy file."""
    return np.load(path)

def save_npy(path, arr):
    """Save a numpy array to .npy."""
    np.save(path, arr)

def file_exists(path):
    return os.path.exists(path)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
