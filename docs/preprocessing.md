
# ETNA Preprocessing Module

The ETNA preprocessing module provides production-grade data preprocessing utilities for machine learning workflows, designed to work seamlessly with the ETNA neural network framework and its Rust core implementation. It is inspired by the scikit-learn API, but optimized for performance and Rust integration.

---

## Features

- **Scalers**: MinMaxScaler, StandardScaler, RobustScaler, L2Normalizer
- **Encoders**: OneHotEncoder (with Rust fallback), one_hot_encode function
- **Imputation**: SimpleImputer
- **Feature Engineering**: LogTransformer, Binarizer, PolynomialFeatures
- **Pipeline**: PreprocessingPipeline for chaining steps

---

## When to Use ETNA Preprocessing

- Preparing data for neural network training or inference
- Integrating with ETNA's Rust-based neural network core for maximum speed
- Replacing or supplementing scikit-learn preprocessing in Python ML pipelines
- Chaining multiple preprocessing steps in a robust, testable way

---

## Installation

The preprocessing module requires NumPy. Install dependencies with:

```bash
pip install numpy
```


## Quick Start


### Basic Scaling Example

```python
from etna.preprocessing import MinMaxScaler

X = [[1, 2], [3, 4]]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
```


### One-Hot Encoding Example

```python
from etna.preprocessing import OneHotEncoder

labels = [0, 1, 2, 1]
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(labels)
print(X_encoded)
```


### Pipeline Example

```python
from etna.preprocessing import MinMaxScaler, OneHotEncoder, PreprocessingPipeline

pipeline = PreprocessingPipeline([
    ("scale", MinMaxScaler()),
    ("encode", OneHotEncoder())
])
X_proc = pipeline.fit_transform([[0], [1], [2], [1]])
print(X_proc)
```


---

## API Reference


### Scalers

- **MinMaxScaler**
    - Scales features to a given range (default [0, 1]).
    - Methods: `fit`, `transform`, `fit_transform`
    - Example: `scaler = MinMaxScaler(); X_scaled = scaler.fit_transform(X)`

- **StandardScaler**
    - Standardizes features by removing the mean and scaling to unit variance.
    - Methods: `fit`, `transform`, `fit_transform`

- **RobustScaler**
    - Scales features using statistics robust to outliers (median and IQR).
    - Methods: `fit`, `transform`, `fit_transform`

- **L2Normalizer**
    - Scales input vectors individually to unit norm (L2).
    - Methods: `transform`


### Encoders

- **OneHotEncoder**
    - Encodes categorical integer features as a one-hot numeric array.
    - Uses Rust backend if available, otherwise falls back to NumPy.
    - Methods: `fit`, `transform`, `fit_transform`
    - Example: `encoder = OneHotEncoder(); X_enc = encoder.fit_transform(labels)`

- **one_hot_encode**
    - Convenience function for one-hot encoding a 1D array.
    - Example: `one_hot_encode([0, 1, 2])`


### Imputation

- **SimpleImputer**
    - Imputes missing values using mean, median, most frequent, or constant value.
    - Methods: `fit`, `transform`, `fit_transform`
    - Example: `imputer = SimpleImputer(strategy='mean'); X_filled = imputer.fit_transform(X)`


### Feature Engineering

- **LogTransformer**
    - Applies log(1 + x) transformation to features.
    - Methods: `transform`

- **Binarizer**
    - Binarizes features based on a threshold.
    - Methods: `transform`

- **PolynomialFeatures**
    - Generates polynomial and interaction features up to a given degree.
    - Methods: `transform`


### Pipeline

- **PreprocessingPipeline**
    - Chains multiple preprocessing steps (fit/transform) in sequence.
    - Methods: `fit`, `transform`, `fit_transform`
    - Example: See Quick Start above.


---

## Rust Integration

- If the Rust core is available and built, heavy numerical operations (such as one-hot encoding) will be delegated to the Rust backend for optimal performance.
- If not, the module will fall back to pure NumPy implementations.
- To enable Rust-backed features, build the Rust extension with maturin and ensure it is importable as `etna._etna_rust`.

---


## Error Handling

- All transformers validate input shapes and types.
- Clear error messages are provided for invalid strategies or parameters.
- NaN and edge cases are handled gracefully in imputers and scalers.

---


## Testing

- Unit tests for all preprocessing utilities should be placed in `tests/test_preprocessing.py`.
- Example tests:

```python
import numpy as np
from etna.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer

def test_minmax_scaler():
    X = np.array([[1, 2], [3, 4]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.min(axis=0), 0)
    assert np.allclose(X_scaled.max(axis=0), 1)

def test_one_hot_encoder():
    X = [0, 1, 2, 1]
    encoder = OneHotEncoder()
    X_enc = encoder.fit_transform(X)
    assert X_enc.shape == (4, 3)
    assert np.all((X_enc.sum(axis=1) == 1))

def test_simple_imputer():
    X = np.array([[1, np.nan], [3, 4]])
    imputer = SimpleImputer()
    X_filled = imputer.fit_transform(X)
    assert not np.isnan(X_filled).any()
```

---


## Contributing

- Ensure new preprocessing utilities include comprehensive tests and docstrings.
- Maintain compatibility with the Rust core implementation.
- Add examples for new functionality.
- Update this documentation as needed.
- Follow PEP8 and project code style.

---


## License

This module is part of the ETNA framework and is licensed under the MIT License.
