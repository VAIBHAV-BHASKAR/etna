# Unit tests for preprocessing logic
# Unit tests for preprocessing logic
# tests/test_preprocessing.py

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etna.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, L2Normalizer,
    OneHotEncoder, LabelEncoder, SimpleImputer,
    LogTransformer, Binarizer, PolynomialFeatures,
    PreprocessingPipeline, one_hot_encode
)


def test_min_max_scaler():
    X = np.array([[1, 2], [3, 4], [5, 6]], float)
    scaler = MinMaxScaler((0, 1)).fit(X)
    Xs = scaler.transform(X)
    # scaled into [0,1]
    assert Xs.min() >= 0 and Xs.max() <= 1
    # fit_transform is equivalent
    Xr = scaler.fit_transform(X)
    assert np.allclose(Xs, Xr, atol=1e-6)


def test_standard_scaler():
    rng = np.random.RandomState(0)
    X = rng.rand(100, 3) * 10 + 5
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # zero mean, unit variance
    assert abs(Xs.mean()) < 1e-6
    assert abs(Xs.std() - 1) < 1e-6


def test_robust_scaler():
    X = np.array([[1, 10], [2, 20], [100, 200]], float)
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    # for first row: (1 - median)/IQR
    median = np.median(X, axis=0)
    iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) + 1e-8
    expected = (X[0] - median) / iqr
    assert np.allclose(Xs[0], expected, atol=1e-6)


def test_l2_normalizer():
    X = np.array([[3., 4.], [0, 0]], float)
    Xn = L2Normalizer().transform(X)
    # first row norm ~1, second stays zero
    assert np.allclose(np.linalg.norm(Xn[0]), 1.0, atol=1e-6)
    assert np.allclose(Xn[1], [0, 0], atol=1e-6)


def test_one_hot_encoder_and_function():
    labels = [0, 2, 1, 2]
    enc = OneHotEncoder().fit(labels)
    X_enc = enc.transform(labels)
    assert X_enc.shape == (4, 3)
    # convenience function matches
    func_enc = one_hot_encode(labels, num_classes=3)
    assert np.array_equal(X_enc, func_enc)


def test_label_encoder():
    labels = ['a', 'b', 'a', 'c']
    le = LabelEncoder().fit(labels)
    nums = le.transform(labels)
    assert set(nums.tolist()) == {0, 1, 2}


def test_simple_imputer_mean_and_median():
    X = np.array([[1, np.nan], [np.nan, 4], [3, 6]], float)
    imp_mean = SimpleImputer(strategy='mean').fit(X)
    Xt = imp_mean.transform(X)
    assert not np.isnan(Xt).any()

    imp_med = SimpleImputer(strategy='median').fit(X)
    Xt2 = imp_med.transform(X)
    assert not np.isnan(Xt2).any()


def test_simple_imputer_constant_and_mode():
    X = np.array([[1, np.nan], [np.nan, 4], [3, 6]], float)
    imp_const = SimpleImputer(strategy='constant', fill_value=0.0).fit(X)
    Xt = imp_const.transform(X)
    assert (Xt[np.isnan(X)] == 0.0).all()

    # most frequent strategy
    X2 = np.array([[1, 2], [1, np.nan], [3, 2]], float)
    imp_mode = SimpleImputer(strategy='most_frequent').fit(X2)
    Xt2 = imp_mode.transform(X2)
    # missing in col1 should fill with 2 (mode of [2,2])
    assert not np.isnan(Xt2).any()


def test_log_transformer():
    X = np.array([[0, 1], [1, 2]], float)
    Lt = LogTransformer().transform(X)
    assert np.allclose(Lt, np.log1p(X), atol=1e-6)


def test_binarizer():
    X = np.array([[0.2, -0.1], [0.9, 1.1]], float)
    B = Binarizer(threshold=0.5).transform(X)
    assert B.dtype == int
    assert B.max() == 1 and B.min() == 0


def test_polynomial_features_degree2():
    X = np.array([[2, 3]], float)
    pf = PolynomialFeatures(degree=2).transform(X)
    # combinations: (0,0), (0,1), (1,1)
    assert pf.shape == (1, 3)
    assert np.allclose(pf, [[4, 6, 9]], atol=1e-6)


def test_pipeline_chain_and_fit_transform():
    X = np.random.rand(5, 2)
    pipeline = PreprocessingPipeline([
        ('scale', MinMaxScaler((0, 1))),
        ('std', StandardScaler())
    ])
    X_out = pipeline.fit_transform(X)
    assert X_out.shape == X.shape


@pytest.mark.parametrize("bad", [[], np.array([]), np.zeros((2,2,2))])
def test_invalid_input_raises(bad):
    with pytest.raises(Exception):
        MinMaxScaler().fit_transform(bad)
