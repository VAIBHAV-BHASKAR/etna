import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Unit tests for utility functions

import pytest
import numpy as np
from etna.utils import (
    load_csv, save_csv, load_json, save_json, load_npy, save_npy,
    file_exists, ensure_dir, set_random_seed, accuracy, argmax_rows, transpose, check_array, safe_divide
)

def test_csv(tmp_path):
    arr = np.array([[1,2],[3,4]])
    f = tmp_path / "test.csv"
    save_csv(f, arr)
    arr2 = load_csv(f)
    assert np.allclose(arr, arr2)

def test_json(tmp_path):
    obj = {"a": 1, "b": [2,3]}
    f = tmp_path / "test.json"
    save_json(f, obj)
    obj2 = load_json(f)
    assert obj == obj2

def test_npy(tmp_path):
    arr = np.arange(6).reshape(2,3)
    f = tmp_path / "test.npy"
    save_npy(f, arr)
    arr2 = load_npy(f)
    assert np.allclose(arr, arr2)

def test_file_exists(tmp_path):
    f = tmp_path / "foo.txt"
    with open(f, "w") as out:
        out.write("hi")
    assert file_exists(f)

def test_ensure_dir(tmp_path):
    d = tmp_path / "subdir"
    ensure_dir(d)
    assert os.path.isdir(d)

def test_set_random_seed():
    set_random_seed(42)
    a = np.random.rand()
    set_random_seed(42)
    b = np.random.rand()
    assert a == b

def test_accuracy():
    assert accuracy([1,0,1],[1,0,0]) == 2/3

def test_argmax_rows():
    arr = np.array([[1,2,3],[3,2,1]])
    assert np.all(argmax_rows(arr) == [2,0])

def test_transpose():
    arr = np.array([[1,2,3],[4,5,6]])
    t = transpose(arr)
    assert np.allclose(t, np.array([[1,4],[2,5],[3,6]]))

def test_check_array():
    arr = [1,2,3]
    out = check_array(arr, dtype="float32", ensure_2d=True)
    assert out.shape == (3,1)
    arr2 = np.array([[1,2],[3,4]])
    out2 = check_array(arr2, dtype="int32", ensure_2d=True)
    assert out2.dtype == np.int32
    assert out2.shape == (2,2)
    with pytest.raises(ValueError):
        check_array(np.zeros((2,2,2)), ensure_2d=True)

def test_safe_divide():
    a = np.array([1,2,3])
    b = np.array([1,0,3])
    out = safe_divide(a, b, fill_value=-1)
    assert np.allclose(out, [1,-1,1])
