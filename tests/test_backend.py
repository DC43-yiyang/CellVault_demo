"""Tests for DuckDBZarrBackend."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellvault.backend import DuckDBZarrBackend


@pytest.fixture
def backend(tmp_path):
    path = str(tmp_path / "test.cvdb")
    b = DuckDBZarrBackend(path)
    yield b
    b.close()


class TestObs:
    def test_write_read_obs(self, backend):
        df = pd.DataFrame({"a": [1, 2, 3], "_index": ["c0", "c1", "c2"]})
        backend.write_obs(df)
        result = backend.read_obs()
        assert len(result) == 3
        assert "a" in result.columns

    def test_read_obs_empty(self, backend):
        result = backend.read_obs()
        assert len(result) == 0

    def test_has_obs(self, backend):
        assert not backend._has_obs()
        df = pd.DataFrame({"a": [1], "_index": ["c0"]})
        backend.write_obs(df)
        assert backend._has_obs()

    def test_obs_columns(self, backend):
        df = pd.DataFrame({"a": [1], "b": [2], "_index": ["c0"]})
        backend.write_obs(df)
        cols = backend.obs_columns
        assert "a" in cols
        assert "b" in cols
        assert "_index" not in cols

    def test_obs_columns_empty(self, backend):
        assert backend.obs_columns == []


class TestVar:
    def test_write_read_var(self, backend):
        df = pd.DataFrame({"gene": ["g1", "g2"]}, index=["v0", "v1"])
        backend.write_var(df)
        result = backend.read_var()
        assert len(result) == 2
        assert "gene" in result.columns

    def test_read_var_empty(self, backend):
        result = backend.read_var()
        assert len(result) == 0


class TestX:
    def test_write_read_dense(self, backend):
        X = np.random.rand(10, 5).astype(np.float32)
        backend.write_X(X)
        result = backend.read_X()
        assert result is not None
        np.testing.assert_array_almost_equal(result, X)

    def test_write_read_sparse(self, backend):
        X = sparse.random(10, 5, density=0.3, format="csr", dtype=np.float32)
        backend.write_X(X)
        result = backend.read_X()
        assert sparse.issparse(result)
        np.testing.assert_array_almost_equal(result.toarray(), X.toarray())

    def test_read_X_empty(self, backend):
        assert backend.read_X() is None

    def test_X_exists(self, backend):
        assert not backend.X_exists
        backend.write_X(np.array([[1, 2], [3, 4]]))
        assert backend.X_exists


class TestObsm:
    def test_write_read_obsm(self, backend):
        data = np.random.rand(10, 3)
        backend.write_obsm("X_pca", data)
        result = backend.read_obsm("X_pca")
        assert result is not None
        np.testing.assert_array_almost_equal(result, data)

    def test_read_obsm_missing(self, backend):
        assert backend.read_obsm("nonexistent") is None

    def test_obsm_keys(self, backend):
        assert backend.obsm_keys == []
        backend.write_obsm("X_pca", np.random.rand(5, 2))
        backend.write_obsm("X_umap", np.random.rand(5, 2))
        keys = backend.obsm_keys
        assert "X_pca" in keys
        assert "X_umap" in keys


class TestObsp:
    def test_write_read_obsp_sparse(self, backend):
        data = sparse.random(10, 10, density=0.3, format="csr")
        backend.write_obsp("connectivities", data)
        result = backend.read_obsp("connectivities")
        assert sparse.issparse(result)
        np.testing.assert_array_almost_equal(result.toarray(), data.toarray())

    def test_write_read_obsp_dense(self, backend):
        data = np.random.rand(5, 5)
        backend.write_obsp("test_dense", data)
        result = backend.read_obsp("test_dense")
        np.testing.assert_array_almost_equal(result, data)

    def test_read_obsp_missing(self, backend):
        assert backend.read_obsp("nonexistent") is None

    def test_obsp_keys(self, backend):
        assert backend.obsp_keys == []
        backend.write_obsp("conn", sparse.eye(3, format="csr"))
        assert "conn" in backend.obsp_keys


class TestUns:
    def test_write_read_uns(self, backend):
        uns = {"key1": "value1", "key2": [1, 2, 3]}
        backend.write_uns(uns)
        result = backend.read_uns()
        assert result == uns

    def test_read_uns_empty(self, backend):
        assert backend.read_uns() == {}

    def test_uns_numpy_serialization(self, backend):
        uns = {"arr": np.array([1, 2, 3]), "val": np.float64(3.14)}
        backend.write_uns(uns)
        result = backend.read_uns()
        assert result["arr"] == [1, 2, 3]
        assert abs(result["val"] - 3.14) < 0.001


class TestRegistry:
    def test_write_read_registry(self, backend):
        reg = {"pca": "X_pca"}
        backend.write_registry(reg)
        result = backend.read_registry()
        assert result == reg

    def test_read_registry_empty(self, backend):
        assert backend.read_registry() == {}
