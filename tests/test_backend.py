"""Tests for DuckDBZarrBackend storage operations."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellvault.backend import DuckDBZarrBackend, _serialize_uns


# ── obs (DuckDB) ────────────────────────────────────────────────────


class TestObsOperations:
    def test_write_and_read_obs(self, backend):
        df = pd.DataFrame(
            {"batch": ["A", "B", "C"], "_index": ["c0", "c1", "c2"]},
        )
        backend.write_obs(df)
        result = backend.read_obs()
        assert len(result) == 3
        assert "batch" in result.columns

    def test_read_obs_empty(self, backend):
        result = backend.read_obs()
        assert len(result) == 0

    def test_has_obs_false(self, backend):
        assert backend._has_obs() is False

    def test_has_obs_true(self, backend):
        df = pd.DataFrame({"_index": ["a"], "x": [1.0]})
        backend.write_obs(df)
        assert backend._has_obs() is True

    def test_obs_columns(self, backend):
        df = pd.DataFrame({"_index": ["a"], "batch": ["A"], "score": [1.0]})
        backend.write_obs(df)
        cols = backend.obs_columns
        assert "batch" in cols
        assert "score" in cols
        assert "_index" not in cols

    def test_update_obs(self, backend):
        df = pd.DataFrame({"_index": ["a", "b", "c"], "score": [1.0, 2.0, 3.0]})
        backend.write_obs(df)
        backend.update_obs("score", ["b"], [99.0])
        result = backend.read_obs()
        assert result.loc["b", "score"] == 99.0
        assert result.loc["a", "score"] == 1.0

    def test_count_obs(self, backend):
        df = pd.DataFrame({"_index": ["a", "b", "c"], "x": [1, 2, 3]})
        backend.write_obs(df)
        assert backend.count_obs() == 3

    def test_count_obs_empty(self, backend):
        assert backend.count_obs() == 0


# ── add_obs_column type inference ───────────────────────────────────


class TestAddObsColumn:
    @pytest.fixture(autouse=True)
    def _setup_obs(self, backend):
        df = pd.DataFrame({"_index": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
        backend.write_obs(df)
        self.backend = backend

    def test_add_double_column(self):
        self.backend.add_obs_column("score", [0.1, 0.2, 0.3])
        result = self.backend.read_obs()
        assert "score" in result.columns
        assert result["score"].dtype in (np.float64, np.float32)

    def test_add_string_column(self):
        self.backend.add_obs_column("cell_type", ["T", "B", "NK"])
        result = self.backend.read_obs()
        assert result["cell_type"].tolist() == ["T", "B", "NK"]

    def test_add_integer_column(self):
        self.backend.add_obs_column("n_genes", [100, 200, 300])
        result = self.backend.read_obs()
        assert "n_genes" in result.columns
        assert result["n_genes"].tolist() == [100, 200, 300]

    def test_add_boolean_column(self):
        self.backend.add_obs_column("is_doublet", [True, False, True])
        result = self.backend.read_obs()
        assert result["is_doublet"].tolist() == [True, False, True]

    def test_add_column_no_obs_raises(self, tmp_path):
        b = DuckDBZarrBackend(str(tmp_path / "empty.cvdb"))
        with pytest.raises(ValueError, match="No obs table"):
            b.add_obs_column("x", [1])
        b.close()


# ── var (Parquet) ───────────────────────────────────────────────────


class TestVarOperations:
    def test_write_and_read_var(self, backend):
        df = pd.DataFrame(
            {"highly_variable": [True, False]},
            index=["gene_0", "gene_1"],
        )
        backend.write_var(df)
        result = backend.read_var()
        assert len(result) == 2
        assert list(result.index) == ["gene_0", "gene_1"]

    def test_read_var_empty(self, backend):
        result = backend.read_var()
        assert len(result) == 0

    def test_count_vars(self, backend):
        df = pd.DataFrame(index=["g0", "g1", "g2"])
        backend.write_var(df)
        assert backend.count_vars() == 3

    def test_count_vars_empty(self, backend):
        assert backend.count_vars() == 0


# ── X (Zarr) ───────────────────────────────────────────────────


class TestXOperations:
    def test_write_read_dense(self, backend):
        X = np.random.randn(10, 5).astype(np.float32)
        backend.write_X(X)
        result = backend.read_X()
        np.testing.assert_array_almost_equal(result, X)

    def test_write_read_sparse(self, backend):
        X = sparse.random(50, 20, density=0.1, format="csr", dtype=np.float32)
        backend.write_X(X)
        result = backend.read_X()
        assert sparse.issparse(result)
        diff = np.abs(result - X).max()
        assert diff == 0.0

    def test_read_X_nonexistent(self, backend):
        assert backend.read_X() is None

    def test_X_exists(self, backend):
        assert backend.X_exists is False
        backend.write_X(np.array([[1.0]]))
        assert backend.X_exists is True

    def test_overwrite_X(self, backend):
        """Overwriting X should work atomically without data loss."""
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        backend.write_X(X1)
        backend.write_X(X2)
        result = backend.read_X()
        np.testing.assert_array_equal(result, X2)


# ── obsm (Zarr per key) ────────────────────────────────────────────


class TestObsmOperations:
    def test_write_read_obsm(self, backend):
        data = np.random.randn(10, 2).astype(np.float32)
        backend.write_obsm("X_pca", data)
        result = backend.read_obsm("X_pca")
        np.testing.assert_array_almost_equal(result, data)

    def test_obsm_keys(self, backend):
        backend.write_obsm("X_pca", np.zeros((5, 2)))
        backend.write_obsm("X_umap", np.zeros((5, 2)))
        keys = backend.obsm_keys
        assert "X_pca" in keys
        assert "X_umap" in keys

    def test_read_obsm_nonexistent(self, backend):
        assert backend.read_obsm("nonexistent") is None

    def test_overwrite_obsm(self, backend):
        d1 = np.array([[1.0, 2.0]])
        d2 = np.array([[3.0, 4.0]])
        backend.write_obsm("X_pca", d1)
        backend.write_obsm("X_pca", d2)
        result = backend.read_obsm("X_pca")
        np.testing.assert_array_equal(result, d2)


# ── obsp (Zarr per key, sparse) ────────────────────────────────────


class TestObspOperations:
    def test_write_read_sparse_obsp(self, backend):
        data = sparse.random(10, 10, density=0.3, format="csr")
        backend.write_obsp("connectivities", data)
        result = backend.read_obsp("connectivities")
        assert sparse.issparse(result)
        assert np.abs(result - data).max() == 0.0

    def test_write_read_dense_obsp(self, backend):
        data = np.random.randn(5, 5)
        backend.write_obsp("distances", data)
        result = backend.read_obsp("distances")
        np.testing.assert_array_almost_equal(result, data)

    def test_obsp_keys(self, backend):
        backend.write_obsp("connectivities", np.zeros((3, 3)))
        keys = backend.obsp_keys
        assert "connectivities" in keys

    def test_read_obsp_nonexistent(self, backend):
        assert backend.read_obsp("nonexistent") is None


# ── uns (JSON) ──────────────────────────────────────────────────────


class TestUnsOperations:
    def test_write_read_uns(self, backend):
        uns = {"project": "test", "version": 1}
        backend.write_uns(uns)
        result = backend.read_uns()
        assert result == uns

    def test_read_uns_empty(self, backend):
        assert backend.read_uns() == {}

    def test_uns_nested(self, backend):
        uns = {"params": {"n_comps": 50, "seed": 42}, "name": "exp1"}
        backend.write_uns(uns)
        result = backend.read_uns()
        assert result["params"]["n_comps"] == 50


class TestSerializeUns:
    def test_numpy_array(self):
        result = _serialize_uns(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_numpy_int(self):
        result = _serialize_uns(np.int64(42))
        assert result == 42 and isinstance(result, int)

    def test_numpy_float(self):
        result = _serialize_uns(np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_bool(self):
        result = _serialize_uns(np.bool_(True))
        assert result is True

    def test_dict_recursive(self):
        result = _serialize_uns({"a": np.int64(1), "b": {"c": np.array([1, 2])}})
        assert result == {"a": 1, "b": {"c": [1, 2]}}

    def test_dataframe(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = _serialize_uns(df)
        assert isinstance(result, dict)

    def test_plain_types_passthrough(self):
        assert _serialize_uns("hello") == "hello"
        assert _serialize_uns(42) == 42
        assert _serialize_uns([1, 2]) == [1, 2]


# ── Close / lifecycle ──────────────────────────────────────────────


class TestBackendLifecycle:
    def test_close_and_reopen(self, tmp_path):
        path = str(tmp_path / "lifecycle.cvdb")
        b = DuckDBZarrBackend(path)
        b.write_obs(pd.DataFrame({"_index": ["a"], "x": [1.0]}))
        b.write_X(np.array([[1.0, 2.0]]))
        b.close()

        # Reopen
        b2 = DuckDBZarrBackend(path)
        assert b2.count_obs() == 1
        assert b2.read_X() is not None
        b2.close()
