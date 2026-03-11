"""Tests for CellDB integration."""

import os
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellvault import CellDB


class TestCellDBCreate:
    def test_create_empty(self, tmp_path):
        path = str(tmp_path / "empty.cvdb")
        cdb = CellDB.create(path)
        assert cdb.n_obs == 0
        assert cdb.n_vars == 0
        cdb.close()

    def test_open_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            CellDB.open("/nonexistent/path.cvdb")

    def test_open_existing(self, sample_celldb, tmp_path_cv):
        sample_celldb.close()
        cdb2 = CellDB.open(tmp_path_cv)
        assert cdb2.n_obs == 100
        assert cdb2.n_vars == 50
        cdb2.close()


class TestFromAnndata:
    def test_from_anndata_shape(self, sample_adata, tmp_path):
        path = str(tmp_path / "test.cvdb")
        cdb = CellDB.from_anndata(sample_adata, path)
        assert cdb.n_obs == 100
        assert cdb.n_vars == 50
        assert cdb.shape == (100, 50)
        cdb.close()

    def test_from_anndata_sparse(self, sample_adata_sparse, tmp_path):
        path = str(tmp_path / "test_sparse.cvdb")
        cdb = CellDB.from_anndata(sample_adata_sparse, path)
        X = cdb.X
        assert sparse.issparse(X)
        assert X.shape == (100, 50)
        cdb.close()

    def test_from_anndata_overwrites(self, sample_adata, tmp_path):
        path = str(tmp_path / "test.cvdb")
        cdb1 = CellDB.from_anndata(sample_adata, path)
        cdb1.close()
        # Create again at same path
        cdb2 = CellDB.from_anndata(sample_adata, path)
        assert cdb2.n_obs == 100
        cdb2.close()

    def test_from_anndata_categorical_obs(self, tmp_path):
        path = str(tmp_path / "test.cvdb")
        obs = pd.DataFrame(
            {"group": pd.Categorical(["A", "B", "A"])},
            index=["c0", "c1", "c2"],
        )
        adata = ad.AnnData(X=np.zeros((3, 2)), obs=obs)
        cdb = CellDB.from_anndata(adata, path)
        result_obs = cdb.obs
        assert "group" in result_obs.columns
        cdb.close()


class TestRoundTrip:
    def test_dense_roundtrip(self, sample_adata, tmp_path):
        path = str(tmp_path / "rt.cvdb")
        cdb = CellDB.from_anndata(sample_adata, path)
        adata2 = cdb.to_anndata()

        np.testing.assert_array_almost_equal(adata2.X, sample_adata.X)
        assert adata2.n_obs == sample_adata.n_obs
        assert adata2.n_vars == sample_adata.n_vars
        assert list(adata2.obs.columns) == list(sample_adata.obs.columns)
        cdb.close()

    def test_sparse_roundtrip(self, sample_adata_sparse, tmp_path):
        path = str(tmp_path / "rt_sparse.cvdb")
        cdb = CellDB.from_anndata(sample_adata_sparse, path)
        adata2 = cdb.to_anndata()

        assert sparse.issparse(adata2.X)
        np.testing.assert_array_almost_equal(
            adata2.X.toarray(), sample_adata_sparse.X.toarray()
        )
        cdb.close()

    def test_obs_index_preserved(self, sample_adata, tmp_path):
        path = str(tmp_path / "rt.cvdb")
        cdb = CellDB.from_anndata(sample_adata, path)
        adata2 = cdb.to_anndata()
        assert list(adata2.obs.index) == list(sample_adata.obs.index)
        cdb.close()

    def test_var_index_preserved(self, sample_adata, tmp_path):
        path = str(tmp_path / "rt.cvdb")
        cdb = CellDB.from_anndata(sample_adata, path)
        adata2 = cdb.to_anndata()
        assert list(adata2.var.index) == list(sample_adata.var.index)
        cdb.close()


class TestProperties:
    def test_obs_setter(self, sample_celldb):
        obs = sample_celldb.obs
        obs["new_col"] = range(len(obs))
        sample_celldb.obs = obs
        assert "new_col" in sample_celldb.obs.columns

    def test_X_setter(self, sample_celldb):
        new_X = np.zeros((100, 50), dtype=np.float32)
        sample_celldb.X = new_X
        np.testing.assert_array_equal(sample_celldb.X, new_X)

    def test_uns_roundtrip(self, sample_celldb):
        sample_celldb.uns = {"test_key": "test_value", "nested": {"a": 1}}
        uns = sample_celldb.uns
        assert uns["test_key"] == "test_value"
        assert uns["nested"]["a"] == 1

    def test_obsm_accessor(self, sample_celldb):
        data = np.random.rand(100, 3)
        sample_celldb.obsm["X_test"] = data
        assert "X_test" in sample_celldb.obsm
        np.testing.assert_array_almost_equal(sample_celldb.obsm["X_test"], data)

    def test_obsm_missing_raises(self, sample_celldb):
        with pytest.raises(KeyError, match="not found"):
            _ = sample_celldb.obsm["nonexistent"]

    def test_obsp_accessor(self, sample_celldb):
        data = sparse.random(100, 100, density=0.1, format="csr")
        sample_celldb.obsp["test_conn"] = data
        assert "test_conn" in sample_celldb.obsp

    def test_obsp_missing_raises(self, sample_celldb):
        with pytest.raises(KeyError, match="not found"):
            _ = sample_celldb.obsp["nonexistent"]


class TestGetState:
    def test_state_dict_keys(self, sample_celldb):
        state = sample_celldb.get_state()
        assert "X_exists" in state
        assert "obsm_keys" in state
        assert "obsp_keys" in state
        assert "obs_columns" in state
        assert "uns_keys" in state

    def test_state_X_exists(self, sample_celldb):
        assert sample_celldb.get_state()["X_exists"] is True


class TestProvenance:
    def test_provenance_logged(self, sample_celldb):
        entries = sample_celldb.provenance.read_log()
        assert len(entries) > 0
        ops = [e["operation"] for e in entries]
        assert "write_obs" in ops
        assert "write_X" in ops


class TestRepr:
    def test_repr(self, sample_celldb):
        r = repr(sample_celldb)
        assert "CellDB" in r
        assert "n_obs=100" in r
        assert "n_vars=50" in r


class TestH5ad:
    def test_to_h5ad(self, sample_celldb, tmp_path):
        h5ad_path = str(tmp_path / "export.h5ad")
        sample_celldb.to_h5ad(h5ad_path)
        assert os.path.exists(h5ad_path)
        adata = ad.read_h5ad(h5ad_path)
        assert adata.n_obs == 100

    def test_from_h5ad(self, sample_adata, tmp_path):
        h5ad_path = str(tmp_path / "input.h5ad")
        sample_adata.write_h5ad(h5ad_path)
        cvdb_path = str(tmp_path / "from_h5ad.cvdb")
        cdb = CellDB.from_h5ad(h5ad_path, cvdb_path)
        assert cdb.n_obs == 100
        assert cdb.n_vars == 50
        cdb.close()
