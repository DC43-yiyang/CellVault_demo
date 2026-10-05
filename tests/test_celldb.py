"""Tests for CellDB user-facing API."""

import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellvault.celldb import CellDB, _convert_categoricals

# ── _convert_categoricals ──────────────────────────────────────────


class TestConvertCategoricals:
    def test_converts_categorical(self):
        df = pd.DataFrame({"col": pd.Categorical(["a", "b", "c"])})
        result = _convert_categoricals(df.copy())
        assert result["col"].dtype == object

    def test_leaves_non_categorical(self):
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        result = _convert_categoricals(df.copy())
        assert result["col"].dtype == np.float64


# ── create / open / from_anndata / from_h5ad ───────────────────────


class TestCellDBCreation:
    def test_create(self, tmp_path):
        cdb = CellDB.create(str(tmp_path / "new.cvdb"))
        assert cdb.n_obs == 0
        assert cdb.n_vars == 0
        cdb.close()

    def test_open_existing(self, celldb_small):
        assert celldb_small.n_obs == 100
        assert celldb_small.n_vars == 50

    def test_open_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CellDB.open(str(tmp_path / "nope.cvdb"))

    def test_from_anndata(self, tmp_path, small_adata):
        cdb = CellDB.from_anndata(small_adata, str(tmp_path / "from_ad.cvdb"))
        assert cdb.n_obs == small_adata.n_obs
        assert cdb.n_vars == small_adata.n_vars
        cdb.close()

    def test_from_anndata_requires_explicit_overwrite(self, tmp_path, small_adata):
        path = str(tmp_path / "overwrite.cvdb")
        cdb1 = CellDB.from_anndata(small_adata, path)
        cdb1.close()

        with pytest.raises(FileExistsError, match="overwrite=True"):
            CellDB.from_anndata(small_adata, path)

        cdb2 = CellDB.from_anndata(small_adata, path, overwrite=True)
        assert cdb2.n_obs == small_adata.n_obs
        cdb2.close()

    def test_from_h5ad(self, tmp_path, small_adata):
        h5ad_path = str(tmp_path / "input.h5ad")
        small_adata.write_h5ad(h5ad_path)
        cdb = CellDB.from_h5ad(h5ad_path, str(tmp_path / "from_h5ad.cvdb"))
        assert cdb.n_obs == small_adata.n_obs
        cdb.close()


# ── Properties ──────────────────────────────────────────────────────


class TestCellDBProperties:
    def test_obs(self, celldb_small):
        obs = celldb_small.obs
        assert len(obs) == 100
        assert "batch" in obs.columns

    def test_var(self, celldb_small):
        var = celldb_small.var
        assert len(var) == 50

    def test_X_sparse(self, celldb_small):
        X = celldb_small.X
        assert sparse.issparse(X)
        assert X.shape == (100, 50)

    def test_uns(self, celldb_small):
        uns = celldb_small.uns
        assert uns["project"] == "test"
        assert uns["params"]["seed"] == 42

    def test_n_obs_efficient(self, celldb_small):
        """n_obs should use SQL COUNT, not full DataFrame read."""
        assert celldb_small.n_obs == 100

    def test_n_vars_efficient(self, celldb_small):
        assert celldb_small.n_vars == 50

    def test_shape(self, celldb_small):
        assert celldb_small.shape == (100, 50)

    def test_obsm_accessor(self, celldb_small):
        assert celldb_small.obsm.keys() == []

    def test_obsp_accessor(self, celldb_small):
        assert celldb_small.obsp.keys() == []

    def test_obs_setter(self, celldb_small):
        obs = celldb_small.obs
        obs["new_col"] = range(100)
        celldb_small.obs = obs
        reread = celldb_small.obs
        assert "new_col" in reread.columns

    def test_X_setter(self, celldb_small):
        new_X = np.zeros((100, 50), dtype=np.float32)
        celldb_small.X = new_X
        result = celldb_small.X
        np.testing.assert_array_equal(result, new_X)

    def test_uns_setter(self, celldb_small):
        celldb_small.uns = {"new_key": "value"}
        assert celldb_small.uns["new_key"] == "value"


# ── Selective to_anndata ────────────────────────────────────────────


class TestSelectiveToAnndata:
    def test_full_load(self, celldb_small):
        adata = celldb_small.to_anndata()
        assert adata.X is not None
        assert len(adata.obs) == 100
        assert len(adata.var) == 50

    def test_slots_X_obs_var_only(self, celldb_small):
        adata = celldb_small.to_anndata(slots={"X", "obs", "var"})
        assert adata.X is not None
        assert len(adata.obs) == 100
        assert len(adata.obsm) == 0
        assert len(adata.obsp) == 0

    def test_slots_skip_X(self, celldb_small):
        adata = celldb_small.to_anndata(slots={"obs", "var"})
        assert adata.X is None
        assert len(adata.obs) == 100

    def test_slots_uns_only(self, celldb_small):
        adata = celldb_small.to_anndata(slots={"obs", "var", "uns"})
        assert adata.X is None
        assert adata.uns["project"] == "test"

    def test_selective_obsm_keys(self, celldb_with_pipeline):
        adata = celldb_with_pipeline.to_anndata(
            slots={"obs", "var", "obsm"}, obsm_keys=["X_pca"]
        )
        assert "X_pca" in adata.obsm
        assert "X_umap" not in adata.obsm

    def test_selective_obsp_keys(self, celldb_with_pipeline):
        adata = celldb_with_pipeline.to_anndata(
            slots={"obs", "var", "obsp"}, obsp_keys=["connectivities"]
        )
        assert "connectivities" in adata.obsp
        assert "distances" not in adata.obsp


# ── uns serialization warnings ─────────────────────────────────────


class TestUnsWarning:
    def test_non_serializable_uns_warns(self, tmp_path):
        """Non-serializable uns values should warn, not silently discard all."""
        adata = ad.AnnData(
            X=np.array([[1.0]]),
            obs=pd.DataFrame(index=["c0"]),
            var=pd.DataFrame(index=["g0"]),
        )
        adata.uns["good"] = "kept"
        adata.uns["bad"] = {1, 2, 3}  # sets are not JSON-serializable

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cdb = CellDB.from_anndata(adata, str(tmp_path / "warn.cvdb"))
            warns = [x for x in w if "uns key" in str(x.message)]
            assert len(warns) >= 1
            assert "bad" in str(warns[0].message)

        # Good key preserved
        assert cdb.uns["good"] == "kept"
        # Bad key dropped (not silently — warned above)
        assert "bad" not in cdb.uns
        cdb.close()


# ── get_state / provenance / repr ──────────────────────────────────


class TestCellDBMisc:
    def test_get_state(self, celldb_small):
        state = celldb_small.get_state()
        assert state["X_exists"] is True
        assert "batch" in state["obs_columns"]
        assert isinstance(state["obsm_keys"], list)

    def test_get_state_with_pipeline(self, celldb_with_pipeline):
        state = celldb_with_pipeline.get_state()
        assert "X_pca" in state["obsm_keys"]
        assert "connectivities" in state["obsp_keys"]
        assert "leiden" in state["obs_columns"]

    def test_provenance_accessible(self, celldb_small):
        log = celldb_small.provenance.read_log()
        assert len(log) > 0

    def test_repr(self, celldb_small):
        r = repr(celldb_small)
        assert "n_obs=100" in r
        assert "n_vars=50" in r

    def test_to_h5ad_roundtrip(self, tmp_path, celldb_small):
        h5ad_path = str(tmp_path / "export.h5ad")
        celldb_small.to_h5ad(h5ad_path)
        assert os.path.exists(h5ad_path)
        adata = ad.read_h5ad(h5ad_path)
        assert adata.shape == (100, 50)
