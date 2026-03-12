"""Round-trip fidelity tests: AnnData → CellDB → AnnData.

These tests verify that data is preserved losslessly through conversion,
covering various data types, sparsity patterns, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import anndata as ad

from cellvault.celldb import CellDB


class TestRoundTripSparseCSR:
    def test_standard_sparse(self, tmp_path):
        np.random.seed(1)
        X = sparse.random(200, 100, density=0.15, format="csr", dtype=np.float32)
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(200)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(100)]),
        )
        cdb = CellDB.from_anndata(adata, str(tmp_path / "csr.cvdb"))
        adata_rt = cdb.to_anndata()
        diff = np.abs(adata_rt.X - X).max()
        assert diff == 0.0
        assert adata_rt.X.shape == X.shape
        cdb.close()

    def test_ultra_sparse(self, tmp_path):
        """Extremely sparse matrix (density=0.001)."""
        X = sparse.random(500, 300, density=0.001, format="csr", dtype=np.float64)
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(500)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(300)]),
        )
        cdb = CellDB.from_anndata(adata, str(tmp_path / "ultra.cvdb"))
        adata_rt = cdb.to_anndata()
        diff = np.abs(adata_rt.X - X).max()
        assert diff == 0.0
        cdb.close()


class TestRoundTripDense:
    def test_dense_small(self, tmp_path):
        X = np.random.randn(20, 10).astype(np.float32)
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(20)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(10)]),
        )
        cdb = CellDB.from_anndata(adata, str(tmp_path / "dense.cvdb"))
        adata_rt = cdb.to_anndata()
        np.testing.assert_array_almost_equal(adata_rt.X, X)
        cdb.close()

    def test_dense_float64(self, tmp_path):
        X = np.random.randn(30, 15).astype(np.float64)
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(15)]),
        )
        cdb = CellDB.from_anndata(adata, str(tmp_path / "f64.cvdb"))
        adata_rt = cdb.to_anndata()
        np.testing.assert_array_almost_equal(adata_rt.X, X)
        cdb.close()


class TestRoundTripObs:
    def test_mixed_obs_types(self, tmp_path):
        """obs with string, int, float, categorical columns."""
        obs = pd.DataFrame(
            {
                "cell_type": pd.Categorical(["T", "B", "NK", "T", "B"]),
                "n_genes": [100, 200, 300, 400, 500],
                "score": [0.1, 0.2, 0.3, 0.4, 0.5],
                "sample": ["S1", "S2", "S1", "S2", "S1"],
            },
            index=[f"c{i}" for i in range(5)],
        )
        adata = ad.AnnData(
            X=np.ones((5, 3), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )
        cdb = CellDB.from_anndata(adata, str(tmp_path / "mixed.cvdb"))
        obs_rt = cdb.obs

        assert obs_rt["n_genes"].tolist() == [100, 200, 300, 400, 500]
        assert obs_rt["score"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
        # Categoricals converted to strings
        assert obs_rt["cell_type"].tolist() == ["T", "B", "NK", "T", "B"]
        assert obs_rt["sample"].tolist() == ["S1", "S2", "S1", "S2", "S1"]
        cdb.close()

    def test_obs_index_preserved(self, tmp_path):
        obs = pd.DataFrame(
            {"x": [1.0, 2.0]}, index=["alpha", "beta"]
        )
        adata = ad.AnnData(
            X=np.ones((2, 2), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=["g0", "g1"]),
        )
        cdb = CellDB.from_anndata(adata, str(tmp_path / "idx.cvdb"))
        obs_rt = cdb.obs
        assert list(obs_rt.index) == ["alpha", "beta"]
        cdb.close()


class TestRoundTripVar:
    def test_var_preserved(self, tmp_path):
        var = pd.DataFrame(
            {"highly_variable": [True, False, True]},
            index=["ACTB", "TP53", "BRCA1"],
        )
        adata = ad.AnnData(
            X=np.ones((2, 3), dtype=np.float32),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=var,
        )
        cdb = CellDB.from_anndata(adata, str(tmp_path / "var.cvdb"))
        var_rt = cdb.var
        assert list(var_rt.index) == ["ACTB", "TP53", "BRCA1"]
        cdb.close()


class TestRoundTripObsm:
    def test_multi_obsm_keys(self, tmp_path):
        n = 10
        adata = ad.AnnData(
            X=np.ones((n, 5), dtype=np.float32),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )
        pca = np.random.randn(n, 50).astype(np.float32)
        umap_emb = np.random.randn(n, 2).astype(np.float32)
        adata.obsm["X_pca"] = pca
        adata.obsm["X_umap"] = umap_emb

        cdb = CellDB.from_anndata(adata, str(tmp_path / "obsm.cvdb"))
        adata_rt = cdb.to_anndata()

        np.testing.assert_array_almost_equal(adata_rt.obsm["X_pca"], pca)
        np.testing.assert_array_almost_equal(adata_rt.obsm["X_umap"], umap_emb)
        cdb.close()


class TestRoundTripObsp:
    def test_sparse_obsp(self, tmp_path):
        n = 10
        adata = ad.AnnData(
            X=np.ones((n, 5), dtype=np.float32),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )
        conn = sparse.random(n, n, density=0.3, format="csr")
        dist = sparse.random(n, n, density=0.3, format="csr")
        adata.obsp["connectivities"] = conn
        adata.obsp["distances"] = dist

        cdb = CellDB.from_anndata(adata, str(tmp_path / "obsp.cvdb"))
        adata_rt = cdb.to_anndata()

        assert np.abs(adata_rt.obsp["connectivities"] - conn).max() == 0.0
        assert np.abs(adata_rt.obsp["distances"] - dist).max() == 0.0
        cdb.close()


class TestRoundTripUns:
    def test_uns_nested_values(self, tmp_path):
        adata = ad.AnnData(
            X=np.ones((2, 2), dtype=np.float32),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1"]),
        )
        adata.uns["str_val"] = "hello"
        adata.uns["int_val"] = 42
        adata.uns["nested"] = {"a": 1, "b": [1, 2, 3]}

        cdb = CellDB.from_anndata(adata, str(tmp_path / "uns.cvdb"))
        uns_rt = cdb.uns

        assert uns_rt["str_val"] == "hello"
        assert uns_rt["int_val"] == 42
        assert uns_rt["nested"]["a"] == 1
        assert uns_rt["nested"]["b"] == [1, 2, 3]
        cdb.close()

    def test_uns_numpy_values(self, tmp_path):
        """numpy types should be serialized to JSON-compatible types."""
        adata = ad.AnnData(
            X=np.ones((2, 2), dtype=np.float32),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1"]),
        )
        adata.uns["arr"] = np.array([1.0, 2.0, 3.0])
        adata.uns["int_val"] = np.int64(99)

        cdb = CellDB.from_anndata(adata, str(tmp_path / "np_uns.cvdb"))
        uns_rt = cdb.uns
        assert uns_rt["arr"] == [1.0, 2.0, 3.0]
        assert uns_rt["int_val"] == 99
        cdb.close()


class TestRoundTripFullPipeline:
    def test_pipeline_state_roundtrips(self, celldb_with_pipeline):
        """After full pipeline, to_anndata should capture all computed results."""
        adata = celldb_with_pipeline.to_anndata()
        assert "X_pca" in adata.obsm
        assert "X_umap" in adata.obsm
        assert "connectivities" in adata.obsp
        assert "distances" in adata.obsp
        assert "leiden" in adata.obs.columns
        assert adata.X is not None
        assert adata.shape[0] == 100
