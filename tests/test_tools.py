"""Tests for Scanpy tool wrappers with selective loading."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellvault.celldb import CellDB
from cellvault.validator import CellVaultStateError
from cellvault import tools


# ── Full pipeline ───────────────────────────────────────────────────


class TestFullPipeline:
    def test_pca_neighbors_umap_leiden(self, celldb_small):
        pca_key = tools.pca(celldb_small, n_comps=20)
        assert pca_key == "X_pca"
        assert "X_pca" in celldb_small.obsm.keys()
        assert celldb_small.obsm["X_pca"].shape == (100, 20)

        nbr_key = tools.neighbors(celldb_small, n_neighbors=10)
        assert nbr_key == "neighbors"
        assert "connectivities" in celldb_small.obsp.keys()
        assert "distances" in celldb_small.obsp.keys()

        umap_key = tools.umap(celldb_small)
        assert umap_key == "X_umap"
        assert celldb_small.obsm["X_umap"].shape == (100, 2)

        leiden_key = tools.leiden(celldb_small, resolution=0.5)
        assert leiden_key == "leiden"
        obs = celldb_small.obs
        assert "leiden" in obs.columns
        assert len(obs["leiden"].unique()) >= 1


# ── Individual tools ────────────────────────────────────────────────


class TestPCA:
    def test_pca_stores_variance(self, celldb_small):
        tools.pca(celldb_small, n_comps=10)
        uns = celldb_small.uns
        assert "X_pca_variance_ratio" in uns
        assert len(uns["X_pca_variance_ratio"]) == 10

    def test_pca_custom_n_comps(self, celldb_small):
        tools.pca(celldb_small, n_comps=5)
        assert celldb_small.obsm["X_pca"].shape[1] == 5

    def test_pca_validates_X(self, celldb_empty):
        with pytest.raises(CellVaultStateError, match="expression matrix"):
            tools.pca(celldb_empty)


class TestNeighbors:
    def test_neighbors_validates_pca(self, celldb_small):
        """neighbors requires PCA to be computed first."""
        with pytest.raises(CellVaultStateError, match="obsm"):
            tools.neighbors(celldb_small)

    def test_neighbors_stores_uns_params(self, celldb_small):
        tools.pca(celldb_small, n_comps=20)
        tools.neighbors(celldb_small, n_neighbors=15)
        uns = celldb_small.uns
        assert "neighbors" in uns
        assert uns["neighbors"]["params"]["n_neighbors"] == 15
        assert uns["neighbors"]["connectivities_key"] == "connectivities"
        assert uns["neighbors"]["distances_key"] == "distances"


class TestUMAP:
    def test_umap_validates_graph(self, celldb_small):
        """umap requires neighbor graph."""
        tools.pca(celldb_small, n_comps=20)
        with pytest.raises(CellVaultStateError, match="obsp"):
            tools.umap(celldb_small)

    def test_umap_output_shape(self, celldb_small):
        tools.pca(celldb_small, n_comps=20)
        tools.neighbors(celldb_small, n_neighbors=10)
        tools.umap(celldb_small)
        assert celldb_small.obsm["X_umap"].shape == (100, 2)


class TestLeiden:
    def test_leiden_validates_graph(self, celldb_small):
        tools.pca(celldb_small, n_comps=20)
        with pytest.raises(CellVaultStateError, match="obsp"):
            tools.leiden(celldb_small)

    def test_leiden_uses_add_obs_column(self, celldb_small):
        """When leiden column doesn't exist, should use add_obs_column (not full rewrite)."""
        tools.pca(celldb_small, n_comps=20)
        tools.neighbors(celldb_small, n_neighbors=10)
        tools.leiden(celldb_small, resolution=0.5)
        obs = celldb_small.obs
        assert "leiden" in obs.columns
        # Values should be string cluster labels
        assert all(isinstance(v, str) for v in obs["leiden"].values)

    def test_leiden_different_resolutions(self, celldb_small):
        tools.pca(celldb_small, n_comps=20)
        tools.neighbors(celldb_small, n_neighbors=10)
        tools.leiden(celldb_small, resolution=0.1)
        n_low = celldb_small.obs["leiden"].nunique()
        # Re-run with higher resolution (overwrites via full obs setter)
        tools.leiden(celldb_small, resolution=2.0)
        n_high = celldb_small.obs["leiden"].nunique()
        assert n_high >= n_low


# ── Provenance logging ──────────────────────────────────────────────


class TestToolsProvenance:
    def test_pipeline_provenance(self, celldb_with_pipeline):
        log = celldb_with_pipeline.provenance.read_log()
        ops = [e["operation"] for e in log]
        assert "pca" in ops
        assert "neighbors" in ops
        assert "umap" in ops
        assert "leiden" in ops

    def test_pca_provenance_params(self, celldb_small):
        tools.pca(celldb_small, n_comps=15)
        log = celldb_small.provenance.read_log()
        pca_entries = [e for e in log if e["operation"] == "pca"]
        assert len(pca_entries) == 1
        assert pca_entries[0]["params"]["n_comps"] == 15
