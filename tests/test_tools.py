"""Tests for Scanpy tool wrappers."""

import numpy as np
import pytest

from cellvault import CellDB
from cellvault.validator import CellVaultStateError


# Check if scanpy is available
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

pytestmark = pytest.mark.skipif(not HAS_SCANPY, reason="scanpy not installed")


@pytest.fixture
def cdb_with_X(sample_adata, tmp_path):
    """CellDB with expression matrix ready for PCA."""
    path = str(tmp_path / "tools_test.cvdb")
    cdb = CellDB.from_anndata(sample_adata, path)
    yield cdb
    cdb.close()


class TestPCA:
    def test_pca_runs(self, cdb_with_X):
        from cellvault.tools import pca
        key = pca(cdb_with_X, n_comps=10)
        assert key == "X_pca"
        assert "X_pca" in cdb_with_X.obsm
        assert cdb_with_X.obsm["X_pca"].shape == (100, 10)

    def test_pca_provenance(self, cdb_with_X):
        from cellvault.tools import pca
        pca(cdb_with_X, n_comps=10)
        entries = cdb_with_X.provenance.query(operation="pca")
        assert len(entries) >= 1

    def test_pca_variance_ratio_in_uns(self, cdb_with_X):
        from cellvault.tools import pca
        pca(cdb_with_X, n_comps=10)
        uns = cdb_with_X.uns
        assert "X_pca_variance_ratio" in uns


class TestNeighbors:
    def test_neighbors_runs(self, cdb_with_X):
        from cellvault.tools import pca, neighbors
        pca(cdb_with_X, n_comps=10)
        key = neighbors(cdb_with_X, n_neighbors=5)
        assert key == "neighbors"
        assert "connectivities" in cdb_with_X.obsp
        assert "distances" in cdb_with_X.obsp

    def test_neighbors_without_pca_fails(self, cdb_with_X):
        from cellvault.tools import neighbors
        with pytest.raises(CellVaultStateError):
            neighbors(cdb_with_X)


class TestUMAP:
    def test_umap_runs(self, cdb_with_X):
        from cellvault.tools import pca, neighbors, umap
        pca(cdb_with_X, n_comps=10)
        neighbors(cdb_with_X, n_neighbors=5)
        key = umap(cdb_with_X)
        assert key == "X_umap"
        assert "X_umap" in cdb_with_X.obsm
        assert cdb_with_X.obsm["X_umap"].shape == (100, 2)

    def test_umap_without_neighbors_fails(self, cdb_with_X):
        from cellvault.tools import pca, umap
        pca(cdb_with_X, n_comps=10)
        with pytest.raises(CellVaultStateError):
            umap(cdb_with_X)


class TestLeiden:
    def test_leiden_runs(self, cdb_with_X):
        from cellvault.tools import pca, neighbors, leiden
        pca(cdb_with_X, n_comps=10)
        neighbors(cdb_with_X, n_neighbors=5)
        key = leiden(cdb_with_X, resolution=0.5)
        assert key == "leiden"
        assert "leiden" in cdb_with_X.obs.columns

    def test_leiden_without_neighbors_fails(self, cdb_with_X):
        from cellvault.tools import leiden
        with pytest.raises(CellVaultStateError):
            leiden(cdb_with_X)


class TestLazyImport:
    def test_import_error_message(self, monkeypatch):
        """Test that missing scanpy gives a clear error."""
        from cellvault import tools
        # Temporarily make _import_scanpy fail
        def mock_import():
            raise ImportError("scanpy is required for cellvault.tools. Install it with: pip install cellvault[scanpy]")
        monkeypatch.setattr(tools, "_import_scanpy", mock_import)
        with pytest.raises(ImportError, match="cellvault.tools"):
            tools.pca(None)
