"""Shared fixtures for CellVault test suite."""

import os
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# Ensure cellvault package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cellvault.backend import DuckDBZarrBackend
from cellvault.celldb import CellDB


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory path as string."""
    return str(tmp_path)


@pytest.fixture
def backend(tmp_path):
    """Create a fresh DuckDBZarrBackend."""
    b = DuckDBZarrBackend(str(tmp_path / "test.cvdb"))
    yield b
    b.close()


@pytest.fixture
def small_adata():
    """Small synthetic AnnData (100 cells x 50 genes) with sparse X."""
    np.random.seed(42)
    n_obs, n_vars = 100, 50
    X = sparse.random(n_obs, n_vars, density=0.2, format="csr", dtype=np.float32)
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(np.random.choice(["A", "B", "C"], n_obs)),
            "n_counts": np.random.randint(100, 5000, n_obs),
            "percent_mito": np.random.uniform(0.0, 0.1, n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"highly_variable": np.random.choice([True, False], n_vars)},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["project"] = "test"
    adata.uns["params"] = {"seed": 42, "version": "1.0"}
    return adata


@pytest.fixture
def dense_adata():
    """Small AnnData with dense X."""
    np.random.seed(99)
    n_obs, n_vars = 30, 20
    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def celldb_empty(tmp_path):
    """Create an empty CellDB."""
    cdb = CellDB.create(str(tmp_path / "empty.cvdb"))
    yield cdb
    cdb.close()


@pytest.fixture
def celldb_small(tmp_path, small_adata):
    """Create a CellDB populated from small_adata."""
    cdb = CellDB.from_anndata(small_adata, str(tmp_path / "small.cvdb"))
    yield cdb
    cdb.close()


@pytest.fixture
def celldb_with_pipeline(tmp_path, small_adata):
    """Create a CellDB with full pipeline run (PCA → neighbors → UMAP → leiden)."""
    from cellvault import tools

    cdb = CellDB.from_anndata(small_adata, str(tmp_path / "pipeline.cvdb"))
    tools.pca(cdb, n_comps=20)
    tools.neighbors(cdb, n_neighbors=10)
    tools.umap(cdb)
    tools.leiden(cdb, resolution=0.5)
    yield cdb
    cdb.close()
