"""Shared test fixtures for CellVault tests."""

import os
import shutil
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


@pytest.fixture
def tmp_path_cv(tmp_path):
    """Provide a temporary path for CellVault databases."""
    cvdb_path = tmp_path / "test.cvdb"
    yield str(cvdb_path)
    if cvdb_path.exists():
        shutil.rmtree(cvdb_path)


@pytest.fixture
def sample_adata():
    """Create a small sample AnnData for testing."""
    n_obs, n_vars = 100, 50
    X = np.random.rand(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": np.random.choice(["A", "B", "C"], n_obs)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def sample_adata_sparse():
    """Create a sample AnnData with sparse X."""
    n_obs, n_vars = 100, 50
    X = sparse.random(n_obs, n_vars, density=0.1, format="csr", dtype=np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def sample_celldb(sample_adata, tmp_path_cv):
    """Create a CellDB instance from sample data."""
    from cellvault import CellDB
    cdb = CellDB.from_anndata(sample_adata, tmp_path_cv)
    yield cdb
    cdb.close()
