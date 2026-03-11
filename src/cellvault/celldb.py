"""CellDB: AnnData-compatible user interface for CellVault."""

import os
import shutil
import time
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .backend import DuckDBZarrBackend
from .registry import NameRegistry
from .validator import PipelineStateValidator
from ._debug import logger


class _ObsmAccessor:
    """Dict-like accessor for obsm, enforcing NameRegistry."""

    def __init__(self, backend: DuckDBZarrBackend):
        self._backend = backend

    def __getitem__(self, key: str) -> np.ndarray:
        data = self._backend.read_obsm(key)
        if data is None:
            raise KeyError(f"obsm key '{key}' not found. Available: {self.keys()}")
        return data

    def __setitem__(self, key: str, value: np.ndarray):
        self._backend.write_obsm(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._backend.obsm_keys

    def keys(self) -> list[str]:
        return self._backend.obsm_keys

    def __repr__(self):
        return f"ObsmAccessor(keys={self.keys()})"


class _ObspAccessor:
    """Dict-like accessor for obsp."""

    def __init__(self, backend: DuckDBZarrBackend):
        self._backend = backend

    def __getitem__(self, key: str):
        data = self._backend.read_obsp(key)
        if data is None:
            raise KeyError(f"obsp key '{key}' not found. Available: {self.keys()}")
        return data

    def __setitem__(self, key: str, value):
        self._backend.write_obsp(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._backend.obsp_keys

    def keys(self) -> list[str]:
        return self._backend.obsp_keys

    def __repr__(self):
        return f"ObspAccessor(keys={self.keys()})"


class CellDB:
    """AnnData-compatible interface backed by CellVault storage.

    Provides .obs, .var, .X, .obsm, .obsp, .uns attributes
    with the same API as AnnData, but backed by DuckDB + Zarr.
    """

    def __init__(self, backend: DuckDBZarrBackend):
        self._backend = backend
        self.obsm = _ObsmAccessor(backend)
        self.obsp = _ObspAccessor(backend)

    @classmethod
    def create(cls, path: str) -> "CellDB":
        """Create a new CellVault database."""
        backend = DuckDBZarrBackend(path)
        return cls(backend)

    @classmethod
    def open(cls, path: str) -> "CellDB":
        """Open an existing CellVault database."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"CellVault database not found: {path}")
        backend = DuckDBZarrBackend(path)
        return cls(backend)

    @classmethod
    def from_h5ad(cls, h5ad_path: str, cvdb_path: str) -> "CellDB":
        """Convert an h5ad file to CellVault format."""
        t0 = time.perf_counter()
        adata = ad.read_h5ad(h5ad_path)
        result = cls.from_anndata(adata, cvdb_path)
        logger.debug(
            "from_h5ad: path=%s, n_obs=%d, n_vars=%d, elapsed=%.3fs",
            h5ad_path, result.n_obs, result.n_vars, time.perf_counter() - t0,
        )
        return result

    @classmethod
    def from_anndata(cls, adata: ad.AnnData, cvdb_path: str) -> "CellDB":
        """Convert an AnnData object to CellVault format."""
        t0 = time.perf_counter()
        if os.path.exists(cvdb_path):
            shutil.rmtree(cvdb_path)

        backend = DuckDBZarrBackend(cvdb_path)

        # obs: store index as _index column
        obs_df = adata.obs.copy()
        obs_df["_index"] = obs_df.index
        # Convert categoricals to strings for DuckDB
        for col in obs_df.columns:
            if isinstance(obs_df[col].dtype, pd.CategoricalDtype):
                obs_df[col] = obs_df[col].astype(str)
        backend.write_obs(obs_df)

        # var
        var_df = adata.var.copy()
        for col in var_df.columns:
            if isinstance(var_df[col].dtype, pd.CategoricalDtype):
                var_df[col] = var_df[col].astype(str)
        backend.write_var(var_df)

        # X
        if adata.X is not None:
            backend.write_X(adata.X)

        # obsm
        for key in adata.obsm.keys():
            backend.write_obsm(key, adata.obsm[key])

        # obsp
        for key in adata.obsp.keys():
            backend.write_obsp(key, adata.obsp[key])

        # uns (best effort - skip non-serializable)
        try:
            backend.write_uns(dict(adata.uns))
        except (TypeError, ValueError):
            backend.write_uns({})

        db = cls(backend)
        logger.debug(
            "from_anndata: path=%s, n_obs=%d, n_vars=%d, elapsed=%.3fs",
            cvdb_path, db.n_obs, db.n_vars, time.perf_counter() - t0,
        )
        return db

    def to_anndata(self) -> ad.AnnData:
        """Convert CellVault database to AnnData object."""
        t0 = time.perf_counter()
        obs = self.obs
        var = self.var
        X = self.X

        adata = ad.AnnData(X=X, obs=obs, var=var)

        for key in self.obsm.keys():
            adata.obsm[key] = self.obsm[key]

        for key in self.obsp.keys():
            adata.obsp[key] = self.obsp[key]

        adata.uns = self.uns

        logger.debug(
            "to_anndata: n_obs=%d, n_vars=%d, elapsed=%.3fs",
            adata.n_obs, adata.n_vars, time.perf_counter() - t0,
        )
        return adata

    def to_h5ad(self, path: str):
        """Export CellVault database to h5ad file."""
        adata = self.to_anndata()
        adata.write_h5ad(path)

    # ── Properties ───────────────────────────────────────────────

    @property
    def obs(self) -> pd.DataFrame:
        return self._backend.read_obs()

    @obs.setter
    def obs(self, df: pd.DataFrame):
        obs_df = df.copy()
        obs_df["_index"] = obs_df.index
        for col in obs_df.columns:
            if isinstance(obs_df[col].dtype, pd.CategoricalDtype):
                obs_df[col] = obs_df[col].astype(str)
        self._backend.write_obs(obs_df)

    @property
    def var(self) -> pd.DataFrame:
        return self._backend.read_var()

    @var.setter
    def var(self, df: pd.DataFrame):
        self._backend.write_var(df)

    @property
    def X(self):
        return self._backend.read_X()

    @X.setter
    def X(self, value):
        self._backend.write_X(value)

    @property
    def uns(self) -> dict:
        return self._backend.read_uns()

    @uns.setter
    def uns(self, value: dict):
        self._backend.write_uns(value)

    @property
    def n_obs(self) -> int:
        obs = self.obs
        return len(obs)

    @property
    def n_vars(self) -> int:
        var = self.var
        return len(var)

    @property
    def shape(self) -> tuple:
        return (self.n_obs, self.n_vars)

    # ── Partial update ───────────────────────────────────────────

    def update_obs(self, column: str, index_mask, values):
        """Partial update: modify specific rows of a specific obs column."""
        self._backend.update_obs(column, index_mask, values)

    # ── State query (for validator) ──────────────────────────────

    def get_state(self) -> dict:
        """Get current state summary for validation."""
        return {
            "X_exists": self._backend.X_exists,
            "obsm_keys": self._backend.obsm_keys,
            "obsp_keys": self._backend.obsp_keys,
            "obs_columns": self._backend.obs_columns,
            "uns_keys": list(self.uns.keys()),
        }

    # ── Provenance ───────────────────────────────────────────────

    @property
    def provenance(self):
        return self._backend.provenance

    def close(self):
        self._backend.close()

    def __repr__(self):
        return f"CellDB(n_obs={self.n_obs}, n_vars={self.n_vars}, path='{self._backend.path}')"
