"""CellDB: AnnData-compatible user interface for CellVault."""

import os
import shutil
import warnings
from pathlib import Path
from typing import Optional, Set

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .backend import DuckDBZarrBackend
from .registry import NameRegistry
from .validator import PipelineStateValidator


def _convert_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to strings for DuckDB compatibility."""
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    return df


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
        if not NameRegistry.is_canonical(key):
            raise ValueError(
                f"obsm key '{key}' is not a registered canonical name. "
                f"Use CellVault tools (pca, umap, ...) or NameRegistry.register() first."
            )
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
        if not NameRegistry.is_canonical(key):
            raise ValueError(
                f"obsp key '{key}' is not a registered canonical name. "
                f"Use CellVault tools (neighbors, ...) or NameRegistry.register() first."
            )
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
        adata = ad.read_h5ad(h5ad_path)
        return cls.from_anndata(adata, cvdb_path)

    @classmethod
    def from_anndata(cls, adata: ad.AnnData, cvdb_path: str) -> "CellDB":
        """Convert an AnnData object to CellVault format."""
        if os.path.exists(cvdb_path):
            shutil.rmtree(cvdb_path)

        backend = DuckDBZarrBackend(cvdb_path)

        # obs: store index as _index column
        obs_df = adata.obs.copy()
        obs_df["_index"] = obs_df.index
        obs_df = _convert_categoricals(obs_df)
        backend.write_obs(obs_df)

        # var
        var_df = adata.var.copy()
        var_df = _convert_categoricals(var_df)
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

        # uns: serialize what we can, warn on failures (never silently discard)
        import warnings
        uns_raw = dict(adata.uns)
        uns_safe = {}
        uns_dropped = []
        for k, v in uns_raw.items():
            try:
                from .backend import _serialize_uns
                serialized = _serialize_uns(v)
                # Verify round-trip via JSON
                import json
                json.dumps(serialized)
                uns_safe[k] = serialized
            except (TypeError, ValueError, OverflowError) as e:
                uns_dropped.append((k, type(v).__name__, str(e)))
        if uns_dropped:
            dropped_keys = [f"'{k}' ({t})" for k, t, _ in uns_dropped]
            warnings.warn(
                f"CellVault: {len(uns_dropped)} uns key(s) could not be serialized "
                f"and were skipped: {', '.join(dropped_keys)}. "
                f"Use JSON-compatible types to preserve all metadata.",
                UserWarning,
                stacklevel=2,
            )
        backend.write_uns(uns_safe)

        db = cls(backend)
        return db

    def to_anndata(
        self,
        slots: Optional[Set[str]] = None,
        obsm_keys: Optional[list[str]] = None,
        obsp_keys: Optional[list[str]] = None,
    ) -> ad.AnnData:
        """Convert CellVault database to AnnData object.

        Selective materialization: only load the data slots you need.

        Args:
            slots: Set of slots to load. Default: all.
                   Valid: {'X', 'obs', 'var', 'obsm', 'obsp', 'uns'}
            obsm_keys: If given, only load these obsm keys (instead of all).
            obsp_keys: If given, only load these obsp keys (instead of all).

        Examples:
            # Full load (backward-compatible)
            adata = cdb.to_anndata()

            # PCA only needs X + obs + var
            adata = cdb.to_anndata(slots={'X', 'obs', 'var'})

            # UMAP only needs specific obsp keys + uns
            adata = cdb.to_anndata(
                slots={'obs', 'obsp', 'uns'},
                obsp_keys=['connectivities', 'distances'],
            )
        """
        load_all = slots is None
        if slots is None:
            slots = {'X', 'obs', 'var', 'obsm', 'obsp', 'uns'}

        # obs/var: always need at least a stub for AnnData shape
        obs = self.obs if 'obs' in slots else pd.DataFrame(index=range(self.n_obs))
        var = self.var if 'var' in slots else pd.DataFrame(index=range(self.n_vars))
        X = self.X if 'X' in slots else None

        adata = ad.AnnData(X=X, obs=obs, var=var)

        if 'obsm' in slots:
            keys_to_load = obsm_keys if obsm_keys is not None else self.obsm.keys()
            for key in keys_to_load:
                adata.obsm[key] = self.obsm[key]

        if 'obsp' in slots:
            keys_to_load = obsp_keys if obsp_keys is not None else self.obsp.keys()
            for key in keys_to_load:
                adata.obsp[key] = self.obsp[key]

        if 'uns' in slots:
            adata.uns = self.uns

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
        obs_df = _convert_categoricals(obs_df)
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
        return self._backend.count_obs()

    @property
    def n_vars(self) -> int:
        return self._backend.count_vars()

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
