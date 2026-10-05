"""CellDB: AnnData-compatible user interface for CellVault."""

import copy
import json
import os
import shutil
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .backend import DuckDBZarrBackend, _serialize_uns
from .registry import NameRegistry


def _convert_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to strings for DuckDB compatibility."""
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str).astype(object)
    return df


def _quote_sql_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


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


class _LayersAccessor:
    def __init__(self, backend: DuckDBZarrBackend):
        self._backend = backend

    def __getitem__(self, key: str):
        data = self._backend.read_layer(key)
        if data is None:
            raise KeyError(f"layer {key!r} not found. Available: {self.keys()}")
        return data

    def __setitem__(self, key: str, value) -> None:
        self._backend.write_layer(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._backend.layer_keys

    def keys(self) -> list[str]:
        return self._backend.layer_keys


class _VarmAccessor:
    def __init__(self, backend: DuckDBZarrBackend):
        self._backend = backend

    def __getitem__(self, key: str):
        data = self._backend.read_varm(key)
        if data is None:
            raise KeyError(f"varm key {key!r} not found. Available: {self.keys()}")
        return data

    def __setitem__(self, key: str, value) -> None:
        self._backend.write_varm(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._backend.varm_keys

    def keys(self) -> list[str]:
        return self._backend.varm_keys


class _VarpAccessor:
    def __init__(self, backend: DuckDBZarrBackend):
        self._backend = backend

    def __getitem__(self, key: str):
        data = self._backend.read_varp(key)
        if data is None:
            raise KeyError(f"varp key {key!r} not found. Available: {self.keys()}")
        return data

    def __setitem__(self, key: str, value) -> None:
        self._backend.write_varp(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._backend.varp_keys

    def keys(self) -> list[str]:
        return self._backend.varp_keys


class _RawVarmAccessor:
    def __init__(self, backend: DuckDBZarrBackend):
        self._backend = backend

    def __getitem__(self, key: str):
        data = self._backend.read_raw_varm(key)
        if data is None:
            raise KeyError(f"raw.varm key {key!r} not found. Available: {self.keys()}")
        return data

    def keys(self) -> list[str]:
        return self._backend.raw_varm_keys

    def __contains__(self, key: str) -> bool:
        return key in self._backend.raw_varm_keys

    def __iter__(self):
        return iter(self.keys())


class _RawAccessor:
    def __init__(self, backend: DuckDBZarrBackend, row_indices=None):
        self._backend = backend
        self._row_indices = row_indices
        self.varm = _RawVarmAccessor(backend)

    @property
    def X(self):
        return self._backend.read_raw_X(row_indices=self._row_indices)

    @property
    def var(self) -> pd.DataFrame:
        return self._backend.read_raw_var()

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    @property
    def n_obs(self) -> int:
        return (
            self._backend.raw_shape[0]
            if self._row_indices is None
            else len(self._row_indices)
        )

    @property
    def n_vars(self) -> int:
        return self._backend.raw_shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_obs, self.n_vars)

    def to_adata(self, obs: pd.DataFrame | None = None) -> ad.AnnData:
        if obs is None:
            obs = pd.DataFrame(index=range(self.n_obs))
        result = ad.AnnData(X=self.X, obs=obs.copy(), var=self.var)
        for key in self.varm:
            result.varm[key] = self.varm[key]
        return result


class _ViewObsmAccessor:
    """Read-only, row-aligned obsm accessor for a CellView."""

    def __init__(self, view):
        self._view = view

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._view._local_obsm:
            return self._view._local_obsm[key]
        data = self._view._parent._backend.read_obsm(
            key,
            row_indices=self._view._row_indices,
        )
        if data is None:
            raise KeyError(f"obsm key '{key}' not found. Available: {self.keys()}")
        return data

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        if not NameRegistry.is_canonical(key):
            raise ValueError(f"obsm key {key!r} is not a registered canonical name")
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[0] != self._view.n_obs:
            raise ValueError(f"view obsm values must have {self._view.n_obs} rows")
        self._view._local_obsm[key] = array

    def __contains__(self, key: str) -> bool:
        return (
            key in self._view._local_obsm
            or key in self._view._parent._backend.obsm_keys
        )

    def keys(self) -> list[str]:
        return sorted(
            set(self._view._parent._backend.obsm_keys) | set(self._view._local_obsm)
        )

    def __repr__(self):
        return f"ViewObsmAccessor(keys={self.keys()})"


class _ViewObspAccessor:
    """Read-only, induced-subgraph obsp accessor for a CellView."""

    def __init__(self, view):
        self._view = view

    def __getitem__(self, key: str):
        if key in self._view._local_obsp:
            return self._view._local_obsp[key]
        data = self._view._parent._backend.read_obsp(
            key,
            row_indices=self._view._row_indices,
        )
        if data is None:
            raise KeyError(f"obsp key '{key}' not found. Available: {self.keys()}")
        return data

    def __setitem__(self, key: str, value) -> None:
        if not NameRegistry.is_canonical(key):
            raise ValueError(f"obsp key {key!r} is not a registered canonical name")
        if getattr(value, "ndim", None) != 2 or value.shape != (
            self._view.n_obs,
            self._view.n_obs,
        ):
            raise ValueError(
                f"view obsp values must have shape {(self._view.n_obs, self._view.n_obs)}"
            )
        self._view._local_obsp[key] = value.copy() if hasattr(value, "copy") else value

    def __contains__(self, key: str) -> bool:
        return (
            key in self._view._local_obsp
            or key in self._view._parent._backend.obsp_keys
        )

    def keys(self) -> list[str]:
        return sorted(
            set(self._view._parent._backend.obsp_keys) | set(self._view._local_obsp)
        )

    def __repr__(self):
        return f"ViewObspAccessor(keys={self.keys()})"


class _ViewLayersAccessor:
    def __init__(self, view):
        self._view = view

    def __getitem__(self, key: str):
        data = self._view._parent._backend.read_layer(
            key,
            row_indices=self._view._row_indices,
        )
        if data is None:
            raise KeyError(f"layer {key!r} not found. Available: {self.keys()}")
        return data

    def __contains__(self, key: str) -> bool:
        return key in self._view._parent._backend.layer_keys

    def keys(self) -> list[str]:
        return self._view._parent._backend.layer_keys


class CellView:
    """Read-only, row-filtered view over a CellDB.

    A view stores only matching matrix row positions and materializes selected
    rows from array slots on demand. It does not copy or mutate the parent DB.
    """

    def __init__(
        self,
        parent: "CellDB",
        obs: pd.DataFrame,
        row_indices: np.ndarray,
        where: str,
        params: Sequence,
        *,
        clauses: Sequence[tuple[str, Sequence]] | None = None,
        name: str | None = None,
    ):
        self._parent = parent
        self._obs = obs
        self._row_indices = np.asarray(row_indices, dtype=np.int64)
        self.where = where
        self.params = tuple(params)
        self._clauses = tuple(clauses or ((where, tuple(params)),))
        self.name = name
        self._local_obsm: dict[str, np.ndarray] = {}
        self._local_obsp: dict[str, object] = {}
        self._local_uns: dict | None = None
        self.obsm = _ViewObsmAccessor(self)
        self.obsp = _ViewObspAccessor(self)
        self.layers = _ViewLayersAccessor(self)
        self.varm = parent.varm
        self.varp = parent.varp

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs.copy()

    @property
    def var(self) -> pd.DataFrame:
        return self._parent.var

    @property
    def X(self):
        return self._parent._backend.read_X(row_indices=self._row_indices)

    @property
    def obs_positions(self) -> np.ndarray:
        return self._row_indices.copy()

    @property
    def obs_names(self) -> pd.Index:
        return self._obs.index.copy()

    @property
    def var_names(self) -> pd.Index:
        return self._parent.var.index

    @property
    def uns(self) -> dict:
        if self._local_uns is None:
            return self._parent.uns
        return copy.deepcopy(self._local_uns)

    @uns.setter
    def uns(self, value: dict) -> None:
        self._local_uns = copy.deepcopy(value)

    @property
    def raw(self) -> _RawAccessor | None:
        if not self._parent._backend.raw_exists:
            return None
        return _RawAccessor(self._parent._backend, self._row_indices)

    @property
    def n_obs(self) -> int:
        return len(self._row_indices)

    @property
    def n_vars(self) -> int:
        return self._parent.n_vars

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_obs, self.n_vars)

    def to_anndata(
        self,
        slots: set[str] | None = None,
        obsm_keys: list[str] | None = None,
        obsp_keys: list[str] | None = None,
        layer_keys: list[str] | None = None,
        varm_keys: list[str] | None = None,
        varp_keys: list[str] | None = None,
    ) -> ad.AnnData:
        """Materialize this row subset as an AnnData object."""
        requested_slots = self._parent._normalize_slots(slots)
        parent_obsm_keys = None
        if "obsm" in requested_slots:
            requested_obsm = self.obsm.keys() if obsm_keys is None else obsm_keys
            parent_obsm_keys = [
                key for key in requested_obsm if key not in self._local_obsm
            ]
        parent_obsp_keys = None
        if "obsp" in requested_slots:
            requested_obsp = self.obsp.keys() if obsp_keys is None else obsp_keys
            parent_obsp_keys = [
                key for key in requested_obsp if key not in self._local_obsp
            ]
        result = self._parent._to_anndata(
            row_indices=self._row_indices,
            obs_override=self._obs,
            slots=requested_slots,
            obsm_keys=parent_obsm_keys,
            obsp_keys=parent_obsp_keys,
            layer_keys=layer_keys,
            varm_keys=varm_keys,
            varp_keys=varp_keys,
        )
        if "obsm" in requested_slots:
            requested_obsm = self.obsm.keys() if obsm_keys is None else obsm_keys
            for key in requested_obsm:
                if key in self._local_obsm:
                    result.obsm[key] = self._local_obsm[key]
        if "obsp" in requested_slots:
            requested_obsp = self.obsp.keys() if obsp_keys is None else obsp_keys
            for key in requested_obsp:
                if key in self._local_obsp:
                    result.obsp[key] = self._local_obsp[key]
        if "uns" in requested_slots and self._local_uns is not None:
            result.uns = self.uns
        return result

    def query_obs(
        self,
        where: str = "TRUE",
        params: Sequence | None = None,
        *,
        columns: Sequence[str] | None = None,
        name: str | None = None,
    ) -> "CellView":
        """Refine this view with another SQL predicate."""
        if isinstance(params, (str, bytes)):
            raise TypeError("params must be a sequence of parameter values")
        query_params = () if params is None else tuple(params)
        clauses = self._clauses + ((where, query_params),)
        combined_where = " AND ".join(f"({clause})" for clause, _ in clauses)
        combined_params = tuple(
            value for _, clause_params in clauses for value in clause_params
        )
        obs, row_indices = self._parent._backend.query_obs(
            combined_where,
            combined_params,
            columns=columns,
        )
        view = CellView(
            self._parent,
            obs,
            row_indices,
            combined_where,
            combined_params,
            clauses=clauses,
            name=name,
        )
        if name is not None:
            self._parent.save_view(name, view)
        return view

    def update_obs(
        self,
        column: str,
        values,
        *,
        create: bool = False,
        fill_value=None,
    ) -> None:
        """Write values back to the parent using this view's stable cell IDs."""
        values = list(values)
        if len(values) != self.n_obs:
            raise ValueError(f"values has length {len(values)}, expected {self.n_obs}")
        if column not in self._parent.obs_columns:
            if not create:
                raise KeyError(
                    f"obs column {column!r} does not exist; pass create=True to add it"
                )
            full_values = np.full(self._parent.n_obs, fill_value, dtype=object)
            full_values[self._row_indices] = values
            self._parent.add_obs_column(column, full_values)
        else:
            self._parent.update_obs(column, self.obs_names, values)
        self._obs[column] = values

    def save(self, name: str) -> "CellView":
        """Persist this view's query definition without copying matrix data."""
        self._parent.save_view(name, self)
        self.name = name
        return self

    def iter_X_batches(
        self,
        batch_size: int = 4096,
        *,
        layer: str | None = None,
    ) -> Iterator[tuple[pd.DataFrame, object]]:
        """Yield bounded-memory observation and matrix batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.n_obs, batch_size):
            stop = min(start + batch_size, self.n_obs)
            rows = self._row_indices[start:stop]
            matrix = (
                self._parent._backend.read_X(row_indices=rows)
                if layer is None
                else self._parent._backend.read_layer(layer, row_indices=rows)
            )
            if matrix is None:
                slot = "X" if layer is None else f"layer {layer!r}"
                raise KeyError(f"{slot} is not available")
            yield self._obs.iloc[start:stop].copy(), matrix

    def aggregate_many(self, tasks, *, batch_size: int = 4096):
        """Execute compatible group-by aggregations with shared matrix scans."""
        from .execution import execute_aggregate_tasks

        return execute_aggregate_tasks(self, tasks, batch_size=batch_size)

    def execute_tasks(
        self,
        tasks,
        *,
        batch_size: int = 4096,
        memory_budget_bytes: int | None = None,
    ):
        """Execute compatible aggregate and materialize tasks jointly."""
        from .execution import execute_tasks

        return execute_tasks(
            self,
            tasks,
            batch_size=batch_size,
            memory_budget_bytes=memory_budget_bytes,
        )

    def get_state(self) -> dict:
        return {
            "X_exists": self._parent._backend.X_exists,
            "obsm_keys": self.obsm.keys(),
            "obsp_keys": self.obsp.keys(),
            "obs_columns": list(self._obs.columns),
            "uns_keys": list(self.uns.keys()),
        }

    @property
    def provenance(self):
        return self._parent.provenance

    def to_h5ad(self, path: str):
        """Materialize this view and write it as h5ad."""
        self.to_anndata().write_h5ad(path)

    def materialize(self, path: str, *, overwrite: bool = False) -> "CellDB":
        """Create an independent CellDB containing only this view."""
        return CellDB.from_anndata(self.to_anndata(), path, overwrite=overwrite)

    def __len__(self) -> int:
        return self.n_obs

    def __repr__(self):
        return (
            f"CellView(n_obs={self.n_obs}, n_vars={self.n_vars}, where={self.where!r})"
        )


class CellDB:
    """AnnData-compatible interface backed by CellVault storage.

    Provides .obs, .var, .X, .obsm, .obsp, .uns attributes
    with the same API as AnnData, but backed by DuckDB + Zarr.
    """

    def __init__(self, backend: DuckDBZarrBackend):
        self._backend = backend
        self.layers = _LayersAccessor(backend)
        self.obsm = _ObsmAccessor(backend)
        self.obsp = _ObspAccessor(backend)
        self.varm = _VarmAccessor(backend)
        self.varp = _VarpAccessor(backend)

    @classmethod
    def create(cls, path: str, *, overwrite: bool = False) -> "CellDB":
        """Create a new CellVault database."""
        target = Path(path)
        if target.exists() and not target.is_dir():
            raise ValueError(f"CellVault path is not a directory: {path}")
        if target.exists() and any(target.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"CellVault path already exists and is not empty: {path}. "
                    "Pass overwrite=True to replace it."
                )
            shutil.rmtree(target)
        backend = DuckDBZarrBackend(path)
        return cls(backend)

    @classmethod
    def open(cls, path: str) -> "CellDB":
        """Open an existing CellVault database."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"CellVault database not found: {path}")
        if not os.path.isdir(path) or not os.path.exists(
            os.path.join(path, "obs.duckdb")
        ):
            raise ValueError(f"Path is not a CellVault database: {path}")
        backend = DuckDBZarrBackend(path)
        return cls(backend)

    @classmethod
    def from_h5ad(
        cls,
        h5ad_path: str,
        cvdb_path: str,
        *,
        overwrite: bool = False,
    ) -> "CellDB":
        """Convert an h5ad file to CellVault format with backed X access."""
        if Path(cvdb_path).exists() and not overwrite:
            raise FileExistsError(
                f"CellVault path already exists: {cvdb_path}. "
                "Pass overwrite=True to replace it."
            )
        adata = ad.read_h5ad(h5ad_path, backed="r")
        try:
            return cls.from_anndata(adata, cvdb_path, overwrite=overwrite)
        finally:
            adata.file.close()

    @classmethod
    def from_anndata(
        cls,
        adata: ad.AnnData,
        cvdb_path: str,
        *,
        overwrite: bool = False,
    ) -> "CellDB":
        """Convert an AnnData object to CellVault format."""
        if os.path.exists(cvdb_path):
            if not overwrite:
                raise FileExistsError(
                    f"CellVault path already exists: {cvdb_path}. "
                    "Pass overwrite=True to replace it."
                )
            if not os.path.isdir(cvdb_path):
                raise ValueError(f"CellVault path is not a directory: {cvdb_path}")
            shutil.rmtree(cvdb_path)

        backend = DuckDBZarrBackend(cvdb_path)

        # obs: store index as _index column
        obs_df = adata.obs.copy()
        obs_df["_index"] = obs_df.index
        backend.write_obs(obs_df)

        # var
        backend.write_var(adata.var.copy())

        # X
        if adata.X is not None:
            backend.write_X(adata.X)

        for key in adata.layers:
            if key is not None:
                backend.write_layer(key, adata.layers[key])

        # obsm
        for key in adata.obsm:
            backend.write_obsm(key, adata.obsm[key])

        # obsp
        for key in adata.obsp:
            backend.write_obsp(key, adata.obsp[key])

        for key in adata.varm:
            backend.write_varm(key, adata.varm[key])

        for key in adata.varp:
            backend.write_varp(key, adata.varp[key])

        if adata.raw is not None:
            backend.write_raw_var(adata.raw.var.copy())
            backend.write_raw_X(adata.raw.X)
            for key in adata.raw.varm:
                backend.write_raw_varm(key, adata.raw.varm[key])

        # uns: serialize what we can, warn on failures (never silently discard)
        uns_raw = dict(adata.uns)
        uns_safe = {}
        uns_dropped = []
        for k, v in uns_raw.items():
            try:
                serialized = _serialize_uns(v)
                # Verify round-trip via JSON
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
        slots: set[str] | None = None,
        obsm_keys: list[str] | None = None,
        obsp_keys: list[str] | None = None,
        layer_keys: list[str] | None = None,
        varm_keys: list[str] | None = None,
        varp_keys: list[str] | None = None,
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
        return self._to_anndata(
            row_indices=None,
            obs_override=None,
            slots=slots,
            obsm_keys=obsm_keys,
            obsp_keys=obsp_keys,
            layer_keys=layer_keys,
            varm_keys=varm_keys,
            varp_keys=varp_keys,
        )

    @staticmethod
    def _normalize_slots(slots: set[str] | None) -> set[str]:
        valid_slots = {
            "X",
            "obs",
            "var",
            "layers",
            "obsm",
            "obsp",
            "varm",
            "varp",
            "raw",
            "uns",
        }
        normalized = valid_slots.copy() if slots is None else set(slots)
        unknown_slots = normalized - valid_slots
        if unknown_slots:
            raise ValueError(
                f"Unknown AnnData slots requested: {sorted(unknown_slots)}"
            )
        return normalized

    def _to_anndata(
        self,
        row_indices: np.ndarray | None,
        obs_override: pd.DataFrame | None,
        slots: set[str] | None,
        obsm_keys: list[str] | None,
        obsp_keys: list[str] | None,
        layer_keys: list[str] | None = None,
        varm_keys: list[str] | None = None,
        varp_keys: list[str] | None = None,
    ) -> ad.AnnData:
        slots = self._normalize_slots(slots)

        selected_n_obs = self.n_obs if row_indices is None else len(row_indices)

        # obs/var: always need at least a stub for AnnData shape
        if "obs" in slots:
            obs = self.obs if obs_override is None else obs_override.copy()
        else:
            obs = pd.DataFrame(index=range(selected_n_obs))
        var = self.var if "var" in slots else pd.DataFrame(index=range(self.n_vars))
        X = self._backend.read_X(row_indices=row_indices) if "X" in slots else None

        adata = ad.AnnData(X=X, obs=obs, var=var)

        if "layers" in slots:
            keys_to_load = layer_keys if layer_keys is not None else self.layers.keys()
            for key in keys_to_load:
                data = self._backend.read_layer(key, row_indices=row_indices)
                if data is None:
                    raise KeyError(
                        f"layer {key!r} not found. Available: {self.layers.keys()}"
                    )
                adata.layers[key] = data

        if "obsm" in slots:
            keys_to_load = obsm_keys if obsm_keys is not None else self.obsm.keys()
            for key in keys_to_load:
                data = self._backend.read_obsm(key, row_indices=row_indices)
                if data is None:
                    raise KeyError(
                        f"obsm key '{key}' not found. Available: {self.obsm.keys()}"
                    )
                adata.obsm[key] = data

        if "obsp" in slots:
            keys_to_load = obsp_keys if obsp_keys is not None else self.obsp.keys()
            for key in keys_to_load:
                data = self._backend.read_obsp(key, row_indices=row_indices)
                if data is None:
                    raise KeyError(
                        f"obsp key '{key}' not found. Available: {self.obsp.keys()}"
                    )
                adata.obsp[key] = data

        if "varm" in slots:
            keys_to_load = varm_keys if varm_keys is not None else self.varm.keys()
            for key in keys_to_load:
                data = self._backend.read_varm(key)
                if data is None:
                    raise KeyError(
                        f"varm key {key!r} not found. Available: {self.varm.keys()}"
                    )
                adata.varm[key] = data

        if "varp" in slots:
            keys_to_load = varp_keys if varp_keys is not None else self.varp.keys()
            for key in keys_to_load:
                data = self._backend.read_varp(key)
                if data is None:
                    raise KeyError(
                        f"varp key {key!r} not found. Available: {self.varp.keys()}"
                    )
                adata.varp[key] = data

        if "raw" in slots and self._backend.raw_exists:
            adata.raw = _RawAccessor(self._backend, row_indices).to_adata(obs=obs)

        if "uns" in slots:
            adata.uns = self.uns

        return adata

    def query_obs(
        self,
        where: str = "TRUE",
        params: Sequence | None = None,
        *,
        columns: Sequence[str] | None = None,
        name: str | None = None,
    ) -> CellView:
        """Create a row-filtered view using a DuckDB SQL predicate.

        Values should be supplied through ``params`` rather than interpolated
        into ``where``. ``columns`` optionally projects metadata columns while
        retaining cell identifiers. Example::

            t_cells = cdb.query_obs('"cell_type" = ?', ["T cell"])
        """
        if isinstance(params, (str, bytes)):
            raise TypeError("params must be a sequence of parameter values")
        query_params = () if params is None else tuple(params)
        obs, row_indices = self._backend.query_obs(
            where,
            query_params,
            columns=columns,
        )
        view = CellView(
            self,
            obs,
            row_indices,
            where,
            query_params,
            name=name,
        )
        if name is not None:
            self.save_view(name, view)
        return view

    def partition_obs(
        self,
        column: str,
        groups: Mapping[str, object],
        *,
        columns: Sequence[str] | None = None,
        require_complete: bool = False,
        persist: bool = False,
    ) -> dict[str, CellView]:
        """Create disjoint lineage views with one metadata scan."""
        if column not in self.obs_columns:
            raise KeyError(f"obs column {column!r} does not exist")
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError("groups must be a non-empty mapping")

        normalized: dict[str, tuple] = {}
        assigned = {}
        for name, raw_values in groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("partition names must be non-empty strings")
            if isinstance(raw_values, np.ndarray):
                if raw_values.ndim == 0:
                    values = (raw_values.item(),)
                elif raw_values.ndim == 1:
                    values = tuple(raw_values.tolist())
                else:
                    raise ValueError(
                        f"partition {name!r} values must be one-dimensional"
                    )
            elif isinstance(raw_values, (str, bytes)):
                values = (raw_values,)
            else:
                try:
                    values = tuple(raw_values)
                except TypeError:
                    values = (raw_values,)
            if not values:
                raise ValueError(f"partition {name!r} has no values")
            for value in values:
                if value in assigned:
                    raise ValueError(
                        f"value {value!r} is assigned to both {assigned[value]!r} and {name!r}"
                    )
                assigned[value] = name
            normalized[name] = values

        if isinstance(columns, (str, bytes)):
            raise TypeError("columns must be a sequence of column names")
        requested_columns = None if columns is None else list(columns)
        scan_columns = None
        if requested_columns is not None:
            scan_columns = list(dict.fromkeys([column, *requested_columns]))
        all_obs, all_positions = self._backend.query_obs(
            "TRUE",
            (),
            columns=scan_columns,
        )
        selected = np.zeros(len(all_obs), dtype=bool)
        result = {}
        quoted_column = _quote_sql_identifier(column)
        for name, values in normalized.items():
            mask = all_obs[column].isin(values).to_numpy()
            selected |= mask
            view_obs = all_obs.loc[mask]
            if requested_columns is not None:
                view_obs = view_obs.loc[:, requested_columns]
            placeholders = ", ".join("?" for _ in values)
            where = f"{quoted_column} IN ({placeholders})"
            view = CellView(
                self,
                view_obs.copy(),
                all_positions[mask],
                where,
                values,
                name=name if persist else None,
            )
            if persist:
                self.save_view(name, view)
            result[name] = view

        if require_complete and not bool(np.all(selected)):
            raise ValueError(
                f"partition does not cover {int((~selected).sum())} observations"
            )
        return result

    def materialize_many(
        self,
        views: Mapping[str, CellView],
        slots: set[str] | None = None,
        *,
        obsm_keys: list[str] | None = None,
        obsp_keys: list[str] | None = None,
        layer_keys: list[str] | None = None,
        varm_keys: list[str] | None = None,
        varp_keys: list[str] | None = None,
    ) -> dict[str, ad.AnnData]:
        """Materialize multiple views while sharing matrix reads."""
        if not isinstance(views, Mapping):
            raise TypeError("views must be a mapping of names to CellView objects")
        for view in views.values():
            if not isinstance(view, CellView) or view._parent is not self:
                raise ValueError("every view must belong to this CellDB")
        normalized_slots = self._normalize_slots(slots)
        selections = {name: view._row_indices for name, view in views.items()}
        matrices = (
            self._backend.read_X_many(selections) if "X" in normalized_slots else {}
        )
        requested_layers = self.layers.keys() if layer_keys is None else layer_keys
        layer_matrices = {}
        if "layers" in normalized_slots:
            for key in requested_layers:
                layer_matrices[key] = self._backend.read_layer_many(key, selections)
        raw_matrices = {}
        if "raw" in normalized_slots and self._backend.raw_exists:
            raw_matrices = self._backend.read_raw_X_many(selections)

        base_slots = normalized_slots - {"X", "layers", "raw"}
        results = {}
        for name, view in views.items():
            result = view.to_anndata(
                slots=base_slots,
                obsm_keys=obsm_keys,
                obsp_keys=obsp_keys,
                varm_keys=varm_keys,
                varp_keys=varp_keys,
            )
            if "X" in normalized_slots:
                result.X = matrices[name]
            if "layers" in normalized_slots:
                for key in requested_layers:
                    result.layers[key] = layer_matrices[key][name]
            if raw_matrices:
                raw_adata = ad.AnnData(
                    X=raw_matrices[name],
                    obs=result.obs.copy(),
                    var=self._backend.read_raw_var(),
                )
                for key in self._backend.raw_varm_keys:
                    raw_adata.varm[key] = self._backend.read_raw_varm(key)
                result.raw = raw_adata
            results[name] = result
        return results

    @property
    def named_views(self) -> list[str]:
        path = self._backend.path / "views.json"
        if not path.exists():
            return []
        return sorted(json.loads(path.read_text(encoding="utf-8")))

    def save_view(self, name: str, view: CellView) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("view name must be a non-empty string")
        if not isinstance(view, CellView) or view._parent is not self:
            raise ValueError("view must belong to this CellDB")
        path = self._backend.path / "views.json"
        definitions = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        definition = {
            "where": view.where,
            "params": list(view.params),
            "columns": list(view._obs.columns),
        }
        try:
            json.dumps(definition)
        except TypeError as error:
            raise TypeError("view parameters must be JSON-serializable") from error
        definitions[name] = definition
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(definitions, indent=2), encoding="utf-8")
        temporary_path.replace(path)

    def load_view(self, name: str) -> CellView:
        path = self._backend.path / "views.json"
        definitions = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        if name not in definitions:
            raise KeyError(f"named view {name!r} not found")
        definition = definitions[name]
        view = self.query_obs(
            definition["where"],
            definition.get("params", ()),
            columns=definition.get("columns"),
        )
        view.name = name
        return view

    def iter_X_batches(
        self,
        batch_size: int = 4096,
        *,
        layer: str | None = None,
    ) -> Iterator[tuple[pd.DataFrame, object]]:
        """Yield bounded-memory observation and matrix batches."""
        return self.query_obs("TRUE").iter_X_batches(batch_size, layer=layer)

    def aggregate_many(self, tasks, *, batch_size: int = 4096):
        """Execute compatible group-by aggregations with shared matrix scans."""
        from .execution import execute_aggregate_tasks

        return execute_aggregate_tasks(self, tasks, batch_size=batch_size)

    def execute_tasks(
        self,
        tasks,
        *,
        batch_size: int = 4096,
        memory_budget_bytes: int | None = None,
    ):
        """Execute compatible aggregate and materialize tasks jointly."""
        from .execution import execute_tasks

        return execute_tasks(
            self,
            tasks,
            batch_size=batch_size,
            memory_budget_bytes=memory_budget_bytes,
        )

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
    def raw(self) -> _RawAccessor | None:
        if not self._backend.raw_exists:
            return None
        return _RawAccessor(self._backend)

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

    @property
    def obs_columns(self) -> list[str]:
        return self._backend.obs_columns

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    # ── Partial update ───────────────────────────────────────────

    def update_obs(self, column: str, index_mask, values):
        """Partial update: modify specific rows of a specific obs column."""
        self._backend.update_obs(column, index_mask, values)

    def update_obs_where(
        self,
        column: str,
        value,
        where: str,
        params: Sequence | None = None,
    ) -> int:
        """Update one metadata value for rows matching a SQL predicate."""
        return self._backend.update_obs_where(column, value, where, params)

    def add_obs_column(self, column: str, values):
        """Add a metadata column without rewriting the full obs table."""
        self._backend.add_obs_column(column, values)

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return f"CellDB(n_obs={self.n_obs}, n_vars={self.n_vars}, path='{self._backend.path}')"
