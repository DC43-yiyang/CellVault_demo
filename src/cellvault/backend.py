"""DuckDBZarrBackend: Storage backend using DuckDB for obs and Zarr for arrays."""

import os
import json
import shutil
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import zarr
from scipy import sparse

from .provenance import ProvenanceLogger, _hash_data


class DuckDBZarrBackend:
    """Storage backend: obs in DuckDB, arrays in Zarr, var in Parquet."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._db_path = self.path / "obs.duckdb"
        self._var_path = self.path / "var.parquet"
        self._x_path = self.path / "X.zarr"
        self._obsm_path = self.path / "obsm"
        self._obsp_path = self.path / "obsp"
        self._uns_path = self.path / "uns.json"
        self._registry_path = self.path / "registry.json"
        self._provenance_path = self.path / "provenance.jsonl"

        self._obsm_path.mkdir(exist_ok=True)
        self._obsp_path.mkdir(exist_ok=True)

        self._conn = duckdb.connect(str(self._db_path))
        self.provenance = ProvenanceLogger(str(self._provenance_path))

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()

    # ── obs (DuckDB) ─────────────────────────────────────────────

    def write_obs(self, df: pd.DataFrame):
        """Write full obs dataframe to DuckDB."""
        old_hash = _hash_data(self.read_obs()) if self._has_obs() else None
        self._conn.execute("DROP TABLE IF EXISTS obs")
        self._conn.execute("CREATE TABLE obs AS SELECT * FROM df")
        self.provenance.log("write_obs", "obs", old_hash=old_hash, new_hash=_hash_data(df))

    def read_obs(self) -> pd.DataFrame:
        """Read full obs dataframe from DuckDB."""
        if not self._has_obs():
            return pd.DataFrame()
        result = self._conn.execute("SELECT * FROM obs").fetchdf()
        if "_index" in result.columns:
            result = result.set_index("_index")
            result.index.name = None
        return result

    def update_obs(self, column: str, index_mask, values):
        """Partial update: modify specific rows of a specific obs column."""
        if not self._has_obs():
            raise ValueError("No obs table exists. Write obs first.")

        old_hash = _hash_data(self._conn.execute(f'SELECT "{column}" FROM obs').fetchdf())

        # Use parameterized update for safety
        if isinstance(index_mask, (list, np.ndarray)):
            df_update = pd.DataFrame({"_index": index_mask, column: values})
            self._conn.execute(f"""
                UPDATE obs SET "{column}" = u."{column}"
                FROM df_update u WHERE obs._index = u._index
            """)
        else:
            raise TypeError("index_mask must be list or array of index values")

        new_hash = _hash_data(self._conn.execute(f'SELECT "{column}" FROM obs').fetchdf())
        self.provenance.log(
            "update_obs", "obs", key=column,
            old_hash=old_hash, new_hash=new_hash,
            params={"n_rows": len(index_mask)},
        )

    def add_obs_column(self, column: str, values):
        """Add a new column to obs with automatic type inference.

        Supports DOUBLE, VARCHAR, INTEGER, BOOLEAN — inferred from the data.
        """
        if not self._has_obs():
            raise ValueError("No obs table exists.")

        # Infer DuckDB type from actual data
        series = pd.Series(values)
        if hasattr(series, "cat"):
            # Categorical → VARCHAR
            series = series.astype(str)
            duck_type = "VARCHAR"
        elif pd.api.types.is_bool_dtype(series):
            duck_type = "BOOLEAN"
        elif pd.api.types.is_integer_dtype(series):
            duck_type = "BIGINT"
        elif pd.api.types.is_float_dtype(series):
            duck_type = "DOUBLE"
        elif pd.api.types.is_string_dtype(series) or series.dtype == object:
            series = series.astype(str)
            duck_type = "VARCHAR"
        else:
            duck_type = "VARCHAR"
            series = series.astype(str)

        self._conn.execute(f'ALTER TABLE obs ADD COLUMN "{column}" {duck_type}')
        temp_df = pd.DataFrame({"_val": series, "_rowid": range(len(series))})
        self._conn.execute(f"""
            UPDATE obs SET "{column}" = t._val
            FROM (SELECT _val, _rowid FROM temp_df) t
            WHERE rowid = t._rowid
        """)
        self.provenance.log("add_obs_column", "obs", key=column, new_hash=_hash_data(values))

    def _has_obs(self) -> bool:
        try:
            self._conn.execute("SELECT count(*) FROM obs")
            return True
        except duckdb.CatalogException:
            return False

    @property
    def obs_columns(self) -> list[str]:
        if not self._has_obs():
            return []
        cols = self._conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='obs'").fetchall()
        return [c[0] for c in cols if c[0] != "_index"]

    def count_obs(self) -> int:
        """Return row count without materializing the full DataFrame."""
        if not self._has_obs():
            return 0
        return self._conn.execute("SELECT count(*) FROM obs").fetchone()[0]

    def count_vars(self) -> int:
        """Return var row count from parquet metadata without full read."""
        if not self._var_path.exists():
            return 0
        # Use DuckDB to read parquet metadata — no full deserialization
        result = self._conn.execute(
            f"SELECT count(*) FROM read_parquet('{self._var_path}')"
        ).fetchone()
        return result[0]

    # ── var (Parquet) ────────────────────────────────────────────

    def write_var(self, df: pd.DataFrame):
        old_hash = _hash_data(self.read_var()) if self._var_path.exists() else None
        df_with_idx = df.copy()
        df_with_idx["_index"] = df.index
        df_with_idx.to_parquet(str(self._var_path))
        self.provenance.log("write_var", "var", old_hash=old_hash, new_hash=_hash_data(df))

    def read_var(self) -> pd.DataFrame:
        if not self._var_path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(str(self._var_path))
        if "_index" in df.columns:
            df = df.set_index("_index")
            df.index.name = None
        return df

    # ── X (Zarr) ─────────────────────────────────────────────────

    def write_X(self, X):
        """Write expression matrix to Zarr. Supports dense and sparse.

        Uses atomic write-to-temp-then-rename to prevent data loss on crash.
        """
        tmp_path = self._x_path.with_suffix(".zarr.tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        store = zarr.storage.LocalStore(str(tmp_path))
        root = zarr.open_group(store, mode="w")

        if sparse.issparse(X):
            csr = X.tocsr()
            root.create_array("data", data=np.asarray(csr.data))
            root.create_array("indices", data=np.asarray(csr.indices))
            root.create_array("indptr", data=np.asarray(csr.indptr))
            root.attrs["encoding_type"] = "csr_matrix"
            root.attrs["shape"] = list(csr.shape)
        else:
            root.create_array("X", data=np.asarray(X))
            root.attrs["encoding_type"] = "dense"
            root.attrs["shape"] = list(X.shape)

        # Atomic swap: write to tmp, then rename over old
        if self._x_path.exists():
            shutil.rmtree(self._x_path)
        tmp_path.rename(self._x_path)

        self.provenance.log("write_X", "X", new_hash=_hash_data(X))

    def read_X(self):
        """Read expression matrix from Zarr."""
        if not self._x_path.exists():
            return None
        store = zarr.storage.LocalStore(str(self._x_path))
        root = zarr.open_group(store, mode="r")
        enc = root.attrs.get("encoding_type", "dense")
        shape = tuple(root.attrs["shape"])

        if enc == "csr_matrix":
            data = np.array(root["data"])
            indices = np.array(root["indices"])
            indptr = np.array(root["indptr"])
            return sparse.csr_matrix((data, indices, indptr), shape=shape)
        else:
            return np.array(root["X"])

    @property
    def X_exists(self) -> bool:
        return self._x_path.exists()

    # ── obsm (Zarr per key) ──────────────────────────────────────

    def write_obsm(self, key: str, data: np.ndarray):
        key_path = self._obsm_path / f"{key}.zarr"
        tmp_path = self._obsm_path / f"{key}.zarr.tmp"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        store = zarr.storage.LocalStore(str(tmp_path))
        root = zarr.open_group(store, mode="w")
        root.create_array("data", data=np.asarray(data))
        root.attrs["shape"] = list(data.shape)
        # Atomic swap
        if key_path.exists():
            shutil.rmtree(key_path)
        tmp_path.rename(key_path)
        self.provenance.log("write_obsm", "obsm", key=key, new_hash=_hash_data(data))

    def read_obsm(self, key: str) -> Optional[np.ndarray]:
        key_path = self._obsm_path / f"{key}.zarr"
        if not key_path.exists():
            return None
        store = zarr.storage.LocalStore(str(key_path))
        root = zarr.open_group(store, mode="r")
        return np.array(root["data"])

    @property
    def obsm_keys(self) -> list[str]:
        return [p.stem for p in self._obsm_path.glob("*.zarr") if p.is_dir()]

    # ── obsp (Zarr per key, sparse) ──────────────────────────────

    def write_obsp(self, key: str, data):
        key_path = self._obsp_path / f"{key}.zarr"
        tmp_path = self._obsp_path / f"{key}.zarr.tmp"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        store = zarr.storage.LocalStore(str(tmp_path))
        root = zarr.open_group(store, mode="w")

        if sparse.issparse(data):
            csr = data.tocsr()
            root.create_array("data", data=np.asarray(csr.data))
            root.create_array("indices", data=np.asarray(csr.indices))
            root.create_array("indptr", data=np.asarray(csr.indptr))
            root.attrs["encoding_type"] = "csr_matrix"
            root.attrs["shape"] = list(csr.shape)
        else:
            root.create_array("data", data=np.asarray(data))
            root.attrs["encoding_type"] = "dense"
            root.attrs["shape"] = list(data.shape)

        # Atomic swap
        if key_path.exists():
            shutil.rmtree(key_path)
        tmp_path.rename(key_path)

        self.provenance.log("write_obsp", "obsp", key=key, new_hash=_hash_data(data))

    def read_obsp(self, key: str):
        key_path = self._obsp_path / f"{key}.zarr"
        if not key_path.exists():
            return None
        store = zarr.storage.LocalStore(str(key_path))
        root = zarr.open_group(store, mode="r")
        enc = root.attrs.get("encoding_type", "dense")
        shape = tuple(root.attrs["shape"])

        if enc == "csr_matrix":
            data = np.array(root["data"])
            indices = np.array(root["indices"])
            indptr = np.array(root["indptr"])
            return sparse.csr_matrix((data, indices, indptr), shape=shape)
        else:
            return np.array(root["data"])

    @property
    def obsp_keys(self) -> list[str]:
        return [p.stem for p in self._obsp_path.glob("*.zarr") if p.is_dir()]

    # ── uns (JSON) ───────────────────────────────────────────────

    def write_uns(self, uns: dict):
        old_hash = _hash_data(str(self.read_uns())) if self._uns_path.exists() else None
        with open(self._uns_path, "w") as f:
            json.dump(_serialize_uns(uns), f, indent=2)
        self.provenance.log("write_uns", "uns", old_hash=old_hash, new_hash=_hash_data(str(uns)))

    def read_uns(self) -> dict:
        if not self._uns_path.exists():
            return {}
        with open(self._uns_path) as f:
            return json.load(f)

    # ── Registry ─────────────────────────────────────────────────

    def write_registry(self, registry: dict):
        with open(self._registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def read_registry(self) -> dict:
        if not self._registry_path.exists():
            return {}
        with open(self._registry_path) as f:
            return json.load(f)


def _serialize_uns(obj):
    """Make uns JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize_uns(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_uns(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, pd.Categorical):
        return obj.tolist()
    return obj
