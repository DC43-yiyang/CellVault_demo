"""DuckDBZarrBackend: Storage backend using DuckDB for obs and Zarr for arrays."""

import json
import shutil
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import sparse

from .provenance import ProvenanceLogger, _hash_data

try:
    import _duckdb as duckdb
except ImportError:
    import duckdb


_ROW_POSITION_COLUMN = "__cellvault_row_position"
_QUERY_POSITION_COLUMN = "__cellvault_query_position"
_TARGET_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_RANGE_READ_BYTES = 2 * _TARGET_CHUNK_BYTES
_BATCH_RANGE_READ_BYTES = 8 * _TARGET_CHUNK_BYTES


def _quote_identifier(value: str) -> str:
    """Quote a DuckDB identifier after escaping embedded quotes."""
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _validate_predicate(where: str, params: Sequence | None) -> list:
    if not isinstance(where, str) or not where.strip():
        raise ValueError("where must be a non-empty SQL predicate")
    if ";" in where:
        raise ValueError("where must contain a single SQL predicate")
    if isinstance(params, (str, bytes)):
        raise TypeError("params must be a sequence of parameter values")
    return [] if params is None else list(params)


def _validate_storage_key(key: str) -> str:
    """Validate a key before using it as part of a storage path."""
    if not isinstance(key, str):
        raise TypeError("storage key must be a string")
    if not key or key in {".", ".."} or "/" in key or "\\" in key or "\0" in key:
        raise ValueError(f"invalid storage key: {key!r}")
    return key


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def _dataframe_schema(frame: pd.DataFrame) -> dict:
    categoricals = {}
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            categoricals[str(column)] = {
                "categories": [
                    _json_scalar(value) for value in frame[column].cat.categories
                ],
                "ordered": bool(frame[column].cat.ordered),
            }
    return {
        "index_name": _json_scalar(frame.index.name),
        "categoricals": categoricals,
    }


def _frame_for_storage(frame: pd.DataFrame) -> pd.DataFrame:
    stored = frame.copy()
    for column in stored.columns:
        if isinstance(stored[column].dtype, pd.CategoricalDtype):
            stored[column] = stored[column].astype(object)
    return stored


def _restore_dataframe_schema(frame: pd.DataFrame, schema: dict | None) -> pd.DataFrame:
    if not schema:
        return frame
    restored = frame.copy()
    for column, categorical in schema.get("categoricals", {}).items():
        if column in restored.columns:
            restored[column] = pd.Categorical(
                restored[column],
                categories=categorical.get("categories", []),
                ordered=bool(categorical.get("ordered", False)),
            )
    restored.index.name = schema.get("index_name")
    return restored


def _chunk_length(length: int, itemsize: int) -> int:
    return max(1, min(length, _TARGET_CHUNK_BYTES // max(1, itemsize)))


def _copy_1d_array(root, name: str, source, *, dtype=None) -> None:
    length = int(source.shape[0])
    target_dtype = np.dtype(source.dtype if dtype is None else dtype)
    chunk_length = _chunk_length(length, target_dtype.itemsize)
    target = root.create_array(
        name,
        shape=(length,),
        dtype=target_dtype,
        chunks=(chunk_length,),
    )
    for start in range(0, length, chunk_length):
        stop = min(start + chunk_length, length)
        target[start:stop] = np.asarray(source[start:stop], dtype=target_dtype)


def _sparse_index_dtype(shape: tuple[int, int], nnz: int) -> np.dtype:
    """Use 32-bit CSR indices whenever both dimensions and nnz allow it."""
    int32_max = np.iinfo(np.int32).max
    if max(shape, default=0) <= int32_max and nnz <= int32_max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def _dense_chunks(shape: tuple[int, int], itemsize: int) -> tuple[int, int]:
    n_rows, n_columns = shape
    column_chunk = max(1, n_columns)
    row_chunk = max(
        1,
        min(n_rows, _TARGET_CHUNK_BYTES // (column_chunk * max(1, itemsize))),
    )
    return row_chunk, column_chunk


def _write_dense_array(root, name: str, source) -> None:
    shape = tuple(source.shape)
    chunks = _dense_chunks(shape, source.dtype.itemsize)
    target = root.create_array(
        name,
        shape=shape,
        dtype=source.dtype,
        chunks=chunks,
    )
    for start in range(0, shape[0], chunks[0]):
        stop = min(start + chunks[0], shape[0])
        target[start:stop, :] = np.asarray(source[start:stop, :])


def _normalize_indices(indices, size: int) -> np.ndarray | None:
    """Normalize integer, boolean, and slice row selectors."""
    if indices is None:
        return None
    if isinstance(indices, slice):
        return np.arange(size, dtype=np.int64)[indices]

    result = np.asarray(indices)
    if result.ndim != 1:
        raise IndexError("indices must be one-dimensional")
    if result.size == 0:
        return np.array([], dtype=np.int64)
    if np.issubdtype(result.dtype, np.bool_):
        if len(result) != size:
            raise IndexError(f"boolean index has length {len(result)}, expected {size}")
        result = np.flatnonzero(result)
    elif not np.issubdtype(result.dtype, np.integer):
        raise TypeError("indices must contain integers or booleans")
    else:
        result = result.astype(np.int64, copy=False)

    result = np.where(result < 0, result + size, result)
    if np.any((result < 0) | (result >= size)):
        raise IndexError(f"index is out of bounds for axis with size {size}")
    return result.astype(np.int64, copy=False)


def _read_array_ranges(array, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Read ordered one-dimensional ranges while avoiding repeated chunk I/O."""
    lengths = stops - starts
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    output = np.empty(int(offsets[-1]), dtype=array.dtype)
    nonempty = np.flatnonzero(lengths)
    if len(nonempty) == 0:
        return output

    chunk_size = int(array.chunks[0])
    max_block_items = max(
        chunk_size,
        _MAX_RANGE_READ_BYTES // array.dtype.itemsize,
    )

    def copy_group(group: list[int]) -> None:
        read_start = int(starts[group[0]])
        read_stop = int(stops[group[-1]])
        block = np.asarray(array[read_start:read_stop])
        for range_index in group:
            output_start = int(offsets[range_index])
            output_stop = int(offsets[range_index + 1])
            source_start = int(starts[range_index]) - read_start
            source_stop = int(stops[range_index]) - read_start
            output[output_start:output_stop] = block[source_start:source_stop]

    group = [int(nonempty[0])]
    group_start = int(starts[group[0]])
    group_stop = int(stops[group[0]])
    last_chunk = (group_stop - 1) // chunk_size
    for range_index_raw in nonempty[1:]:
        range_index = int(range_index_raw)
        start = int(starts[range_index])
        stop = int(stops[range_index])
        first_chunk = start // chunk_size
        merged_stop = max(group_stop, stop)
        if (
            first_chunk <= last_chunk + 1
            and merged_stop - group_start <= max_block_items
        ):
            group.append(range_index)
            group_stop = merged_stop
            last_chunk = max(last_chunk, (stop - 1) // chunk_size)
        else:
            copy_group(group)
            group = [range_index]
            group_start = start
            group_stop = stop
            last_chunk = (stop - 1) // chunk_size
    copy_group(group)
    return output


def _read_dense_array(array, row_indices=None, column_indices=None) -> np.ndarray:
    """Read an optional orthogonal subset from a Zarr array."""
    rows = _normalize_indices(row_indices, array.shape[0])
    columns = _normalize_indices(column_indices, array.shape[1])
    selection = (
        rows if rows is not None else slice(None),
        columns if columns is not None else slice(None),
    )
    return np.asarray(array.get_orthogonal_selection(selection))


def _rows_are_sorted(rows: np.ndarray) -> bool:
    return len(rows) < 2 or bool(np.all(rows[:-1] <= rows[1:]))


def _read_csr_group(
    root,
    row_indices=None,
    column_indices=None,
    *,
    source_indptr: np.ndarray | None = None,
):
    """Read selected CSR rows without materializing the complete matrix."""
    shape = tuple(root.attrs["shape"])
    rows = _normalize_indices(row_indices, shape[0])
    columns = _normalize_indices(column_indices, shape[1])

    if rows is None:
        data = np.asarray(root["data"])
        indices = np.asarray(root["indices"])
        indptr = np.asarray(root["indptr"]) if source_indptr is None else source_indptr
        matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
    elif len(rows) == 0:
        matrix = sparse.csr_matrix((0, shape[1]), dtype=root["data"].dtype)
    else:
        sorted_input = _rows_are_sorted(rows)
        sort_order = None if sorted_input else np.argsort(rows, kind="stable")
        sorted_rows = rows if sorted_input else rows[sort_order]
        if source_indptr is None:
            source_indptr = np.asarray(root["indptr"])
        starts = source_indptr[sorted_rows]
        stops = source_indptr[sorted_rows + 1]
        lengths = stops - starts
        index_dtype = _sparse_index_dtype(shape, int(lengths.sum()))
        contiguous = len(sorted_rows) == 1 or bool(
            np.all(np.diff(sorted_rows) == 1)
        )
        if contiguous:
            first_row = int(sorted_rows[0])
            last_row = int(sorted_rows[-1]) + 1
            read_start = int(source_indptr[first_row])
            read_stop = int(source_indptr[last_row])
            indptr = (
                source_indptr[first_row : last_row + 1] - read_start
            ).astype(index_dtype, copy=False)
            data = np.asarray(root["data"][read_start:read_stop])
            indices = np.asarray(root["indices"][read_start:read_stop])
        else:
            indptr = np.empty(len(sorted_rows) + 1, dtype=index_dtype)
            indptr[0] = 0
            np.cumsum(lengths, dtype=index_dtype, out=indptr[1:])
            data = _read_array_ranges(root["data"], starts, stops)
            indices = _read_array_ranges(root["indices"], starts, stops)
        matrix = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(sorted_rows), shape[1]),
            copy=False,
        )
        if not sorted_input:
            matrix = matrix[np.argsort(sort_order, kind="stable")]

    if columns is not None:
        matrix = matrix[:, columns]
    return matrix


def _read_dense_array_many(array, selections, column_indices=None):
    """Read many dense row selections while reading every source row once."""
    normalized = {
        key: _normalize_indices(rows, array.shape[0])
        for key, rows in selections.items()
    }
    columns = _normalize_indices(column_indices, array.shape[1])
    column_selector = columns if columns is not None else slice(None)
    n_columns = array.shape[1] if columns is None else len(columns)
    outputs = {
        key: np.empty((len(rows), n_columns), dtype=array.dtype)
        for key, rows in normalized.items()
    }
    occurrences: dict[int, list[tuple[Hashable, int]]] = {}
    for key, rows in normalized.items():
        for output_row, source_row in enumerate(rows):
            occurrences.setdefault(int(source_row), []).append((key, output_row))
    if not occurrences:
        return outputs

    source_rows = np.fromiter(sorted(occurrences), dtype=np.int64)
    chunk_rows = int(array.chunks[0])
    block_rows = max(
        chunk_rows,
        _MAX_RANGE_READ_BYTES // max(1, array.dtype.itemsize * array.shape[1]),
    )
    block_start = 0
    while block_start < len(source_rows):
        first_row = int(source_rows[block_start])
        block_stop = block_start + 1
        while block_stop < len(source_rows):
            next_row = int(source_rows[block_stop])
            if (
                next_row // chunk_rows
                > int(source_rows[block_stop - 1]) // chunk_rows + 1
                or next_row - first_row >= block_rows
            ):
                break
            block_stop += 1
        last_row = int(source_rows[block_stop - 1]) + 1
        block = np.asarray(
            array.get_orthogonal_selection(
                (slice(first_row, last_row), column_selector)
            )
        )
        for source_row in source_rows[block_start:block_stop]:
            values = block[int(source_row) - first_row]
            for key, output_row in occurrences[int(source_row)]:
                outputs[key][output_row] = values
        block_start = block_stop
    return outputs


def _read_csr_group_many(
    root,
    selections: Mapping[Hashable, object],
    column_indices=None,
    *,
    source_indptr: np.ndarray | None = None,
):
    """Materialize many CSR row selections with one pass over source ranges."""
    shape = tuple(root.attrs["shape"])
    normalized = {
        key: _normalize_indices(rows, shape[0]) for key, rows in selections.items()
    }
    columns = _normalize_indices(column_indices, shape[1])
    if source_indptr is None:
        source_indptr = np.asarray(root["indptr"])

    occurrences: dict[int, list[tuple[Hashable, int]]] = {}
    output_indptr: dict[Hashable, np.ndarray] = {}
    output_data: dict[Hashable, np.ndarray] = {}
    output_indices: dict[Hashable, np.ndarray] = {}
    data_array = root["data"]
    indices_array = root["indices"]

    for key, rows in normalized.items():
        lengths = source_indptr[rows + 1] - source_indptr[rows]
        nnz = int(lengths.sum())
        index_dtype = _sparse_index_dtype(shape, nnz)
        indptr = np.empty(len(rows) + 1, dtype=index_dtype)
        indptr[0] = 0
        np.cumsum(lengths, dtype=index_dtype, out=indptr[1:])
        output_indptr[key] = indptr
        output_data[key] = np.empty(nnz, dtype=data_array.dtype)
        output_indices[key] = np.empty(nnz, dtype=indices_array.dtype)
        for output_row, source_row in enumerate(rows):
            occurrences.setdefault(int(source_row), []).append((key, output_row))

    nonempty_rows = np.fromiter(
        (
            row
            for row in sorted(occurrences)
            if source_indptr[row + 1] > source_indptr[row]
        ),
        dtype=np.int64,
    )
    if len(nonempty_rows):
        max_itemsize = max(data_array.dtype.itemsize, indices_array.dtype.itemsize)
        max_block_items = max(
            min(int(data_array.chunks[0]), int(indices_array.chunks[0])),
            _BATCH_RANGE_READ_BYTES // max_itemsize,
        )
        chunk_size = min(int(data_array.chunks[0]), int(indices_array.chunks[0]))
        group_start = 0
        while group_start < len(nonempty_rows):
            first_row = int(nonempty_rows[group_start])
            read_start = int(source_indptr[first_row])
            group_stop = group_start + 1
            last_stop = int(source_indptr[first_row + 1])
            last_chunk = (last_stop - 1) // chunk_size
            while group_stop < len(nonempty_rows):
                next_row = int(nonempty_rows[group_stop])
                next_start = int(source_indptr[next_row])
                next_stop = int(source_indptr[next_row + 1])
                if (
                    next_start // chunk_size > last_chunk + 1
                    or next_stop - read_start > max_block_items
                ):
                    break
                last_stop = next_stop
                last_chunk = max(last_chunk, (next_stop - 1) // chunk_size)
                group_stop += 1

            data_block = np.asarray(data_array[read_start:last_stop])
            indices_block = np.asarray(indices_array[read_start:last_stop])
            for source_row in nonempty_rows[group_start:group_stop]:
                source_start = int(source_indptr[source_row])
                source_stop = int(source_indptr[source_row + 1])
                local_start = source_start - read_start
                local_stop = source_stop - read_start
                for key, output_row in occurrences[int(source_row)]:
                    output_start = int(output_indptr[key][output_row])
                    output_stop = int(output_indptr[key][output_row + 1])
                    output_data[key][output_start:output_stop] = data_block[
                        local_start:local_stop
                    ]
                    output_indices[key][output_start:output_stop] = indices_block[
                        local_start:local_stop
                    ]
            group_start = group_stop

    results = {}
    for key, rows in normalized.items():
        matrix = sparse.csr_matrix(
            (output_data[key], output_indices[key], output_indptr[key]),
            shape=(len(rows), shape[1]),
            copy=False,
        )
        if columns is not None:
            matrix = matrix[:, columns]
        results[key] = matrix
    return results


class DuckDBZarrBackend:
    """Storage backend: obs in DuckDB, arrays in Zarr, var in Parquet."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._db_path = self.path / "obs.duckdb"
        self._var_path = self.path / "var.parquet"
        self._x_path = self.path / "X.zarr"
        self._layers_path = self.path / "layers"
        self._obsm_path = self.path / "obsm"
        self._obsp_path = self.path / "obsp"
        self._varm_path = self.path / "varm"
        self._varp_path = self.path / "varp"
        self._raw_path = self.path / "raw"
        self._uns_path = self.path / "uns.json"
        self._registry_path = self.path / "registry.json"
        self._frame_schema_path = self.path / "dataframe_schema.json"
        self._provenance_path = self.path / "provenance.jsonl"

        self._layers_path.mkdir(exist_ok=True)
        self._obsm_path.mkdir(exist_ok=True)
        self._obsp_path.mkdir(exist_ok=True)
        self._varm_path.mkdir(exist_ok=True)
        self._varp_path.mkdir(exist_ok=True)
        self._raw_path.mkdir(exist_ok=True)

        self._matrix_roots: dict[Path, object] = {}
        self._indptr_cache: dict[Path, np.ndarray] = {}
        self._var_cache: pd.DataFrame | None = None
        self._raw_var_cache: pd.DataFrame | None = None

        self._conn = duckdb.connect(
            str(self._db_path),
            config={"threads": 1},
        )
        self.provenance = ProvenanceLogger(str(self._provenance_path))

    def _read_frame_schemas(self) -> dict:
        if not self._frame_schema_path.exists():
            return {}
        with self._frame_schema_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _write_frame_schema(self, key: str, frame: pd.DataFrame) -> None:
        schemas = self._read_frame_schemas()
        schemas[key] = _dataframe_schema(frame)
        temporary_path = self._frame_schema_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(schemas, handle, indent=2)
        temporary_path.replace(self._frame_schema_path)

    def _drop_categorical_schema(self, key: str, column: str) -> None:
        schemas = self._read_frame_schemas()
        categoricals = schemas.get(key, {}).get("categoricals", {})
        if column not in categoricals:
            return
        del categoricals[column]
        temporary_path = self._frame_schema_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(schemas, handle, indent=2)
        temporary_path.replace(self._frame_schema_path)

    def _set_categorical_schema(self, key: str, column: str, series: pd.Series) -> None:
        schemas = self._read_frame_schemas()
        schema = schemas.setdefault(key, {"index_name": None, "categoricals": {}})
        schema.setdefault("categoricals", {})[column] = {
            "categories": [_json_scalar(value) for value in series.cat.categories],
            "ordered": bool(series.cat.ordered),
        }
        temporary_path = self._frame_schema_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(schemas, handle, indent=2)
        temporary_path.replace(self._frame_schema_path)

    def _invalidate_matrix_cache(self, path: Path) -> None:
        self._matrix_roots.pop(path, None)
        self._indptr_cache.pop(path, None)

    def _open_matrix(self, path: Path):
        root = self._matrix_roots.get(path)
        if root is None:
            store = zarr.storage.LocalStore(str(path))
            root = zarr.open_group(store, mode="r")
            self._matrix_roots[path] = root
        return root

    def _matrix_indptr(self, path: Path, root) -> np.ndarray:
        indptr = self._indptr_cache.get(path)
        if indptr is None:
            indptr = np.asarray(root["indptr"])
            self._indptr_cache[path] = indptr
        return indptr

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
        self._matrix_roots.clear()
        self._indptr_cache.clear()
        self._var_cache = None
        self._raw_var_cache = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()

    def _aligned_row_counts(
        self,
        exclude: str | None = None,
    ) -> list[tuple[str, int]]:
        counts: list[tuple[str, int]] = []
        if exclude != "X" and self.X_exists:
            counts.append(("X", self.X_shape[0]))
        for path in self._layers_path.glob("*.zarr"):
            name = f"layers[{path.stem!r}]"
            if name != exclude:
                counts.append((name, self._matrix_shape(path)[0]))
        raw_x_path = self._raw_path / "X.zarr"
        if raw_x_path.exists() and exclude != "raw.X":
            counts.append(("raw.X", self._matrix_shape(raw_x_path)[0]))
        for path in self._obsm_path.glob("*.zarr"):
            name = f"obsm[{path.stem!r}]"
            if name != exclude:
                counts.append((name, self._zarr_shape(path)[0]))
        for path in self._obsp_path.glob("*.zarr"):
            name = f"obsp[{path.stem!r}]"
            if name != exclude:
                counts.append((name, self._zarr_shape(path)[0]))
        return counts

    @staticmethod
    def _zarr_shape(path: Path) -> tuple[int, ...]:
        store = zarr.storage.LocalStore(str(path))
        root = zarr.open_group(store, mode="r")
        return tuple(root.attrs["shape"])

    def _validate_aligned_rows(
        self,
        target: str,
        row_count: int,
        exclude: str | None = None,
        include_obs: bool = True,
    ) -> None:
        if include_obs and self._has_obs() and row_count != self.count_obs():
            raise ValueError(
                f"{target} has {row_count} rows but obs has {self.count_obs()} rows"
            )
        for existing_target, existing_count in self._aligned_row_counts(exclude):
            if row_count != existing_count:
                raise ValueError(
                    f"{target} has {row_count} rows but {existing_target} has "
                    f"{existing_count} rows"
                )

    # ── obs (DuckDB) ─────────────────────────────────────────────

    def write_obs(self, df: pd.DataFrame):
        """Write full obs dataframe to DuckDB."""
        if _ROW_POSITION_COLUMN in df.columns:
            raise ValueError(f"'{_ROW_POSITION_COLUMN}' is reserved for CellVault")
        self._validate_aligned_rows(
            "obs",
            len(df),
            include_obs=False,
        )
        old_hash = _hash_data(self.read_obs()) if self._has_obs() else None
        self._write_frame_schema("obs", df)
        stored_df = _frame_for_storage(df)
        stored_df.insert(0, _ROW_POSITION_COLUMN, np.arange(len(df), dtype=np.int64))
        self._conn.register("_cellvault_obs_df", stored_df)
        try:
            self._conn.execute("BEGIN TRANSACTION")
            self._conn.execute("DROP TABLE IF EXISTS obs")
            self._conn.execute("CREATE TABLE obs AS SELECT * FROM _cellvault_obs_df")
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        finally:
            self._conn.unregister("_cellvault_obs_df")
        self.provenance.log(
            "write_obs", "obs", old_hash=old_hash, new_hash=_hash_data(df)
        )

    def read_obs(self) -> pd.DataFrame:
        """Read full obs dataframe from DuckDB."""
        if not self._has_obs():
            return pd.DataFrame()
        order_by = (
            f" ORDER BY {_quote_identifier(_ROW_POSITION_COLUMN)}"
            if self._has_row_positions()
            else ""
        )
        result = self._conn.execute(f"SELECT * FROM obs{order_by}").fetchdf()
        result = result.drop(columns=[_ROW_POSITION_COLUMN], errors="ignore")
        if "_index" in result.columns:
            result = result.set_index("_index")
            result.index.name = None
        return _restore_dataframe_schema(
            result,
            self._read_frame_schemas().get("obs"),
        )

    def update_obs(self, column: str, index_mask, values):
        """Partial update: modify specific rows of a specific obs column."""
        if not self._has_obs():
            raise ValueError("No obs table exists. Write obs first.")

        if column not in self.obs_columns:
            raise KeyError(f"obs column '{column}' does not exist")
        if isinstance(index_mask, (str, bytes)):
            raise TypeError("index_mask must be a sequence of index values")
        if isinstance(values, (str, bytes)):
            raise TypeError("values must be a sequence")
        try:
            index_values = list(index_mask)
            update_values = list(values)
        except TypeError as error:
            raise TypeError(
                "index_mask and values must be one-dimensional sequences"
            ) from error
        if len(index_values) != len(update_values):
            raise ValueError("index_mask and values must have the same length")
        if pd.Index(index_values).has_duplicates:
            raise ValueError("index_mask must not contain duplicate cell identifiers")
        if not index_values:
            return

        quoted_column = _quote_identifier(column)
        quoted_index = _quote_identifier("_index")
        update_frame = pd.DataFrame(
            {
                "__cellvault_update_index": index_values,
                "__cellvault_update_value": update_values,
                "__cellvault_update_order": np.arange(len(index_values)),
            }
        )
        self._conn.register("_cellvault_obs_update", update_frame)
        try:
            old_values = self._conn.execute(f"""
                SELECT obs.{quoted_column}
                FROM obs
                INNER JOIN _cellvault_obs_update AS u
                    ON obs.{quoted_index} = u.__cellvault_update_index
                ORDER BY u.__cellvault_update_order
            """).fetchdf()
            if len(old_values) != len(index_values):
                raise KeyError(
                    "one or more cell identifiers were missing or non-unique"
                )
            old_hash = _hash_data(old_values)
            self._conn.execute("BEGIN TRANSACTION")
            self._conn.execute(f"""
                UPDATE obs SET {quoted_column} = u.__cellvault_update_value
                FROM _cellvault_obs_update AS u
                WHERE obs.{quoted_index} = u.__cellvault_update_index
            """)
            self._conn.execute("COMMIT")
            new_values = self._conn.execute(f"""
                SELECT obs.{quoted_column}
                FROM obs
                INNER JOIN _cellvault_obs_update AS u
                    ON obs.{quoted_index} = u.__cellvault_update_index
                ORDER BY u.__cellvault_update_order
            """).fetchdf()
            new_hash = _hash_data(new_values)
        except Exception:
            try:
                self._conn.execute("ROLLBACK")
            except duckdb.TransactionException:
                pass
            raise
        finally:
            self._conn.unregister("_cellvault_obs_update")
        self.provenance.log(
            "update_obs",
            "obs",
            key=column,
            old_hash=old_hash,
            new_hash=new_hash,
            params={"n_rows": len(index_values)},
        )
        self._drop_categorical_schema("obs", column)

    def update_obs_where(
        self,
        column: str,
        value,
        where: str,
        params: Sequence | None = None,
    ) -> int:
        """Set one obs value for all rows matching a SQL predicate."""
        if not self._has_obs():
            raise ValueError("No obs table exists. Write obs first.")
        if column not in self.obs_columns:
            raise KeyError(f"obs column '{column}' does not exist")
        query_params = _validate_predicate(where, params)
        quoted_column = _quote_identifier(column)
        position_expression = (
            _quote_identifier(_ROW_POSITION_COLUMN)
            if self._has_row_positions()
            else "rowid"
        )
        selected = self._conn.execute(
            f"SELECT {position_expression}, {quoted_column} "
            f"FROM obs WHERE {where} ORDER BY {position_expression}",
            query_params,
        ).fetchdf()
        if selected.empty:
            return 0

        position_frame = pd.DataFrame(
            {"__cellvault_update_position": selected.iloc[:, 0].to_numpy()}
        )
        old_hash = _hash_data(selected.iloc[:, 1:])
        self._conn.register("_cellvault_obs_positions", position_frame)
        try:
            self._conn.execute("BEGIN TRANSACTION")
            self._conn.execute(
                f"""
                UPDATE obs SET {quoted_column} = ?
                FROM _cellvault_obs_positions AS p
                WHERE obs.{position_expression} = p.__cellvault_update_position
                """,
                [value],
            )
            self._conn.execute("COMMIT")
            new_values = self._conn.execute(f"""
                SELECT obs.{quoted_column}
                FROM obs
                INNER JOIN _cellvault_obs_positions AS p
                    ON obs.{position_expression} = p.__cellvault_update_position
                ORDER BY p.__cellvault_update_position
            """).fetchdf()
            new_hash = _hash_data(new_values)
        except Exception:
            try:
                self._conn.execute("ROLLBACK")
            except duckdb.TransactionException:
                pass
            raise
        finally:
            self._conn.unregister("_cellvault_obs_positions")

        row_count = len(position_frame)
        self.provenance.log(
            "update_obs_where",
            "obs",
            key=column,
            old_hash=old_hash,
            new_hash=new_hash,
            params={"where": where, "n_rows": row_count},
        )
        self._drop_categorical_schema("obs", column)
        return row_count

    def add_obs_column(self, column: str, values):
        """Add a new column to obs with automatic type inference.

        Supports DOUBLE, VARCHAR, INTEGER, BOOLEAN — inferred from the data.
        """
        if not self._has_obs():
            raise ValueError("No obs table exists.")
        if column in self._stored_obs_columns():
            raise ValueError(f"obs column '{column}' already exists")
        if column == _ROW_POSITION_COLUMN:
            raise ValueError(f"'{_ROW_POSITION_COLUMN}' is reserved for CellVault")
        if len(values) != self.count_obs():
            raise ValueError(
                f"values has length {len(values)}, expected {self.count_obs()}"
            )

        # Infer DuckDB type from actual data
        series = pd.Series(values)
        if isinstance(series.dtype, pd.CategoricalDtype):
            # Categorical → VARCHAR
            self._set_categorical_schema("obs", column, series)
            series = series.astype(object)
            duck_type = "VARCHAR"
        elif pd.api.types.is_bool_dtype(series):
            duck_type = "BOOLEAN"
        elif pd.api.types.is_integer_dtype(series):
            duck_type = "BIGINT"
        elif pd.api.types.is_float_dtype(series):
            duck_type = "DOUBLE"
        elif pd.api.types.is_string_dtype(series) or series.dtype == object:
            series = series.astype("string")
            duck_type = "VARCHAR"
        else:
            duck_type = "VARCHAR"
            series = series.astype(str)

        quoted_column = _quote_identifier(column)
        temp_df = pd.DataFrame({"_val": series, "_rowid": range(len(series))})
        self._conn.register("_cellvault_new_obs_column", temp_df)
        try:
            self._conn.execute("BEGIN TRANSACTION")
            self._conn.execute(
                f"ALTER TABLE obs ADD COLUMN {quoted_column} {duck_type}"
            )
            position_column = (
                _quote_identifier(_ROW_POSITION_COLUMN)
                if self._has_row_positions()
                else "rowid"
            )
            self._conn.execute(f"""
                UPDATE obs SET {quoted_column} = t._val
                FROM _cellvault_new_obs_column t
                WHERE obs.{position_column} = t._rowid
            """)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        finally:
            self._conn.unregister("_cellvault_new_obs_column")
        self.provenance.log(
            "add_obs_column", "obs", key=column, new_hash=_hash_data(values)
        )

    def _has_obs(self) -> bool:
        try:
            self._conn.execute("SELECT count(*) FROM obs")
            return True
        except duckdb.CatalogException:
            return False

    def _stored_obs_columns(self) -> list[str]:
        if not self._has_obs():
            return []
        rows = self._conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='obs' ORDER BY ordinal_position"
        ).fetchall()
        return [row[0] for row in rows]

    def _has_row_positions(self) -> bool:
        return _ROW_POSITION_COLUMN in self._stored_obs_columns()

    @property
    def obs_columns(self) -> list[str]:
        return [
            column
            for column in self._stored_obs_columns()
            if column not in {"_index", _ROW_POSITION_COLUMN}
        ]

    def query_obs(
        self,
        where: str = "TRUE",
        params: Sequence | None = None,
        columns: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return matching obs rows and their stable matrix row positions."""
        query_params = _validate_predicate(where, params)
        if not self._has_obs():
            return pd.DataFrame(), np.array([], dtype=np.int64)

        position_expression = (
            _quote_identifier(_ROW_POSITION_COLUMN)
            if self._has_row_positions()
            else "rowid"
        )
        stored_columns = self._stored_obs_columns()
        if columns is None:
            projected_columns = [
                column for column in stored_columns if column != _ROW_POSITION_COLUMN
            ]
        else:
            if isinstance(columns, str):
                raise TypeError("columns must be a sequence of column names")
            requested_columns = list(columns)
            if len(requested_columns) != len(set(requested_columns)):
                raise ValueError("columns must not contain duplicates")
            available_columns = set(self.obs_columns)
            missing_columns = [
                column
                for column in requested_columns
                if column not in available_columns
            ]
            if missing_columns:
                raise KeyError(f"obs columns not found: {missing_columns}")
            projected_columns = (
                ["_index"] if "_index" in stored_columns else []
            ) + requested_columns
        projection = ", ".join(
            _quote_identifier(column) for column in projected_columns
        )
        if projection:
            projection = f", {projection}"
        result = self._conn.execute(
            f"SELECT {position_expression} AS {_quote_identifier(_QUERY_POSITION_COLUMN)}"
            f"{projection} "
            f"FROM obs WHERE {where} ORDER BY {position_expression}",
            query_params,
        ).fetchdf()
        positions = result.iloc[:, 0].to_numpy(dtype=np.int64)
        result = result.iloc[:, 1:]
        if "_index" in result.columns:
            result = result.set_index("_index")
            result.index.name = None
        return _restore_dataframe_schema(
            result,
            self._read_frame_schemas().get("obs"),
        ), positions

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
            "SELECT count(*) FROM read_parquet(?)",
            [str(self._var_path)],
        ).fetchone()
        return result[0]

    # ── var (Parquet) ────────────────────────────────────────────

    def write_var(self, df: pd.DataFrame):
        if self.X_exists and len(df) != self.X_shape[1]:
            raise ValueError(
                f"var has {len(df)} rows but X has {self.X_shape[1]} columns"
            )
        old_hash = _hash_data(self.read_var()) if self._var_path.exists() else None
        self._write_frame_schema("var", df)
        df_with_idx = _frame_for_storage(df)
        df_with_idx["_index"] = df.index
        df_with_idx.to_parquet(str(self._var_path))
        self._var_cache = df.copy()
        self.provenance.log(
            "write_var", "var", old_hash=old_hash, new_hash=_hash_data(df)
        )

    def read_var(self) -> pd.DataFrame:
        if not self._var_path.exists():
            return pd.DataFrame()
        if self._var_cache is None:
            df = pd.read_parquet(str(self._var_path))
            if "_index" in df.columns:
                df = df.set_index("_index")
                df.index.name = None
            self._var_cache = _restore_dataframe_schema(
                df,
                self._read_frame_schemas().get("var"),
            )
        return self._var_cache.copy()

    # ── X (Zarr) ─────────────────────────────────────────────────

    def _write_matrix(self, path: Path, value) -> None:
        raw_shape = getattr(value, "shape", None)
        if raw_shape is None or len(raw_shape) != 2:
            raise ValueError("matrix values must be two-dimensional")
        shape = tuple(int(dimension) for dimension in raw_shape)
        tmp_path = path.with_name(f"{path.name}.tmp")
        self._invalidate_matrix_cache(path)
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        store = zarr.storage.LocalStore(str(tmp_path))
        root = zarr.open_group(store, mode="w")
        if sparse.issparse(value):
            csr = value.tocsr()
            index_dtype = _sparse_index_dtype(shape, int(csr.nnz))
            _copy_1d_array(root, "data", csr.data)
            _copy_1d_array(root, "indices", csr.indices, dtype=index_dtype)
            _copy_1d_array(root, "indptr", csr.indptr, dtype=index_dtype)
            root.attrs["encoding_type"] = "csr_matrix"
        elif getattr(value, "format", None) == "csr" and hasattr(value, "group"):
            nnz = int(value.group["data"].shape[0])
            index_dtype = _sparse_index_dtype(shape, nnz)
            _copy_1d_array(root, "data", value.group["data"])
            _copy_1d_array(root, "indices", value.group["indices"], dtype=index_dtype)
            _copy_1d_array(root, "indptr", value.group["indptr"], dtype=index_dtype)
            root.attrs["encoding_type"] = "csr_matrix"
        elif getattr(value, "format", None) == "csc" and hasattr(value, "to_memory"):
            csr = value.to_memory().tocsr()
            index_dtype = _sparse_index_dtype(shape, int(csr.nnz))
            _copy_1d_array(root, "data", csr.data)
            _copy_1d_array(root, "indices", csr.indices, dtype=index_dtype)
            _copy_1d_array(root, "indptr", csr.indptr, dtype=index_dtype)
            root.attrs["encoding_type"] = "csr_matrix"
        else:
            _write_dense_array(root, "X", value)
            root.attrs["encoding_type"] = "dense"
        root.attrs["shape"] = list(shape)

        if path.exists():
            shutil.rmtree(path)
        tmp_path.rename(path)

    def _read_matrix(self, path: Path, row_indices=None, column_indices=None):
        if not path.exists():
            return None
        root = self._open_matrix(path)
        encoding = root.attrs.get("encoding_type", "dense")
        if encoding == "csr_matrix":
            return _read_csr_group(
                root,
                row_indices,
                column_indices,
                source_indptr=self._matrix_indptr(path, root),
            )
        array_key = "X" if "X" in root else "data"
        return _read_dense_array(root[array_key], row_indices, column_indices)

    def _read_matrix_many(
        self,
        path: Path,
        selections: Mapping[Hashable, object],
        column_indices=None,
    ) -> dict[Hashable, object]:
        if not path.exists():
            return {key: None for key in selections}
        root = self._open_matrix(path)
        encoding = root.attrs.get("encoding_type", "dense")
        if encoding == "csr_matrix":
            return _read_csr_group_many(
                root,
                selections,
                column_indices,
                source_indptr=self._matrix_indptr(path, root),
            )
        array_key = "X" if "X" in root else "data"
        return _read_dense_array_many(root[array_key], selections, column_indices)

    def _matrix_shape(self, path: Path) -> tuple[int, int]:
        if not path.exists():
            return (0, 0)
        return tuple(self._open_matrix(path).attrs["shape"])

    def write_X(self, X):
        """Write expression matrix to Zarr. Supports dense and sparse.

        Uses atomic write-to-temp-then-rename to prevent data loss on crash.
        """
        raw_shape = getattr(X, "shape", None)
        if raw_shape is None or len(raw_shape) != 2:
            raise ValueError("X must be a two-dimensional matrix")
        shape = tuple(raw_shape)
        self._validate_aligned_rows("X", shape[0], exclude="X")
        if self._var_path.exists() and shape[1] != self.count_vars():
            raise ValueError(
                f"X has {shape[1]} columns but var has {self.count_vars()} rows"
            )

        self._write_matrix(self._x_path, X)

        self.provenance.log("write_X", "X", new_hash=_hash_data(X))

    def read_X(self, row_indices=None, column_indices=None):
        """Read expression matrix from Zarr."""
        return self._read_matrix(self._x_path, row_indices, column_indices)

    def read_X_many(self, selections, column_indices=None):
        """Read multiple matrix row selections with shared source I/O."""
        if not isinstance(selections, Mapping):
            raise TypeError("selections must be a mapping of names to row selectors")
        return self._read_matrix_many(self._x_path, selections, column_indices)

    @property
    def X_exists(self) -> bool:
        return self._x_path.exists()

    @property
    def X_shape(self) -> tuple[int, int]:
        return self._matrix_shape(self._x_path)

    # ── layers (Zarr per key) ─────────────────────────────────────

    def write_layer(self, key: str, data) -> None:
        key = _validate_storage_key(key)
        shape = tuple(getattr(data, "shape", ()))
        if len(shape) != 2:
            raise ValueError("layer values must be two-dimensional")
        if self.X_exists and shape != self.X_shape:
            raise ValueError(
                f"layer has shape {shape}, expected X shape {self.X_shape}"
            )
        self._validate_aligned_rows(
            f"layers[{key!r}]", shape[0], exclude=f"layers[{key!r}]"
        )
        path = self._layers_path / f"{key}.zarr"
        self._write_matrix(path, data)
        self.provenance.log("write_layer", "layers", key=key, new_hash=_hash_data(data))

    def read_layer(self, key: str, row_indices=None, column_indices=None):
        key = _validate_storage_key(key)
        return self._read_matrix(
            self._layers_path / f"{key}.zarr",
            row_indices,
            column_indices,
        )

    def read_layer_many(self, key: str, selections, column_indices=None):
        key = _validate_storage_key(key)
        return self._read_matrix_many(
            self._layers_path / f"{key}.zarr",
            selections,
            column_indices,
        )

    @property
    def layer_keys(self) -> list[str]:
        return sorted(path.stem for path in self._layers_path.glob("*.zarr"))

    # ── obsm (Zarr per key) ──────────────────────────────────────

    def write_obsm(self, key: str, data: np.ndarray):
        key = _validate_storage_key(key)
        if getattr(data, "ndim", None) != 2:
            raise ValueError("obsm values must be two-dimensional")
        target = f"obsm[{key!r}]"
        self._validate_aligned_rows(target, data.shape[0], exclude=target)
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

    def read_obsm(self, key: str, row_indices=None) -> np.ndarray | None:
        key = _validate_storage_key(key)
        key_path = self._obsm_path / f"{key}.zarr"
        if not key_path.exists():
            return None
        store = zarr.storage.LocalStore(str(key_path))
        root = zarr.open_group(store, mode="r")
        return _read_dense_array(root["data"], row_indices)

    @property
    def obsm_keys(self) -> list[str]:
        return [p.stem for p in self._obsm_path.glob("*.zarr") if p.is_dir()]

    # ── obsp (Zarr per key, sparse) ──────────────────────────────

    def write_obsp(self, key: str, data):
        key = _validate_storage_key(key)
        if getattr(data, "ndim", None) != 2 or data.shape[0] != data.shape[1]:
            raise ValueError("obsp values must be square two-dimensional matrices")
        target = f"obsp[{key!r}]"
        self._validate_aligned_rows(target, data.shape[0], exclude=target)
        key_path = self._obsp_path / f"{key}.zarr"
        self._write_matrix(key_path, data)

        self.provenance.log("write_obsp", "obsp", key=key, new_hash=_hash_data(data))

    def read_obsp(self, key: str, row_indices=None):
        key = _validate_storage_key(key)
        key_path = self._obsp_path / f"{key}.zarr"
        if not key_path.exists():
            return None
        return self._read_matrix(key_path, row_indices, row_indices)

    @property
    def obsp_keys(self) -> list[str]:
        return [p.stem for p in self._obsp_path.glob("*.zarr") if p.is_dir()]

    # ── variable-aligned arrays ───────────────────────────────────

    def write_varm(self, key: str, data) -> None:
        key = _validate_storage_key(key)
        if getattr(data, "ndim", None) != 2:
            raise ValueError("varm values must be two-dimensional")
        if self._var_path.exists() and data.shape[0] != self.count_vars():
            raise ValueError(
                f"varm[{key!r}] has {data.shape[0]} rows but var has "
                f"{self.count_vars()} rows"
            )
        self._write_matrix(self._varm_path / f"{key}.zarr", data)
        self.provenance.log("write_varm", "varm", key=key, new_hash=_hash_data(data))

    def read_varm(self, key: str):
        key = _validate_storage_key(key)
        return self._read_matrix(self._varm_path / f"{key}.zarr")

    @property
    def varm_keys(self) -> list[str]:
        return sorted(path.stem for path in self._varm_path.glob("*.zarr"))

    def write_varp(self, key: str, data) -> None:
        key = _validate_storage_key(key)
        if getattr(data, "ndim", None) != 2 or data.shape[0] != data.shape[1]:
            raise ValueError("varp values must be square two-dimensional matrices")
        if self._var_path.exists() and data.shape[0] != self.count_vars():
            raise ValueError(
                f"varp[{key!r}] has {data.shape[0]} rows but var has "
                f"{self.count_vars()} rows"
            )
        self._write_matrix(self._varp_path / f"{key}.zarr", data)
        self.provenance.log("write_varp", "varp", key=key, new_hash=_hash_data(data))

    def read_varp(self, key: str):
        key = _validate_storage_key(key)
        return self._read_matrix(self._varp_path / f"{key}.zarr")

    @property
    def varp_keys(self) -> list[str]:
        return sorted(path.stem for path in self._varp_path.glob("*.zarr"))

    # ── raw ───────────────────────────────────────────────────────

    @property
    def raw_exists(self) -> bool:
        return (self._raw_path / "X.zarr").exists() and (
            self._raw_path / "var.parquet"
        ).exists()

    @property
    def raw_shape(self) -> tuple[int, int]:
        return self._matrix_shape(self._raw_path / "X.zarr")

    def write_raw_X(self, data) -> None:
        shape = tuple(getattr(data, "shape", ()))
        if len(shape) != 2:
            raise ValueError("raw.X must be two-dimensional")
        self._validate_aligned_rows("raw.X", shape[0], exclude="raw.X")
        raw_var_path = self._raw_path / "var.parquet"
        if raw_var_path.exists() and shape[1] != self.count_raw_vars():
            raise ValueError(
                f"raw.X has {shape[1]} columns but raw.var has "
                f"{self.count_raw_vars()} rows"
            )
        self._write_matrix(self._raw_path / "X.zarr", data)
        self.provenance.log("write_raw_X", "raw.X", new_hash=_hash_data(data))

    def read_raw_X(self, row_indices=None, column_indices=None):
        return self._read_matrix(
            self._raw_path / "X.zarr",
            row_indices,
            column_indices,
        )

    def read_raw_X_many(self, selections, column_indices=None):
        return self._read_matrix_many(
            self._raw_path / "X.zarr",
            selections,
            column_indices,
        )

    def write_raw_var(self, df: pd.DataFrame) -> None:
        if (self._raw_path / "X.zarr").exists() and len(df) != self.raw_shape[1]:
            raise ValueError(
                f"raw.var has {len(df)} rows but raw.X has {self.raw_shape[1]} columns"
            )
        self._write_frame_schema("raw_var", df)
        frame = _frame_for_storage(df)
        frame["_index"] = df.index
        frame.to_parquet(str(self._raw_path / "var.parquet"))
        self._raw_var_cache = df.copy()
        self.provenance.log("write_raw_var", "raw.var", new_hash=_hash_data(df))

    def read_raw_var(self) -> pd.DataFrame:
        path = self._raw_path / "var.parquet"
        if not path.exists():
            return pd.DataFrame()
        if self._raw_var_cache is None:
            frame = pd.read_parquet(path)
            if "_index" in frame.columns:
                frame = frame.set_index("_index")
                frame.index.name = None
            self._raw_var_cache = _restore_dataframe_schema(
                frame,
                self._read_frame_schemas().get("raw_var"),
            )
        return self._raw_var_cache.copy()

    def count_raw_vars(self) -> int:
        path = self._raw_path / "var.parquet"
        if not path.exists():
            return 0
        return self._conn.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()[0]

    def write_raw_varm(self, key: str, data) -> None:
        key = _validate_storage_key(key)
        if getattr(data, "ndim", None) != 2:
            raise ValueError("raw.varm values must be two-dimensional")
        if self.count_raw_vars() and data.shape[0] != self.count_raw_vars():
            raise ValueError(
                f"raw.varm[{key!r}] has {data.shape[0]} rows but raw.var has "
                f"{self.count_raw_vars()} rows"
            )
        path = self._raw_path / "varm"
        path.mkdir(exist_ok=True)
        self._write_matrix(path / f"{key}.zarr", data)
        self.provenance.log(
            "write_raw_varm", "raw.varm", key=key, new_hash=_hash_data(data)
        )

    def read_raw_varm(self, key: str):
        key = _validate_storage_key(key)
        return self._read_matrix(self._raw_path / "varm" / f"{key}.zarr")

    @property
    def raw_varm_keys(self) -> list[str]:
        path = self._raw_path / "varm"
        if not path.exists():
            return []
        return sorted(item.stem for item in path.glob("*.zarr"))

    # ── uns (JSON) ───────────────────────────────────────────────

    def write_uns(self, uns: dict):
        old_hash = _hash_data(str(self.read_uns())) if self._uns_path.exists() else None
        with open(self._uns_path, "w") as f:
            json.dump(_serialize_uns(uns), f, indent=2)
        self.provenance.log(
            "write_uns", "uns", old_hash=old_hash, new_hash=_hash_data(str(uns))
        )

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
