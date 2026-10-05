"""Shared-scan execution for bounded-memory CellVault aggregations."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    from .celldb import CellDB, CellView


_VALID_METRICS = frozenset({"sum", "mean", "count_nonzero"})
_MISSING_GROUP = "<NA>"


@runtime_checkable
class Task(Protocol):
    """Common structural contract for executable CellVault tasks."""

    name: str
    source: str
    features: tuple[Hashable, ...] | None
    where: str
    params: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible task definition."""
        ...


def _as_tuple(value, *, field_name: str, allow_string: bool) -> tuple:
    if isinstance(value, str):
        if allow_string:
            return (value,)
        raise TypeError(f"{field_name} must be a sequence, not a string")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence") from exc


def _normalize_optional_values(value, *, field_name: str) -> tuple | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        values = (value.item(),)
    elif isinstance(value, str):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError:
            values = (value,)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return values


@dataclass(frozen=True, slots=True)
class AggregateTask:
    """Description of one group-by aggregation over a CellDB or CellView.

    ``metrics`` may be empty when only group keys and ``n_cells`` are needed.
    ``features`` contains var names and preserves the requested order.
    """

    name: str
    groupby: tuple[str, ...] | Sequence[str] | str
    source: str = "X"
    metrics: tuple[str, ...] | Sequence[str] | str = ("sum",)
    features: tuple[Hashable, ...] | Sequence[Hashable] | Hashable | None = None
    where: str = "TRUE"
    params: tuple[Any, ...] | Sequence[Any] = ()
    membership: pd.DataFrame | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    membership_cell_id: str = "cell_id"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("AggregateTask.name must be a non-empty string")

        groupby = _as_tuple(
            self.groupby,
            field_name="AggregateTask.groupby",
            allow_string=True,
        )
        if not groupby:
            raise ValueError("AggregateTask.groupby must not be empty")
        if any(not isinstance(column, str) or not column for column in groupby):
            raise ValueError("AggregateTask.groupby must contain non-empty strings")
        if len(groupby) != len(set(groupby)):
            raise ValueError("AggregateTask.groupby must not contain duplicates")

        metrics = _as_tuple(
            self.metrics,
            field_name="AggregateTask.metrics",
            allow_string=True,
        )
        if any(not isinstance(metric, str) for metric in metrics):
            raise TypeError("AggregateTask.metrics must contain strings")
        unknown_metrics = set(metrics) - _VALID_METRICS
        if unknown_metrics:
            raise ValueError(
                f"Unknown aggregate metrics: {sorted(unknown_metrics)}; "
                f"expected a subset of {sorted(_VALID_METRICS)}"
            )
        if len(metrics) != len(set(metrics)):
            raise ValueError("AggregateTask.metrics must not contain duplicates")

        if not isinstance(self.source, str):
            raise TypeError("AggregateTask.source must be a string")
        if self.source != "X" and not (
            self.source.startswith("layers:") and self.source.removeprefix("layers:")
        ):
            raise ValueError(
                "AggregateTask.source must be 'X' or 'layers:<name>'"
            )

        if not isinstance(self.where, str) or not self.where.strip():
            raise ValueError("AggregateTask.where must be a non-empty SQL predicate")
        if isinstance(self.params, (str, bytes)):
            raise TypeError("AggregateTask.params must be a sequence")
        try:
            params = tuple(self.params)
        except TypeError as exc:
            raise TypeError("AggregateTask.params must be a sequence") from exc

        membership = self.membership
        if membership is not None:
            if not isinstance(membership, pd.DataFrame):
                raise TypeError("AggregateTask.membership must be a pandas DataFrame")
            if (
                not isinstance(self.membership_cell_id, str)
                or not self.membership_cell_id
            ):
                raise ValueError(
                    "AggregateTask.membership_cell_id must be a non-empty string"
                )
            if self.membership_cell_id not in membership.columns:
                raise KeyError(
                    f"membership cell ID column {self.membership_cell_id!r} not found"
                )
            if "membership" in groupby:
                raise ValueError("membership is reserved and cannot be a groupby column")
            membership = membership.copy(deep=True)
            if membership[self.membership_cell_id].isna().any():
                raise ValueError("membership cell IDs must not be missing")
            if "membership" in membership.columns:
                weights = membership["membership"]
                if weights.isna().any() or not weights.isin([0, 1, False, True]).all():
                    raise ValueError(
                        "membership values must be boolean or binary; weighted "
                        "memberships are not supported"
                    )
                membership = membership.loc[weights.astype(bool)].drop(
                    columns="membership"
                )
            membership_groupby = [
                column for column in groupby if column in membership.columns
            ]
            edge_columns = [self.membership_cell_id, *membership_groupby]
            if membership.duplicated(edge_columns).any():
                raise ValueError("membership contains duplicate cell/group edges")

        features = self.features
        if features is not None:
            if isinstance(features, np.ndarray) and features.ndim == 0:
                features = (features.item(),)
            elif isinstance(features, str):
                features = (features,)
            else:
                try:
                    features = tuple(features)
                except TypeError:
                    features = (features,)
            if not features:
                raise ValueError("AggregateTask.features must not be empty")
            if any(not isinstance(feature, Hashable) for feature in features):
                raise TypeError(
                    "AggregateTask.features must contain hashable var names"
                )
            if len(features) != len(set(features)):
                raise ValueError(
                    "AggregateTask.features must not contain duplicates"
                )

        object.__setattr__(self, "groupby", groupby)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "membership", membership)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible task definition."""
        membership = None
        if self.membership is not None:
            hashed = pd.util.hash_pandas_object(
                self.membership,
                index=True,
            ).to_numpy()
            membership = {
                "cell_id_column": self.membership_cell_id,
                "columns": list(self.membership.columns),
                "n_edges": len(self.membership),
                "sha256": hashlib.sha256(hashed.tobytes()).hexdigest(),
            }
        return {
            "name": self.name,
            "groupby": list(self.groupby),
            "source": self.source,
            "metrics": list(self.metrics),
            "features": None if self.features is None else list(self.features),
            "where": self.where,
            "params": list(self.params),
            "membership": membership,
        }


@dataclass(frozen=True, slots=True)
class MaterializeTask:
    """Description of one bounded, cell-level matrix materialization."""

    name: str
    where: str = "TRUE"
    params: tuple[Any, ...] | Sequence[Any] = ()
    source: str = "X"
    features: tuple[Hashable, ...] | Sequence[Hashable] | Hashable | None = None
    obs_columns: tuple[str, ...] | Sequence[str] | None = None
    cell_ids: tuple[Hashable, ...] | Sequence[Hashable] | Hashable | None = None
    consumer: Callable[[ad.AnnData], Any] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("MaterializeTask.name must be a non-empty string")
        if not isinstance(self.where, str) or not self.where.strip():
            raise ValueError("MaterializeTask.where must be a non-empty SQL predicate")
        if isinstance(self.params, (str, bytes)):
            raise TypeError("MaterializeTask.params must be a sequence")
        try:
            params = tuple(self.params)
        except TypeError as exc:
            raise TypeError("MaterializeTask.params must be a sequence") from exc
        if not isinstance(self.source, str):
            raise TypeError("MaterializeTask.source must be a string")
        if self.source != "X" and not (
            self.source.startswith("layers:") and self.source.removeprefix("layers:")
        ):
            raise ValueError("MaterializeTask.source must be 'X' or 'layers:<name>'")
        if self.consumer is not None and not callable(self.consumer):
            raise TypeError("MaterializeTask.consumer must be callable or None")

        features = _normalize_optional_values(
            self.features,
            field_name="MaterializeTask.features",
        )
        if features is not None:
            if any(not isinstance(feature, Hashable) for feature in features):
                raise TypeError(
                    "MaterializeTask.features must contain hashable var names"
                )
            if len(features) != len(set(features)):
                raise ValueError(
                    "MaterializeTask.features must not contain duplicates"
                )

        obs_columns = self.obs_columns
        if obs_columns is not None:
            obs_columns = _as_tuple(
                obs_columns,
                field_name="MaterializeTask.obs_columns",
                allow_string=False,
            )
            if any(not isinstance(column, str) or not column for column in obs_columns):
                raise ValueError(
                    "MaterializeTask.obs_columns must contain non-empty strings"
                )
            if len(obs_columns) != len(set(obs_columns)):
                raise ValueError(
                    "MaterializeTask.obs_columns must not contain duplicates"
                )

        cell_ids = _normalize_optional_values(
            self.cell_ids,
            field_name="MaterializeTask.cell_ids",
        )
        if cell_ids is not None:
            if any(not isinstance(cell_id, Hashable) for cell_id in cell_ids):
                raise TypeError("MaterializeTask.cell_ids must be hashable")
            if len(cell_ids) != len(set(cell_ids)):
                raise ValueError("MaterializeTask.cell_ids must not contain duplicates")

        object.__setattr__(self, "params", params)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "obs_columns", obs_columns)
        object.__setattr__(self, "cell_ids", cell_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible task definition."""
        return {
            "name": self.name,
            "where": self.where,
            "params": list(self.params),
            "source": self.source,
            "features": None if self.features is None else list(self.features),
            "obs_columns": (
                None if self.obs_columns is None else list(self.obs_columns)
            ),
            "cell_ids": None if self.cell_ids is None else list(self.cell_ids),
            "consumer": (
                None
                if self.consumer is None
                else getattr(self.consumer, "__name__", type(self.consumer).__name__)
            ),
        }


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Immutable execution statistics for one aggregate run.

    ``matrix_bytes_read`` counts decoded matrix buffers returned by the
    backend. It is a logical data-volume metric, not physical storage I/O.
    """

    run_id: str
    task_count: int
    source_scan_count: int
    matrix_batch_reads: int
    requested_rows: int
    unique_rows: int
    matrix_bytes_read: int
    reuse_ratio: float
    elapsed_seconds: float
    result_bytes: Mapping[str, int] = field(default_factory=dict)
    task_rows: Mapping[str, int] = field(default_factory=dict)
    task_memberships: Mapping[str, int] = field(default_factory=dict)
    memory_budget_bytes: int | None = None
    peak_buffer_bytes: int = 0
    execution_waves: int = 0
    scan_plan: tuple[Mapping[str, Any], ...] = ()
    degradation_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_bytes",
            MappingProxyType(dict(self.result_bytes)),
        )
        object.__setattr__(
            self,
            "task_rows",
            MappingProxyType(dict(self.task_rows)),
        )
        object.__setattr__(
            self,
            "task_memberships",
            MappingProxyType(dict(self.task_memberships)),
        )
        object.__setattr__(
            self,
            "scan_plan",
            tuple(MappingProxyType(dict(item)) for item in self.scan_plan),
        )
        object.__setattr__(
            self,
            "degradation_reasons",
            tuple(self.degradation_reasons),
        )

    @property
    def batch_count(self) -> int:
        """Alias for ``matrix_batch_reads``."""
        return self.matrix_batch_reads

    @property
    def total_result_bytes(self) -> int:
        """Combined in-memory size estimate for all results."""
        return sum(self.result_bytes.values())

    @property
    def overlap_rows(self) -> int:
        """Logical row requests avoided by compatible shared scans."""
        return max(0, self.requested_rows - self.unique_rows)

    @property
    def redundant_rows(self) -> int:
        """Rows reread because memory-budget waves split compatible tasks."""
        if self.scan_plan:
            return sum(
                int(item.get("repeated_rows", 0)) for item in self.scan_plan
            )
        return max(0, self.unique_rows - self.requested_rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""
        return {
            "run_id": self.run_id,
            "task_count": self.task_count,
            "source_scan_count": self.source_scan_count,
            "matrix_batch_reads": self.matrix_batch_reads,
            "requested_rows": self.requested_rows,
            "unique_rows": self.unique_rows,
            "matrix_bytes_read": self.matrix_bytes_read,
            "reuse_ratio": self.reuse_ratio,
            "elapsed_seconds": self.elapsed_seconds,
            "result_bytes": dict(self.result_bytes),
            "total_result_bytes": self.total_result_bytes,
            "task_rows": dict(self.task_rows),
            "task_memberships": dict(self.task_memberships),
            "overlap_rows": self.overlap_rows,
            "redundant_rows": self.redundant_rows,
            "memory_budget_bytes": self.memory_budget_bytes,
            "peak_buffer_bytes": self.peak_buffer_bytes,
            "execution_waves": self.execution_waves,
            "scan_plan": [dict(item) for item in self.scan_plan],
            "degradation_reasons": list(self.degradation_reasons),
        }


@dataclass(frozen=True, slots=True)
class AggregationRun:
    """Results and execution report returned by a joint aggregation."""

    results: Mapping[str, ad.AnnData]
    report: ExecutionReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))


@dataclass(frozen=True, slots=True)
class TaskRun:
    """Results and execution report returned by mixed task execution."""

    results: Mapping[str, Any]
    report: ExecutionReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))


@dataclass(slots=True)
class _Grouping:
    obs: pd.DataFrame
    row_to_group: np.ndarray
    n_cells: np.ndarray


@dataclass(slots=True)
class _FeatureSelection:
    positions: np.ndarray | None
    var: pd.DataFrame

    @property
    def n_features(self) -> int:
        return len(self.var)


@dataclass(slots=True)
class _PreparedTask:
    task: AggregateTask
    grouping: _Grouping
    selection: _FeatureSelection
    row_indices: np.ndarray
    sum_values: np.ndarray | None = None
    count_nonzero_values: np.ndarray | None = None

    @property
    def needs_sum(self) -> bool:
        return "sum" in self.task.metrics or "mean" in self.task.metrics

    @property
    def needs_count_nonzero(self) -> bool:
        return "count_nonzero" in self.task.metrics


@dataclass(slots=True)
class _PreparedMaterializeTask:
    task: MaterializeTask
    obs: pd.DataFrame
    selection: _FeatureSelection
    row_indices: np.ndarray
    chunks: list[Any] = field(default_factory=list)


def _target_backend(target: CellDB | CellView):
    backend = getattr(target, "_backend", None)
    if backend is not None:
        return backend
    parent = getattr(target, "_parent", None)
    backend = getattr(parent, "_backend", None)
    if backend is None:
        raise TypeError("target must be a CellDB or CellView")
    return backend


def _read_target_obs(
    target: CellDB | CellView,
    columns: Sequence[str] | None,
    where: str = "TRUE",
    params: Sequence[Any] = (),
) -> tuple[pd.DataFrame, np.ndarray]:
    backend = _target_backend(target)
    row_indices = getattr(target, "_row_indices", None)
    if row_indices is None:
        return backend.query_obs(where, params, columns=columns)

    local_obs = getattr(target, "_obs", None)
    if (
        where.strip().upper() == "TRUE"
        and not params
        and local_obs is not None
        and columns is not None
        and set(columns).issubset(local_obs.columns)
    ):
        return (
            local_obs.loc[:, list(columns)].copy(),
            np.asarray(row_indices, dtype=np.int64),
        )

    combined_where = f"({target.where}) AND ({where})"
    combined_params = (*target.params, *params)
    obs, queried_rows = backend.query_obs(
        combined_where,
        combined_params,
        columns=columns,
    )
    return obs, queried_rows


def _cohort_key(task: AggregateTask | MaterializeTask) -> tuple[str, str]:
    try:
        params_key = json.dumps(task.params, sort_keys=True, default=str)
    except (TypeError, ValueError):
        params_key = repr(task.params)
    return task.where, params_key


def _membership_grouping(
    task: AggregateTask,
    cohort_obs: pd.DataFrame,
    cohort_rows: np.ndarray,
) -> tuple[_Grouping, np.ndarray]:
    membership = task.membership
    if membership is None:
        return _build_grouping(cohort_obs, task.groupby), cohort_rows
    if not cohort_obs.index.is_unique:
        raise ValueError("obs names must be unique when using membership")

    cell_ids = membership[task.membership_cell_id]
    positions = cohort_obs.index.get_indexer(cell_ids)
    if np.any(positions < 0):
        unknown = pd.Index(cell_ids[positions < 0]).drop_duplicates().tolist()
        preview = unknown[:10]
        suffix = "" if len(unknown) <= len(preview) else " ..."
        raise KeyError(
            "membership contains cell IDs outside the target cohort: "
            f"{preview}{suffix}"
        )

    expanded = pd.DataFrame(index=pd.RangeIndex(len(membership)))
    for column in task.groupby:
        if column in membership.columns:
            expanded[column] = membership[column].reset_index(drop=True)
        else:
            expanded[column] = cohort_obs.iloc[positions][column].reset_index(
                drop=True
            )
    grouping = _build_grouping(expanded, task.groupby)
    member_rows = cohort_rows[positions]
    order = np.argsort(member_rows, kind="stable")
    return (
        _Grouping(
            grouping.obs,
            grouping.row_to_group[order],
            grouping.n_cells,
        ),
        member_rows[order],
    )


def _materialize_cohort(
    target: CellDB | CellView,
    task: MaterializeTask,
) -> tuple[pd.DataFrame, np.ndarray]:
    obs, rows = _read_target_obs(
        target,
        task.obs_columns,
        task.where,
        task.params,
    )
    if task.cell_ids is None:
        return obs, rows
    if not obs.index.is_unique:
        raise ValueError("obs names must be unique when selecting cell_ids")
    requested = pd.Index(task.cell_ids)
    missing = requested.difference(obs.index)
    if len(missing):
        preview = missing[:10].tolist()
        suffix = "" if len(missing) <= len(preview) else " ..."
        raise KeyError(
            f"cell_ids are outside the target cohort: {preview}{suffix}"
        )
    selected = obs.index.isin(requested)
    return obs.loc[selected].copy(), rows[selected]


def _read_source_batch(backend, source: str, rows, positions):
    if source == "X":
        return backend.read_X(row_indices=rows, column_indices=positions)
    return backend.read_layer(
        source.removeprefix("layers:"),
        row_indices=rows,
        column_indices=positions,
    )


def _rows_within_batch(
    task_rows: np.ndarray,
    batch_rows: np.ndarray,
) -> tuple[slice, np.ndarray]:
    member_start = np.searchsorted(task_rows, batch_rows[0], side="left")
    member_stop = np.searchsorted(task_rows, batch_rows[-1], side="right")
    selected_rows = task_rows[member_start:member_stop]
    if not len(selected_rows):
        return slice(member_start, member_stop), np.asarray([], dtype=np.int64)
    matrix_positions = np.searchsorted(batch_rows, selected_rows)
    if not np.array_equal(batch_rows[matrix_positions], selected_rows):
        raise RuntimeError("task rows are not contained in the planned batch")
    return slice(member_start, member_stop), matrix_positions


def _select_matrix_rows(matrix, positions: np.ndarray):
    """Select matrix rows without copying a batch that is already exact."""
    if not len(positions):
        return matrix[:0]
    first = int(positions[0])
    stop = int(positions[-1]) + 1
    contiguous = stop - first == len(positions) and (
        len(positions) == 1 or bool(np.all(positions[1:] == positions[:-1] + 1))
    )
    if contiguous:
        if first == 0 and stop == matrix.shape[0]:
            return matrix
        return matrix[first:stop]
    return matrix[positions]


def _factorize_column(
    series: pd.Series,
) -> tuple[np.ndarray, list[Any], bool, bool]:
    categorical = isinstance(series.dtype, pd.CategoricalDtype)
    if categorical:
        raw_codes = series.cat.codes.to_numpy(dtype=np.int64, copy=True)
        observed_codes = set(raw_codes[raw_codes >= 0].tolist())
        original_categories = series.cat.categories
        category_positions = [
            position
            for position in range(len(original_categories))
            if position in observed_codes
        ]
        remap = {
            old_position: new_position
            for new_position, old_position in enumerate(category_positions)
        }
        codes = np.fromiter(
            (remap.get(int(code), -1) for code in raw_codes),
            dtype=np.int64,
            count=len(raw_codes),
        )
        levels = [original_categories[position] for position in category_positions]
        ordered = series.cat.ordered
    else:
        codes, uniques = pd.factorize(series, sort=False, use_na_sentinel=True)
        codes = codes.astype(np.int64, copy=False)
        levels = uniques.tolist()
        ordered = False

    has_missing = bool(np.any(codes < 0))
    if has_missing:
        if any(isinstance(level, str) and level == _MISSING_GROUP for level in levels):
            raise ValueError(
                f"obs column {series.name!r} contains both missing values and the "
                f"reserved missing-group label {_MISSING_GROUP!r}"
            )
        missing_code = len(levels)
        codes = codes.copy()
        codes[codes < 0] = missing_code
        levels.append(_MISSING_GROUP)
    return codes, levels, categorical, ordered


def _build_grouping(obs: pd.DataFrame, groupby: tuple[str, ...]) -> _Grouping:
    column_codes: list[np.ndarray] = []
    column_levels: list[list[Any]] = []
    categorical: list[bool] = []
    ordered: list[bool] = []

    for column in groupby:
        codes, levels, is_categorical, is_ordered = _factorize_column(obs[column])
        column_codes.append(codes)
        column_levels.append(levels)
        categorical.append(is_categorical)
        ordered.append(is_ordered)

    if len(obs) == 0:
        combinations = np.empty((0, len(groupby)), dtype=np.int64)
        row_to_group = np.empty(0, dtype=np.int64)
    else:
        dimensions = tuple(len(levels) for levels in column_levels)
        try:
            flat_codes = np.ravel_multi_index(tuple(column_codes), dimensions)
            observed_codes, row_to_group = np.unique(
                flat_codes,
                return_inverse=True,
            )
            combinations = np.column_stack(
                np.unravel_index(observed_codes, dimensions)
            )
        except ValueError:
            combinations, row_to_group = np.unique(
                np.column_stack(column_codes),
                axis=0,
                return_inverse=True,
            )
        row_to_group = row_to_group.astype(np.int64, copy=False)
    n_cells = np.bincount(row_to_group, minlength=len(combinations)).astype(
        np.int64,
        copy=False,
    )

    group_obs = pd.DataFrame(index=pd.RangeIndex(len(combinations)))
    for column_index, column in enumerate(groupby):
        values = [
            column_levels[column_index][combination[column_index]]
            for combination in combinations
        ]
        if categorical[column_index]:
            group_obs[column] = pd.Categorical(
                values,
                categories=column_levels[column_index],
                ordered=ordered[column_index],
            )
        elif _MISSING_GROUP in values:
            group_obs[column] = pd.Series(values, dtype=object)
        else:
            try:
                group_obs[column] = pd.Series(values, dtype=obs[column].dtype)
            except (TypeError, ValueError):
                group_obs[column] = values
    group_obs["n_cells"] = n_cells
    return _Grouping(group_obs, row_to_group, n_cells)


def _validate_features(
    task: AggregateTask | MaterializeTask,
    var: pd.DataFrame,
) -> _FeatureSelection:
    if task.features is None:
        return _FeatureSelection(None, var.copy())

    for feature in task.features:
        if not isinstance(feature, Hashable):
            raise TypeError(f"{type(task).__name__}.features must contain hashable var names")
    if len(task.features) != len(set(task.features)):
        raise ValueError(
            f"{type(task).__name__} {task.name!r} features must not contain duplicates"
        )
    if not var.index.is_unique:
        raise ValueError("var names must be unique when selecting features")
    positions = var.index.get_indexer(task.features)
    if np.any(positions < 0):
        missing = [
            task.features[index]
            for index, position in enumerate(positions)
            if position < 0
        ]
        raise KeyError(f"var features not found for task {task.name!r}: {missing}")
    return _FeatureSelection(
        positions.astype(np.int64, copy=False),
        var.iloc[positions].copy(),
    )


def _validate_matrix_source(backend, source: str) -> None:
    if source == "X":
        if not backend.X_exists:
            raise KeyError("X is not available")
        return
    layer = source.removeprefix("layers:")
    if layer not in backend.layer_keys:
        raise KeyError(f"layer {layer!r} not found. Available: {backend.layer_keys}")


def _prepare_aggregate_tasks(
    target: CellDB | CellView,
    task_list: tuple[AggregateTask, ...],
    backend,
    var: pd.DataFrame,
) -> list[_PreparedTask]:
    obs_columns = set(backend.obs_columns)
    task_obs_columns: dict[str, tuple[str, ...]] = {}
    for task in task_list:
        membership_columns = (
            set() if task.membership is None else set(task.membership.columns)
        )
        columns = tuple(
            column for column in task.groupby if column not in membership_columns
        )
        missing_obs_columns = [
            column for column in columns if column not in obs_columns
        ]
        if missing_obs_columns:
            raise KeyError(
                f"obs columns not found for task {task.name!r}: "
                f"{missing_obs_columns}"
            )
        task_obs_columns[task.name] = columns
        if task.metrics:
            _validate_matrix_source(backend, task.source)

    selection_cache: dict[tuple[Hashable, ...] | None, _FeatureSelection] = {}
    for task in task_list:
        if task.features not in selection_cache:
            selection_cache[task.features] = _validate_features(task, var)

    cohort_tasks: dict[tuple[str, str], list[AggregateTask]] = {}
    for task in task_list:
        cohort_tasks.setdefault(_cohort_key(task), []).append(task)

    cohort_data: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for cohort in cohort_tasks.values():
        cohort_columns = tuple(
            dict.fromkeys(
                column for task in cohort for column in task_obs_columns[task.name]
            )
        )
        obs, row_indices = _read_target_obs(
            target,
            cohort_columns,
            cohort[0].where,
            cohort[0].params,
        )
        if len(row_indices) > 1 and np.any(row_indices[1:] <= row_indices[:-1]):
            raise RuntimeError("cohort row positions must be unique and sorted")
        for task in cohort:
            cohort_data[task.name] = (obs, row_indices)

    grouping_cache: dict[tuple[tuple[str, str], tuple[str, ...]], _Grouping] = {}
    prepared_tasks: list[_PreparedTask] = []
    for task in task_list:
        obs, row_indices = cohort_data[task.name]
        grouping_key = (_cohort_key(task), task.groupby)
        if task.membership is None:
            grouping = grouping_cache.get(grouping_key)
            if grouping is None:
                grouping = _build_grouping(obs, task.groupby)
                grouping_cache[grouping_key] = grouping
            task_row_indices = row_indices
        else:
            grouping, task_row_indices = _membership_grouping(
                task,
                obs,
                row_indices,
            )
        prepared_tasks.append(
            _PreparedTask(
                task,
                grouping,
                selection_cache[task.features],
                task_row_indices,
            )
        )
    return prepared_tasks


def _prepare_materialize_tasks(
    target: CellDB | CellView,
    task_list: tuple[MaterializeTask, ...],
    backend,
    var: pd.DataFrame,
) -> list[_PreparedMaterializeTask]:
    selection_cache: dict[tuple[Hashable, ...] | None, _FeatureSelection] = {}
    prepared_tasks: list[_PreparedMaterializeTask] = []
    for task in task_list:
        _validate_matrix_source(backend, task.source)
        if task.features not in selection_cache:
            selection_cache[task.features] = _validate_features(task, var)
        if task.obs_columns is not None:
            missing_columns = [
                column
                for column in task.obs_columns
                if column not in backend.obs_columns
            ]
            if missing_columns:
                raise KeyError(
                    f"obs columns not found for task {task.name!r}: "
                    f"{missing_columns}"
                )
        obs, row_indices = _materialize_cohort(target, task)
        if len(row_indices) > 1 and np.any(row_indices[1:] <= row_indices[:-1]):
            raise RuntimeError("materialize row positions must be unique and sorted")
        prepared_tasks.append(
            _PreparedMaterializeTask(
                task,
                obs,
                selection_cache[task.features],
                row_indices,
            )
        )
    return prepared_tasks


def _sum_dtype(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return np.dtype(np.int64)
    if dtype.kind == "i":
        return np.dtype(np.int64) if dtype.itemsize < 8 else dtype
    if dtype.kind == "u":
        return np.dtype(np.uint64)
    if dtype.kind == "f":
        return np.dtype(np.float64)
    if dtype.kind == "c":
        return np.dtype(np.complex128)
    return np.dtype(np.float64)


def _matrix_nbytes(matrix) -> int:
    if sparse.issparse(matrix):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return int(np.asarray(matrix).nbytes)


def _initialize_accumulators(task: _PreparedTask, dtype: np.dtype) -> None:
    shape = (len(task.grouping.n_cells), task.selection.n_features)
    if task.needs_sum and task.sum_values is None:
        task.sum_values = np.zeros(shape, dtype=_sum_dtype(dtype))
    if task.needs_count_nonzero and task.count_nonzero_values is None:
        task.count_nonzero_values = np.zeros(shape, dtype=np.int64)


def _accumulate_dense(
    prepared: _PreparedTask,
    matrix: np.ndarray,
    group_codes: np.ndarray,
) -> None:
    for group in np.unique(group_codes):
        block = matrix[group_codes == group]
        if prepared.needs_sum:
            prepared.sum_values[group] += np.sum(
                block,
                axis=0,
                dtype=prepared.sum_values.dtype,
            )
        if prepared.needs_count_nonzero:
            prepared.count_nonzero_values[group] += np.count_nonzero(block, axis=0)


def _accumulate_sparse(
    prepared: _PreparedTask,
    matrix,
    group_codes: np.ndarray,
) -> None:
    csr = matrix.tocsr(copy=False)
    for group in np.unique(group_codes):
        block = csr[group_codes == group]
        if prepared.needs_sum:
            prepared.sum_values[group] += np.asarray(
                block.sum(axis=0, dtype=prepared.sum_values.dtype)
            ).reshape(-1)
        if prepared.needs_count_nonzero:
            canonical = block.copy()
            canonical.sum_duplicates()
            canonical.eliminate_zeros()
            prepared.count_nonzero_values[group] += np.bincount(
                canonical.indices,
                minlength=prepared.selection.n_features,
            )


def _accumulate(
    prepared: _PreparedTask,
    matrix,
    group_codes: np.ndarray,
) -> None:
    _initialize_accumulators(prepared, matrix.dtype)
    if sparse.issparse(matrix):
        _accumulate_sparse(prepared, matrix, group_codes)
    else:
        _accumulate_dense(prepared, np.asarray(matrix), group_codes)


def _empty_accumulators(prepared: _PreparedTask) -> None:
    _initialize_accumulators(prepared, np.dtype(np.float64))


def _result_nbytes(result: ad.AnnData) -> int:
    size = int(result.obs.memory_usage(index=True, deep=True).sum())
    size += int(result.var.memory_usage(index=True, deep=True).sum())
    if result.X is not None:
        size += _matrix_nbytes(result.X)
    for layer in result.layers.values():
        size += _matrix_nbytes(layer)
    try:
        size += len(json.dumps(result.uns, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        pass
    return size


def _output_nbytes(value: Any) -> int:
    if isinstance(value, ad.AnnData):
        return _result_nbytes(value)
    if sparse.issparse(value) or isinstance(value, np.ndarray):
        return _matrix_nbytes(value)
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    try:
        return len(json.dumps(value, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _build_result(
    prepared: _PreparedTask,
    run_id: str,
) -> ad.AnnData:
    _empty_accumulators(prepared)
    obs = prepared.grouping.obs.copy()
    obs.index = pd.Index(
        [f"{prepared.task.name}:{group}" for group in range(len(obs))],
        name="group",
    )
    if prepared.task.membership is not None:
        obs["view_id"] = obs.index.astype(str)
    result = ad.AnnData(X=None, obs=obs, var=prepared.selection.var.copy())
    if "sum" in prepared.task.metrics:
        result.layers["sum"] = prepared.sum_values
    if "mean" in prepared.task.metrics:
        mean_dtype = np.result_type(prepared.sum_values.dtype, np.float64)
        result.layers["mean"] = np.divide(
            prepared.sum_values.astype(mean_dtype, copy=False),
            prepared.grouping.n_cells[:, None],
        )
    if "count_nonzero" in prepared.task.metrics:
        result.layers["count_nonzero"] = prepared.count_nonzero_values
    result.uns["cellvault_aggregation"] = {
        "run_id": run_id,
        "analysis_id": run_id,
        "task_id": prepared.task.name,
        "task": prepared.task.to_dict(),
        "source": prepared.task.source,
        "n_input_cells": int(len(np.unique(prepared.row_indices))),
        "n_memberships": int(prepared.grouping.n_cells.sum()),
        "n_groups": len(prepared.grouping.n_cells),
    }
    return result


def _build_materialize_result(
    prepared: _PreparedMaterializeTask,
    run_id: str,
) -> ad.AnnData:
    if prepared.chunks:
        if sparse.issparse(prepared.chunks[0]):
            matrix = sparse.vstack(prepared.chunks, format="csr")
        else:
            matrix = np.concatenate(prepared.chunks, axis=0)
    else:
        matrix = np.empty((0, prepared.selection.n_features), dtype=np.float64)
    result = ad.AnnData(
        X=matrix,
        obs=prepared.obs.copy(),
        var=prepared.selection.var.copy(),
    )
    result.uns["cellvault_materialization"] = {
        "run_id": run_id,
        "analysis_id": run_id,
        "task_id": prepared.task.name,
        "task": prepared.task.to_dict(),
        "source": prepared.task.source,
        "n_input_cells": len(prepared.row_indices),
    }
    return result


def _prepared_buffer_bytes(
    aggregate_tasks: Sequence[_PreparedTask],
    materialize_tasks: Sequence[_PreparedMaterializeTask],
) -> int:
    return _aggregate_buffer_bytes(aggregate_tasks) + _consumer_buffer_bytes(
        materialize_tasks
    )


def _aggregate_buffer_bytes(
    aggregate_tasks: Sequence[_PreparedTask],
) -> int:
    size = 0
    for prepared in aggregate_tasks:
        if prepared.sum_values is not None:
            size += int(prepared.sum_values.nbytes)
        if prepared.count_nonzero_values is not None:
            size += int(prepared.count_nonzero_values.nbytes)
    return size


def _consumer_buffer_bytes(
    materialize_tasks: Sequence[_PreparedMaterializeTask],
) -> int:
    size = 0
    for prepared in materialize_tasks:
        size += sum(_matrix_nbytes(chunk) for chunk in prepared.chunks)
    return size


def _aggregate_finalization_bytes(
    aggregate_tasks: Sequence[_PreparedTask],
) -> int:
    """Return buffers allocated in addition to existing accumulators."""
    size = 0
    for prepared in aggregate_tasks:
        if "mean" not in prepared.task.metrics:
            continue
        if prepared.sum_values is None:
            _empty_accumulators(prepared)
        mean_dtype = np.result_type(prepared.sum_values.dtype, np.float64)
        size += prepared.sum_values.size * mean_dtype.itemsize
    return int(size)


def _source_matrix_root(backend, source: str):
    path = (
        backend._x_path
        if source == "X"
        else backend._layers_path / f"{source.removeprefix('layers:')}.zarr"
    )
    return backend._open_matrix(path), path


def _source_dtype(backend, source: str) -> np.dtype:
    root, _ = _source_matrix_root(backend, source)
    if root.attrs.get("encoding_type", "dense") == "csr_matrix":
        return np.dtype(root["data"].dtype)
    array_key = "X" if "X" in root else "data"
    return np.dtype(root[array_key].dtype)


def _estimate_matrix_bytes(
    backend,
    source: str,
    selection: _FeatureSelection,
    row_indices: np.ndarray,
    *,
    chunked: bool = False,
) -> int:
    """Estimate a decoded selection without loading expression values.

    Dense estimates are exact. Sparse estimates use the selected rows' source
    nnz and are conservative when a feature subset removes stored values.
    """
    rows = np.asarray(row_indices, dtype=np.int64)
    if not len(rows):
        return 0
    root, path = _source_matrix_root(backend, source)
    encoding = root.attrs.get("encoding_type", "dense")
    if encoding != "csr_matrix":
        array_key = "X" if "X" in root else "data"
        return int(
            len(rows)
            * selection.n_features
            * np.dtype(root[array_key].dtype).itemsize
        )

    indptr = backend._matrix_indptr(path, root)
    nnz = int(np.sum(indptr[rows + 1] - indptr[rows], dtype=np.int64))
    index_itemsize = max(
        np.dtype(root["indices"].dtype).itemsize,
        np.dtype(root["indptr"].dtype).itemsize,
    )
    indptr_entries = 2 * len(rows) if chunked else len(rows) + 1
    return int(
        nnz
        * (np.dtype(root["data"].dtype).itemsize + index_itemsize)
        + indptr_entries * index_itemsize
    )


def _estimate_aggregate_bytes(
    prepared_tasks: Sequence[_PreparedTask],
    dtype: np.dtype,
) -> int:
    size = 0
    for prepared in prepared_tasks:
        elements = len(prepared.grouping.n_cells) * prepared.selection.n_features
        if prepared.needs_sum:
            size += elements * _sum_dtype(dtype).itemsize
        if prepared.needs_count_nonzero:
            size += elements * np.dtype(np.int64).itemsize
    return int(size)


def _estimate_wave_memory(
    backend,
    source: str,
    aggregate_tasks: Sequence[_PreparedTask],
    materialize_tasks: Sequence[_PreparedMaterializeTask],
    *,
    batch_size: int,
    retained_bytes: int,
) -> tuple[int, int]:
    all_consumers = [*aggregate_tasks, *materialize_tasks]
    selection = all_consumers[0].selection
    union_rows = np.unique(
        np.concatenate([prepared.row_indices for prepared in all_consumers])
    )
    aggregate_bytes = _estimate_aggregate_bytes(
        aggregate_tasks,
        _source_dtype(backend, source),
    )
    persistent_bytes = retained_bytes + aggregate_bytes

    peak_batch_bytes = 0
    for start in range(0, len(union_rows), batch_size):
        rows = union_rows[start : start + batch_size]
        peak_batch_bytes = max(
            peak_batch_bytes,
            _estimate_matrix_bytes(backend, source, selection, rows),
        )

    chunk_bytes = [
        _estimate_matrix_bytes(
            backend,
            source,
            prepared.selection,
            prepared.row_indices,
            chunked=True,
        )
        for prepared in materialize_tasks
    ]
    result_matrix_bytes = [
        _estimate_matrix_bytes(
            backend,
            source,
            prepared.selection,
            prepared.row_indices,
        )
        for prepared in materialize_tasks
    ]
    remaining_chunks = sum(chunk_bytes)
    peak_bytes = persistent_bytes + remaining_chunks + peak_batch_bytes
    retained_after = persistent_bytes
    for prepared, buffered, result_size in zip(
        materialize_tasks,
        chunk_bytes,
        result_matrix_bytes,
        strict=True,
    ):
        peak_bytes = max(
            peak_bytes,
            retained_after + remaining_chunks + result_size,
        )
        remaining_chunks -= buffered
        if prepared.task.consumer is None:
            retained_after += result_size
    return int(peak_bytes), int(retained_after)


def _plan_execution_waves(
    backend,
    scan_groups: Mapping[
        tuple[str, tuple[Hashable, ...] | None],
        Mapping[str, list[Any]],
    ],
    *,
    batch_size: int,
    memory_budget_bytes: int | None,
):
    planned_waves = []
    degradation_reasons = []
    retained_bytes = 0

    for key, consumers in scan_groups.items():
        source, features = key
        aggregate_tasks = list(consumers["aggregate"])
        materialize_tasks = list(consumers["materialize"])
        group_waves = []

        def add_wave(aggregates, materializers) -> None:
            nonlocal retained_bytes
            peak_bytes, retained_after = _estimate_wave_memory(
                backend,
                source,
                aggregates,
                materializers,
                batch_size=batch_size,
                retained_bytes=retained_bytes,
            )
            group_waves.append((key, aggregates, materializers, peak_bytes))
            retained_bytes = retained_after

        combined_peak, _ = _estimate_wave_memory(
            backend,
            source,
            aggregate_tasks,
            materialize_tasks,
            batch_size=batch_size,
            retained_bytes=retained_bytes,
        )
        if memory_budget_bytes is None or combined_peak <= memory_budget_bytes:
            add_wave(aggregate_tasks, materialize_tasks)
        elif not materialize_tasks:
            add_wave(aggregate_tasks, [])
        else:
            current_aggregates = aggregate_tasks
            current_materializers = []
            for materializer in materialize_tasks:
                candidate = [*current_materializers, materializer]
                candidate_peak, _ = _estimate_wave_memory(
                    backend,
                    source,
                    current_aggregates,
                    candidate,
                    batch_size=batch_size,
                    retained_bytes=retained_bytes,
                )
                if candidate_peak <= memory_budget_bytes:
                    current_materializers = candidate
                    continue

                if current_materializers:
                    add_wave(current_aggregates, current_materializers)
                    current_aggregates = []
                    current_materializers = [materializer]
                    continue

                if current_aggregates:
                    aggregate_peak, _ = _estimate_wave_memory(
                        backend,
                        source,
                        current_aggregates,
                        [],
                        batch_size=batch_size,
                        retained_bytes=retained_bytes,
                    )
                    if aggregate_peak <= memory_budget_bytes:
                        add_wave(current_aggregates, [])
                        current_aggregates = []
                current_materializers = [materializer]

            if current_aggregates or current_materializers:
                add_wave(current_aggregates, current_materializers)

        if len(group_waves) > 1:
            feature_scope = (
                "all features"
                if features is None
                else f"{len(features)} selected features"
            )
            degradation_reasons.append(
                "memory_budget_bytes split "
                f"{len(materialize_tasks)} compatible materialization tasks for "
                f"source {source!r} ({feature_scope}) across "
                f"{len(group_waves)} waves"
            )
        planned_waves.extend(group_waves)

    return planned_waves, degradation_reasons


def execute_aggregate_tasks(
    target: CellDB | CellView,
    tasks: Sequence[AggregateTask],
    *,
    batch_size: int = 4096,
) -> AggregationRun:
    """Execute compatible aggregate tasks with one matrix scan per source/axis.

    Compatible tasks have the same ``source`` and identical ``features`` request.
    Matrix rows are read once per compatible batch and reused by every task in it.
    """
    started = time.perf_counter()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(tasks, (str, bytes)):
        raise TypeError("tasks must be a sequence of AggregateTask objects")
    task_list = tuple(tasks)
    if not task_list:
        raise ValueError("tasks must not be empty")
    if any(not isinstance(task, AggregateTask) for task in task_list):
        raise TypeError("tasks must contain only AggregateTask objects")
    names = [task.name for task in task_list]
    if len(names) != len(set(names)):
        raise ValueError("AggregateTask names must be unique within a run")

    backend = _target_backend(target)
    var = backend.read_var()
    prepared_tasks = _prepare_aggregate_tasks(target, task_list, backend, var)

    scan_groups: dict[
        tuple[str, tuple[Hashable, ...] | None],
        list[_PreparedTask],
    ] = {}
    for prepared in prepared_tasks:
        if prepared.task.metrics:
            scan_groups.setdefault(
                (prepared.task.source, prepared.task.features),
                [],
            ).append(prepared)

    matrix_batch_reads = 0
    matrix_bytes_read = 0
    unique_rows = 0
    for (source, _), compatible_tasks in scan_groups.items():
        selection = compatible_tasks[0].selection
        row_arrays = [prepared.row_indices for prepared in compatible_tasks]
        union_rows = (
            np.unique(np.concatenate(row_arrays))
            if row_arrays
            else np.asarray([], dtype=np.int64)
        )
        unique_rows += len(union_rows)
        for start in range(0, len(union_rows), batch_size):
            stop = min(start + batch_size, len(union_rows))
            rows = union_rows[start:stop]
            matrix = _read_source_batch(
                backend,
                source,
                rows,
                selection.positions,
            )
            if matrix is None:
                raise KeyError(f"matrix source {source!r} is not available")
            expected_shape = (len(rows), selection.n_features)
            if tuple(matrix.shape) != expected_shape:
                raise RuntimeError(
                    f"matrix source {source!r} returned shape {matrix.shape}, "
                    f"expected {expected_shape}"
                )
            matrix_batch_reads += 1
            matrix_bytes_read += _matrix_nbytes(matrix)
            for prepared in compatible_tasks:
                member_slice, matrix_positions = _rows_within_batch(
                    prepared.row_indices,
                    rows,
                )
                if not len(matrix_positions):
                    continue
                _accumulate(
                    prepared,
                    _select_matrix_rows(matrix, matrix_positions),
                    prepared.grouping.row_to_group[member_slice],
                )

    run_id = uuid.uuid4().hex
    results = {
        prepared.task.name: _build_result(prepared, run_id)
        for prepared in prepared_tasks
    }
    result_bytes = {
        name: _result_nbytes(result) for name, result in results.items()
    }
    task_rows = {
        prepared.task.name: int(len(np.unique(prepared.row_indices)))
        for prepared in prepared_tasks
    }
    task_memberships = {
        prepared.task.name: int(len(prepared.row_indices))
        for prepared in prepared_tasks
    }
    requested_rows = sum(
        task_rows[prepared.task.name]
        for prepared in prepared_tasks
        if prepared.task.metrics
    )
    reuse_ratio = requested_rows / unique_rows if unique_rows else 1.0
    report = ExecutionReport(
        run_id=run_id,
        task_count=len(task_list),
        source_scan_count=len(scan_groups),
        matrix_batch_reads=matrix_batch_reads,
        requested_rows=requested_rows,
        unique_rows=unique_rows,
        matrix_bytes_read=matrix_bytes_read,
        reuse_ratio=reuse_ratio,
        elapsed_seconds=time.perf_counter() - started,
        result_bytes=result_bytes,
        task_rows=task_rows,
        task_memberships=task_memberships,
    )
    return AggregationRun(results, report)


def execute_tasks(
    target: CellDB | CellView,
    tasks: Sequence[Task],
    *,
    batch_size: int = 4096,
    memory_budget_bytes: int | None = None,
) -> TaskRun:
    """Execute aggregate and cell-level tasks through compatible shared scans.

    The optional memory budget is a hard guard over decoded batches,
    accumulators, and materialized matrix buffers. A task that cannot fit is
    rejected with a deterministic ``MemoryError`` rather than overcommitting.
    """
    started = time.perf_counter()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if memory_budget_bytes is not None:
        if isinstance(memory_budget_bytes, bool) or not isinstance(
            memory_budget_bytes, int
        ):
            raise TypeError("memory_budget_bytes must be an integer or None")
        if memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
    if isinstance(tasks, (str, bytes)):
        raise TypeError("tasks must be a sequence of task objects")
    task_list = tuple(tasks)
    if not task_list:
        raise ValueError("tasks must not be empty")
    valid_types = (AggregateTask, MaterializeTask)
    if any(not isinstance(task, valid_types) for task in task_list):
        raise TypeError(
            "tasks must contain only AggregateTask or MaterializeTask objects"
        )
    names = [task.name for task in task_list]
    if len(names) != len(set(names)):
        raise ValueError("task names must be unique within a run")

    aggregate_tasks = tuple(
        task for task in task_list if isinstance(task, AggregateTask)
    )
    materialize_tasks = tuple(
        task for task in task_list if isinstance(task, MaterializeTask)
    )
    backend = _target_backend(target)
    var = backend.read_var()
    prepared_aggregates = _prepare_aggregate_tasks(
        target,
        aggregate_tasks,
        backend,
        var,
    )
    prepared_materializations = _prepare_materialize_tasks(
        target,
        materialize_tasks,
        backend,
        var,
    )

    scan_groups: dict[
        tuple[str, tuple[Hashable, ...] | None],
        dict[str, list[Any]],
    ] = {}
    for prepared in prepared_aggregates:
        if prepared.task.metrics:
            group = scan_groups.setdefault(
                (prepared.task.source, prepared.task.features),
                {"aggregate": [], "materialize": []},
            )
            group["aggregate"].append(prepared)
    for prepared in prepared_materializations:
        group = scan_groups.setdefault(
            (prepared.task.source, prepared.task.features),
            {"aggregate": [], "materialize": []},
        )
        group["materialize"].append(prepared)

    planned_waves, degradation_reasons = _plan_execution_waves(
        backend,
        scan_groups,
        batch_size=batch_size,
        memory_budget_bytes=memory_budget_bytes,
    )

    matrix_batch_reads = 0
    matrix_bytes_read = 0
    unique_rows = 0
    peak_buffer_bytes = 0
    retained_result_bytes = 0
    run_id = uuid.uuid4().hex
    built_results: dict[str, Any] = {}
    scan_plan = []
    seen_rows: dict[tuple[str, tuple[Hashable, ...] | None], np.ndarray] = {}
    for wave, (
        (source, features),
        aggregate_consumers,
        materialize_consumers,
        estimated_peak_buffer_bytes,
    ) in enumerate(planned_waves, start=1):
        all_consumers = [*aggregate_consumers, *materialize_consumers]
        selection = all_consumers[0].selection
        union_rows = np.unique(
            np.concatenate([prepared.row_indices for prepared in all_consumers])
        )
        unique_rows += len(union_rows)
        scan_key = (source, features)
        prior_rows = seen_rows.get(scan_key)
        repeated_rows = (
            0
            if prior_rows is None
            else int(np.intersect1d(prior_rows, union_rows, assume_unique=True).size)
        )
        seen_rows[scan_key] = (
            union_rows
            if prior_rows is None
            else np.union1d(prior_rows, union_rows)
        )
        wave_plan = {
            "wave": wave,
            "source": source,
            "features": None if features is None else tuple(features),
            "n_features": selection.n_features,
            "rows": int(len(union_rows)),
            "repeated_rows": repeated_rows,
            "batch_count": 0,
            "aggregate_tasks": tuple(
                prepared.task.name for prepared in aggregate_consumers
            ),
            "materialize_tasks": tuple(
                prepared.task.name for prepared in materialize_consumers
            ),
            "peak_batch_bytes": 0,
            "peak_accumulator_bytes": 0,
            "peak_consumer_bytes": 0,
            "peak_buffer_bytes": 0,
            "estimated_peak_buffer_bytes": estimated_peak_buffer_bytes,
        }
        source_dtype = _source_dtype(backend, source)
        for prepared in aggregate_consumers:
            _initialize_accumulators(prepared, source_dtype)
        initial_buffered = (
            _aggregate_buffer_bytes(prepared_aggregates)
            + _consumer_buffer_bytes(prepared_materializations)
            + retained_result_bytes
        )
        peak_buffer_bytes = max(peak_buffer_bytes, initial_buffered)
        wave_plan["peak_accumulator_bytes"] = _aggregate_buffer_bytes(
            prepared_aggregates
        )
        wave_plan["peak_consumer_bytes"] = (
            _consumer_buffer_bytes(prepared_materializations)
            + retained_result_bytes
        )
        wave_plan["peak_buffer_bytes"] = initial_buffered
        if (
            memory_budget_bytes is not None
            and initial_buffered > memory_budget_bytes
        ):
            raise MemoryError(
                "task execution exceeded memory_budget_bytes before reading a "
                "matrix batch; reduce features, groups, cohort size, or retained "
                "result size"
            )
        for start in range(0, len(union_rows), batch_size):
            stop = min(start + batch_size, len(union_rows))
            rows = union_rows[start:stop]
            matrix = _read_source_batch(
                backend,
                source,
                rows,
                selection.positions,
            )
            if matrix is None:
                raise KeyError(f"matrix source {source!r} is not available")
            expected_shape = (len(rows), selection.n_features)
            if tuple(matrix.shape) != expected_shape:
                raise RuntimeError(
                    f"matrix source {source!r} returned shape {matrix.shape}, "
                    f"expected {expected_shape}"
                )
            matrix_batch_reads += 1
            matrix_size = _matrix_nbytes(matrix)
            matrix_bytes_read += matrix_size
            wave_plan["batch_count"] += 1
            wave_plan["peak_batch_bytes"] = max(
                wave_plan["peak_batch_bytes"],
                matrix_size,
            )

            for prepared in aggregate_consumers:
                member_slice, matrix_positions = _rows_within_batch(
                    prepared.row_indices,
                    rows,
                )
                if len(matrix_positions):
                    _accumulate(
                        prepared,
                        _select_matrix_rows(matrix, matrix_positions),
                        prepared.grouping.row_to_group[member_slice],
                    )
            for prepared in materialize_consumers:
                _, matrix_positions = _rows_within_batch(
                    prepared.row_indices,
                    rows,
                )
                if len(matrix_positions):
                    chunk = matrix[matrix_positions]
                    projected = (
                        matrix_size
                        + _aggregate_buffer_bytes(prepared_aggregates)
                        + _consumer_buffer_bytes(prepared_materializations)
                        + retained_result_bytes
                        + _matrix_nbytes(chunk)
                    )
                    peak_buffer_bytes = max(peak_buffer_bytes, projected)
                    wave_plan["peak_buffer_bytes"] = max(
                        wave_plan["peak_buffer_bytes"],
                        projected,
                    )
                    if (
                        memory_budget_bytes is not None
                        and projected > memory_budget_bytes
                    ):
                        raise MemoryError(
                            "task execution exceeded memory_budget_bytes while "
                            "buffering a materialization; reduce batch_size, "
                            "features, cohort size, or concurrent-task size"
                        )
                    prepared.chunks.append(chunk)
                    del chunk

            accumulator_bytes = _aggregate_buffer_bytes(prepared_aggregates)
            consumer_bytes = (
                _consumer_buffer_bytes(prepared_materializations)
                + retained_result_bytes
            )
            buffered = matrix_size + accumulator_bytes + consumer_bytes
            peak_buffer_bytes = max(peak_buffer_bytes, buffered)
            wave_plan["peak_accumulator_bytes"] = max(
                wave_plan["peak_accumulator_bytes"],
                accumulator_bytes,
            )
            wave_plan["peak_consumer_bytes"] = max(
                wave_plan["peak_consumer_bytes"],
                consumer_bytes,
            )
            wave_plan["peak_buffer_bytes"] = max(
                wave_plan["peak_buffer_bytes"],
                buffered,
            )
            if memory_budget_bytes is not None and buffered > memory_budget_bytes:
                raise MemoryError(
                    "task execution exceeded memory_budget_bytes; reduce batch_size, "
                    "features, cohort size, or use a MaterializeTask consumer"
                )
            del matrix

        for prepared in materialize_consumers:
            chunk_bytes = sum(_matrix_nbytes(chunk) for chunk in prepared.chunks)
            accumulator_bytes = _aggregate_buffer_bytes(prepared_aggregates)
            consumer_bytes = (
                _consumer_buffer_bytes(prepared_materializations)
                + retained_result_bytes
                + chunk_bytes
            )
            projected = accumulator_bytes + consumer_bytes
            peak_buffer_bytes = max(peak_buffer_bytes, projected)
            wave_plan["peak_accumulator_bytes"] = max(
                wave_plan["peak_accumulator_bytes"],
                accumulator_bytes,
            )
            wave_plan["peak_consumer_bytes"] = max(
                wave_plan["peak_consumer_bytes"],
                consumer_bytes,
            )
            wave_plan["peak_buffer_bytes"] = max(
                wave_plan["peak_buffer_bytes"],
                projected,
            )
            if memory_budget_bytes is not None and projected > memory_budget_bytes:
                raise MemoryError(
                    "final materialization would exceed memory_budget_bytes; reduce "
                    "the cohort/features or provide a streaming-sized consumer task"
                )
            materialized = _build_materialize_result(prepared, run_id)
            prepared.chunks.clear()
            output = (
                materialized
                if prepared.task.consumer is None
                else prepared.task.consumer(materialized)
            )
            if output is not materialized:
                del materialized
            built_results[prepared.task.name] = output
            retained_result_bytes += _output_nbytes(output)
            retained_buffered = (
                _aggregate_buffer_bytes(prepared_aggregates)
                + _consumer_buffer_bytes(prepared_materializations)
                + retained_result_bytes
            )
            peak_buffer_bytes = max(peak_buffer_bytes, retained_buffered)
            wave_plan["peak_consumer_bytes"] = max(
                wave_plan["peak_consumer_bytes"],
                _consumer_buffer_bytes(prepared_materializations)
                + retained_result_bytes,
            )
            wave_plan["peak_buffer_bytes"] = max(
                wave_plan["peak_buffer_bytes"],
                retained_buffered,
            )
            if (
                memory_budget_bytes is not None
                and retained_buffered > memory_budget_bytes
            ):
                raise MemoryError(
                    "retained consumer output exceeded memory_budget_bytes; return "
                    "a smaller consumer result or increase the budget"
                )
        scan_plan.append(wave_plan)

    aggregate_finalization_bytes = _aggregate_finalization_bytes(
        prepared_aggregates
    )
    finalizing_buffered = (
        _aggregate_buffer_bytes(prepared_aggregates)
        + retained_result_bytes
        + aggregate_finalization_bytes
    )
    peak_buffer_bytes = max(peak_buffer_bytes, finalizing_buffered)
    if scan_plan:
        scan_plan[-1]["peak_accumulator_bytes"] = max(
            scan_plan[-1]["peak_accumulator_bytes"],
            _aggregate_buffer_bytes(prepared_aggregates)
            + aggregate_finalization_bytes,
        )
        scan_plan[-1]["peak_buffer_bytes"] = max(
            scan_plan[-1]["peak_buffer_bytes"],
            finalizing_buffered,
        )
    if (
        memory_budget_bytes is not None
        and finalizing_buffered > memory_budget_bytes
    ):
        raise MemoryError(
            "aggregate result finalization would exceed memory_budget_bytes; "
            "reduce metrics, features, groups, or retained consumer output"
        )

    for prepared in prepared_aggregates:
        built_results[prepared.task.name] = _build_result(prepared, run_id)

    results = {task.name: built_results[task.name] for task in task_list}
    result_bytes = {
        name: _output_nbytes(result) for name, result in results.items()
    }
    task_rows = {
        prepared.task.name: int(len(np.unique(prepared.row_indices)))
        for prepared in [*prepared_aggregates, *prepared_materializations]
    }
    task_memberships = {
        prepared.task.name: int(len(prepared.row_indices))
        for prepared in [*prepared_aggregates, *prepared_materializations]
    }
    requested_rows = sum(
        task_rows[prepared.task.name]
        for prepared in prepared_aggregates
        if prepared.task.metrics
    ) + sum(
        task_rows[prepared.task.name] for prepared in prepared_materializations
    )
    reuse_ratio = requested_rows / unique_rows if unique_rows else 1.0
    report = ExecutionReport(
        run_id=run_id,
        task_count=len(task_list),
        source_scan_count=len(planned_waves),
        matrix_batch_reads=matrix_batch_reads,
        requested_rows=requested_rows,
        unique_rows=unique_rows,
        matrix_bytes_read=matrix_bytes_read,
        reuse_ratio=reuse_ratio,
        elapsed_seconds=time.perf_counter() - started,
        result_bytes=result_bytes,
        task_rows=task_rows,
        task_memberships=task_memberships,
        memory_budget_bytes=memory_budget_bytes,
        peak_buffer_bytes=peak_buffer_bytes,
        execution_waves=len(planned_waves),
        scan_plan=tuple(scan_plan),
        degradation_reasons=tuple(degradation_reasons),
    )
    return TaskRun(results, report)


aggregate_many = execute_aggregate_tasks


__all__ = [
    "AggregateTask",
    "AggregationRun",
    "ExecutionReport",
    "MaterializeTask",
    "Task",
    "TaskRun",
    "aggregate_many",
    "execute_aggregate_tasks",
    "execute_tasks",
]
