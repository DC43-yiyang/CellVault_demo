#!/usr/bin/env python3
"""Benchmark independent, hand-fused, and CellVault joint aggregation.

Each method executes the same list of group-by tasks in a fresh process:

1. ``independent`` scans the backed H5AD matrix once per task.
2. ``manual-single-scan`` scans it once and updates every task by hand.
3. ``zarr-independent`` scans the CellVault Zarr matrix once per task.
4. ``zarr-single-scan`` scans the same Zarr matrix once with a light executor.
5. ``cellvault-joint`` calls :meth:`CellDB.aggregate_many` once.

CellVault conversion is measured separately because it is a one-time ingest
cost. Worker order is randomized for every repeat, and numerical outputs are
validated from sidecar arrays in addition to recording compact fingerprints.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

METHODS = (
    "independent",
    "manual-single-scan",
    "zarr-independent",
    "zarr-single-scan",
    "cellvault-joint",
)
DEFAULT_METHODS = ("independent", "manual-single-scan", "cellvault-joint")
METRICS = ("sum", "mean", "count_nonzero")
PHASES = (
    "import",
    "open",
    "grouping",
    "matrix_read",
    "dispatch_aggregation",
    "finalize",
    "framework_overhead",
    "scan",
    "validation",
    "close",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--cellvault-path", default="")
    parser.add_argument(
        "--output-json",
        default="benchmark_outputs/aggregate_workflow/results.json",
    )
    parser.add_argument(
        "--groupby",
        action="append",
        required=True,
        metavar="NAME=COLUMN[,COLUMN...]",
        help=(
            "Aggregation task and its obs columns. Repeat for multiple tasks, "
            "for example --groupby sample_lineage=donor_id,celltype_major."
        ),
    )
    parser.add_argument(
        "--source",
        default="",
        help="Matrix source: X or layers:<name> (default: X).",
    )
    parser.add_argument(
        "--layer",
        default="",
        help="Convenience alias for --source layers:<name>.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", "--repeat", dest="repeats", type=int, default=5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rebuild-cellvault", action="store_true")
    parser.add_argument(
        "--keep-validation-arrays",
        action="store_true",
        help="Keep temporary NPZ arrays used for cross-method validation.",
    )
    parser.add_argument(
        "--worker-method",
        choices=(*METHODS, "prepare"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_groupbys(specs: list[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, spec in enumerate(specs, start=1):
        raw_name, separator, raw_columns = spec.partition("=")
        if separator:
            name = raw_name.strip()
        else:
            name = f"groupby_{index}"
            raw_columns = raw_name
        columns = tuple(
            column.strip() for column in raw_columns.split(",") if column.strip()
        )
        if not name or not columns:
            raise ValueError(
                f"invalid groupby {spec!r}; expected NAME=COLUMN[,COLUMN...]"
            )
        if len(set(columns)) != len(columns):
            raise ValueError(f"groupby {name!r} contains duplicate columns")
        if name in names:
            raise ValueError(f"duplicate task name: {name!r}")
        names.add(name)
        tasks.append({"name": name, "groupby": columns})
    return tasks


def resolve_source(args: argparse.Namespace) -> str:
    if args.layer and args.source:
        raise ValueError("use either --source or --layer, not both")
    source = f"layers:{args.layer}" if args.layer else (args.source or "X")
    if source == "X":
        return source
    prefix, separator, layer = source.partition(":")
    if prefix != "layers" or not separator or not layer:
        raise ValueError("source must be 'X' or 'layers:<name>'")
    return source


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(rss) / divisor


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    first_quartile = percentile(values, 0.25)
    third_quartile = percentile(values, 0.75)
    return {
        "median": statistics.median(values),
        "q1": first_quartile,
        "q3": third_quartile,
        "iqr": third_quartile - first_quartile,
        "min": min(values),
        "max": max(values),
    }


def matrix_nbytes(matrix: Any, sparse: Any, np: Any) -> int:
    if sparse.issparse(matrix):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return int(np.asarray(matrix).nbytes)


def dense_array(matrix: Any, sparse: Any, np: Any) -> Any:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def feature_names_sha256(names: Any) -> str:
    payload = "\0".join(map(str, names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_scalar(value: Any, pd: Any, np: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if bool(pd.isna(value)):
            return {"type": "missing"}
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {"type": "timedelta", "value": value.total_seconds()}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    return {"type": "string", "value": str(value)}


def group_token(values: list[Any]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encode_groups(
    obs: Any, columns: tuple[str, ...], pd: Any, np: Any
) -> dict[str, Any]:
    missing = [column for column in columns if column not in obs.columns]
    if missing:
        raise KeyError(f"groupby columns not found: {missing}")
    frame = obs.loc[:, list(columns)].astype(object)
    for column in columns:
        missing = frame[column].isna()
        if bool(missing.any()):
            if bool((frame.loc[~missing, column] == "<NA>").any()):
                raise ValueError(
                    f"obs column {column!r} contains both missing values and '<NA>'"
                )
            frame.loc[missing, column] = "<NA>"
    keys = pd.MultiIndex.from_frame(frame)
    codes, unique_keys = pd.factorize(keys, sort=False)
    if bool(np.any(codes < 0)):
        raise RuntimeError("failed to assign one aggregation group to every row")
    values = [tuple(key) for key in unique_keys.tolist()]
    return {
        "codes": np.asarray(codes, dtype=np.int64),
        "values": values,
        "n_cells": np.bincount(codes, minlength=len(values)).astype(np.int64),
    }


def new_accumulator(
    grouping: dict[str, Any],
    n_vars: int,
    metrics: tuple[str, ...],
    matrix_dtype: Any,
    np: Any,
) -> dict[str, Any]:
    n_groups = len(grouping["values"])
    accumulator: dict[str, Any] = {
        "grouping": grouping,
        "matrix_dtype": np.dtype(matrix_dtype),
    }
    if "sum" in metrics or "mean" in metrics:
        dtype = np.dtype(matrix_dtype)
        if dtype.kind == "b" or (dtype.kind == "i" and dtype.itemsize < 8):
            sum_dtype = np.int64
        elif dtype.kind == "u":
            sum_dtype = np.uint64
        elif dtype.kind == "i":
            sum_dtype = dtype
        elif dtype.kind == "c":
            sum_dtype = np.complex128
        else:
            sum_dtype = np.float64
        accumulator["sum"] = np.zeros((n_groups, n_vars), dtype=sum_dtype)
    if "count_nonzero" in metrics:
        accumulator["count_nonzero"] = np.zeros(
            (n_groups, n_vars), dtype=np.int64
        )
    return accumulator


def update_accumulator(
    accumulator: dict[str, Any],
    matrix: Any,
    row_codes: Any,
    sparse: Any,
    np: Any,
) -> None:
    for group_index in np.unique(row_codes):
        selected = matrix[row_codes == group_index]
        if "sum" in accumulator:
            values = np.asarray(
                selected.sum(axis=0, dtype=accumulator["sum"].dtype)
            ).ravel()
            accumulator["sum"][group_index] += values
        if "count_nonzero" in accumulator:
            if sparse.issparse(selected):
                canonical = selected.copy()
                canonical.sum_duplicates()
                canonical.eliminate_zeros()
                values = np.bincount(
                    canonical.indices, minlength=accumulator["count_nonzero"].shape[1]
                )
            else:
                values = np.count_nonzero(np.asarray(selected), axis=0)
            accumulator["count_nonzero"][group_index] += np.asarray(values).ravel()


def finalize_accumulator(
    accumulator: dict[str, Any], metrics: tuple[str, ...], np: Any
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if "sum" in metrics:
        values["sum"] = accumulator["sum"]
    if "mean" in metrics:
        counts = accumulator["grouping"]["n_cells"].astype(np.float64)
        values["mean"] = accumulator["sum"].astype(np.float64) / counts[:, None]
    if "count_nonzero" in metrics:
        values["count_nonzero"] = accumulator["count_nonzero"]
    return values


def array_fingerprint(array: Any, np: Any) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    payload = memoryview(contiguous).cast("B")
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "nnz": int(np.count_nonzero(contiguous)),
        "sum": float(np.sum(contiguous, dtype=np.float64)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def canonicalize_task_result(
    task: dict[str, Any],
    group_values: list[tuple[Any, ...]],
    n_cells: Any,
    metric_values: dict[str, Any],
    var_names: Any,
    task_index: int,
    pd: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_keys = [
        [normalize_scalar(value, pd, np) for value in row] for row in group_values
    ]
    tokens = [group_token(row) for row in normalized_keys]
    if len(tokens) != len(set(tokens)):
        raise RuntimeError(f"task {task['name']!r} produced duplicate group keys")
    order = np.argsort(np.asarray(tokens, dtype=object), kind="stable")
    ordered_keys = [normalized_keys[int(index)] for index in order]
    ordered_counts = np.asarray(n_cells, dtype=np.int64)[order]
    arrays: dict[str, Any] = {}
    metric_records: dict[str, Any] = {}
    final_bytes = int(ordered_counts.nbytes)
    for metric in task["metrics"]:
        values = np.asarray(metric_values[metric])[order]
        key = f"task_{task_index:03d}_{metric}"
        arrays[key] = values
        final_bytes += int(values.nbytes)
        metric_records[metric] = {
            **array_fingerprint(values, np),
            "array_key": key,
        }
    key_payload = json.dumps(
        ordered_keys, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    record = {
        "name": task["name"],
        "groupby": list(task["groupby"]),
        "n_groups": len(ordered_keys),
        "n_vars": len(var_names),
        "group_keys": ordered_keys,
        "group_keys_sha256": hashlib.sha256(key_payload).hexdigest(),
        "n_cells": ordered_counts.tolist(),
        "n_cells_sha256": hashlib.sha256(
            memoryview(np.ascontiguousarray(ordered_counts)).cast("B")
        ).hexdigest(),
        "var_names_sha256": feature_names_sha256(var_names),
        "metrics": metric_records,
        "final_artifact_bytes": final_bytes,
    }
    return record, arrays


def reference_worker(
    args: argparse.Namespace, method: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_started = time.perf_counter()
    phases = {phase: 0.0 for phase in PHASES}
    source_name = resolve_source(args)
    task_specs = parse_groupbys(args.groupby)
    metrics = tuple(dict.fromkeys(args.metrics))
    tasks = [{**task, "source": source_name, "metrics": metrics} for task in task_specs]

    started = time.perf_counter()
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse

    phases["import"] = time.perf_counter() - started
    started = time.perf_counter()
    source = ad.read_h5ad(args.input_h5ad, backed="r")
    phases["open"] = time.perf_counter() - started
    source_shape = [int(source.n_obs), int(source.n_vars)]
    arrays: dict[str, Any] = {}
    task_records: dict[str, Any] = {}
    matrix_bytes_returned = 0
    logical_reads = 0
    source_scans = len(tasks) if method == "independent" else 1

    try:
        if source_name == "X":
            matrix_source = source.X
        else:
            layer_name = source_name.split(":", 1)[1]
            if layer_name not in source.layers:
                raise KeyError(
                    f"layer {layer_name!r} not found; available: "
                    f"{list(source.layers.keys())}"
                )
            matrix_source = source.layers[layer_name]
        if matrix_source is None:
            raise ValueError(f"matrix source {source_name!r} is empty")
        matrix_dtype = np.dtype(matrix_source.dtype)
        var_names = source.var_names.copy()

        if method == "independent":
            for task_index, task in enumerate(tasks):
                started = time.perf_counter()
                grouping = encode_groups(source.obs, task["groupby"], pd, np)
                accumulator = new_accumulator(
                    grouping, source.n_vars, metrics, matrix_dtype, np
                )
                phases["grouping"] += time.perf_counter() - started

                started = time.perf_counter()
                for begin in range(0, source.n_obs, args.batch_size):
                    end = min(begin + args.batch_size, source.n_obs)
                    phase_started = time.perf_counter()
                    batch = matrix_source[begin:end, :]
                    phases["matrix_read"] += time.perf_counter() - phase_started
                    logical_reads += 1
                    matrix_bytes_returned += matrix_nbytes(batch, sparse, np)
                    phase_started = time.perf_counter()
                    update_accumulator(
                        accumulator,
                        batch,
                        grouping["codes"][begin:end],
                        sparse,
                        np,
                    )
                    phases["dispatch_aggregation"] += (
                        time.perf_counter() - phase_started
                    )
                phases["scan"] += time.perf_counter() - started

                started = time.perf_counter()
                metric_values = finalize_accumulator(accumulator, metrics, np)
                phases["finalize"] += time.perf_counter() - started
                started = time.perf_counter()
                record, task_arrays = canonicalize_task_result(
                    task,
                    grouping["values"],
                    grouping["n_cells"],
                    metric_values,
                    var_names,
                    task_index,
                    pd,
                    np,
                )
                phases["validation"] += time.perf_counter() - started
                task_records[task["name"]] = record
                arrays.update(task_arrays)
                del accumulator, metric_values
                gc.collect()
        elif method == "manual-single-scan":
            started = time.perf_counter()
            groupings = {
                task["name"]: encode_groups(source.obs, task["groupby"], pd, np)
                for task in tasks
            }
            accumulators = {
                task["name"]: new_accumulator(
                    groupings[task["name"]],
                    source.n_vars,
                    metrics,
                    matrix_dtype,
                    np,
                )
                for task in tasks
            }
            phases["grouping"] = time.perf_counter() - started

            started = time.perf_counter()
            for begin in range(0, source.n_obs, args.batch_size):
                end = min(begin + args.batch_size, source.n_obs)
                phase_started = time.perf_counter()
                batch = matrix_source[begin:end, :]
                phases["matrix_read"] += time.perf_counter() - phase_started
                logical_reads += 1
                matrix_bytes_returned += matrix_nbytes(batch, sparse, np)
                phase_started = time.perf_counter()
                for task in tasks:
                    grouping = groupings[task["name"]]
                    update_accumulator(
                        accumulators[task["name"]],
                        batch,
                        grouping["codes"][begin:end],
                        sparse,
                        np,
                    )
                phases["dispatch_aggregation"] += (
                    time.perf_counter() - phase_started
                )
            phases["scan"] = time.perf_counter() - started

            for task_index, task in enumerate(tasks):
                grouping = groupings[task["name"]]
                started = time.perf_counter()
                metric_values = finalize_accumulator(
                    accumulators[task["name"]], metrics, np
                )
                phases["finalize"] += time.perf_counter() - started
                started = time.perf_counter()
                record, task_arrays = canonicalize_task_result(
                    task,
                    grouping["values"],
                    grouping["n_cells"],
                    metric_values,
                    var_names,
                    task_index,
                    pd,
                    np,
                )
                phases["validation"] += time.perf_counter() - started
                task_records[task["name"]] = record
                arrays.update(task_arrays)
        else:
            raise ValueError(f"unknown reference method: {method}")
    finally:
        started = time.perf_counter()
        source.file.close()
        source = None
        gc.collect()
        phases["close"] = time.perf_counter() - started

    requested_rows = source_shape[0] * len(tasks)
    unique_rows = source_shape[0]
    aggregation_seconds = sum(
        phases[phase] for phase in ("grouping", "scan", "finalize")
    )
    final_bytes = sum(
        record["final_artifact_bytes"] for record in task_records.values()
    )
    return (
        {
            "method": method,
            "source_shape": source_shape,
            "source": source_name,
            "matrix_dtype": str(matrix_dtype),
            "integral_input": bool(np.issubdtype(matrix_dtype, np.integer)),
            "phase_seconds": phases,
            "aggregation_seconds": aggregation_seconds,
            "data_workflow_seconds": phases["open"] + aggregation_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "peak_rss_mb": peak_rss_mb(),
            "access": {
                "requested_rows": requested_rows,
                "unique_rows": unique_rows,
                "row_reuse_factor": requested_rows / unique_rows,
                "row_reuse_fraction": 1.0 - unique_rows / requested_rows,
                "source_scans": source_scans,
                "logical_matrix_batch_reads": logical_reads,
                "matrix_bytes_returned": matrix_bytes_returned,
            },
            "intermediate_artifact_bytes": 0,
            "final_artifact_bytes": final_bytes,
            "tasks": task_records,
        },
        arrays,
    )


def zarr_reference_worker(
    args: argparse.Namespace, method: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the reference accumulator directly against CellVault's Zarr backend."""
    total_started = time.perf_counter()
    phases = {phase: 0.0 for phase in PHASES}
    source_name = resolve_source(args)
    task_specs = parse_groupbys(args.groupby)
    metrics = tuple(dict.fromkeys(args.metrics))
    tasks = [{**task, "source": source_name, "metrics": metrics} for task in task_specs]

    started = time.perf_counter()
    import numpy as np
    import pandas as pd
    from scipy import sparse

    from cellvault import CellDB
    from cellvault.execution import _source_dtype

    phases["import"] = time.perf_counter() - started
    started = time.perf_counter()
    database = CellDB.open(args.cellvault_path)
    phases["open"] = time.perf_counter() - started
    backend = database._backend
    source_shape = [int(database.n_obs), int(database.n_vars)]
    arrays: dict[str, Any] = {}
    task_records: dict[str, Any] = {}
    matrix_bytes_returned = 0
    logical_reads = 0
    independent = method == "zarr-independent"
    source_scans = len(tasks) if independent else 1

    def read_batch(begin: int, end: int):
        rows = np.arange(begin, end, dtype=np.int64)
        if source_name == "X":
            return backend.read_X(row_indices=rows)
        return backend.read_layer(
            source_name.removeprefix("layers:"),
            row_indices=rows,
        )

    try:
        started = time.perf_counter()
        matrix_dtype = _source_dtype(backend, source_name)
        var_names = database.var_names.copy()
        obs = database.obs
        groupings = {
            task["name"]: encode_groups(obs, task["groupby"], pd, np)
            for task in tasks
        }
        phases["grouping"] = time.perf_counter() - started

        if independent:
            for task_index, task in enumerate(tasks):
                grouping = groupings[task["name"]]
                accumulator = new_accumulator(
                    grouping,
                    source_shape[1],
                    metrics,
                    matrix_dtype,
                    np,
                )
                started = time.perf_counter()
                for begin in range(0, source_shape[0], args.batch_size):
                    end = min(begin + args.batch_size, source_shape[0])
                    phase_started = time.perf_counter()
                    batch = read_batch(begin, end)
                    phases["matrix_read"] += time.perf_counter() - phase_started
                    logical_reads += 1
                    matrix_bytes_returned += matrix_nbytes(batch, sparse, np)
                    phase_started = time.perf_counter()
                    update_accumulator(
                        accumulator,
                        batch,
                        grouping["codes"][begin:end],
                        sparse,
                        np,
                    )
                    phases["dispatch_aggregation"] += (
                        time.perf_counter() - phase_started
                    )
                phases["scan"] += time.perf_counter() - started

                started = time.perf_counter()
                metric_values = finalize_accumulator(accumulator, metrics, np)
                phases["finalize"] += time.perf_counter() - started
                started = time.perf_counter()
                record, task_arrays = canonicalize_task_result(
                    task,
                    grouping["values"],
                    grouping["n_cells"],
                    metric_values,
                    var_names,
                    task_index,
                    pd,
                    np,
                )
                phases["validation"] += time.perf_counter() - started
                task_records[task["name"]] = record
                arrays.update(task_arrays)
                del accumulator, metric_values
                gc.collect()
        else:
            accumulators = {
                task["name"]: new_accumulator(
                    groupings[task["name"]],
                    source_shape[1],
                    metrics,
                    matrix_dtype,
                    np,
                )
                for task in tasks
            }
            started = time.perf_counter()
            for begin in range(0, source_shape[0], args.batch_size):
                end = min(begin + args.batch_size, source_shape[0])
                phase_started = time.perf_counter()
                batch = read_batch(begin, end)
                phases["matrix_read"] += time.perf_counter() - phase_started
                logical_reads += 1
                matrix_bytes_returned += matrix_nbytes(batch, sparse, np)
                phase_started = time.perf_counter()
                for task in tasks:
                    grouping = groupings[task["name"]]
                    update_accumulator(
                        accumulators[task["name"]],
                        batch,
                        grouping["codes"][begin:end],
                        sparse,
                        np,
                    )
                phases["dispatch_aggregation"] += (
                    time.perf_counter() - phase_started
                )
            phases["scan"] = time.perf_counter() - started

            for task_index, task in enumerate(tasks):
                grouping = groupings[task["name"]]
                started = time.perf_counter()
                metric_values = finalize_accumulator(
                    accumulators[task["name"]], metrics, np
                )
                phases["finalize"] += time.perf_counter() - started
                started = time.perf_counter()
                record, task_arrays = canonicalize_task_result(
                    task,
                    grouping["values"],
                    grouping["n_cells"],
                    metric_values,
                    var_names,
                    task_index,
                    pd,
                    np,
                )
                phases["validation"] += time.perf_counter() - started
                task_records[task["name"]] = record
                arrays.update(task_arrays)
    finally:
        started = time.perf_counter()
        database.close()
        database = None
        gc.collect()
        phases["close"] = time.perf_counter() - started

    requested_rows = source_shape[0] * len(tasks)
    unique_rows = source_shape[0]
    aggregation_seconds = sum(
        phases[phase] for phase in ("grouping", "scan", "finalize")
    )
    final_bytes = sum(
        record["final_artifact_bytes"] for record in task_records.values()
    )
    return (
        {
            "method": method,
            "source_shape": source_shape,
            "source": source_name,
            "matrix_dtype": str(matrix_dtype),
            "integral_input": bool(np.issubdtype(matrix_dtype, np.integer)),
            "phase_seconds": phases,
            "aggregation_seconds": aggregation_seconds,
            "data_workflow_seconds": phases["open"] + aggregation_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "peak_rss_mb": peak_rss_mb(),
            "access": {
                "requested_rows": requested_rows,
                "unique_rows": unique_rows,
                "row_reuse_factor": requested_rows / unique_rows,
                "row_reuse_fraction": 1.0 - unique_rows / requested_rows,
                "source_scans": source_scans,
                "logical_matrix_batch_reads": logical_reads,
                "matrix_bytes_returned": matrix_bytes_returned,
            },
            "intermediate_artifact_bytes": 0,
            "final_artifact_bytes": final_bytes,
            "tasks": task_records,
        },
        arrays,
    )


def report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        return dict(report.to_dict())
    if dataclasses.is_dataclass(report) and not isinstance(report, type):
        return dataclasses.asdict(report)
    if hasattr(report, "__dict__"):
        return dict(vars(report))
    raise TypeError(f"unsupported execution report type: {type(report).__name__}")


def report_number(report: dict[str, Any], names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if name in report and report[name] is not None:
            return report[name]
    return default


def cellvault_worker(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    total_started = time.perf_counter()
    phases = {phase: 0.0 for phase in PHASES}
    source_name = resolve_source(args)
    task_specs = parse_groupbys(args.groupby)
    metrics = tuple(dict.fromkeys(args.metrics))

    started = time.perf_counter()
    import numpy as np
    import pandas as pd
    from scipy import sparse

    from cellvault import AggregateTask, CellDB
    from cellvault import execution as execution_module

    phases["import"] = time.perf_counter() - started
    started = time.perf_counter()
    database = CellDB.open(args.cellvault_path)
    phases["open"] = time.perf_counter() - started
    source_shape = [int(database.n_obs), int(database.n_vars)]
    task_records: dict[str, Any] = {}
    arrays: dict[str, Any] = {}

    original_build_grouping = execution_module._build_grouping
    original_read_target_obs = execution_module._read_target_obs
    original_accumulate = execution_module._accumulate
    original_build_result = execution_module._build_result
    backend = database._backend
    read_method_name = "read_X" if source_name == "X" else "read_layer"
    original_read = getattr(backend, read_method_name)

    def timed_build_grouping(*values, **options):
        started = time.perf_counter()
        try:
            return original_build_grouping(*values, **options)
        finally:
            phases["grouping"] += time.perf_counter() - started

    def timed_read_target_obs(*values, **options):
        started = time.perf_counter()
        try:
            return original_read_target_obs(*values, **options)
        finally:
            phases["grouping"] += time.perf_counter() - started

    def timed_read(*values, **options):
        started = time.perf_counter()
        try:
            return original_read(*values, **options)
        finally:
            phases["matrix_read"] += time.perf_counter() - started

    def timed_accumulate(*values, **options):
        started = time.perf_counter()
        try:
            return original_accumulate(*values, **options)
        finally:
            phases["dispatch_aggregation"] += time.perf_counter() - started

    def timed_build_result(*values, **options):
        started = time.perf_counter()
        try:
            return original_build_result(*values, **options)
        finally:
            phases["finalize"] += time.perf_counter() - started

    execution_module._build_grouping = timed_build_grouping
    execution_module._read_target_obs = timed_read_target_obs
    execution_module._accumulate = timed_accumulate
    execution_module._build_result = timed_build_result
    setattr(backend, read_method_name, timed_read)

    try:
        tasks = [
            AggregateTask(
                name=task["name"],
                groupby=task["groupby"],
                source=source_name,
                metrics=metrics,
            )
            for task in task_specs
        ]
        started = time.perf_counter()
        run = database.aggregate_many(tasks, batch_size=args.batch_size)
        phases["scan"] = time.perf_counter() - started
        measured = sum(
            phases[phase]
            for phase in (
                "grouping",
                "matrix_read",
                "dispatch_aggregation",
                "finalize",
            )
        )
        phases["framework_overhead"] = max(0.0, phases["scan"] - measured)
        execution_report = report_dict(run.report)

        started = time.perf_counter()
        for task_index, task in enumerate(task_specs):
            result = run.results[task["name"]]
            group_values = list(
                result.obs.loc[:, list(task["groupby"])].itertuples(
                    index=False, name=None
                )
            )
            metric_values = {
                metric: dense_array(result.layers[metric], sparse, np)
                for metric in metrics
            }
            benchmark_task = {**task, "source": source_name, "metrics": metrics}
            record, task_arrays = canonicalize_task_result(
                benchmark_task,
                group_values,
                result.obs["n_cells"].to_numpy(),
                metric_values,
                result.var_names,
                task_index,
                pd,
                np,
            )
            task_records[task["name"]] = record
            arrays.update(task_arrays)
        phases["validation"] = time.perf_counter() - started
    finally:
        execution_module._build_grouping = original_build_grouping
        execution_module._read_target_obs = original_read_target_obs
        execution_module._accumulate = original_accumulate
        execution_module._build_result = original_build_result
        setattr(backend, read_method_name, original_read)
        started = time.perf_counter()
        database.close()
        database = None
        gc.collect()
        phases["close"] = time.perf_counter() - started

    requested_rows_default = source_shape[0] * len(task_specs)
    unique_rows_default = source_shape[0]
    requested_rows = int(
        report_number(execution_report, ("requested_rows",), requested_rows_default)
    )
    unique_rows = int(
        report_number(execution_report, ("unique_rows",), unique_rows_default)
    )
    final_bytes = sum(
        record["final_artifact_bytes"] for record in task_records.values()
    )
    matrix_dtype = str(report_number(execution_report, ("matrix_dtype",), "unknown"))
    aggregation_seconds = phases["scan"]
    return (
        {
            "method": "cellvault-joint",
            "source_shape": source_shape,
            "source": source_name,
            "matrix_dtype": matrix_dtype,
            "integral_input": bool(
                report_number(execution_report, ("integral_input",), False)
            ),
            "phase_seconds": phases,
            "aggregation_seconds": aggregation_seconds,
            "data_workflow_seconds": phases["open"] + aggregation_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "peak_rss_mb": peak_rss_mb(),
            "access": {
                "requested_rows": requested_rows,
                "unique_rows": unique_rows,
                "row_reuse_factor": requested_rows / unique_rows,
                "row_reuse_fraction": 1.0 - unique_rows / requested_rows,
                "source_scans": int(
                    report_number(
                        execution_report,
                        ("source_scans", "source_scan_count", "matrix_scans"),
                        1,
                    )
                ),
                "logical_matrix_batch_reads": int(
                    report_number(
                        execution_report,
                        (
                            "logical_matrix_batch_reads",
                            "matrix_batch_reads",
                            "batch_count",
                            "batches",
                        ),
                        math.ceil(source_shape[0] / args.batch_size),
                    )
                ),
                "matrix_bytes_returned": int(
                    report_number(
                        execution_report,
                        (
                            "matrix_bytes_returned",
                            "matrix_bytes_read",
                            "bytes_read",
                        ),
                        0,
                    )
                ),
            },
            "intermediate_artifact_bytes": 0,
            "final_artifact_bytes": final_bytes,
            "execution_report": execution_report,
            "tasks": task_records,
        },
        arrays,
    )


def prepare_cellvault(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    from cellvault import CellDB

    with CellDB.from_h5ad(
        args.input_h5ad,
        args.cellvault_path,
        overwrite=True,
    ) as database:
        shape = [int(database.n_obs), int(database.n_vars)]
    return {
        "method": "prepare",
        "shape": shape,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb(),
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.worker_method in {"independent", "manual-single-scan"}:
        record, arrays = reference_worker(args, args.worker_method)
    elif args.worker_method in {"zarr-independent", "zarr-single-scan"}:
        record, arrays = zarr_reference_worker(args, args.worker_method)
    elif args.worker_method == "cellvault-joint":
        record, arrays = cellvault_worker(args)
    else:
        raise ValueError(f"unknown worker method: {args.worker_method}")
    array_path = Path(args.worker_output).with_suffix(".npz")
    import numpy as np

    np.savez(array_path, **arrays)
    record["validation_array_path"] = str(array_path.resolve())
    return record


def worker_command(
    args: argparse.Namespace, method: str, output_path: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input-h5ad",
        str(Path(args.input_h5ad).expanduser().resolve()),
        "--cellvault-path",
        str(Path(args.cellvault_path).expanduser().resolve()),
        "--output-json",
        str(Path(args.output_json).expanduser().resolve()),
        "--source",
        resolve_source(args),
        "--batch-size",
        str(args.batch_size),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--worker-method",
        method,
        "--worker-output",
        str(output_path),
        "--metrics",
        *args.metrics,
    ]
    for groupby in args.groupby:
        command.extend(("--groupby", groupby))
    return command


def execute_worker(
    args: argparse.Namespace, method: str, output_path: Path
) -> dict[str, Any]:
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        environment[variable] = str(args.threads)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH", "")))
    )
    started = time.perf_counter()
    completed = subprocess.run(
        worker_command(args, method, output_path),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    process_wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{method} worker failed\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    record = json.loads(output_path.read_text(encoding="utf-8"))
    record["process_wall_seconds"] = process_wall_seconds
    return record


def validate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    expected = runs[0]
    rtol = 1e-6
    atol = 1e-8
    comparisons = 0
    with np.load(expected["validation_array_path"]) as expected_arrays:
        for run in runs[1:]:
            if run["source_shape"] != expected["source_shape"]:
                raise RuntimeError(
                    f"source shape mismatch for {run['method']}: "
                    f"{run['source_shape']} != {expected['source_shape']}"
                )
            if run["source"] != expected["source"]:
                raise RuntimeError(f"source mismatch for {run['method']}")
            if run["tasks"].keys() != expected["tasks"].keys():
                raise RuntimeError(f"task mismatch for {run['method']}")
            with np.load(run["validation_array_path"]) as observed_arrays:
                for name, expected_task in expected["tasks"].items():
                    observed_task = run["tasks"][name]
                    for key in (
                        "groupby",
                        "n_groups",
                        "n_vars",
                        "group_keys",
                        "n_cells",
                        "var_names_sha256",
                    ):
                        if observed_task[key] != expected_task[key]:
                            raise RuntimeError(
                                f"result mismatch for {run['method']} / {name} / {key}"
                            )
                    if (
                        observed_task["metrics"].keys()
                        != expected_task["metrics"].keys()
                    ):
                        raise RuntimeError(
                            f"metric mismatch for {run['method']} / {name}"
                        )
                    for metric, expected_metric in expected_task["metrics"].items():
                        observed_metric = observed_task["metrics"][metric]
                        expected_array = expected_arrays[expected_metric["array_key"]]
                        observed_array = observed_arrays[observed_metric["array_key"]]
                        integral_sum = metric == "sum" and (
                            expected_array.dtype.kind in "biu"
                            and observed_array.dtype.kind in "biu"
                        )
                        if metric == "count_nonzero" or integral_sum:
                            np.testing.assert_array_equal(
                                observed_array, expected_array
                            )
                        else:
                            np.testing.assert_allclose(
                                observed_array,
                                expected_array,
                                rtol=rtol,
                                atol=atol,
                                equal_nan=True,
                            )
                        comparisons += 1
    return {
        "status": "passed",
        "array_comparisons": comparisons,
        "rtol": rtol,
        "atol": atol,
        "integer_sum_and_count_nonzero": "exact",
    }


def package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "cellvault",
        "anndata",
        "duckdb",
        "zarr",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.threads < 1:
        raise ValueError("threads must be at least 1")
    task_specs = parse_groupbys(args.groupby)
    source_name = resolve_source(args)
    args.metrics = list(dict.fromkeys(args.metrics))

    input_path = Path(args.input_h5ad).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input h5ad not found: {input_path}")
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.cellvault_path:
        args.cellvault_path = str(output_path.parent / f"{input_path.stem}.cvdb")
    cellvault_path = Path(args.cellvault_path).expanduser().resolve()
    raw_dir = output_path.parent / f"{output_path.stem}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    preparation = None
    cellvault_reused = False
    if any(
        method in {"zarr-independent", "zarr-single-scan", "cellvault-joint"}
        for method in args.methods
    ):
        needs_preparation = (
            args.rebuild_cellvault or not (cellvault_path / "obs.duckdb").is_file()
        )
        if needs_preparation:
            preparation_path = raw_dir / "cellvault_preparation.json"
            print(f"[prepare] {input_path.name} -> {cellvault_path}", flush=True)
            preparation = execute_worker(args, "prepare", preparation_path)
            write_json(preparation_path, preparation)
        else:
            cellvault_reused = True
            print(
                f"[reuse] {cellvault_path}; output validation will verify its contents",
                flush=True,
            )

    runs: list[dict[str, Any]] = []
    generator = random.Random(args.seed)
    methods = list(dict.fromkeys(args.methods))
    for repeat in range(1, args.repeats + 1):
        order = methods.copy()
        generator.shuffle(order)
        for method in order:
            raw_path = raw_dir / f"repeat_{repeat:02d}_{method}.json"
            print(f"[run {repeat}/{args.repeats}] {method}", flush=True)
            record = execute_worker(args, method, raw_path)
            record["repeat"] = repeat
            write_json(raw_path, record)
            runs.append(record)
            access = record["access"]
            print(
                f"  aggregate={record['aggregation_seconds']:.3f}s "
                f"with_open={record['data_workflow_seconds']:.3f}s "
                f"wall={record['process_wall_seconds']:.3f}s "
                f"peak_rss={record['peak_rss_mb']:.1f}MB "
                f"reads={access['logical_matrix_batch_reads']}",
                flush=True,
            )

    validation = validate_runs(runs)
    if not args.keep_validation_arrays:
        for run in runs:
            Path(run["validation_array_path"]).unlink(missing_ok=True)
    metric_names = (
        "aggregation_seconds",
        "data_workflow_seconds",
        "process_wall_seconds",
        "total_seconds",
        "peak_rss_mb",
        "intermediate_artifact_bytes",
        "final_artifact_bytes",
    )
    access_names = (
        "requested_rows",
        "unique_rows",
        "row_reuse_factor",
        "row_reuse_fraction",
        "source_scans",
        "logical_matrix_batch_reads",
        "matrix_bytes_returned",
    )
    summary: dict[str, Any] = {}
    for method in methods:
        method_runs = [run for run in runs if run["method"] == method]
        method_summary: dict[str, Any] = {
            metric: summarize([float(run[metric]) for run in method_runs])
            for metric in metric_names
        }
        method_summary["phase_seconds"] = {
            phase: summarize(
                [float(run["phase_seconds"][phase]) for run in method_runs]
            )
            for phase in PHASES
        }
        method_summary["access"] = {
            name: summarize([float(run["access"][name]) for run in method_runs])
            for name in access_names
        }
        summary[method] = method_summary

    derived: dict[str, Any] = {}
    if "independent" in summary and "cellvault-joint" in summary:
        independent = summary["independent"]["aggregation_seconds"]["median"]
        joint = summary["cellvault-joint"]["aggregation_seconds"]["median"]
        derived["independent_vs_cellvault_joint_speedup"] = independent / joint
        independent_reads = summary["independent"]["access"][
            "logical_matrix_batch_reads"
        ]["median"]
        joint_reads = summary["cellvault-joint"]["access"][
            "logical_matrix_batch_reads"
        ]["median"]
        derived["independent_vs_cellvault_joint_logical_read_reduction"] = (
            independent_reads / joint_reads
        )
    if "manual-single-scan" in summary and "cellvault-joint" in summary:
        manual = summary["manual-single-scan"]["aggregation_seconds"]["median"]
        joint = summary["cellvault-joint"]["aggregation_seconds"]["median"]
        derived["cellvault_joint_vs_manual_time_ratio"] = joint / manual
    if "independent" in summary and "manual-single-scan" in summary:
        independent = summary["independent"]["aggregation_seconds"]["median"]
        shared = summary["manual-single-scan"]["aggregation_seconds"]["median"]
        derived["h5ad_shared_scan_speedup"] = independent / shared
    if "zarr-independent" in summary and "zarr-single-scan" in summary:
        independent = summary["zarr-independent"]["aggregation_seconds"]["median"]
        shared = summary["zarr-single-scan"]["aggregation_seconds"]["median"]
        derived["zarr_shared_scan_speedup"] = independent / shared
    if "manual-single-scan" in summary and "zarr-single-scan" in summary:
        h5ad = summary["manual-single-scan"]["aggregation_seconds"]["median"]
        zarr = summary["zarr-single-scan"]["aggregation_seconds"]["median"]
        derived["zarr_vs_h5ad_shared_time_ratio"] = zarr / h5ad
    if "zarr-single-scan" in summary and "cellvault-joint" in summary:
        zarr = summary["zarr-single-scan"]["aggregation_seconds"]["median"]
        joint = summary["cellvault-joint"]["aggregation_seconds"]["median"]
        derived["cellvault_vs_zarr_shared_time_ratio"] = joint / zarr
        derived["cellvault_vs_zarr_shared_seconds"] = joint - zarr

    dataset_shape = runs[0]["source_shape"]
    public_runs = []
    for run in runs:
        public_run = dict(run)
        public_run.pop("validation_array_path", None)
        public_runs.append(public_run)
    payload = {
        "benchmark": "multi-task joint aggregation",
        "input_h5ad": str(input_path),
        "cellvault_path": str(cellvault_path),
        "source": source_name,
        "tasks": [
            {
                "name": task["name"],
                "groupby": list(task["groupby"]),
                "metrics": list(args.metrics),
            }
            for task in task_specs
        ],
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "methods": methods,
        "threads": args.threads,
        "seed": args.seed,
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpus": os.cpu_count(),
            "packages": package_versions(),
        },
        "dataset": {
            "n_obs": dataset_shape[0],
            "n_vars": dataset_shape[1],
            "input_h5ad_bytes": input_path.stat().st_size,
            "cellvault_bytes": sum(
                path.stat().st_size
                for path in cellvault_path.rglob("*")
                if path.is_file()
            )
            if cellvault_path.exists()
            else None,
        },
        "preparation": preparation,
        "cellvault_reused": cellvault_reused,
        "validation": validation,
        "summary": summary,
        "derived": derived,
        "runs": public_runs,
        "notes": [
            (
                "Every timed method runs in a fresh process; order is randomized "
                "per repeat."
            ),
            "Independent and manual baselines stream the H5AD in backed mode.",
            (
                "Zarr reference methods use the CellVault storage backend directly "
                "with the same lightweight accumulator as the H5AD baselines."
            ),
            (
                "matrix_read and dispatch_aggregation are instrumented separately; "
                "CellVault framework_overhead is the remaining public-API run time."
            ),
            "CellVault conversion is reported separately from recurring aggregation.",
            (
                "requested_rows counts logical task membership; unique_rows counts "
                "source rows."
            ),
            (
                "logical matrix reads are API batch reads, not physical chunks or "
                "decompressions."
            ),
            (
                "Benchmark JSON/NPZ validation files are not counted as workflow "
                "artifacts."
            ),
            (
                "Validation NPZ files are deleted after comparison unless "
                "--keep-validation-arrays is set."
            ),
            "Floating aggregates use rtol=1e-6 and atol=1e-8; integer sums are exact.",
        ],
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    if args.worker_method:
        if not args.worker_output:
            raise ValueError("worker-output is required in worker mode")
        result = (
            prepare_cellvault(args)
            if args.worker_method == "prepare"
            else run_worker(args)
        )
        write_json(Path(args.worker_output), result)
        return

    result = run_parent(args)
    print("\nMedian aggregation / including-open time / peak RSS")
    for method, metrics in result["summary"].items():
        print(
            f"{method:20s} "
            f"{metrics['aggregation_seconds']['median']:.3f}s / "
            f"{metrics['data_workflow_seconds']['median']:.3f}s / "
            f"{metrics['peak_rss_mb']['median']:.1f}MB"
        )
    speedup = result["derived"].get("independent_vs_cellvault_joint_speedup")
    if speedup is not None:
        print(f"independent / CellVault joint speedup: {speedup:.2f}x")
    print(f"results: {Path(args.output_json).expanduser().resolve()}")


if __name__ == "__main__":
    main()
