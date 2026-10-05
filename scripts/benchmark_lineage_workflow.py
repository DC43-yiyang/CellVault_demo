#!/usr/bin/env python3
"""Benchmark a multi-lineage annotation fan-out workflow.

The benchmark compares four paths over the same lineages and payload:

1. ``anndata-direct``: subset the parent AnnData and continue in memory.
2. ``anndata-saved``: subset, save each H5AD, release the parent, then reload.
3. ``cellvault-sql``: query each lineage and materialize the SQL view in memory.
4. ``cellvault-batch``: partition all lineages with one metadata scan and
   materialize them with shared matrix reads.

CellVault conversion is reported separately because it is a one-time ingestion
cost, not a per-annotation-round subset cost.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

METHODS = (
    "anndata-direct",
    "anndata-saved",
    "cellvault-sql",
    "cellvault-batch",
)
PHASES = (
    "import",
    "open",
    "select",
    "materialize",
    "save",
    "reload",
    "analysis",
    "validation",
    "release_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--cellvault-path", default="")
    parser.add_argument(
        "--output-json",
        default="benchmark_outputs/lineage_workflow/results.json",
    )
    parser.add_argument("--column", default="main_lineage")
    parser.add_argument(
        "--lineage",
        action="append",
        required=True,
        metavar="NAME=VALUE[|VALUE...]",
        help=(
            "Lineage name and one or more source labels. Repeat this option, "
            "for example: --lineage 'T/NK=T cell|NK cell' --lineage 'B=B cell'."
        ),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="lzf",
        help="Compression used for traditional intermediate H5AD files.",
    )
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--n-comps", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rebuild-cellvault", action="store_true")
    parser.add_argument("--scratch-dir", default="")
    parser.add_argument(
        "--worker-method",
        choices=(*METHODS, "prepare"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_lineages(specs: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    lineages: list[tuple[str, tuple[str, ...]]] = []
    names: set[str] = set()
    assigned_values: dict[str, str] = {}
    for spec in specs:
        name, separator, raw_values = spec.partition("=")
        name = name.strip()
        values = tuple(
            value.strip() for value in raw_values.split("|") if value.strip()
        )
        if not separator or not name or not values:
            raise ValueError(
                f"invalid lineage {spec!r}; expected NAME=VALUE[|VALUE...]"
            )
        if name in names:
            raise ValueError(f"duplicate lineage name: {name!r}")
        for value in values:
            if value in assigned_values:
                raise ValueError(
                    f"label {value!r} is assigned to both {assigned_values[value]!r} "
                    f"and {name!r}"
                )
            assigned_values[value] = name
        names.add(name)
        lineages.append((name, values))
    return lineages


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(rss) / divisor


def quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def add_elapsed(
    totals: dict[str, float],
    lineage_phases: dict[str, float],
    phase: str,
    started: float,
) -> None:
    elapsed = time.perf_counter() - started
    totals[phase] += elapsed
    lineage_phases[phase] += elapsed


def materialize_anndata_subset(source, positions, ad):
    matrix = None
    if source.X is not None:
        matrix = source.X[positions, :]
        if hasattr(matrix, "copy"):
            matrix = matrix.copy()
    return ad.AnnData(
        X=matrix,
        obs=source.obs.iloc[positions].copy(),
        var=source.var.copy(),
    )


def subset_fingerprint(adata, np, sparse) -> dict[str, Any]:
    obs_names = "\0".join(map(str, adata.obs_names)).encode("utf-8")
    var_names = "\0".join(map(str, adata.var_names)).encode("utf-8")
    matrix_hash = hashlib.sha256()
    matrix_hash.update(str(adata.shape).encode("ascii"))
    if adata.X is None:
        nonzero = 0
        matrix_sum = 0.0
    elif sparse.issparse(adata.X):
        matrix = adata.X.tocsr()
        nonzero = int(matrix.nnz)
        matrix_sum = float(matrix.sum())
        for values in (matrix.data, matrix.indices, matrix.indptr):
            contiguous = np.ascontiguousarray(values)
            matrix_hash.update(memoryview(contiguous).cast("B"))
    else:
        matrix = np.asarray(adata.X)
        nonzero = int(np.count_nonzero(matrix))
        matrix_sum = float(np.sum(matrix))
        if matrix.flags.c_contiguous:
            matrix_hash.update(memoryview(matrix).cast("B"))
        else:
            for row in matrix:
                contiguous = np.ascontiguousarray(row)
                matrix_hash.update(memoryview(contiguous).cast("B"))
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_names_sha256": hashlib.sha256(obs_names).hexdigest(),
        "var_names_sha256": hashlib.sha256(var_names).hexdigest(),
        "x_nnz": nonzero,
        "x_sum": matrix_sum,
        "x_sha256": matrix_hash.hexdigest(),
    }


def analyze_subset(adata, args: argparse.Namespace) -> dict[str, Any]:
    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError("analysis requires at least 3 cells and 2 variables")

    import scanpy as sc

    n_comps = min(args.n_comps, adata.n_obs - 1, adata.n_vars - 1)
    n_neighbors = min(args.n_neighbors, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_comps, random_state=args.seed)
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        random_state=args.seed,
    )
    sc.tl.umap(adata, random_state=args.seed)
    sc.tl.leiden(
        adata,
        resolution=args.resolution,
        flavor="igraph",
        n_iterations=2,
        random_state=args.seed,
    )
    labels = "\0".join(adata.obs["leiden"].astype(str)).encode("utf-8")
    return {
        "n_comps": n_comps,
        "n_neighbors": n_neighbors,
        "n_clusters": int(adata.obs["leiden"].nunique()),
        "leiden_sha256": hashlib.sha256(labels).hexdigest(),
    }


def prepare_cellvault(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    from cellvault import CellDB

    with CellDB.from_h5ad(
        args.input_h5ad,
        args.cellvault_path,
        overwrite=True,
    ) as cdb:
        shape = [cdb.n_obs, cdb.n_vars]
    return {
        "method": "prepare",
        "shape": shape,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb(),
    }


def new_lineage_record(values: tuple[str, ...]) -> dict[str, Any]:
    return {
        "values": list(values),
        "phase_seconds": {
            phase: 0.0 for phase in PHASES if phase not in {"import", "open"}
        },
        "subset": None,
        "analysis": None,
        "artifact_bytes": 0,
    }


def validate_column(source, method: str, column: str) -> None:
    columns = (
        source.obs_columns
        if method in {"cellvault-sql", "cellvault-batch"}
        else source.obs.columns
    )
    if column not in columns:
        raise KeyError(f"column {column!r} not found; available: {list(columns)}")


def process_in_memory_lineages(
    source,
    method: str,
    lineages: list[tuple[str, tuple[str, ...]]],
    records: dict[str, dict[str, Any]],
    totals: dict[str, float],
    args: argparse.Namespace,
    ad,
    np,
    sparse,
) -> None:
    labels = None
    if method == "anndata-direct":
        started = time.perf_counter()
        labels = source.obs[args.column].astype(str)
        totals["select"] += time.perf_counter() - started

    for name, values in lineages:
        record = records[name]
        lineage_phases = record["phase_seconds"]

        started = time.perf_counter()
        if method == "cellvault-sql":
            placeholders = ", ".join("?" for _ in values)
            predicate = (
                f"CAST({quote_identifier(args.column)} AS VARCHAR) IN ({placeholders})"
            )
            selection = source.query_obs(predicate, values)
        else:
            selection = np.flatnonzero(labels.isin(values).to_numpy())
        add_elapsed(totals, lineage_phases, "select", started)

        started = time.perf_counter()
        if method == "cellvault-sql":
            subset = selection.to_anndata(slots={"X", "obs", "var"})
        else:
            subset = materialize_anndata_subset(source, selection, ad)
        add_elapsed(totals, lineage_phases, "materialize", started)
        if subset.n_obs == 0:
            raise ValueError(f"lineage {name!r} selected zero cells")

        started = time.perf_counter()
        record["subset"] = subset_fingerprint(subset, np, sparse)
        add_elapsed(totals, lineage_phases, "validation", started)

        if args.run_analysis:
            started = time.perf_counter()
            record["analysis"] = analyze_subset(subset, args)
            add_elapsed(totals, lineage_phases, "analysis", started)

        subset = None
        selection = None
        gc.collect()


def process_cellvault_batch_lineages(
    source,
    lineages: list[tuple[str, tuple[str, ...]]],
    records: dict[str, dict[str, Any]],
    totals: dict[str, float],
    args: argparse.Namespace,
    np,
    sparse,
) -> None:
    groups = {name: values for name, values in lineages}

    started = time.perf_counter()
    views = source.partition_obs(args.column, groups)
    totals["select"] += time.perf_counter() - started
    if any(view.n_obs == 0 for view in views.values()):
        empty = next(name for name, view in views.items() if view.n_obs == 0)
        raise ValueError(f"lineage {empty!r} selected zero cells")

    started = time.perf_counter()
    subsets = source.materialize_many(views, slots={"X", "obs", "var"})
    totals["materialize"] += time.perf_counter() - started

    for name, _ in lineages:
        record = records[name]
        lineage_phases = record["phase_seconds"]
        subset = subsets[name]

        started = time.perf_counter()
        record["subset"] = subset_fingerprint(subset, np, sparse)
        add_elapsed(totals, lineage_phases, "validation", started)

        if args.run_analysis:
            started = time.perf_counter()
            record["analysis"] = analyze_subset(subset, args)
            add_elapsed(totals, lineage_phases, "analysis", started)

        subsets[name] = None
        gc.collect()


def save_lineage_subsets(
    source,
    lineages: list[tuple[str, tuple[str, ...]]],
    records: dict[str, dict[str, Any]],
    totals: dict[str, float],
    args: argparse.Namespace,
    ad,
    np,
    scratch_path: Path,
) -> dict[str, Path]:
    started = time.perf_counter()
    labels = source.obs[args.column].astype(str)
    totals["select"] += time.perf_counter() - started
    compression = None if args.compression == "none" else args.compression
    subset_paths: dict[str, Path] = {}

    for index, (name, values) in enumerate(lineages, start=1):
        record = records[name]
        lineage_phases = record["phase_seconds"]

        started = time.perf_counter()
        positions = np.flatnonzero(labels.isin(values).to_numpy())
        add_elapsed(totals, lineage_phases, "select", started)

        started = time.perf_counter()
        subset = materialize_anndata_subset(source, positions, ad)
        add_elapsed(totals, lineage_phases, "materialize", started)
        if subset.n_obs == 0:
            raise ValueError(f"lineage {name!r} selected zero cells")

        subset_path = scratch_path / f"lineage_{index:02d}.h5ad"
        started = time.perf_counter()
        subset.write_h5ad(subset_path, compression=compression)
        add_elapsed(totals, lineage_phases, "save", started)
        record["artifact_bytes"] = subset_path.stat().st_size
        subset_paths[name] = subset_path
        subset = None
        positions = None
        gc.collect()

    return subset_paths


def reload_lineage_subsets(
    lineages: list[tuple[str, tuple[str, ...]]],
    records: dict[str, dict[str, Any]],
    totals: dict[str, float],
    args: argparse.Namespace,
    ad,
    np,
    sparse,
    subset_paths: dict[str, Path],
) -> None:
    for name, _ in lineages:
        record = records[name]
        lineage_phases = record["phase_seconds"]

        started = time.perf_counter()
        subset = ad.read_h5ad(subset_paths[name])
        add_elapsed(totals, lineage_phases, "reload", started)

        started = time.perf_counter()
        record["subset"] = subset_fingerprint(subset, np, sparse)
        add_elapsed(totals, lineage_phases, "validation", started)

        if args.run_analysis:
            started = time.perf_counter()
            record["analysis"] = analyze_subset(subset, args)
            add_elapsed(totals, lineage_phases, "analysis", started)

        subset = None
        gc.collect()


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter()
    lineages = parse_lineages(args.lineage)
    totals = {phase: 0.0 for phase in PHASES}
    records = {name: new_lineage_record(values) for name, values in lineages}

    started = time.perf_counter()
    import anndata as ad
    import numpy as np
    from scipy import sparse

    CellDB: Any = None
    if args.worker_method in {"cellvault-sql", "cellvault-batch"}:
        from cellvault import CellDB as CellDBClass

        CellDB = CellDBClass
    totals["import"] = time.perf_counter() - started

    source: Any = None
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    started = time.perf_counter()
    if args.worker_method in {"anndata-direct", "anndata-saved"}:
        source = ad.read_h5ad(args.input_h5ad)
    elif args.worker_method in {"cellvault-sql", "cellvault-batch"}:
        source = CellDB.open(args.cellvault_path)
    else:
        raise ValueError(f"unknown worker method: {args.worker_method}")
    totals["open"] = time.perf_counter() - started
    source_shape = [int(source.n_obs), int(source.n_vars)]

    try:
        validate_column(source, args.worker_method, args.column)
        if args.worker_method == "anndata-saved":
            scratch_parent = Path(args.scratch_dir).expanduser().resolve()
            scratch_parent.mkdir(parents=True, exist_ok=True)
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="cellvault-lineages-",
                dir=scratch_parent,
            )
            subset_paths = save_lineage_subsets(
                source,
                lineages,
                records,
                totals,
                args,
                ad,
                np,
                Path(temporary_directory.name),
            )
            started = time.perf_counter()
            source = None
            gc.collect()
            totals["release_source"] += time.perf_counter() - started
            reload_lineage_subsets(
                lineages,
                records,
                totals,
                args,
                ad,
                np,
                sparse,
                subset_paths,
            )
        elif args.worker_method == "cellvault-batch":
            process_cellvault_batch_lineages(
                source,
                lineages,
                records,
                totals,
                args,
                np,
                sparse,
            )
        else:
            process_in_memory_lineages(
                source,
                args.worker_method,
                lineages,
                records,
                totals,
                args,
                ad,
                np,
                sparse,
            )
    finally:
        started = time.perf_counter()
        if (
            args.worker_method in {"cellvault-sql", "cellvault-batch"}
            and source is not None
        ):
            source.close()
        source = None
        gc.collect()
        totals["release_source"] += time.perf_counter() - started
        if temporary_directory is not None:
            temporary_directory.cleanup()

    fanout_seconds = sum(
        totals[phase] for phase in ("select", "materialize", "save", "reload")
    )
    data_workflow_seconds = totals["open"] + fanout_seconds
    return {
        "method": args.worker_method,
        "phase_seconds": totals,
        "fanout_seconds": fanout_seconds,
        "data_workflow_seconds": data_workflow_seconds,
        "workflow_with_analysis_seconds": data_workflow_seconds + totals["analysis"],
        "total_seconds": time.perf_counter() - total_started,
        "peak_rss_mb": peak_rss_mb(),
        "source_shape": source_shape,
        "artifact_bytes": sum(record["artifact_bytes"] for record in records.values()),
        "lineages": records,
    }


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


def worker_command(
    args: argparse.Namespace,
    method: str,
    output_path: Path,
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
        "--column",
        args.column,
        "--compression",
        args.compression,
        "--n-comps",
        str(args.n_comps),
        "--n-neighbors",
        str(args.n_neighbors),
        "--resolution",
        str(args.resolution),
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--scratch-dir",
        str(Path(args.scratch_dir).expanduser().resolve()),
        "--worker-method",
        method,
        "--worker-output",
        str(output_path),
    ]
    for lineage in args.lineage:
        command.extend(("--lineage", lineage))
    if args.run_analysis:
        command.append("--run-analysis")
    return command


def execute_worker(
    args: argparse.Namespace,
    method: str,
    output_path: Path,
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


def validate_runs(runs: list[dict[str, Any]]) -> None:
    expected_shape = runs[0]["source_shape"]
    expected_lineages = runs[0]["lineages"]
    for run in runs[1:]:
        if run["source_shape"] != expected_shape:
            raise RuntimeError(
                f"source shape mismatch for {run['method']}: "
                f"{run['source_shape']} != {expected_shape}"
            )
        if run["lineages"].keys() != expected_lineages.keys():
            raise RuntimeError(f"lineage mismatch for {run['method']}")
        for name, expected_record in expected_lineages.items():
            observed_record = run["lineages"][name]
            expected = expected_record["subset"]
            observed = observed_record["subset"]
            exact_keys = (
                "n_obs",
                "n_vars",
                "obs_names_sha256",
                "var_names_sha256",
                "x_nnz",
                "x_sha256",
            )
            if any(observed[key] != expected[key] for key in exact_keys):
                raise RuntimeError(
                    f"subset mismatch for {run['method']} / {name}: "
                    f"{observed} != {expected}"
                )
            tolerance = max(1e-6, abs(expected["x_sum"]) * 1e-12)
            if abs(observed["x_sum"] - expected["x_sum"]) > tolerance:
                raise RuntimeError(
                    f"matrix checksum mismatch for {run['method']} / {name}: "
                    f"{observed['x_sum']} != {expected['x_sum']}"
                )
            if observed_record["analysis"] != expected_record["analysis"]:
                raise RuntimeError(f"analysis mismatch for {run['method']} / {name}")


def package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "cellvault",
        "anndata",
        "scanpy",
        "duckdb",
        "zarr",
        "numpy",
        "pandas",
        "scipy",
        "igraph",
        "leidenalg",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    if args.threads < 1:
        raise ValueError("threads must be at least 1")
    lineages = parse_lineages(args.lineage)

    input_path = Path(args.input_h5ad).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input h5ad not found: {input_path}")

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.cellvault_path:
        args.cellvault_path = str(output_path.parent / f"{input_path.stem}.cvdb")
    if not args.scratch_dir:
        args.scratch_dir = str(output_path.parent)
    scratch_path = Path(args.scratch_dir).expanduser().resolve()
    scratch_path.mkdir(parents=True, exist_ok=True)
    cellvault_path = Path(args.cellvault_path).expanduser().resolve()
    raw_dir = output_path.parent / f"{output_path.stem}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    preparation = None
    cellvault_reused = False
    cellvault_methods = {"cellvault-sql", "cellvault-batch"}
    if cellvault_methods.intersection(args.methods):
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
            verification = (
                "cross-method fingerprints will verify that it matches the H5AD"
                if any(method.startswith("anndata-") for method in args.methods)
                else "no AnnData baseline was requested; pass --rebuild-cellvault "
                "unless this store is known to match the H5AD"
            )
            print(
                f"[reuse] {cellvault_path}; {verification}",
                flush=True,
            )

    runs: list[dict[str, Any]] = []
    generator = random.Random(args.seed)
    for repeat in range(1, args.repeats + 1):
        order = list(dict.fromkeys(args.methods))
        generator.shuffle(order)
        for method in order:
            raw_path = raw_dir / f"repeat_{repeat:02d}_{method}.json"
            print(f"[run {repeat}/{args.repeats}] {method}", flush=True)
            record = execute_worker(args, method, raw_path)
            record["repeat"] = repeat
            write_json(raw_path, record)
            runs.append(record)
            print(
                f"  fanout={record['fanout_seconds']:.3f}s "
                f"with_open={record['data_workflow_seconds']:.3f}s "
                f"wall={record['process_wall_seconds']:.3f}s "
                f"peak_rss={record['peak_rss_mb']:.1f}MB "
                f"saved={record['artifact_bytes'] / (1024**2):.1f}MB",
                flush=True,
            )

    validate_runs(runs)
    metric_names = (
        "fanout_seconds",
        "data_workflow_seconds",
        "workflow_with_analysis_seconds",
        "process_wall_seconds",
        "total_seconds",
        "peak_rss_mb",
        "artifact_bytes",
    )
    summary: dict[str, Any] = {}
    for method in dict.fromkeys(args.methods):
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
        summary[method] = method_summary

    derived: dict[str, Any] = {}
    if "anndata-saved" in summary and "cellvault-sql" in summary:
        traditional_fanout = summary["anndata-saved"]["fanout_seconds"]["median"]
        cellvault_fanout = summary["cellvault-sql"]["fanout_seconds"]["median"]
        traditional_with_open = summary["anndata-saved"]["data_workflow_seconds"][
            "median"
        ]
        cellvault_with_open = summary["cellvault-sql"]["data_workflow_seconds"][
            "median"
        ]
        derived["saved_vs_cellvault_fanout_speedup"] = (
            traditional_fanout / cellvault_fanout
        )
        derived["saved_vs_cellvault_including_open_speedup"] = (
            traditional_with_open / cellvault_with_open
        )
        difference = traditional_with_open - cellvault_with_open
        if preparation is not None and difference > 0:
            derived["cellvault_conversion_break_even_including_open_batches"] = (
                preparation["total_seconds"] / difference
            )
    if "cellvault-sql" in summary and "cellvault-batch" in summary:
        sql_fanout = summary["cellvault-sql"]["fanout_seconds"]["median"]
        batch_fanout = summary["cellvault-batch"]["fanout_seconds"]["median"]
        sql_with_open = summary["cellvault-sql"]["data_workflow_seconds"]["median"]
        batch_with_open = summary["cellvault-batch"]["data_workflow_seconds"]["median"]
        derived["cellvault_sql_vs_batch_fanout_speedup"] = sql_fanout / batch_fanout
        derived["cellvault_sql_vs_batch_including_open_speedup"] = (
            sql_with_open / batch_with_open
        )

    dataset_shape = runs[0]["source_shape"]
    payload = {
        "benchmark": "multi-lineage annotation fan-out",
        "input_h5ad": str(input_path),
        "cellvault_path": str(cellvault_path),
        "selection": {
            "column": args.column,
            "lineages": [
                {"name": name, "values": list(values)} for name, values in lineages
            ],
        },
        "repeats": args.repeats,
        "methods": list(dict.fromkeys(args.methods)),
        "compression": args.compression,
        "payload_slots": ["X", "obs", "var"],
        "run_analysis": bool(args.run_analysis),
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
        "summary": summary,
        "derived": derived,
        "runs": runs,
        "notes": [
            (
                "fanout_seconds assumes the parent source is already open and includes "
                "selection, materialization, intermediate H5AD save, and reload."
            ),
            "data_workflow_seconds additionally includes opening the parent source.",
            (
                "anndata-direct is the no-intermediate-file control; AnnData does not "
                "technically require saving subsets."
            ),
            "CellVault conversion is reported separately from recurring lineage rounds.",
            "All methods materialize only X, obs, and var for a common payload.",
            (
                "cellvault-batch uses CellDB.partition_obs followed by "
                "CellDB.materialize_many."
            ),
            (
                "H5AD writes return after library-level close; no device-level fsync or "
                "filesystem cache purge is forced."
            ),
            "Each method runs in a fresh process; method order is randomized per repeat.",
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
    print("\nMedian fan-out / including-open time / peak RSS")
    for method, metrics in result["summary"].items():
        print(
            f"{method:16s} "
            f"{metrics['fanout_seconds']['median']:.3f}s / "
            f"{metrics['data_workflow_seconds']['median']:.3f}s / "
            f"{metrics['peak_rss_mb']['median']:.1f}MB"
        )
    speedup = result["derived"].get("saved_vs_cellvault_fanout_speedup")
    if speedup is not None:
        print(f"saved AnnData / CellVault fan-out speedup: {speedup:.2f}x")
    print(f"results: {Path(args.output_json).expanduser().resolve()}")


if __name__ == "__main__":
    main()
