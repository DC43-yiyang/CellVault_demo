#!/usr/bin/env python3
"""Benchmark SQL-driven subset extraction against AnnData access modes.

Each timed run executes in a fresh subprocess so peak RSS and import/open costs
do not leak between methods. CellVault conversion is treated as preparation and
is excluded from query benchmarks.
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
import time
from pathlib import Path
from typing import Any

METHODS = ("anndata-memory", "anndata-backed", "cellvault")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark subset extraction on a large single-cell dataset"
    )
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--cellvault-path", default="")
    parser.add_argument(
        "--output-json",
        default="benchmark_outputs/subset_357k/results.json",
    )
    parser.add_argument("--column", default="leiden")
    parser.add_argument("--value", default="2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--n-comps", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rebuild-cellvault", action="store_true")
    parser.add_argument(
        "--worker-method", choices=(*METHODS, "prepare"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(rss) / divisor


def quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


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


def subset_fingerprint(adata, np, sparse) -> dict[str, Any]:
    names = "\0".join(map(str, adata.obs_names)).encode("utf-8")
    if adata.X is None:
        nnz = 0
        matrix_sum = 0.0
    elif sparse.issparse(adata.X):
        nnz = int(adata.X.nnz)
        matrix_sum = float(adata.X.sum())
    else:
        nnz = int(np.count_nonzero(adata.X))
        matrix_sum = float(np.sum(adata.X))
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_names_sha256": hashlib.sha256(names).hexdigest(),
        "x_nnz": nnz,
        "x_sum": matrix_sum,
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


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    phase_started = time.perf_counter()
    import anndata as ad
    import numpy as np
    from scipy import sparse

    CellDB: Any = None
    if args.worker_method == "cellvault":
        from cellvault import CellDB as CellDBClass

        CellDB = CellDBClass
    phase_seconds["import"] = time.perf_counter() - phase_started

    source: Any = None
    phase_started = time.perf_counter()
    if args.worker_method == "anndata-memory":
        source = ad.read_h5ad(args.input_h5ad)
    elif args.worker_method == "anndata-backed":
        source = ad.read_h5ad(args.input_h5ad, backed="r")
    elif args.worker_method == "cellvault":
        source = CellDB.open(args.cellvault_path)
    else:
        raise ValueError(f"unknown worker method: {args.worker_method}")
    phase_seconds["open"] = time.perf_counter() - phase_started
    source_shape = [int(source.n_obs), int(source.n_vars)]

    try:
        phase_started = time.perf_counter()
        if args.worker_method == "cellvault":
            predicate = f"CAST({quote_identifier(args.column)} AS VARCHAR) = ?"
            selection: Any = source.query_obs(predicate, [str(args.value)])
        else:
            if args.column not in source.obs.columns:
                raise KeyError(
                    f"column {args.column!r} not found; available: {list(source.obs.columns)}"
                )
            mask = source.obs[args.column].astype(str).to_numpy() == str(args.value)
            selection = np.flatnonzero(mask)
        phase_seconds["select"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        if args.worker_method == "cellvault":
            subset = selection.to_anndata(slots={"X", "obs", "var"})
        else:
            subset = materialize_anndata_subset(source, selection, ad)
        phase_seconds["materialize"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        if args.worker_method == "cellvault":
            source.close()
        elif args.worker_method == "anndata-backed":
            source.file.close()
        source = None
        selection = None
        gc.collect()
        phase_seconds["release_source"] = time.perf_counter() - phase_started

        analysis = None
        phase_seconds["analysis"] = 0.0
        if args.run_analysis:
            phase_started = time.perf_counter()
            analysis = analyze_subset(subset, args)
            phase_seconds["analysis"] = time.perf_counter() - phase_started

        fingerprint = subset_fingerprint(subset, np, sparse)
    finally:
        if args.worker_method == "cellvault" and source is not None:
            source.close()
        elif args.worker_method == "anndata-backed" and source is not None:
            source.file.close()

    return {
        "method": args.worker_method,
        "phase_seconds": phase_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "peak_rss_mb": peak_rss_mb(),
        "source_shape": source_shape,
        "subset": fingerprint,
        "analysis": analysis,
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
        "--column",
        args.column,
        "--value",
        str(args.value),
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
        "--worker-method",
        method,
        "--worker-output",
        str(output_path),
    ]
    if args.run_analysis:
        command.append("--run-analysis")
    return command


def execute_worker(
    args: argparse.Namespace,
    method: str,
    output_path: Path,
) -> dict[str, Any]:
    environment = os.environ.copy()
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
    for variable in thread_variables:
        environment[variable] = str(args.threads)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source_root, environment.get("PYTHONPATH", "")])
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
            f"{method} worker failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    record = json.loads(output_path.read_text(encoding="utf-8"))
    record["process_wall_seconds"] = process_wall_seconds
    return record


def validate_fingerprints(runs: list[dict[str, Any]]) -> None:
    expected = runs[0]["subset"]
    for run in runs[1:]:
        observed = run["subset"]
        exact_keys = ("n_obs", "n_vars", "obs_names_sha256", "x_nnz")
        if any(observed[key] != expected[key] for key in exact_keys):
            raise RuntimeError(
                f"subset mismatch for {run['method']}: {observed} != {expected}"
            )
        tolerance = max(1e-6, abs(expected["x_sum"]) * 1e-12)
        if abs(observed["x_sum"] - expected["x_sum"]) > tolerance:
            raise RuntimeError(
                f"matrix checksum mismatch for {run['method']}: "
                f"{observed['x_sum']} != {expected['x_sum']}"
            )
        if run["analysis"] != runs[0]["analysis"]:
            raise RuntimeError(
                f"analysis result mismatch for {run['method']}: "
                f"{run['analysis']} != {runs[0]['analysis']}"
            )


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

    input_path = Path(args.input_h5ad).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input h5ad not found: {input_path}")

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.cellvault_path:
        args.cellvault_path = str(output_path.parent / f"{input_path.stem}.cvdb")
    cvdb_path = Path(args.cellvault_path).expanduser().resolve()
    raw_dir = output_path.parent / f"{output_path.stem}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    preparation = None
    if "cellvault" in args.methods:
        needs_preparation = (
            args.rebuild_cellvault or not (cvdb_path / "obs.duckdb").is_file()
        )
        if needs_preparation:
            preparation_path = raw_dir / "cellvault_preparation.json"
            print(f"[prepare] {input_path.name} -> {cvdb_path}", flush=True)
            preparation = execute_worker(args, "prepare", preparation_path)
            write_json(preparation_path, preparation)

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
                f"  cells={record['subset']['n_obs']:,} "
                f"wall={record['process_wall_seconds']:.3f}s "
                f"peak_rss={record['peak_rss_mb']:.1f}MB",
                flush=True,
            )

    validate_fingerprints(runs)
    metric_names = (
        "process_wall_seconds",
        "total_seconds",
        "peak_rss_mb",
    )
    summary: dict[str, Any] = {}
    for method in dict.fromkeys(args.methods):
        method_runs = [run for run in runs if run["method"] == method]
        method_summary: dict[str, Any] = {
            metric: summarize([float(run[metric]) for run in method_runs])
            for metric in metric_names
        }
        phase_names = method_runs[0]["phase_seconds"].keys()
        method_summary["phase_seconds"] = {
            phase: summarize(
                [float(run["phase_seconds"][phase]) for run in method_runs]
            )
            for phase in phase_names
        }
        summary[method] = method_summary

    dataset_shape = runs[0]["source_shape"]
    payload = {
        "benchmark": "single-cell subset extraction",
        "input_h5ad": str(input_path),
        "cellvault_path": str(cvdb_path),
        "selection": {"column": args.column, "value": str(args.value)},
        "repeats": args.repeats,
        "methods": list(dict.fromkeys(args.methods)),
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
                path.stat().st_size for path in cvdb_path.rglob("*") if path.is_file()
            )
            if cvdb_path.exists()
            else None,
        },
        "subset": runs[0]["subset"],
        "preparation": preparation,
        "summary": summary,
        "runs": runs,
        "notes": [
            "CellVault conversion is excluded from timed query runs.",
            "Method order is randomized per repeat; filesystem cache is not cleared.",
            "Each method materializes only X, obs, and var before optional analysis.",
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
    print("\nMedian process wall time / peak RSS")
    for method, metrics in result["summary"].items():
        print(
            f"{method:16s} "
            f"{metrics['process_wall_seconds']['median']:.3f}s / "
            f"{metrics['peak_rss_mb']['median']:.1f}MB"
        )
    print(f"results: {Path(args.output_json).expanduser().resolve()}")


if __name__ == "__main__":
    main()
