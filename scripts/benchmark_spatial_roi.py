#!/usr/bin/env python3
"""Validate overlapping ROI aggregation and mixed execution on public IMC data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from cellvault import AggregateTask, CellDB, MaterializeTask


DATASET_URL = "https://exampledata.scverse.org/squidpy/imc.h5ad"
DATASET_SHA256 = "950c44c785ea86c4262140b0229e0b4f77110a765c3b6874cdb5e0e52973c6fe"
ROI_DEFINITIONS = {
    "left_wide": {"x_max": 0.65},
    "right_wide": {"x_min": 0.35},
    "central_rectangle": {
        "x_min": 0.20,
        "x_max": 0.80,
        "y_min": 0.20,
        "y_max": 0.80,
    },
    "upper_band": {"y_min": 0.40},
}
LOCAL_ROIS = ("left_wide", "central_rectangle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        default="benchmark_data/squidpy_imc.h5ad",
    )
    parser.add_argument(
        "--cellvault-path",
        default="benchmark_outputs/imc_spatial/imc.cvdb",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_results/squidpy_imc_spatial_roi.json",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-budget-mib", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--rebuild-cellvault", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(DATASET_URL, partial)
        partial.replace(path)
    observed = sha256_file(path)
    if observed != DATASET_SHA256:
        raise RuntimeError(
            f"dataset SHA256 mismatch: expected {DATASET_SHA256}, got {observed}"
        )


def add_normalized_coordinates(adata: ad.AnnData) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise KeyError("public IMC input is missing obsm['spatial']")
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coordinates.shape != (adata.n_obs, 2):
        raise ValueError("obsm['spatial'] must have shape (n_obs, 2)")
    lower = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - lower
    if np.any(span <= 0):
        raise ValueError("spatial coordinates must span both axes")
    normalized = (coordinates - lower) / span
    adata.obs["spatial_x_normalized"] = normalized[:, 0]
    adata.obs["spatial_y_normalized"] = normalized[:, 1]
    return normalized


def roi_mask(coordinates: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    mask = np.ones(len(coordinates), dtype=bool)
    if "x_min" in bounds:
        mask &= coordinates[:, 0] >= bounds["x_min"]
    if "x_max" in bounds:
        mask &= coordinates[:, 0] <= bounds["x_max"]
    if "y_min" in bounds:
        mask &= coordinates[:, 1] >= bounds["y_min"]
    if "y_max" in bounds:
        mask &= coordinates[:, 1] <= bounds["y_max"]
    return mask


def build_membership(adata: ad.AnnData, coordinates: np.ndarray) -> pd.DataFrame:
    frames = []
    for roi, bounds in ROI_DEFINITIONS.items():
        selected = roi_mask(coordinates, bounds)
        if not selected.any():
            raise RuntimeError(f"controlled ROI {roi!r} is empty")
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": adata.obs_names[selected].astype(str),
                    "roi": roi,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def canonical_aggregate(results: list[ad.AnnData]) -> dict:
    records = []
    metric_names = sorted({name for result in results for name in result.layers})
    for result in results:
        for position, (_, row) in enumerate(result.obs.iterrows()):
            key = (str(row["roi"]), str(row["cell type"]))
            metrics = {
                name: np.asarray(result.layers[name][position], dtype=np.float64)
                for name in metric_names
            }
            records.append((key, int(row["n_cells"]), metrics))
    records.sort(key=lambda item: item[0])
    keys = [record[0] for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate ROI/cell-type result groups")
    return {
        "keys": keys,
        "n_cells": np.asarray([record[1] for record in records], dtype=np.int64),
        "metrics": {
            name: np.stack([record[2][name] for record in records])
            for name in metric_names
        },
    }


def assert_aggregate_equal(left: dict, right: dict) -> None:
    if left["keys"] != right["keys"]:
        raise RuntimeError("ROI group keys differ between execution methods")
    np.testing.assert_array_equal(left["n_cells"], right["n_cells"])
    if set(left["metrics"]) != set(right["metrics"]):
        raise RuntimeError("ROI metric sets differ between execution methods")
    for metric in left["metrics"]:
        if metric == "count_nonzero":
            np.testing.assert_array_equal(
                left["metrics"][metric],
                right["metrics"][metric],
            )
        else:
            np.testing.assert_allclose(
                left["metrics"][metric],
                right["metrics"][metric],
                rtol=1e-6,
                atol=1e-8,
            )


def aggregate_fingerprint(value: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(value["keys"], separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(value["n_cells"]).view(np.uint8))
    for metric in sorted(value["metrics"]):
        digest.update(metric.encode())
        rounded = np.round(value["metrics"][metric], 8)
        digest.update(np.ascontiguousarray(rounded).view(np.uint8))
    return digest.hexdigest()


def local_reanalysis(adata: ad.AnnData) -> dict:
    matrix = _dense(adata.X).astype(np.float64, copy=False)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, adata.obs_names)).encode())
    digest.update(np.ascontiguousarray(np.round(singular_values[:3], 8)).view(np.uint8))
    return {
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "top_singular_values": singular_values[:3].tolist(),
        "fingerprint": digest.hexdigest(),
    }


def aggregate_task(membership: pd.DataFrame, name: str = "roi_summary") -> AggregateTask:
    return AggregateTask(
        name=name,
        groupby=("roi", "cell type"),
        membership=membership,
        metrics=("sum", "mean", "count_nonzero"),
    )


def run_aggregation(database: CellDB, membership: pd.DataFrame, method: str, batch_size: int) -> dict:
    started = time.perf_counter()
    if method == "joint":
        run = database.aggregate_many(
            [aggregate_task(membership)],
            batch_size=batch_size,
        )
        canonical = canonical_aggregate([run.results["roi_summary"]])
        report = run.report.to_dict()
    else:
        results = []
        reports = []
        for roi in ROI_DEFINITIONS:
            roi_membership = membership.loc[membership["roi"].eq(roi)].copy()
            run = database.aggregate_many(
                [aggregate_task(roi_membership, name=roi)],
                batch_size=batch_size,
            )
            results.append(run.results[roi])
            reports.append(run.report)
        canonical = canonical_aggregate(results)
        report = {
            "source_scan_count": sum(item.source_scan_count for item in reports),
            "matrix_batch_reads": sum(item.matrix_batch_reads for item in reports),
            "requested_rows": sum(item.requested_rows for item in reports),
            "unique_rows": sum(item.unique_rows for item in reports),
            "matrix_bytes_read": sum(item.matrix_bytes_read for item in reports),
        }
    return {
        "method": method,
        "seconds": time.perf_counter() - started,
        "report": report,
        "canonical": canonical,
    }


def local_tasks(
    membership: pd.DataFrame,
    *,
    consume: bool = True,
) -> list[MaterializeTask]:
    return [
        MaterializeTask(
            name=f"local_{roi}",
            cell_ids=tuple(
                membership.loc[membership["roi"].eq(roi), "cell_id"].astype(str)
            ),
            obs_columns=("cell type",),
            consumer=local_reanalysis if consume else None,
        )
        for roi in LOCAL_ROIS
    ]


def run_mixed(
    database: CellDB,
    membership: pd.DataFrame,
    method: str,
    batch_size: int,
    memory_budget_bytes: int,
) -> dict:
    started = time.perf_counter()
    materializations = local_tasks(
        membership,
        consume=method != "all-materialized",
    )
    if method in {"mixed", "all-materialized"}:
        run = database.execute_tasks(
            [aggregate_task(membership), *materializations],
            batch_size=batch_size,
            memory_budget_bytes=(
                memory_budget_bytes if method == "mixed" else None
            ),
        )
        canonical = canonical_aggregate([run.results["roi_summary"]])
        local_results = {}
        for task in materializations:
            value = run.results[task.name]
            local_results[task.name] = (
                local_reanalysis(value)
                if method == "all-materialized"
                else value
            )
        report = run.report.to_dict()
    else:
        aggregate_run = database.aggregate_many(
            [aggregate_task(membership)],
            batch_size=batch_size,
        )
        canonical = canonical_aggregate([aggregate_run.results["roi_summary"]])
        local_results = {}
        reports = [aggregate_run.report]
        for task in materializations:
            run = database.execute_tasks(
                [task],
                batch_size=batch_size,
                memory_budget_bytes=memory_budget_bytes,
            )
            local_results[task.name] = run.results[task.name]
            reports.append(run.report)
        report = {
            "source_scan_count": sum(item.source_scan_count for item in reports),
            "matrix_batch_reads": sum(item.matrix_batch_reads for item in reports),
            "requested_rows": sum(item.requested_rows for item in reports),
            "unique_rows": sum(item.unique_rows for item in reports),
            "matrix_bytes_read": sum(item.matrix_bytes_read for item in reports),
            "peak_buffer_bytes": max(item.peak_buffer_bytes for item in reports),
            "execution_waves": sum(item.execution_waves for item in reports),
        }
    return {
        "method": method,
        "seconds": time.perf_counter() - started,
        "report": report,
        "canonical": canonical,
        "local_results": local_results,
    }


def summarize(runs: list[dict], method: str) -> dict:
    selected = [run for run in runs if run["method"] == method]
    times = [run["seconds"] for run in selected]
    return {
        "seconds": {
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
        },
        "matrix_batch_reads": statistics.median(
            [run["report"]["matrix_batch_reads"] for run in selected]
        ),
        "matrix_bytes_read": statistics.median(
            [run["report"]["matrix_bytes_read"] for run in selected]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("batch-size and repeats must be positive")
    if args.memory_budget_mib <= 0:
        raise ValueError("memory-budget-mib must be positive")
    input_path = Path(args.input_h5ad).expanduser().resolve()
    database_path = Path(args.cellvault_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_dataset(input_path)

    adata = ad.read_h5ad(input_path)
    coordinates = add_normalized_coordinates(adata)
    membership = build_membership(adata, coordinates)
    preparation = None
    if args.rebuild_cellvault or not (database_path / "obs.duckdb").exists():
        database_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with CellDB.from_anndata(
            adata,
            str(database_path),
            overwrite=True,
        ) as database:
            shape = list(database.shape)
        preparation = {
            "seconds": time.perf_counter() - started,
            "shape": shape,
        }

    memory_budget_bytes = int(args.memory_budget_mib * 1024**2)
    generator = random.Random(args.seed)
    aggregation_runs = []
    mixed_runs = []
    with CellDB.open(str(database_path)) as database:
        for repeat in range(1, args.repeats + 1):
            methods = ["independent", "joint"]
            generator.shuffle(methods)
            repeat_runs = [
                run_aggregation(database, membership, method, args.batch_size)
                for method in methods
            ]
            assert_aggregate_equal(
                repeat_runs[0]["canonical"],
                repeat_runs[1]["canonical"],
            )
            for run in repeat_runs:
                run["repeat"] = repeat
                run["fingerprint"] = aggregate_fingerprint(run.pop("canonical"))
                aggregation_runs.append(run)

            methods = ["sequential", "all-materialized", "mixed"]
            generator.shuffle(methods)
            repeat_runs = [
                run_mixed(
                    database,
                    membership,
                    method,
                    args.batch_size,
                    memory_budget_bytes,
                )
                for method in methods
            ]
            for candidate in repeat_runs[1:]:
                assert_aggregate_equal(
                    repeat_runs[0]["canonical"],
                    candidate["canonical"],
                )
                if repeat_runs[0]["local_results"] != candidate["local_results"]:
                    raise RuntimeError("local ROI analysis differs between methods")
            for run in repeat_runs:
                run["repeat"] = repeat
                run["aggregate_fingerprint"] = aggregate_fingerprint(
                    run.pop("canonical")
                )
                mixed_runs.append(run)

    aggregation_summary = {
        method: summarize(aggregation_runs, method)
        for method in ("independent", "joint")
    }
    aggregation_summary["speedup"] = (
        aggregation_summary["independent"]["seconds"]["median"]
        / aggregation_summary["joint"]["seconds"]["median"]
    )
    mixed_summary = {
        method: summarize(mixed_runs, method)
        for method in ("sequential", "all-materialized", "mixed")
    }
    for method, summary in mixed_summary.items():
        selected = [run for run in mixed_runs if run["method"] == method]
        summary["peak_buffer_bytes"] = statistics.median(
            [run["report"]["peak_buffer_bytes"] for run in selected]
        )
        summary["execution_waves"] = statistics.median(
            [run["report"]["execution_waves"] for run in selected]
        )
    mixed_summary["speedup"] = (
        mixed_summary["sequential"]["seconds"]["median"]
        / mixed_summary["mixed"]["seconds"]["median"]
    )
    roi_counts = membership.groupby("roi", observed=True).size()
    unique_cells = membership["cell_id"].nunique()
    payload = {
        "benchmark": "overlapping spatial ROI aggregation and mixed execution",
        "dataset": {
            "title": "Squidpy IMC example: Jackson et al. breast cancer subset",
            "publication_doi": "10.1038/s41586-019-1876-x",
            "url": DATASET_URL,
            "sha256": DATASET_SHA256,
            "file_bytes": input_path.stat().st_size,
            "shape": [adata.n_obs, adata.n_vars],
        },
        "roi_design": {
            "kind": "controlled fixed rectangles on normalized public coordinates",
            "biological_claim": "none; these are not curated tissue annotations",
            "definitions": ROI_DEFINITIONS,
            "cell_counts": {name: int(roi_counts[name]) for name in ROI_DEFINITIONS},
            "membership_edges": len(membership),
            "unique_cells": unique_cells,
            "overlap_edges": len(membership) - unique_cells,
        },
        "configuration": {
            "batch_size": args.batch_size,
            "memory_budget_bytes": memory_budget_bytes,
            "local_rois": list(LOCAL_ROIS),
            "repeats": args.repeats,
            "seed": args.seed,
            "method_order_randomized": True,
            "filesystem_cache_cleared": False,
        },
        "preparation": preparation,
        "aggregation_summary": aggregation_summary,
        "mixed_summary": mixed_summary,
        "validation": {
            "status": "passed",
            "aggregate_rtol": 1e-6,
            "aggregate_atol": 1e-8,
            "local_analysis_exact": True,
        },
        "aggregation_runs": aggregation_runs,
        "mixed_runs": mixed_runs,
        "notes": [
            "The source measurements and coordinates are real public IMC data.",
            "ROI rectangles are fixed controlled geometries, not author-curated biological regions.",
            "Logical decoded reads are reported; they are not physical storage I/O.",
            "Warm-cache timings on this small dataset validate behavior, not scale claims.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"aggregation": aggregation_summary, "mixed": mixed_summary}, indent=2))
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
