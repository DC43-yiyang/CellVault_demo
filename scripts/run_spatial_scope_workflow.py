#!/usr/bin/env python3
"""Run regional summaries and local analyses on predefined MIBI-TOF scopes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from cellvault import AggregateTask, CellDB, MaterializeTask

try:
    from scripts.validate_predefined_spatial_rois import (
        DATASET_SHA256,
        DATASET_URL,
        build_membership,
        ensure_dataset,
    )
except ModuleNotFoundError:
    from validate_predefined_spatial_rois import (
        DATASET_SHA256,
        DATASET_URL,
        build_membership,
        ensure_dataset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        default="benchmark_data/squidpy_mibitof.h5ad",
    )
    parser.add_argument(
        "--cellvault-path",
        default="benchmark_outputs/mibitof_spatial/mibitof.cvdb",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_results/mibitof_complete_spatial_workflow.json",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--memory-budget-mib", type=float, default=0.9)
    parser.add_argument("--local-scopes", type=int, default=2)
    parser.add_argument("--rebuild-cellvault", action="store_true")
    return parser.parse_args()


def matrix_fingerprint(matrix) -> str:
    digest = hashlib.sha256()
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=False)
        digest.update(np.ascontiguousarray(matrix.data).view(np.uint8))
        digest.update(np.ascontiguousarray(matrix.indices).view(np.uint8))
        digest.update(np.ascontiguousarray(matrix.indptr).view(np.uint8))
    else:
        digest.update(np.ascontiguousarray(matrix).view(np.uint8))
    return digest.hexdigest()


def local_scope_analysis(adata: ad.AnnData) -> dict:
    context_columns = ("library_id", "donor")
    context = {
        column: sorted(adata.obs[column].astype(str).unique().tolist())
        for column in context_columns
    }
    if any(len(values) != 1 for values in context.values()):
        raise ValueError("a local scope must retain exactly one library_id and donor")
    matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    centered = matrix.astype(np.float64, copy=False) - matrix.mean(axis=0)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular_values**2
    variance_ratio = variance / variance.sum() if variance.sum() else variance
    clusters = adata.obs["Cluster"].astype(str)
    cluster_profiles = {}
    for cluster in sorted(clusters.unique()):
        mask = clusters.eq(cluster).to_numpy()
        means = matrix[mask].mean(axis=0)
        top = np.argsort(-means, kind="stable")[:3]
        cluster_profiles[cluster] = {
            "n_cells": int(mask.sum()),
            "top_markers": [
                {
                    "marker": str(adata.var_names[position]),
                    "mean": float(means[position]),
                }
                for position in top
            ],
        }
    return {
        "scope_id": context["library_id"][0],
        "donor": context["donor"][0],
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "matrix_sha256": matrix_fingerprint(adata.X),
        "pca_variance_ratio": [float(value) for value in variance_ratio[:5]],
        "cluster_profiles": cluster_profiles,
    }


def aggregate_fingerprint(result: ad.AnnData) -> dict:
    keys = [
        [str(value) for value in row]
        for row in result.obs[
            ["scope_level", "roi_id", "Cluster"]
        ].itertuples(index=False, name=None)
    ]
    return {
        "groups": int(result.n_obs),
        "features": int(result.n_vars),
        "group_keys_sha256": hashlib.sha256(
            json.dumps(keys, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "n_cells_sha256": hashlib.sha256(
            np.ascontiguousarray(result.obs["n_cells"].to_numpy()).view(np.uint8)
        ).hexdigest(),
        "metrics": {
            metric: matrix_fingerprint(result.layers[metric])
            for metric in ("sum", "mean", "count_nonzero")
        },
    }


def build_tasks(adata: ad.AnnData, membership: pd.DataFrame, local_scopes: int):
    scope_counts = adata.obs["library_id"].astype(str).value_counts()
    selected_scopes = scope_counts.index[:local_scopes].tolist()
    aggregate = AggregateTask(
        "predefined_spatial_scopes",
        ("scope_level", "roi_id", "Cluster"),
        membership=membership,
        metrics=("sum", "mean", "count_nonzero"),
    )
    materializers = [
        MaterializeTask(
            f"local_{scope}",
            where='"library_id" = ?',
            params=(scope,),
            obs_columns=("library_id", "donor", "Cluster"),
            consumer=local_scope_analysis,
        )
        for scope in selected_scopes
    ]
    return aggregate, materializers, selected_scopes


def run_sequential(database: CellDB, aggregate, materializers, batch_size: int):
    started = time.perf_counter()
    aggregate_run = database.aggregate_many([aggregate], batch_size=batch_size)
    reports = [aggregate_run.report]
    local_results = {}
    for task in materializers:
        run = database.execute_tasks([task], batch_size=batch_size)
        reports.append(run.report)
        local_results[task.name] = run.results[task.name]
    return {
        "seconds": time.perf_counter() - started,
        "aggregate": aggregate_run.results[aggregate.name],
        "local": local_results,
        "report": {
            "source_scan_count": sum(report.source_scan_count for report in reports),
            "matrix_batch_reads": sum(report.matrix_batch_reads for report in reports),
            "matrix_bytes_read": sum(report.matrix_bytes_read for report in reports),
            "peak_buffer_bytes": max(report.peak_buffer_bytes for report in reports),
            "execution_waves": sum(report.execution_waves for report in reports),
        },
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.memory_budget_mib <= 0 or args.local_scopes <= 0:
        raise ValueError("batch-size, memory-budget-mib, and local-scopes must be positive")
    input_path = Path(args.input_h5ad).expanduser().resolve()
    database_path = Path(args.cellvault_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_dataset(input_path)
    adata = ad.read_h5ad(input_path)
    membership = build_membership(adata)

    preparation_seconds = None
    if args.rebuild_cellvault or not (database_path / "obs.duckdb").is_file():
        started = time.perf_counter()
        with CellDB.from_anndata(adata, str(database_path), overwrite=True):
            pass
        preparation_seconds = time.perf_counter() - started

    aggregate, materializers, selected_scopes = build_tasks(
        adata, membership, args.local_scopes
    )
    with CellDB.open(str(database_path)) as database:
        sequential = run_sequential(database, aggregate, materializers, args.batch_size)
        started = time.perf_counter()
        joint_run = database.execute_tasks(
            [aggregate, *materializers],
            batch_size=args.batch_size,
            memory_budget_bytes=int(args.memory_budget_mib * 1024**2),
        )
        joint_seconds = time.perf_counter() - started

    sequential_aggregate = aggregate_fingerprint(sequential["aggregate"])
    joint_aggregate = aggregate_fingerprint(joint_run.results[aggregate.name])
    joint_local = {
        task.name: joint_run.results[task.name] for task in materializers
    }
    if sequential_aggregate != joint_aggregate:
        raise RuntimeError("sequential and joint regional summaries differ")
    if sequential["local"] != joint_local:
        raise RuntimeError("sequential and joint local analyses differ")

    result = joint_run.results[aggregate.name]
    abundance = []
    for (scope_level, roi_id), rows in result.obs.groupby(
        ["scope_level", "roi_id"], sort=False, observed=True
    ):
        total = int(rows["n_cells"].sum())
        for index, row in rows.iterrows():
            position = result.obs.index.get_loc(index)
            mean_values = np.asarray(result.layers["mean"][position])
            top_marker = int(np.argmax(mean_values))
            abundance.append(
                {
                    "scope_level": str(scope_level),
                    "roi_id": str(roi_id),
                    "cell_type": str(row["Cluster"]),
                    "n_cells": int(row["n_cells"]),
                    "fraction": float(row["n_cells"] / total),
                    "top_mean_marker": str(result.var_names[top_marker]),
                    "top_mean_value": float(mean_values[top_marker]),
                }
            )

    payload = {
        "workflow": "MIBI-TOF predefined spatial summaries and local analyses",
        "dataset": {
            "title": "Squidpy MIBI-TOF example from Hartmann et al.",
            "publication_doi": "10.1101/2020.01.17.909796",
            "url": DATASET_URL,
            "sha256": DATASET_SHA256,
            "shape": [int(adata.n_obs), int(adata.n_vars)],
        },
        "preparation_seconds": preparation_seconds,
        "scope_design": {
            "source": "author-provided library_id nested in donor",
            "selected_local_scopes": selected_scopes,
            "membership_edges": int(len(membership)),
            "unique_cells": int(membership["cell_id"].nunique()),
        },
        "sequential": {
            "seconds": sequential["seconds"],
            "report": sequential["report"],
        },
        "budgeted_joint": {
            "seconds": joint_seconds,
            "report": joint_run.report.to_dict(),
        },
        "regional_abundance_and_expression": abundance,
        "local_analyses": joint_local,
        "fingerprint": joint_aggregate,
        "validation": {
            "status": "passed",
            "aggregate_exact": True,
            "local_analysis_exact": True,
            "scope_context_preserved": True,
            "each_cell_contributes_to_fov_and_donor": (
                joint_run.report.task_memberships[aggregate.name] == 2 * adata.n_obs
            ),
        },
        "notes": [
            "The primary scopes are author-provided fields of view and donor specimens, not synthetic rectangles.",
            "Local PCA variance and cluster marker profiles are descriptive outputs within each selected FOV.",
            "The memory budget covers CellVault-managed buffers rather than total process RSS.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "validation": payload["validation"],
                "sequential_seconds": sequential["seconds"],
                "joint_seconds": joint_seconds,
                "joint_waves": joint_run.report.execution_waves,
            },
            indent=2,
        )
    )
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
