#!/usr/bin/env python3
"""Validate overlapping predefined spatial scopes on public MIBI-TOF data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from cellvault import AggregateTask, CellDB


DATASET_URL = "https://exampledata.scverse.org/squidpy/mibitof.h5ad"
DATASET_SHA256 = "3f125c51695d78ed1c36d5485dc390ab400154d021f0c7715b89f8ee83978690"


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
        default="benchmark_results/squidpy_mibitof_predefined_roi.json",
    )
    parser.add_argument("--batch-size", type=int, default=512)
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


def build_membership(adata: ad.AnnData) -> pd.DataFrame:
    required = {"library_id", "donor", "Cluster"}
    missing = required - set(adata.obs)
    if missing:
        raise KeyError(f"MIBI-TOF obs is missing columns: {sorted(missing)}")
    cell_ids = adata.obs_names.astype(str)
    fov = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "scope_level": "field_of_view",
            "roi_id": "fov:" + adata.obs["library_id"].astype(str).to_numpy(),
        }
    )
    donor = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "scope_level": "donor_specimen",
            "roi_id": "donor:" + adata.obs["donor"].astype(str).to_numpy(),
        }
    )
    return pd.concat([fov, donor], ignore_index=True)


def direct_reference(adata: ad.AnnData, membership: pd.DataFrame) -> dict:
    positions = pd.Series(np.arange(adata.n_obs), index=adata.obs_names.astype(str))
    joined = membership.copy()
    joined["position"] = joined["cell_id"].map(positions)
    joined["Cluster"] = joined["cell_id"].map(
        adata.obs["Cluster"].astype(str).set_axis(adata.obs_names.astype(str))
    )
    reference = {}
    for key, edges in joined.groupby(
        ["scope_level", "roi_id", "Cluster"],
        sort=False,
        observed=True,
    ):
        matrix = adata.X[edges["position"].to_numpy()]
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        else:
            matrix = np.asarray(matrix)
        reference[tuple(map(str, key))] = {
            "n_cells": len(edges),
            "sum": matrix.sum(axis=0, dtype=np.float64),
            "mean": matrix.mean(axis=0, dtype=np.float64),
            "count_nonzero": np.count_nonzero(matrix, axis=0),
        }
    return reference


def validate_result(result: ad.AnnData, reference: dict) -> None:
    observed = {}
    for position, (_, row) in enumerate(result.obs.iterrows()):
        key = tuple(str(row[column]) for column in ("scope_level", "roi_id", "Cluster"))
        observed[key] = {
            "n_cells": int(row["n_cells"]),
            "sum": result.layers["sum"][position],
            "mean": result.layers["mean"][position],
            "count_nonzero": result.layers["count_nonzero"][position],
        }
    if set(observed) != set(reference):
        raise RuntimeError("predefined spatial scope keys differ from reference")
    for key, expected in reference.items():
        actual = observed[key]
        if actual["n_cells"] != expected["n_cells"]:
            raise RuntimeError(f"n_cells mismatch for {key}")
        np.testing.assert_allclose(actual["sum"], expected["sum"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(actual["mean"], expected["mean"], rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(
            actual["count_nonzero"], expected["count_nonzero"]
        )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    input_path = Path(args.input_h5ad).expanduser().resolve()
    database_path = Path(args.cellvault_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_dataset(input_path)
    adata = ad.read_h5ad(input_path)
    membership = build_membership(adata)
    reference = direct_reference(adata, membership)

    preparation = None
    if args.rebuild_cellvault or not (database_path / "obs.duckdb").exists():
        database_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with CellDB.from_anndata(
            adata,
            str(database_path),
            overwrite=True,
        ):
            pass
        preparation = time.perf_counter() - started

    started = time.perf_counter()
    with CellDB.open(str(database_path)) as database:
        run = database.aggregate_many(
            [
                AggregateTask(
                    "predefined_spatial_scopes",
                    ("scope_level", "roi_id", "Cluster"),
                    membership=membership,
                    metrics=("sum", "mean", "count_nonzero"),
                )
            ],
            batch_size=args.batch_size,
        )
    wall_seconds = time.perf_counter() - started
    result = run.results["predefined_spatial_scopes"]
    validate_result(result, reference)

    scope_counts = (
        membership[["scope_level", "roi_id"]]
        .drop_duplicates()
        .groupby("scope_level", observed=True)
        .size()
    )
    payload = {
        "validation": "predefined overlapping spatial scopes",
        "dataset": {
            "title": "Squidpy MIBI-TOF example from Hartmann et al.",
            "publication_doi": "10.1101/2020.01.17.909796",
            "url": DATASET_URL,
            "sha256": DATASET_SHA256,
            "file_bytes": input_path.stat().st_size,
            "shape": [adata.n_obs, adata.n_vars],
        },
        "scope_design": {
            "field_of_view_column": "library_id",
            "specimen_column": "donor",
            "scope_counts": {str(key): int(value) for key, value in scope_counts.items()},
            "membership_edges": len(membership),
            "unique_cells": membership["cell_id"].nunique(),
            "memberships_per_cell": 2,
            "random_geometry": False,
        },
        "preparation_seconds": preparation,
        "wall_seconds": wall_seconds,
        "report": run.report.to_dict(),
        "result": {
            "groups": result.n_obs,
            "features": result.n_vars,
        },
        "checks": {
            "status": "passed",
            "group_keys_match": True,
            "n_cells_exact": True,
            "count_nonzero_exact": True,
            "sum_mean_rtol": 1e-6,
            "sum_mean_atol": 1e-8,
            "each_cell_read_once": run.report.unique_rows == adata.n_obs,
            "each_cell_contributes_to_two_scopes": (
                run.report.task_memberships["predefined_spatial_scopes"]
                == 2 * adata.n_obs
            ),
        },
        "notes": [
            "library_id identifies author-provided fields of view.",
            "donor identifies the containing specimen scope.",
            "Each cell contributes once to its FOV and once to its donor scope while its matrix row is read once.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wall_seconds": wall_seconds, "checks": payload["checks"]}, indent=2))
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
