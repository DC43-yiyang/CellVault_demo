#!/usr/bin/env python3
"""Validate sample-input and leave-one-donor adapters on the public Wu atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from cellvault import (
    AggregateTask,
    CellDB,
    leave_one_out_pseudobulk,
    prepare_communication_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cellvault-path",
        default="benchmark_outputs/wu2021_breast_cancer/wu2021_optimized.cvdb",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_results/wu2021_mvp5_workflows.json",
    )
    parser.add_argument("--feature-limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--memory-budget-gib", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def matrix_nbytes(matrix) -> int:
    if sparse.issparse(matrix):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return int(np.asarray(matrix).nbytes)


def sample_input_summary(adata) -> dict:
    donors = adata.obs["donor_id"].astype(str).unique().tolist()
    if len(donors) != 1:
        raise RuntimeError("a communication input contains multiple donors")
    counts = (
        adata.obs["celltype_major"]
        .astype(str)
        .value_counts(sort=False)
        .sort_index()
    )
    matrix = adata.X
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, adata.obs_names)).encode())
    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=False)
        digest.update(np.ascontiguousarray(csr.data).view(np.uint8))
        digest.update(np.ascontiguousarray(csr.indices).view(np.uint8))
        digest.update(np.ascontiguousarray(csr.indptr).view(np.uint8))
    else:
        digest.update(np.ascontiguousarray(matrix).view(np.uint8))
    return {
        "donor_id": donors[0],
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "cell_type_counts": {str(key): int(value) for key, value in counts.items()},
        "matrix_bytes": matrix_nbytes(matrix),
        "fingerprint": digest.hexdigest(),
    }


def pseudobulk_fingerprint(adata) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, adata.obs_names)).encode())
    digest.update(np.ascontiguousarray(adata.layers["sum"]).view(np.uint8))
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    if args.feature_limit <= 0 or args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("feature-limit, batch-size, and repeats must be positive")
    if args.memory_budget_gib <= 0:
        raise ValueError("memory-budget-gib must be positive")
    database_path = Path(args.cellvault_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not (database_path / "obs.duckdb").exists():
        raise FileNotFoundError(f"CellVault store not found: {database_path}")

    runs = []
    memory_budget_bytes = int(args.memory_budget_gib * 1024**3)
    with CellDB.open(str(database_path)) as database:
        required = {"donor_id", "celltype_major", "subtype"}
        missing = required - set(database.obs_columns)
        if missing:
            raise KeyError(f"Wu store is missing obs columns: {sorted(missing)}")
        features = tuple(database.var_names[: args.feature_limit])
        expected_donor_counts = {
            str(key): int(value)
            for key, value in database.obs["donor_id"]
            .astype(str)
            .value_counts(sort=False)
            .items()
        }

        for repeat in range(1, args.repeats + 1):
            communication_started = time.perf_counter()
            communication = prepare_communication_inputs(
                database,
                sample_column="donor_id",
                cell_type_column="celltype_major",
                source="X",
                features=features,
                obs_columns=("subtype",),
                batch_size=args.batch_size,
                memory_budget_bytes=memory_budget_bytes,
                consumer=sample_input_summary,
            )
            communication_seconds = time.perf_counter() - communication_started
            observed_donor_counts = {
                str(donor): int(result["n_obs"])
                for donor, result in communication.results.items()
            }
            if observed_donor_counts != expected_donor_counts:
                raise RuntimeError("per-donor communication input membership mismatch")

            pseudobulk_started = time.perf_counter()
            aggregation = database.aggregate_many(
                [
                    AggregateTask(
                        "donor_cell_type",
                        ("donor_id", "celltype_major"),
                        features=features,
                        metrics=("sum",),
                    )
                ],
                batch_size=args.batch_size,
            )
            pseudobulk = aggregation.results["donor_cell_type"]
            aggregation_seconds = time.perf_counter() - pseudobulk_started

            leave_out_started = time.perf_counter()
            leave_out = leave_one_out_pseudobulk(
                pseudobulk,
                donor_column="donor_id",
            )
            leave_out_seconds = time.perf_counter() - leave_out_started
            for donor, selected in leave_out.items():
                if selected.obs["donor_id"].astype(str).eq(str(donor)).any():
                    raise RuntimeError(f"leave-one-out result retained donor {donor!r}")

            runs.append(
                {
                    "repeat": repeat,
                    "communication_seconds": communication_seconds,
                    "communication_report": communication.report.to_dict(),
                    "communication_fingerprints": {
                        str(donor): result["fingerprint"]
                        for donor, result in communication.results.items()
                    },
                    "pseudobulk_seconds": aggregation_seconds,
                    "pseudobulk_report": aggregation.report.to_dict(),
                    "pseudobulk_fingerprint": pseudobulk_fingerprint(pseudobulk),
                    "leave_one_out_seconds": leave_out_seconds,
                    "leave_one_out_donors": len(leave_out),
                    "leave_one_out_rows": {
                        str(donor): selected.n_obs
                        for donor, selected in leave_out.items()
                    },
                }
            )

        communication_fingerprints = {
            json.dumps(run["communication_fingerprints"], sort_keys=True)
            for run in runs
        }
        pseudobulk_fingerprints = {
            run["pseudobulk_fingerprint"] for run in runs
        }
        if len(communication_fingerprints) != 1 or len(pseudobulk_fingerprints) != 1:
            raise RuntimeError("workflow outputs changed between repeats")

        payload = {
            "benchmark": "MVP-5 sample inputs and leave-one-donor reuse",
            "dataset": {
                "title": "Wu et al. 2021 breast cancer atlas",
                "doi": "10.1038/s41588-021-00911-1",
                "cellvault_path": str(database_path),
                "shape": [database.n_obs, database.n_vars],
                "donors": len(expected_donor_counts),
            },
            "configuration": {
                "features": len(features),
                "batch_size": args.batch_size,
                "memory_budget_bytes": memory_budget_bytes,
                "repeats": args.repeats,
            },
            "summary": {
                "communication_seconds": summarize(
                    [run["communication_seconds"] for run in runs]
                ),
                "pseudobulk_seconds": summarize(
                    [run["pseudobulk_seconds"] for run in runs]
                ),
                "leave_one_out_seconds": summarize(
                    [run["leave_one_out_seconds"] for run in runs]
                ),
                "communication_source_scans": runs[0]["communication_report"][
                    "source_scan_count"
                ],
                "communication_matrix_batch_reads": runs[0][
                    "communication_report"
                ]["matrix_batch_reads"],
                "pseudobulk_matrix_batch_reads": runs[0]["pseudobulk_report"][
                    "matrix_batch_reads"
                ],
            },
            "validation": {
                "status": "passed",
                "all_cells_assigned_to_one_donor_input": sum(
                    expected_donor_counts.values()
                )
                == database.n_obs,
                "leave_one_out_uses_group_level_object": True,
                "core_executor_changes_required": False,
            },
            "runs": runs,
            "notes": [
                "Communication inputs retain every selected cell and cell-type label per donor.",
                "No claim is made that every communication method can consume pseudobulk summaries.",
                "Leave-one-donor variants are sliced from the existing pseudobulk result without rereading X.",
            ],
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
