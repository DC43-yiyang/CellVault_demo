#!/usr/bin/env python3
"""Run matched treatment-control response summaries on McFarland MIX-Seq."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

from cellvault import AggregateTask, CellDB
try:
    from scripts.benchmark_shared_control import (
        DATASET_SHA256,
        DATASET_URL,
        DEFAULT_TREATMENTS,
        MATCH_COLUMNS,
        matched_pairs,
        quote_identifier,
        sha256_file,
    )
except ModuleNotFoundError:
    from benchmark_shared_control import (
        DATASET_SHA256,
        DATASET_URL,
        DEFAULT_TREATMENTS,
        MATCH_COLUMNS,
        matched_pairs,
        quote_identifier,
        sha256_file,
    )


RESPONSE_GENES = (
    "DUSP6",
    "EGR1",
    "FOS",
    "JUN",
    "SPRY2",
    "ETV4",
    "ETV5",
    "BCL2",
    "BCL2L1",
    "MCL1",
    "BAX",
    "PMAIP1",
    "MKI67",
    "TOP2A",
    "CDKN1A",
    "ATF3",
    "FOSL1",
    "BRAF",
    "MAP2K1",
    "MDM2",
    "GPX4",
    "MTOR",
    "BRD4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        default="benchmark_data/mcfarland_2020.h5ad",
    )
    parser.add_argument(
        "--cellvault-path",
        default="benchmark_outputs/mcfarland_2020/mcfarland.cvdb",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_results/mcfarland_2020_response_workflow.json",
    )
    parser.add_argument("--treatment", action="append", dest="treatments")
    parser.add_argument("--control", default="control")
    parser.add_argument("--quality", default="normal")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-genes", type=int, default=8)
    parser.add_argument("--rebuild-cellvault", action="store_true")
    return parser.parse_args()


def build_response_tasks(database: CellDB, args: argparse.Namespace):
    obs = database.obs
    available = set(map(str, database.var_names))
    features = tuple(gene for gene in RESPONSE_GENES if gene in available)
    missing_features = tuple(gene for gene in RESPONSE_GENES if gene not in available)
    if not features:
        raise KeyError("none of the response genes are present")
    tasks = []
    audit = []
    for treatment in args.treatments or DEFAULT_TREATMENTS:
        pairs = matched_pairs(obs, treatment, args.control, args.quality)
        if not pairs:
            raise ValueError(f"no matched control pairs for {treatment!r}")
        pair_predicate = " OR ".join(
            "(" + " AND ".join(
                f"{quote_identifier(column)} = ?" for column in MATCH_COLUMNS
            ) + ")"
            for _ in pairs
        )
        pair_params = tuple(value for pair in pairs for value in pair)
        task = AggregateTask(
            name=treatment,
            groupby=(*MATCH_COLUMNS, "perturbation"),
            where=(
                '"cell_quality" = ? AND "perturbation" IN (?, ?) AND '
                f"({pair_predicate})"
            ),
            params=(args.quality, args.control, treatment, *pair_params),
            metrics=("sum", "count_nonzero"),
            features=features,
        )
        selected = obs.loc[
            obs["cell_quality"].astype(str).eq(args.quality)
            & obs["perturbation"].astype(str).isin([args.control, treatment])
            & obs[list(MATCH_COLUMNS)].apply(tuple, axis=1).isin(pairs)
        ]
        audit.append(
            {
                "treatment": treatment,
                "matched_pairs": len(pairs),
                "treatment_cells": int(
                    selected["perturbation"].astype(str).eq(treatment).sum()
                ),
                "control_cells": int(
                    selected["perturbation"].astype(str).eq(args.control).sum()
                ),
            }
        )
        tasks.append(task)
    return tasks, audit, features, missing_features


def response_summary(result, treatment: str, control: str, top_genes: int) -> dict:
    obs = result.obs.reset_index(drop=True)
    sums = np.asarray(result.layers["sum"], dtype=np.float64)
    means = sums / obs["n_cells"].to_numpy(dtype=np.float64)[:, None]
    rows = {}
    for position, row in obs.iterrows():
        key = tuple(str(row[column]) for column in MATCH_COLUMNS)
        rows.setdefault(key, {})[str(row["perturbation"])] = position

    pair_effects = []
    pair_records = []
    for key in sorted(rows, key=lambda values: tuple(map(str, values))):
        arms = rows[key]
        if control not in arms or treatment not in arms:
            raise RuntimeError(f"incomplete matched pair {key} for {treatment}")
        control_position = arms[control]
        treatment_position = arms[treatment]
        effect = means[treatment_position] - means[control_position]
        pair_effects.append(effect)
        pair_records.append(
            {
                "cell_line": key[0],
                "time": key[1],
                "control_cells": int(obs.loc[control_position, "n_cells"]),
                "treatment_cells": int(obs.loc[treatment_position, "n_cells"]),
            }
        )
    effects = np.vstack(pair_effects)
    mean_effect = effects.mean(axis=0)
    increased = np.argsort(-mean_effect, kind="stable")[:top_genes]
    decreased = np.argsort(mean_effect, kind="stable")[:top_genes]

    def genes(positions) -> list[dict]:
        return [
            {
                "gene": str(result.var_names[position]),
                "mean_matched_difference": float(mean_effect[position]),
                "positive_pair_fraction": float(np.mean(effects[:, position] > 0)),
            }
            for position in positions
        ]

    contiguous = np.ascontiguousarray(mean_effect)
    return {
        "treatment": treatment,
        "control": control,
        "matching_columns": list(MATCH_COLUMNS),
        "matched_pairs": pair_records,
        "top_increased": genes(increased),
        "top_decreased": genes(decreased),
        "effect_sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
        "effect_sum": float(mean_effect.sum(dtype=np.float64)),
        "interpretation": (
            "Unweighted mean of cell-line/time-specific treated-minus-control "
            "mean raw-count expression; descriptive, not a fitted response model"
        ),
    }


def assert_results_equal(left, right) -> None:
    if left.obs[list(MATCH_COLUMNS) + ["perturbation", "n_cells"]].astype(
        str
    ).to_dict("records") != right.obs[
        list(MATCH_COLUMNS) + ["perturbation", "n_cells"]
    ].astype(str).to_dict("records"):
        raise RuntimeError("independent and joint group membership differs")
    if not left.var_names.equals(right.var_names):
        raise RuntimeError("independent and joint feature axes differ")
    for metric in ("sum", "count_nonzero"):
        np.testing.assert_array_equal(left.layers[metric], right.layers[metric])


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def run_path(database: CellDB, tasks, method: str, batch_size: int):
    started = time.perf_counter()
    reports = []
    if method == "joint":
        run = database.aggregate_many(tasks, batch_size=batch_size)
        results = dict(run.results)
        reports.append(run.report)
    else:
        results = {}
        for task in tasks:
            run = database.aggregate_many([task], batch_size=batch_size)
            results[task.name] = run.results[task.name]
            reports.append(run.report)
    access_seconds = time.perf_counter() - started
    report = {
        "source_scan_count": sum(item.source_scan_count for item in reports),
        "matrix_batch_reads": sum(item.matrix_batch_reads for item in reports),
        "matrix_bytes_read": sum(item.matrix_bytes_read for item in reports),
        "requested_rows": sum(item.requested_rows for item in reports),
        "unique_rows": sum(item.unique_rows for item in reports),
    }
    return results, report, access_seconds


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.repeats <= 0 or args.top_genes <= 0:
        raise ValueError("batch-size, repeats, and top-genes must be positive")
    input_path = Path(args.input_h5ad).expanduser().resolve()
    database_path = Path(args.cellvault_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed_sha256 = sha256_file(input_path)
    if observed_sha256 != DATASET_SHA256:
        raise RuntimeError(
            f"dataset SHA256 mismatch: expected {DATASET_SHA256}, got {observed_sha256}"
        )

    preparation_seconds = None
    if args.rebuild_cellvault or not (database_path / "obs.duckdb").is_file():
        started = time.perf_counter()
        with CellDB.from_h5ad(str(input_path), str(database_path), overwrite=True):
            pass
        preparation_seconds = time.perf_counter() - started

    runs = []
    final_responses = None
    audit = None
    features = None
    missing_features = None
    generator = random.Random(args.seed)
    with CellDB.open(str(database_path)) as database:
        tasks, audit, features, missing_features = build_response_tasks(database, args)
        for repeat in range(1, args.repeats + 1):
            order = ["independent", "joint"]
            generator.shuffle(order)
            repeat_results = {}
            for method in order:
                results, report, access_seconds = run_path(
                    database, tasks, method, args.batch_size
                )
                downstream_started = time.perf_counter()
                responses = {
                    task.name: response_summary(
                        results[task.name], task.name, args.control, args.top_genes
                    )
                    for task in tasks
                }
                downstream_seconds = time.perf_counter() - downstream_started
                repeat_results[method] = results
                runs.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "access_seconds": access_seconds,
                        "downstream_seconds": downstream_seconds,
                        "total_seconds": access_seconds + downstream_seconds,
                        "report": report,
                        "response_fingerprints": {
                            name: response["effect_sha256"]
                            for name, response in responses.items()
                        },
                    }
                )
                if method == "joint":
                    final_responses = responses
            for task in tasks:
                assert_results_equal(
                    repeat_results["independent"][task.name],
                    repeat_results["joint"][task.name],
                )

    summary = {}
    for method in ("independent", "joint"):
        selected = [run for run in runs if run["method"] == method]
        summary[method] = {
            metric: summarize([float(run[metric]) for run in selected])
            for metric in ("access_seconds", "downstream_seconds", "total_seconds")
        }
        for metric in ("source_scan_count", "matrix_batch_reads", "matrix_bytes_read"):
            summary[method][metric] = summarize(
                [float(run["report"][metric]) for run in selected]
            )
    summary["access_speedup"] = (
        summary["independent"]["access_seconds"]["median"]
        / summary["joint"]["access_seconds"]["median"]
    )

    payload = {
        "workflow": "McFarland matched treatment-control response summaries",
        "dataset": {
            "title": "McFarland et al. 2020 MIX-Seq perturbation response",
            "doi": "10.1038/s41467-020-17440-w",
            "url": DATASET_URL,
            "sha256": observed_sha256,
            "shape": [182875, 32738],
        },
        "preparation_seconds": preparation_seconds,
        "control": args.control,
        "quality": args.quality,
        "matching_columns": list(MATCH_COLUMNS),
        "features": list(features),
        "missing_features": list(missing_features),
        "task_audit": audit,
        "summary": summary,
        "responses": final_responses,
        "validation": {
            "status": "passed",
            "cohort_membership_equal": True,
            "aggregate_values_exact": True,
            "response_rankings_equal": True,
        },
        "runs": runs,
        "notes": [
            "Controls are matched independently for every treatment by cell_line and time.",
            "The channel field is not treated as a universal batch key.",
            "Response values are descriptive matched summaries, not a model correcting all experimental confounders.",
            "matrix_bytes_read is decoded backend output rather than physical storage I/O.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "validation": payload["validation"]}, indent=2))
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
