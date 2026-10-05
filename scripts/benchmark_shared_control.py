#!/usr/bin/env python3
"""Benchmark matched perturbation comparisons that reuse control cells."""

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


DEFAULT_TREATMENTS = (
    "BRD3379",
    "Dabrafenib",
    "Navitoclax",
    "AZD5591",
    "JQ1",
)
DATASET_URL = "https://exampledata.scverse.org/pertpy/mcfarland_2020.h5ad"
DATASET_SHA256 = "94a7240047b9ce6822dc2aa1e7a66c1fd8b00be842e5a95fc72cbeb6f2834d2c"
MATCH_COLUMNS = ("cell_line", "time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--cellvault-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--treatment",
        action="append",
        dest="treatments",
        help="Treatment to compare with matched controls; repeat as needed.",
    )
    parser.add_argument("--control", default="control")
    parser.add_argument("--quality", default="normal")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--feature-limit", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-cellvault", action="store_true")
    return parser.parse_args()


def quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_fingerprint(matrix) -> dict:
    values = np.asarray(matrix)
    rounded = np.ascontiguousarray(np.round(values.astype(np.float64), 8))
    return {
        "shape": list(values.shape),
        "sum": float(values.sum(dtype=np.float64)),
        "nnz": int(np.count_nonzero(values)),
        "rounded_sha256": hashlib.sha256(rounded.view(np.uint8)).hexdigest(),
    }


def result_signature(result) -> dict:
    group_columns = [column for column in result.obs if column != "n_cells"]
    group_keys = [
        [str(value) for value in row]
        for row in result.obs[group_columns].itertuples(index=False, name=None)
    ]
    order = np.argsort(
        np.asarray([json.dumps(key) for key in group_keys], dtype=object),
        kind="stable",
    )
    return {
        "groupby": group_columns,
        "group_keys": [group_keys[int(index)] for index in order],
        "n_cells": result.obs["n_cells"].to_numpy()[order].tolist(),
        "metrics": {
            metric: matrix_fingerprint(np.asarray(result.layers[metric])[order])
            for metric in result.layers
        },
    }


def matched_pairs(obs, treatment: str, control: str, quality: str):
    selected = obs.loc[obs["cell_quality"].astype(str).eq(quality)]
    control_rows = selected.loc[selected["perturbation"].astype(str).eq(control)]
    treatment_rows = selected.loc[
        selected["perturbation"].astype(str).eq(treatment)
    ]
    control_pairs = set(
        control_rows[list(MATCH_COLUMNS)].itertuples(index=False, name=None)
    )
    treatment_pairs = set(
        treatment_rows[list(MATCH_COLUMNS)].itertuples(index=False, name=None)
    )
    return sorted(control_pairs & treatment_pairs, key=lambda pair: tuple(map(str, pair)))


def build_tasks(database: CellDB, args: argparse.Namespace):
    obs = database.obs
    features = None
    if args.feature_limit:
        features = tuple(database.var_names[: args.feature_limit])
    tasks = []
    audit = []
    for treatment in args.treatments or DEFAULT_TREATMENTS:
        pairs = matched_pairs(obs, treatment, args.control, args.quality)
        if not pairs:
            raise ValueError(
                f"no matched cell_line/time control pairs for {treatment!r}"
            )
        pair_predicate = " OR ".join(
            "(" + " AND ".join(
                f"{quote_identifier(column)} = ?" for column in MATCH_COLUMNS
            ) + ")"
            for _ in pairs
        )
        where = (
            '"cell_quality" = ? AND "perturbation" IN (?, ?) AND '
            f"({pair_predicate})"
        )
        pair_params = tuple(value for pair in pairs for value in pair)
        task = AggregateTask(
            name=treatment,
            groupby=(*MATCH_COLUMNS, "perturbation"),
            where=where,
            params=(args.quality, args.control, treatment, *pair_params),
            metrics=("sum", "count_nonzero"),
            features=features,
        )
        task_obs = obs.loc[
            obs["cell_quality"].astype(str).eq(args.quality)
            & obs["perturbation"].astype(str).isin([args.control, treatment])
            & obs[list(MATCH_COLUMNS)].apply(tuple, axis=1).isin(pairs)
        ]
        audit.append(
            {
                "treatment": treatment,
                "matched_pairs": len(pairs),
                "treatment_cells": int(
                    task_obs["perturbation"].astype(str).eq(treatment).sum()
                ),
                "control_cells": int(
                    task_obs["perturbation"].astype(str).eq(args.control).sum()
                ),
            }
        )
        tasks.append(task)
    return tasks, audit


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def run_method(database: CellDB, tasks, method: str, batch_size: int):
    started = time.perf_counter()
    if method == "joint":
        run = database.aggregate_many(tasks, batch_size=batch_size)
        signatures = {
            name: result_signature(result) for name, result in run.results.items()
        }
        report = run.report.to_dict()
    else:
        signatures = {}
        reports = []
        for task in tasks:
            run = database.aggregate_many([task], batch_size=batch_size)
            signatures[task.name] = result_signature(run.results[task.name])
            reports.append(run.report)
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
        "signatures": signatures,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.repeats <= 0 or args.feature_limit < 0:
        raise ValueError("batch-size/repeats must be positive and feature-limit nonnegative")
    input_path = Path(args.input_h5ad).expanduser().resolve()
    database_path = Path(args.cellvault_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed_sha256 = sha256_file(input_path)
    if observed_sha256 != DATASET_SHA256:
        raise RuntimeError(
            f"dataset SHA256 mismatch: expected {DATASET_SHA256}, "
            f"got {observed_sha256}"
        )

    preparation = None
    if args.rebuild_cellvault or not (database_path / "obs.duckdb").exists():
        started = time.perf_counter()
        with CellDB.from_h5ad(
            str(input_path),
            str(database_path),
            overwrite=True,
        ) as database:
            shape = list(database.shape)
        preparation = {
            "seconds": time.perf_counter() - started,
            "shape": shape,
        }

    with CellDB.open(str(database_path)) as database:
        tasks, audit = build_tasks(database, args)
        runs = []
        generator = random.Random(args.seed)
        for repeat in range(1, args.repeats + 1):
            methods = ["independent", "joint"]
            generator.shuffle(methods)
            repeat_runs = [
                run_method(database, tasks, method, args.batch_size)
                for method in methods
            ]
            reference = repeat_runs[0]["signatures"]
            for run in repeat_runs[1:]:
                if run["signatures"] != reference:
                    raise RuntimeError(
                        f"result mismatch between {repeat_runs[0]['method']} and "
                        f"{run['method']}"
                    )
            for run in repeat_runs:
                run["repeat"] = repeat
                run.pop("signatures")
                runs.append(run)

    summary = {
        method: {
            "seconds": summarize(
                [run["seconds"] for run in runs if run["method"] == method]
            ),
            "matrix_batch_reads": summarize(
                [
                    float(run["report"]["matrix_batch_reads"])
                    for run in runs
                    if run["method"] == method
                ]
            ),
            "matrix_bytes_read": summarize(
                [
                    float(run["report"]["matrix_bytes_read"])
                    for run in runs
                    if run["method"] == method
                ]
            ),
        }
        for method in ("independent", "joint")
    }
    summary["speedup"] = (
        summary["independent"]["seconds"]["median"]
        / summary["joint"]["seconds"]["median"]
    )
    payload = {
        "benchmark": "matched shared-control aggregation",
        "input_h5ad": str(input_path),
        "cellvault_path": str(database_path),
        "dataset": {
            "title": "McFarland et al. 2020 MIX-Seq perturbation response",
            "doi": "10.1038/s41467-020-17440-w",
            "url": DATASET_URL,
            "expected_sha256": DATASET_SHA256,
            "observed_sha256": observed_sha256,
            "file_bytes": input_path.stat().st_size,
        },
        "preparation": preparation,
        "control": args.control,
        "quality": args.quality,
        "matching_columns": list(MATCH_COLUMNS),
        "features": args.feature_limit or "all",
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "task_audit": audit,
        "summary": summary,
        "runs": runs,
        "notes": [
            "Controls are matched by cell_line and time before task creation.",
            "The channel field is not a valid universal batch key here: it is missing for several drug arms and disjoint for others.",
            "CRISPR negative controls are not merged with the drug control label.",
            "matrix_bytes_read is decoded backend output, not physical disk I/O.",
            "Methods are randomized within one process; this is an application check, not a cold-cache benchmark.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
