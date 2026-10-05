#!/usr/bin/env python3
"""Benchmark CellVault workflow on a large h5ad dataset.

Measures two workflows:
1) Pipeline: from_h5ad -> pca/neighbors/umap/leiden -> export h5ad
2) Cluster edit flow: open -> SQL select -> SQL metadata updates -> close
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# Keep matplotlib cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

from cellvault import CellDB
from cellvault.tools import leiden, neighbors, pca, umap


class StageTimer:
    def __init__(self, timings: dict[str, float], name: str):
        self.timings = timings
        self.name = name
        self.start = 0.0

    def __enter__(self):
        print(f"[START] {self.name}", flush=True)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        self.timings[self.name] = elapsed
        print(f"[DONE]  {self.name}: {elapsed:.3f}s", flush=True)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CellVault workflow")
    parser.add_argument("--input-h5ad", required=True, help="Path to input h5ad")
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/cellvault",
        help="Directory to write cvdb artifacts and timing json",
    )
    parser.add_argument(
        "--cvdb-path",
        default="benchmark_outputs/cellvault/integrated.cvdb",
        help="Output CellVault database path",
    )
    parser.add_argument("--n-comps", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--cluster-col", default="leiden")
    parser.add_argument("--cluster-value", default="2")
    parser.add_argument("--metadata-col", default="benchmark_metadata")
    parser.add_argument("--metadata-value", default="edited_cluster2")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--new-cell-type", default="cluster2_reannotated")
    parser.add_argument(
        "--default-col-value",
        default="",
        help="Default value used when creating new obs columns",
    )
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def ensure_obs_column(cdb: CellDB, column: str, default_value: str) -> None:
    if column in cdb.obs_columns:
        return
    fill = [default_value] * cdb.n_obs
    cdb.add_obs_column(column, fill)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_h5ad).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cvdb_path = Path(args.cvdb_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cvdb_path.parent.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    results: dict[str, Any] = {
        "framework": "cellvault",
        "input_h5ad": str(input_path),
        "output_dir": str(output_dir),
        "cvdb_path": str(cvdb_path),
        "params": {
            "n_comps": args.n_comps,
            "n_neighbors": args.n_neighbors,
            "resolution": args.resolution,
            "cluster_col": args.cluster_col,
            "cluster_value": args.cluster_value,
            "metadata_col": args.metadata_col,
            "metadata_value": args.metadata_value,
            "cell_type_col": args.cell_type_col,
            "new_cell_type": args.new_cell_type,
            "default_col_value": args.default_col_value,
        },
        "timings_seconds": timings,
    }

    processed_h5ad_path = output_dir / "cellvault_processed.h5ad"

    # Workflow A: build db + run pipeline
    with StageTimer(timings, "a1_from_h5ad_to_cvdb"):
        cdb = CellDB.from_h5ad(str(input_path), str(cvdb_path), overwrite=True)

    results["dataset"] = {
        "n_obs": int(cdb.n_obs),
        "n_vars": int(cdb.n_vars),
    }

    with StageTimer(timings, "a2_run_pca"):
        pca_key = pca(cdb, n_comps=args.n_comps)

    with StageTimer(timings, "a3_run_neighbors"):
        neighbors_key = neighbors(cdb, n_neighbors=args.n_neighbors, use_rep=pca_key)

    with StageTimer(timings, "a4_run_umap"):
        umap_key = umap(cdb)

    with StageTimer(timings, "a5_run_leiden"):
        leiden_key = leiden(cdb, resolution=args.resolution)

    with StageTimer(timings, "a6_export_processed_h5ad"):
        cdb.to_h5ad(str(processed_h5ad_path))

    with StageTimer(timings, "a7_close_after_pipeline"):
        cdb.close()

    # Workflow B: SQL subset -> modify in place (db persistence via close)
    with StageTimer(timings, "b1_open_cvdb_for_edit"):
        cdb_edit = CellDB.open(str(cvdb_path))

    if args.cluster_col not in cdb_edit.obs_columns:
        cdb_edit.close()
        raise KeyError(
            f"Cluster column '{args.cluster_col}' not found in obs. "
            f"Available columns: {cdb_edit.obs_columns}"
        )
    predicate = f"CAST({quote_identifier(args.cluster_col)} AS VARCHAR) = ?"
    predicate_params = [str(args.cluster_value)]

    with StageTimer(timings, "b2_query_cluster"):
        cluster_view = cdb_edit.query_obs(
            predicate,
            predicate_params,
            columns=[],
        )
        subset_n_obs = cluster_view.n_obs

    if subset_n_obs == 0:
        cdb_edit.close()
        raise ValueError(
            f"No cells found for {args.cluster_col} == '{args.cluster_value}'. "
            "Please check cluster value."
        )

    with StageTimer(timings, "b3_ensure_metadata_column"):
        ensure_obs_column(cdb_edit, args.metadata_col, args.default_col_value)

    with StageTimer(timings, "b4_update_metadata_cluster"):
        updated_metadata = cdb_edit.update_obs_where(
            args.metadata_col,
            args.metadata_value,
            predicate,
            predicate_params,
        )

    with StageTimer(timings, "b5_ensure_cell_type_column"):
        ensure_obs_column(cdb_edit, args.cell_type_col, args.default_col_value)

    with StageTimer(timings, "b6_update_cell_type_cluster"):
        updated_cell_type = cdb_edit.update_obs_where(
            args.cell_type_col,
            args.new_cell_type,
            predicate,
            predicate_params,
        )

    if updated_metadata != subset_n_obs or updated_cell_type != subset_n_obs:
        cdb_edit.close()
        raise RuntimeError(
            "SQL update count changed after selection: "
            f"selected={subset_n_obs}, metadata={updated_metadata}, "
            f"cell_type={updated_cell_type}"
        )

    with StageTimer(timings, "b7_close_after_edit"):
        cdb_edit.close()

    total = sum(timings.values())
    results["pipeline_keys"] = {
        "pca_key": pca_key,
        "neighbors_key": neighbors_key,
        "umap_key": umap_key,
        "leiden_key": leiden_key,
    }
    results["cluster_edit"] = {
        "cluster_col": args.cluster_col,
        "cluster_value": str(args.cluster_value),
        "subset_n_obs": subset_n_obs,
    }
    results["artifacts"] = {
        "cvdb_path": str(cvdb_path),
        "processed_h5ad": str(processed_h5ad_path),
        "timing_json": str(output_dir / "cellvault_timings.json"),
    }
    results["total_seconds"] = total

    write_json(output_dir / "cellvault_timings.json", results)
    return results


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    results = run_benchmark(args)
    elapsed = time.perf_counter() - start

    print("=== CellVault Benchmark Complete ===")
    for k, v in results["timings_seconds"].items():
        print(f"{k}: {v:.3f}s")
    print(f"total_stages: {results['total_seconds']:.3f}s")
    print(f"wall_clock: {elapsed:.3f}s")
    print(f"subset_n_obs: {results['cluster_edit']['subset_n_obs']}")
    print(f"timing_json: {results['artifacts']['timing_json']}")


if __name__ == "__main__":
    main()
