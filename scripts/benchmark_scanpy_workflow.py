#!/usr/bin/env python3
"""Benchmark traditional Scanpy workflow on a large h5ad dataset.

Measures two workflows:
1) Analysis pipeline: read -> pca -> neighbors -> umap -> leiden -> save
2) Cluster edit flow: read processed -> subset cluster -> modify metadata -> save
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

import anndata as ad
import scanpy as sc


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
    parser = argparse.ArgumentParser(description="Benchmark traditional Scanpy workflow")
    parser.add_argument("--input-h5ad", required=True, help="Path to input h5ad")
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/scanpy",
        help="Directory to write processed files and timing json",
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
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_h5ad).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    results: dict[str, Any] = {
        "framework": "scanpy",
        "input_h5ad": str(input_path),
        "output_dir": str(output_dir),
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
        },
        "timings_seconds": timings,
    }

    processed_path = output_dir / "scanpy_processed.h5ad"
    edited_subset_path = output_dir / f"scanpy_cluster_{args.cluster_value}_edited.h5ad"

    # Workflow A: traditional scanpy analysis pipeline
    with StageTimer(timings, "a1_read_input_h5ad"):
        adata = ad.read_h5ad(input_path)

    results["dataset"] = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }

    with StageTimer(timings, "a2_run_pca"):
        sc.tl.pca(adata, n_comps=args.n_comps)

    with StageTimer(timings, "a3_run_neighbors"):
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors)

    with StageTimer(timings, "a4_run_umap"):
        sc.tl.umap(adata)

    with StageTimer(timings, "a5_run_leiden"):
        sc.tl.leiden(adata, resolution=args.resolution)

    with StageTimer(timings, "a6_save_processed_h5ad"):
        adata.write_h5ad(processed_path)

    del adata

    # Workflow B: read -> subset cluster 2 -> modify -> save
    with StageTimer(timings, "b1_read_processed_h5ad"):
        edited = ad.read_h5ad(processed_path)

    with StageTimer(timings, "b2_subset_cluster"):
        if args.cluster_col not in edited.obs.columns:
            raise KeyError(
                f"Cluster column '{args.cluster_col}' not found in obs. "
                f"Available columns: {list(edited.obs.columns)}"
            )
        mask = edited.obs[args.cluster_col].astype(str) == str(args.cluster_value)
        subset = edited[mask].copy()

    results["cluster_edit"] = {
        "cluster_col": args.cluster_col,
        "cluster_value": str(args.cluster_value),
        "subset_n_obs": int(subset.n_obs),
    }

    with StageTimer(timings, "b3_modify_subset_metadata"):
        subset.obs[args.metadata_col] = args.metadata_value
        subset.obs[args.cell_type_col] = args.new_cell_type

    with StageTimer(timings, "b4_save_edited_subset_h5ad"):
        subset.write_h5ad(edited_subset_path)

    total = sum(timings.values())
    results["artifacts"] = {
        "processed_h5ad": str(processed_path),
        "edited_subset_h5ad": str(edited_subset_path),
        "timing_json": str(output_dir / "scanpy_timings.json"),
    }
    results["total_seconds"] = total

    write_json(output_dir / "scanpy_timings.json", results)
    return results


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    results = run_benchmark(args)
    elapsed = time.perf_counter() - start

    print("=== Scanpy Benchmark Complete ===")
    for k, v in results["timings_seconds"].items():
        print(f"{k}: {v:.3f}s")
    print(f"total_stages: {results['total_seconds']:.3f}s")
    print(f"wall_clock: {elapsed:.3f}s")
    print(f"subset_n_obs: {results['cluster_edit']['subset_n_obs']}")
    print(f"timing_json: {results['artifacts']['timing_json']}")


if __name__ == "__main__":
    main()
