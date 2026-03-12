#!/usr/bin/env python3
"""Cold-start benchmark for processed.h5ad cluster edit workflow.

Supports two usage patterns:
1) Single run with internal purge (requires privilege):
   mode=all, purge before each read.
2) Strict manual cold-start with reboot between phases:
   mode=extract (run after cold boot #1), then mode=writeback
   (run after cold boot #2), no internal purge.

Optional cache-buster files can be read before each phase to evict
filesystem cache without requiring purge privilege.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import anndata as ad


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
    parser = argparse.ArgumentParser(
        description="Cold-start benchmark for processed.h5ad cluster edit workflow"
    )
    parser.add_argument("--processed-h5ad", required=True, help="Path to processed.h5ad")
    parser.add_argument(
        "--output-h5ad",
        default="benchmark_outputs/cold_edit/processed_cluster2_edited.h5ad",
        help="Output path for edited h5ad",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_outputs/cold_edit/cold_edit_timings.json",
        help="Path to timing json output",
    )
    parser.add_argument(
        "--cluster-ids-path",
        default="benchmark_outputs/cold_edit/cluster2_ids.json",
        help="Intermediate file storing extracted cluster cell ids",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "extract", "writeback"],
        default="all",
        help="all: full flow; extract: only first cold-start phase; writeback: only second phase",
    )
    parser.add_argument(
        "--skip-purge",
        action="store_true",
        help="Skip internal purge (use when you do manual reboot-based cold start)",
    )
    parser.add_argument(
        "--cache-buster-before-extract",
        default="",
        help="Optional large file to sequentially read before extract phase (not timed)",
    )
    parser.add_argument(
        "--cache-buster-before-writeback",
        default="",
        help="Optional large file to sequentially read before writeback phase (not timed)",
    )
    parser.add_argument(
        "--cache-buster-chunk-mb",
        type=int,
        default=256,
        help="Chunk size in MB when reading cache-buster files",
    )
    parser.add_argument("--cluster-col", default="leiden")
    parser.add_argument("--cluster-value", default="2")
    parser.add_argument("--metadata-col", default="benchmark_metadata")
    parser.add_argument("--metadata-value", default="edited_cluster2")
    return parser.parse_args()


def run_purge() -> None:
    subprocess.run(["sync"], check=True)
    subprocess.run(["purge"], check=True)


def maybe_purge(timings: dict[str, float], stage_name: str, skip_purge: bool) -> None:
    if skip_purge:
        print(f"[SKIP]  {stage_name} (skip-purge enabled)", flush=True)
        timings[stage_name] = 0.0
        return

    with StageTimer(timings, stage_name):
        try:
            run_purge()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "purge failed due insufficient privilege. "
                "Either run with sudo/root capability, or rerun with --skip-purge "
                "and perform manual cold reboot between phases."
            ) from e


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_cache_buster(path_str: str, phase_label: str, chunk_mb: int) -> None:
    if not path_str:
        return

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"cache-buster file not found: {path}")

    chunk_size = max(1, chunk_mb) * 1024 * 1024
    total = 0
    start = time.perf_counter()
    print(f"[CACHE] start {phase_label}: reading {path}", flush=True)
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            total += len(buf)
    elapsed = time.perf_counter() - start
    print(
        f"[CACHE] done  {phase_label}: read {total / 1024**3:.2f}GB in {elapsed:.3f}s (not timed)",
        flush=True,
    )


def run_extract_phase(args: argparse.Namespace, timings: dict[str, float]) -> dict[str, Any]:
    processed_path = Path(args.processed_h5ad).expanduser().resolve()
    cluster_ids_path = Path(args.cluster_ids_path).expanduser().resolve()

    maybe_purge(timings, "c1_purge_before_read_extract", args.skip_purge)
    read_cache_buster(
        args.cache_buster_before_extract,
        phase_label="before_extract",
        chunk_mb=args.cache_buster_chunk_mb,
    )

    with StageTimer(timings, "c2_cold_read_processed_h5ad_extract"):
        adata = ad.read_h5ad(processed_path)

    with StageTimer(timings, "c3_extract_cluster2_subset"):
        if args.cluster_col not in adata.obs.columns:
            raise KeyError(
                f"Cluster column '{args.cluster_col}' not found. "
                f"Available: {list(adata.obs.columns)}"
            )
        mask = adata.obs[args.cluster_col].astype(str) == str(args.cluster_value)
        cluster_ids = adata.obs.index[mask].tolist()
        subset = adata[mask].copy()

    with StageTimer(timings, "c4_modify_subset_metadata"):
        subset.obs[args.metadata_col] = args.metadata_value

    with StageTimer(timings, "c4b_write_cluster_ids"):
        cluster_ids_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_ids_path.write_text(json.dumps(cluster_ids), encoding="utf-8")

    subset_n_obs = int(subset.n_obs)
    del subset, adata
    gc.collect()

    return {
        "cluster_ids_path": str(cluster_ids_path),
        "subset_n_obs": subset_n_obs,
    }


def run_writeback_phase(args: argparse.Namespace, timings: dict[str, float]) -> dict[str, Any]:
    processed_path = Path(args.processed_h5ad).expanduser().resolve()
    output_h5ad = Path(args.output_h5ad).expanduser().resolve()
    cluster_ids_path = Path(args.cluster_ids_path).expanduser().resolve()

    maybe_purge(timings, "c5_purge_before_read_writeback", args.skip_purge)
    read_cache_buster(
        args.cache_buster_before_writeback,
        phase_label="before_writeback",
        chunk_mb=args.cache_buster_chunk_mb,
    )

    with StageTimer(timings, "c6_cold_read_processed_h5ad_writeback"):
        adata2 = ad.read_h5ad(processed_path)

    with StageTimer(timings, "c6b_read_cluster_ids"):
        if not cluster_ids_path.exists():
            raise FileNotFoundError(
                f"cluster ids file not found: {cluster_ids_path}. "
                "Run --mode extract first."
            )
        cluster_ids = json.loads(cluster_ids_path.read_text(encoding="utf-8"))

    with StageTimer(timings, "c7_apply_metadata_to_full_object"):
        adata2.obs.loc[cluster_ids, args.metadata_col] = args.metadata_value

    with StageTimer(timings, "c8_write_back_h5ad"):
        output_h5ad.parent.mkdir(parents=True, exist_ok=True)
        adata2.write_h5ad(output_h5ad)

    return {
        "output_h5ad": str(output_h5ad),
        "subset_n_obs": int(len(cluster_ids)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    timings: dict[str, float] = {}
    result: dict[str, Any] = {
        "framework": "scanpy_cold_only",
        "mode": args.mode,
        "processed_h5ad": str(Path(args.processed_h5ad).expanduser().resolve()),
        "cluster_col": args.cluster_col,
        "cluster_value": str(args.cluster_value),
        "metadata_col": args.metadata_col,
        "metadata_value": args.metadata_value,
        "skip_purge": bool(args.skip_purge),
        "cache_buster_before_extract": args.cache_buster_before_extract,
        "cache_buster_before_writeback": args.cache_buster_before_writeback,
        "cache_buster_chunk_mb": args.cache_buster_chunk_mb,
        "timings_seconds": timings,
    }

    if args.mode in ("all", "extract"):
        extract_info = run_extract_phase(args, timings)
        result.update(extract_info)

    if args.mode in ("all", "writeback"):
        writeback_info = run_writeback_phase(args, timings)
        result.update(writeback_info)

    result["total_seconds"] = sum(timings.values())
    return result


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    result = run(args)
    wall_clock = time.perf_counter() - start

    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, result)

    print("=== Cold Processed.h5ad Benchmark Complete ===")
    for k, v in result["timings_seconds"].items():
        print(f"{k}: {v:.3f}s")
    print(f"total_stages: {result['total_seconds']:.3f}s")
    print(f"wall_clock: {wall_clock:.3f}s")
    if "subset_n_obs" in result:
        print(f"subset_n_obs: {result['subset_n_obs']}")
    if "cluster_ids_path" in result:
        print(f"cluster_ids_path: {result['cluster_ids_path']}")
    if "output_h5ad" in result:
        print(f"output_h5ad: {result['output_h5ad']}")
    print(f"timing_json: {output_json}")


if __name__ == "__main__":
    main()
