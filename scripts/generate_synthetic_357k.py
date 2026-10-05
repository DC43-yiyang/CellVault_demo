#!/usr/bin/env python3
"""Generate a deterministic 357k-cell sparse h5ad benchmark dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-h5ad",
        default="benchmark_data/synthetic_357k.h5ad",
    )
    parser.add_argument("--n-cells", type=int, default=357_000)
    parser.add_argument("--n-genes", type=int, default=2_000)
    parser.add_argument("--nnz-per-cell", type=int, default=100)
    parser.add_argument("--target-cells", type=int, default=14_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-batch-size", type=int, default=4_096)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_cells < 1 or args.n_genes < 1:
        raise ValueError("n-cells and n-genes must be positive")
    if not 1 <= args.nnz_per_cell <= args.n_genes:
        raise ValueError("nnz-per-cell must be between 1 and n-genes")
    if not 1 <= args.target_cells <= args.n_cells:
        raise ValueError("target-cells must be between 1 and n-cells")
    if args.row_batch_size < 1:
        raise ValueError("row-batch-size must be positive")


def generate_matrix(args: argparse.Namespace, rng: np.random.Generator):
    total_nonzero = args.n_cells * args.nnz_per_cell
    indptr = np.arange(
        0,
        total_nonzero + 1,
        args.nnz_per_cell,
        dtype=np.int64,
    )
    indices = np.empty(total_nonzero, dtype=np.int32)
    values = np.empty(total_nonzero, dtype=np.float32)
    gene_order = rng.permutation(args.n_genes).astype(np.int32)
    within_row = np.arange(args.nnz_per_cell, dtype=np.int64)

    for row_start in range(0, args.n_cells, args.row_batch_size):
        row_stop = min(row_start + args.row_batch_size, args.n_cells)
        row_count = row_stop - row_start
        offsets = rng.integers(0, args.n_genes, size=row_count)
        positions = (offsets[:, None] + within_row[None, :]) % args.n_genes
        block_indices = gene_order[positions]
        block_indices.sort(axis=1)
        block_values = rng.negative_binomial(
            n=2,
            p=0.5,
            size=(row_count, args.nnz_per_cell),
        ).astype(np.float32)
        block_values += 1
        flat_start = row_start * args.nnz_per_cell
        flat_stop = row_stop * args.nnz_per_cell
        indices[flat_start:flat_stop] = block_indices.reshape(-1)
        values[flat_start:flat_stop] = block_values.reshape(-1)

    return sparse.csr_matrix(
        (values, indices, indptr),
        shape=(args.n_cells, args.n_genes),
    )


def generate_obs(
    args: argparse.Namespace, matrix, rng: np.random.Generator
) -> pd.DataFrame:
    other_clusters = np.asarray([str(index) for index in range(20) if index != 2])
    leiden = rng.choice(other_clusters, size=args.n_cells)
    target_positions = np.sort(
        rng.choice(args.n_cells, size=args.target_cells, replace=False)
    )
    leiden[target_positions] = "2"

    cell_type = rng.choice(
        np.asarray(["B cell", "NK cell", "myeloid", "stromal"]),
        size=args.n_cells,
    )
    cell_type[target_positions] = "T cell"
    n_counts = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int32)

    return pd.DataFrame(
        {
            "leiden": pd.Categorical(leiden),
            "cell_type": pd.Categorical(cell_type),
            "sample": pd.Categorical(
                [f"sample_{index}" for index in rng.integers(1, 17, args.n_cells)]
            ),
            "donor": pd.Categorical(
                [f"donor_{index}" for index in rng.integers(1, 9, args.n_cells)]
            ),
            "n_counts": n_counts,
            "percent_mito": rng.uniform(0.0, 0.2, args.n_cells).astype(np.float32),
        },
        index=[f"cell_{index:07d}" for index in range(args.n_cells)],
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_path = Path(args.output_h5ad).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    matrix = generate_matrix(args, rng)
    obs = generate_obs(args, matrix, rng)
    var = pd.DataFrame(
        {
            "highly_variable": np.ones(args.n_genes, dtype=bool),
        },
        index=[f"gene_{index:05d}" for index in range(args.n_genes)],
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["synthetic_benchmark"] = {
        "seed": args.seed,
        "target_cluster": "2",
        "target_cells": args.target_cells,
        "nnz_per_cell": args.nnz_per_cell,
    }
    compression = None if args.compression == "none" else args.compression
    adata.write_h5ad(output_path, compression=compression)
    elapsed = time.perf_counter() - started

    result = {
        "output_h5ad": str(output_path),
        "n_cells": args.n_cells,
        "n_genes": args.n_genes,
        "nnz": int(matrix.nnz),
        "target_cells": args.target_cells,
        "file_bytes": output_path.stat().st_size,
        "elapsed_seconds": elapsed,
        "density": matrix.nnz / math.prod(matrix.shape),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
