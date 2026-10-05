#!/usr/bin/env python3
"""Download and preprocess the public 10x Genomics PBMC 68k dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

DATASET_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "fresh_68k_pbmc_donor_a/"
    "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
)
DATASET_SHA256 = "3f35f37ff344bc9b32f97cd003ac986ebb9b5d7f31006c53dff4cb38da267931"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="benchmark_outputs/pbmc68k")
    parser.add_argument("--n-top-genes", type=int, default=2_000)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--max-pct-mito", type=float, default=20.0)
    parser.add_argument("--target-sum", type=float, default=10_000.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(archive_path: Path) -> None:
    if archive_path.is_file() and sha256_file(archive_path) == DATASET_SHA256:
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "CellVault"})
    with (
        urllib.request.urlopen(request) as response,
        temporary_path.open("wb") as output,
    ):
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
    observed_hash = sha256_file(temporary_path)
    if observed_hash != DATASET_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"download checksum mismatch: {observed_hash} != {DATASET_SHA256}"
        )
    os.replace(temporary_path, archive_path)


def extract_dataset(archive_path: Path, raw_dir: Path) -> Path:
    matrix_dir = raw_dir / "filtered_matrices_mex" / "hg19"
    if (matrix_dir / "matrix.mtx").is_file():
        return matrix_dir
    root = raw_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            destination = (raw_dir / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(raw_dir)
    return matrix_dir


def preprocess(
    matrix_dir: Path, output_path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    import scanpy as sc

    adata = sc.read_10x_mtx(
        matrix_dir,
        var_names="gene_symbols",
        make_unique=True,
        cache=False,
    )
    raw_shape = [int(adata.n_obs), int(adata.n_vars)]
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    adata = adata[adata.obs["pct_counts_mt"] < args.max_pct_mito].copy()
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_top_genes,
        flavor="seurat",
        subset=True,
    )
    adata.write_h5ad(output_path, compression="lzf")
    return {
        "raw_shape": raw_shape,
        "processed_shape": [int(adata.n_obs), int(adata.n_vars)],
        "nnz": int(adata.X.nnz),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_dir = output_dir / "raw"
    archive_path = raw_dir / "pbmc68k_filtered_gene_bc_matrices.tar.gz"
    output_path = output_dir / "pbmc68k_processed.h5ad"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )

    started = time.perf_counter()
    download_dataset(archive_path)
    matrix_dir = extract_dataset(archive_path, raw_dir)
    result = preprocess(matrix_dir, output_path, args)
    result.update(
        {
            "dataset_url": DATASET_URL,
            "dataset_sha256": DATASET_SHA256,
            "output_h5ad": str(output_path),
            "file_bytes": output_path.stat().st_size,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preprocess.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
