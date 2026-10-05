#!/usr/bin/env python3
"""Validate one cohort definition across public CITE-seq RNA and ADT stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from cellvault import CellDB, aggregate_modalities


DATASET_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/3.1.0/5k_pbmc_protein_v3/"
    "5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5"
)
DATASET_SHA256 = "3b290ad9605b96974c9c16e5ae3427e5e5c496a66e55223df135b388b6d61417"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        default="benchmark_data/5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/pbmc5k_citeseq",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_results/pbmc5k_citeseq_multimodal.json",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
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


def read_modalities(path: Path) -> tuple[ad.AnnData, ad.AnnData, dict]:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError(
            "this benchmark requires the optional scanpy dependency"
        ) from exc

    combined = sc.read_10x_h5(path, gex_only=False)
    combined.var_names_make_unique()
    feature_types = combined.var["feature_types"].astype(str)
    rna = combined[:, feature_types.eq("Gene Expression")].copy()
    adt = combined[:, feature_types.eq("Antibody Capture")].copy()
    if rna.n_obs != adt.n_obs or not rna.obs_names.equals(adt.obs_names):
        raise RuntimeError("RNA and ADT observations are not aligned")

    rna_totals = np.asarray(rna.X.sum(axis=1)).reshape(-1)
    first, second = np.quantile(rna_totals, [1 / 3, 2 / 3])
    labels = np.select(
        [rna_totals <= first, rna_totals <= second],
        ["lower_rna_library", "middle_rna_library"],
        default="upper_rna_library",
    )
    categories = [
        "lower_rna_library",
        "middle_rna_library",
        "upper_rna_library",
    ]
    cohort = pd.Categorical(labels, categories=categories, ordered=True)
    rna.obs["qc_cohort"] = cohort
    adt.obs["qc_cohort"] = cohort.copy()
    return rna, adt, {
        "definition": "tertiles of per-cell RNA library size",
        "thresholds": [float(first), float(second)],
        "cell_counts": {
            label: int(np.count_nonzero(labels == label)) for label in categories
        },
    }


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def direct_reference(adata: ad.AnnData) -> dict[str, dict[str, np.ndarray | int]]:
    reference = {}
    for cohort in adata.obs["qc_cohort"].cat.categories:
        mask = adata.obs["qc_cohort"].eq(cohort).to_numpy()
        matrix = dense(adata.X[mask])
        reference[str(cohort)] = {
            "n_cells": int(mask.sum()),
            "sum": matrix.sum(axis=0, dtype=np.float64),
            "mean": matrix.mean(axis=0, dtype=np.float64),
            "count_nonzero": np.count_nonzero(matrix, axis=0),
        }
    return reference


def validate_result(result: ad.AnnData, reference: dict) -> None:
    observed_groups = result.obs["qc_cohort"].astype(str).tolist()
    if observed_groups != list(reference):
        raise RuntimeError("modality cohort order differs from the direct reference")
    for position, cohort in enumerate(observed_groups):
        expected = reference[cohort]
        if int(result.obs.iloc[position]["n_cells"]) != expected["n_cells"]:
            raise RuntimeError(f"cell count mismatch for cohort {cohort!r}")
        np.testing.assert_allclose(
            result.layers["sum"][position],
            expected["sum"],
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result.layers["mean"][position],
            expected["mean"],
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_array_equal(
            result.layers["count_nonzero"][position],
            expected["count_nonzero"],
        )


def result_fingerprint(result: ad.AnnData) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(result.obs["qc_cohort"].astype(str)).encode())
    digest.update(np.ascontiguousarray(result.obs["n_cells"].to_numpy()).view(np.uint8))
    for metric in ("sum", "mean", "count_nonzero"):
        values = np.round(np.asarray(result.layers[metric], dtype=np.float64), 8)
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("batch-size and repeats must be positive")
    input_path = Path(args.input_h5).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dataset(input_path)
    rna, adt, cohort_definition = read_modalities(input_path)
    references = {"rna": direct_reference(rna), "adt": direct_reference(adt)}

    stores = {"rna": output_dir / "rna.cvdb", "adt": output_dir / "adt.cvdb"}
    preparation = {}
    for name, data in (("rna", rna), ("adt", adt)):
        if args.rebuild_cellvault or not (stores[name] / "obs.duckdb").exists():
            started = time.perf_counter()
            with CellDB.from_anndata(
                data,
                str(stores[name]),
                overwrite=True,
            ):
                pass
            preparation[name] = time.perf_counter() - started

    runs = []
    with (
        CellDB.open(str(stores["rna"])) as rna_db,
        CellDB.open(str(stores["adt"])) as adt_db,
    ):
        for repeat in range(1, args.repeats + 1):
            started = time.perf_counter()
            run = aggregate_modalities(
                {"rna": rna_db, "adt": adt_db},
                groupby="qc_cohort",
                metrics=("sum", "mean", "count_nonzero"),
                batch_size=args.batch_size,
            )
            elapsed = time.perf_counter() - started
            for modality in ("rna", "adt"):
                validate_result(run.results[modality], references[modality])
            runs.append(
                {
                    "repeat": repeat,
                    "seconds": elapsed,
                    "source_scan_count": run.source_scan_count,
                    "matrix_batch_reads": run.matrix_batch_reads,
                    "reports": {
                        modality: report.to_dict()
                        for modality, report in run.reports.items()
                    },
                    "fingerprints": {
                        modality: result_fingerprint(result)
                        for modality, result in run.results.items()
                    },
                }
            )

    fingerprints = {json.dumps(run["fingerprints"], sort_keys=True) for run in runs}
    if len(fingerprints) != 1:
        raise RuntimeError("multimodal output changed between repeats")
    times = [run["seconds"] for run in runs]
    payload = {
        "benchmark": "same cohort across separate CITE-seq modalities",
        "dataset": {
            "title": "10x 5k PBMCs from a healthy donor with cell-surface proteins",
            "url": DATASET_URL,
            "sha256": DATASET_SHA256,
            "file_bytes": input_path.stat().st_size,
            "n_obs": rna.n_obs,
            "rna_features": rna.n_vars,
            "adt_features": adt.n_vars,
        },
        "cohort": cohort_definition,
        "configuration": {
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "metrics": ["sum", "mean", "count_nonzero"],
        },
        "preparation_seconds": preparation or None,
        "summary": {
            "seconds": {
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
            },
            "source_scan_count": runs[0]["source_scan_count"],
            "matrix_batch_reads": runs[0]["matrix_batch_reads"],
        },
        "validation": {
            "status": "passed",
            "same_cell_ids": rna.obs_names.equals(adt.obs_names),
            "separate_feature_axes": not rna.var_names.equals(adt.var_names),
            "direct_reference_rtol": 1e-6,
            "direct_reference_atol": 1e-8,
        },
        "runs": runs,
        "notes": [
            "RNA and ADT remain separate matrices and are each read once per run.",
            "Only the cohort declaration is reused; no cross-modality fusion is implied.",
            "QC tertiles are a controlled execution validation, not a biological grouping claim.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
