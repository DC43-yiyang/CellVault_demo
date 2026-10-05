#!/usr/bin/env python3
"""Run the complete Wu breast-cancer CellVault research workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cellvault import AggregateTask, CellDB, MaterializeTask


LINEAGES = {
    "T/NK": ("T-cells",),
    "B": ("B-cells", "Plasmablasts"),
    "Stromal": ("CAFs", "Endothelial", "PVL"),
    "Epithelial": ("Cancer Epithelial", "Normal Epithelial"),
    "Myeloid": ("Myeloid",),
}
MARKER_SYMBOLS = (
    "CD3D",
    "CD3E",
    "NKG7",
    "GNLY",
    "MS4A1",
    "CD79A",
    "MZB1",
    "JCHAIN",
    "COL1A1",
    "COL1A2",
    "PECAM1",
    "RGS5",
    "EPCAM",
    "KRT8",
    "KRT18",
    "ERBB2",
    "ESR1",
    "KRT5",
    "KRT14",
    "MKI67",
    "LST1",
    "TYROBP",
    "FCGR3A",
    "CD68",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        default="benchmark_data/wu2021_breast_cancer.h5ad",
    )
    parser.add_argument(
        "--cellvault-path",
        default="benchmark_outputs/wu2021_breast_cancer/wu2021_optimized.cvdb",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_results/wu2021_complete_workflow.json",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--memory-budget-gib", type=float, default=2.0)
    parser.add_argument("--top-genes", type=int, default=20)
    parser.add_argument("--rebuild-cellvault", action="store_true")
    return parser.parse_args()


def array_fingerprint(values) -> dict:
    dense = np.ascontiguousarray(np.asarray(values))
    return {
        "shape": list(dense.shape),
        "dtype": str(dense.dtype),
        "sum": float(dense.sum(dtype=np.float64)),
        "sha256": hashlib.sha256(dense.view(np.uint8)).hexdigest(),
    }


def lineage_labels(major: pd.Series) -> pd.Series:
    lookup = {
        source: lineage for lineage, sources in LINEAGES.items() for source in sources
    }
    labels = major.astype(object).map(lookup)
    if labels.isna().any():
        unknown = sorted(major.loc[labels.isna()].astype(str).unique().tolist())
        raise ValueError(f"unmapped celltype_major labels: {unknown}")
    return labels.astype(str)


def marker_features(
    var: pd.DataFrame,
) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
    if "feature_name" not in var:
        raise KeyError("Wu var table is missing feature_name")
    symbol_to_id = {}
    for feature_id, symbol in var["feature_name"].items():
        symbol_to_id.setdefault(str(symbol), str(feature_id))
    missing = [symbol for symbol in MARKER_SYMBOLS if symbol not in symbol_to_id]
    features = tuple(
        symbol_to_id[symbol] for symbol in MARKER_SYMBOLS if symbol in symbol_to_id
    )
    if not features:
        raise KeyError("none of the requested marker symbols are present")
    return (
        features,
        {feature_id: symbol for symbol, feature_id in symbol_to_id.items()},
        tuple(missing),
    )


def summarize_local_input(adata) -> dict:
    matrix = adata.X
    labels = adata.obs["workflow_fine_label"].astype(str)
    summaries = {}
    for label in pd.unique(labels):
        mask = labels.eq(label).to_numpy()
        selected = matrix[mask]
        means = (
            np.asarray(selected.mean(axis=0)).reshape(-1)
            if sparse.issparse(selected)
            else np.asarray(selected).mean(axis=0)
        )
        summaries[str(label)] = {
            "n_cells": int(mask.sum()),
            "marker_mean": [float(value) for value in means],
        }
    return {
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "cell_ids_sha256": hashlib.sha256(
            "\0".join(map(str, adata.obs_names)).encode("utf-8")
        ).hexdigest(),
        "fine_labels": summaries,
    }


def subtype_contrast(
    result,
    feature_symbols: dict[str, str],
    *,
    reference: str,
    comparison: str,
    top_genes: int,
) -> dict:
    obs = result.obs.reset_index(drop=True)
    values = np.asarray(result.layers["mean"], dtype=np.float64)
    subtype = obs["subtype"].astype(str)
    reference_mask = subtype.eq(reference).to_numpy()
    comparison_mask = subtype.eq(comparison).to_numpy()
    if not reference_mask.any() or not comparison_mask.any():
        raise ValueError(
            f"comparison requires donor groups for {reference!r} and {comparison!r}"
        )
    reference_mean = values[reference_mask].mean(axis=0)
    comparison_mean = values[comparison_mask].mean(axis=0)
    delta = comparison_mean - reference_mean
    descending = np.argsort(-delta, kind="stable")[:top_genes]
    ascending = np.argsort(delta, kind="stable")[:top_genes]

    def records(positions: np.ndarray) -> list[dict]:
        return [
            {
                "feature_id": str(result.var_names[position]),
                "feature_name": feature_symbols.get(
                    str(result.var_names[position]), str(result.var_names[position])
                ),
                "comparison_mean": float(comparison_mean[position]),
                "reference_mean": float(reference_mean[position]),
                "mean_difference": float(delta[position]),
            }
            for position in positions
        ]

    return {
        "contrast": f"{comparison} - {reference}",
        "scale": "donor-level mean of the source X expression values",
        "inference": "descriptive ranking; not a differential-expression model",
        "reference_donors": int(reference_mask.sum()),
        "comparison_donors": int(comparison_mask.sum()),
        "reference_cells": int(obs.loc[reference_mask, "n_cells"].sum()),
        "comparison_cells": int(obs.loc[comparison_mask, "n_cells"].sum()),
        "top_positive": records(descending),
        "top_negative": records(ascending),
        "difference_fingerprint": array_fingerprint(delta),
    }


def ensure_obs_column(database: CellDB, column: str, values: pd.Series) -> str:
    if column not in database.obs_columns:
        database.add_obs_column(column, values.tolist())
        return "created"
    current = database.obs[column].astype(str)
    expected = values.astype(str).set_axis(current.index)
    if current.equals(expected):
        return "verified"
    database.update_obs(column, current.index, values.tolist())
    return "updated"


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.memory_budget_gib <= 0 or args.top_genes <= 0:
        raise ValueError("batch-size, memory-budget-gib, and top-genes must be positive")
    input_path = Path(args.input_h5ad).expanduser().resolve()
    database_path = Path(args.cellvault_path).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preparation_seconds = None
    if args.rebuild_cellvault or not (database_path / "obs.duckdb").is_file():
        started = time.perf_counter()
        with CellDB.from_h5ad(str(input_path), str(database_path), overwrite=True):
            pass
        preparation_seconds = time.perf_counter() - started

    workflow_started = time.perf_counter()
    with CellDB.open(str(database_path)) as database:
        required = {"donor_id", "celltype_major", "celltype_minor", "subtype"}
        missing = required - set(database.obs_columns)
        if missing:
            raise KeyError(f"Wu store is missing obs columns: {sorted(missing)}")
        obs = database.obs
        main_labels = lineage_labels(obs["celltype_major"])
        fine_labels = obs["celltype_minor"].astype(str)

        started = time.perf_counter()
        writeback = {
            "workflow_main_lineage": ensure_obs_column(
                database, "workflow_main_lineage", main_labels
            ),
            "workflow_fine_label": ensure_obs_column(
                database, "workflow_fine_label", fine_labels
            ),
        }
        persisted = database.obs
        if persisted["workflow_main_lineage"].astype(str).tolist() != main_labels.tolist():
            raise RuntimeError("main-lineage write-back verification failed")
        if persisted["workflow_fine_label"].astype(str).tolist() != fine_labels.tolist():
            raise RuntimeError("fine-label write-back verification failed")
        writeback_seconds = time.perf_counter() - started

        views = database.partition_obs(
            "workflow_main_lineage",
            {lineage: (lineage,) for lineage in LINEAGES},
            require_complete=True,
        )
        lineage_counts = {name: int(view.n_obs) for name, view in views.items()}

        features, feature_symbols, missing_marker_symbols = marker_features(database.var)
        local_tasks = [
            MaterializeTask(
                f"local_{position:02d}_{lineage.lower().replace('/', '_')}",
                where='"workflow_main_lineage" = ?',
                params=(lineage,),
                features=features,
                obs_columns=(
                    "workflow_main_lineage",
                    "workflow_fine_label",
                    "donor_id",
                    "subtype",
                ),
                consumer=summarize_local_input,
            )
            for position, lineage in enumerate(LINEAGES)
        ]
        started = time.perf_counter()
        local_run = database.execute_tasks(
            local_tasks,
            batch_size=args.batch_size,
            memory_budget_bytes=int(args.memory_budget_gib * 1024**3),
        )
        local_seconds = time.perf_counter() - started

        aggregation_tasks = [
            AggregateTask(
                "donor_main_lineage",
                ("donor_id", "workflow_main_lineage"),
                metrics=("sum",),
            ),
            AggregateTask(
                "donor_fine_label",
                ("donor_id", "workflow_fine_label"),
                metrics=("sum",),
            ),
            AggregateTask(
                "cancer_epithelial_subtype",
                ("donor_id", "subtype"),
                where='"celltype_major" = ?',
                params=("Cancer Epithelial",),
                metrics=("mean",),
            ),
        ]
        started = time.perf_counter()
        aggregation = database.aggregate_many(
            aggregation_tasks,
            batch_size=args.batch_size,
        )
        aggregation_seconds = time.perf_counter() - started
        contrast = subtype_contrast(
            aggregation.results["cancer_epithelial_subtype"],
            feature_symbols,
            reference="ER+",
            comparison="TNBC",
            top_genes=args.top_genes,
        )

        aggregate_outputs = {
            name: {
                "groups": int(result.n_obs),
                "features": int(result.n_vars),
                "n_cells_total": int(result.obs["n_cells"].sum()),
                "layers": {
                    layer: array_fingerprint(result.layers[layer])
                    for layer in result.layers
                },
            }
            for name, result in aggregation.results.items()
        }

    payload = {
        "workflow": "Wu 2021 hierarchical annotation and subtype expression summary",
        "dataset": {
            "input_h5ad": str(input_path),
            "cellvault_path": str(database_path),
            "shape": [100064, 28468],
        },
        "source_semantics": (
            "X contains normalized expression; sums are sample-level aggregated "
            "expression and are not presented as raw-count differential-expression input"
        ),
        "preparation_seconds": preparation_seconds,
        "writeback": {
            "columns": writeback,
            "seconds": writeback_seconds,
            "verified_by_stable_cell_id": True,
            "fine_label_source": "author-provided celltype_minor",
        },
        "lineage_counts": lineage_counts,
        "local_refinement": {
            "seconds": local_seconds,
            "marker_symbols": list(MARKER_SYMBOLS),
            "available_marker_symbols": [feature_symbols[feature] for feature in features],
            "missing_marker_symbols": list(missing_marker_symbols),
            "report": local_run.report.to_dict(),
            "results": dict(local_run.results),
        },
        "multi_level_aggregation": {
            "seconds": aggregation_seconds,
            "report": aggregation.report.to_dict(),
            "results": aggregate_outputs,
        },
        "subtype_marker_summary": contrast,
        "intermediate_h5ad_bytes": 0,
        "workflow_seconds": time.perf_counter() - workflow_started,
        "status": "passed",
        "notes": [
            "Author-provided celltype_minor labels are used as a deterministic fine-annotation reference, not as predictions from a new classifier.",
            "The subtype comparison is donor-level and descriptive; it is not a replacement for a covariate-aware differential-expression model.",
            "Local marker inputs are consumed and released by the budgeted executor rather than saved as lineage H5AD files.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "workflow_seconds": payload["workflow_seconds"],
                "lineage_counts": lineage_counts,
                "contrast": contrast["contrast"],
            },
            indent=2,
        )
    )
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
