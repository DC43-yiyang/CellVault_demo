"""Thin workflow adapters built from CellVault's public task APIs."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .execution import AggregateTask, ExecutionReport, MaterializeTask


def _quote_sql_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _stable_values(values: pd.Series, *, column: str) -> tuple[Hashable, ...]:
    if values.isna().any():
        raise ValueError(f"{column!r} contains missing values")
    unique = pd.unique(values)
    result = tuple(
        value.item() if isinstance(value, np.generic) else value
        for value in unique
    )
    if any(not isinstance(value, Hashable) for value in result):
        raise TypeError(f"{column!r} values must be hashable")
    return result


@dataclass(frozen=True, slots=True)
class SampleInputRun:
    """Per-sample cell-level inputs plus their shared execution report."""

    results: Mapping[Hashable, Any]
    report: ExecutionReport
    task_names: Mapping[Hashable, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(
            self,
            "task_names",
            MappingProxyType(dict(self.task_names)),
        )


@dataclass(frozen=True, slots=True)
class MultiModalAggregationRun:
    """Results from applying one cohort/grouping definition to each modality."""

    results: Mapping[str, ad.AnnData]
    reports: Mapping[str, ExecutionReport]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(self, "reports", MappingProxyType(dict(self.reports)))

    @property
    def source_scan_count(self) -> int:
        """Total scans across physically separate modality stores."""
        return sum(report.source_scan_count for report in self.reports.values())

    @property
    def matrix_batch_reads(self) -> int:
        """Total logical matrix reads across modality stores."""
        return sum(report.matrix_batch_reads for report in self.reports.values())


def prepare_communication_inputs(
    target,
    *,
    sample_column: str,
    cell_type_column: str,
    source: str = "X",
    features: Sequence[Hashable] | Hashable | None = None,
    obs_columns: Sequence[str] = (),
    batch_size: int = 4096,
    memory_budget_bytes: int | None = None,
    consumer: Callable[[ad.AnnData], Any] | None = None,
) -> SampleInputRun:
    """Prepare one cell-level input per observed sample through shared scans.

    The adapter deliberately does not replace a communication method with group
    summaries. Pass a ``consumer`` only when the downstream method can consume
    each complete per-sample AnnData independently.
    """
    if not isinstance(sample_column, str) or not sample_column:
        raise ValueError("sample_column must be a non-empty string")
    if not isinstance(cell_type_column, str) or not cell_type_column:
        raise ValueError("cell_type_column must be a non-empty string")
    if isinstance(obs_columns, str):
        raise TypeError("obs_columns must be a sequence, not a string")
    obs = target.obs
    available_columns = set(obs.columns)
    required_columns = (sample_column, cell_type_column, *tuple(obs_columns))
    missing_columns = [
        column for column in required_columns if column not in available_columns
    ]
    if missing_columns:
        raise KeyError(f"obs columns not found: {missing_columns}")

    samples = _stable_values(obs[sample_column], column=sample_column)
    if not samples:
        raise ValueError("no samples are available in the target")
    selected_obs_columns = tuple(dict.fromkeys(required_columns))
    task_names = {
        sample: f"communication_sample_{position:04d}"
        for position, sample in enumerate(samples)
    }
    tasks = [
        MaterializeTask(
            name=task_names[sample],
            where=f"{_quote_sql_identifier(sample_column)} = ?",
            params=(sample,),
            source=source,
            features=features,
            obs_columns=selected_obs_columns,
            consumer=consumer,
        )
        for sample in samples
    ]
    run = target.execute_tasks(
        tasks,
        batch_size=batch_size,
        memory_budget_bytes=memory_budget_bytes,
    )
    return SampleInputRun(
        results={sample: run.results[task_names[sample]] for sample in samples},
        report=run.report,
        task_names=task_names,
    )


def leave_one_out_pseudobulk(
    pseudobulk: ad.AnnData,
    *,
    donor_column: str,
) -> Mapping[Hashable, ad.AnnData]:
    """Create leave-one-donor datasets from an existing group-level result.

    This operates only on the small pseudobulk object and never returns to the
    cell-level matrix.
    """
    if not isinstance(pseudobulk, ad.AnnData):
        raise TypeError("pseudobulk must be an AnnData object")
    if donor_column not in pseudobulk.obs:
        raise KeyError(f"pseudobulk obs column {donor_column!r} not found")
    donors = _stable_values(pseudobulk.obs[donor_column], column=donor_column)
    if len(donors) < 2:
        raise ValueError("leave-one-out analysis requires at least two donors")

    results = {
        donor: pseudobulk[
            ~pseudobulk.obs[donor_column].eq(donor).to_numpy()
        ].copy()
        for donor in donors
    }
    return MappingProxyType(results)


def aggregate_modalities(
    modalities: Mapping[str, Any],
    *,
    groupby: Sequence[str] | str,
    sources: Mapping[str, str] | None = None,
    features: Mapping[str, Sequence[Hashable] | Hashable | None] | None = None,
    metrics: Sequence[str] | str = ("sum",),
    where: str = "TRUE",
    params: Sequence[Any] = (),
    batch_size: int = 4096,
) -> MultiModalAggregationRun:
    """Apply one cohort and grouping definition to separate modality stores.

    Modalities are intentionally scanned separately. The adapter reuses the
    cohort declaration without claiming that RNA, ADT, or other matrices share
    physical storage or feature axes.
    """
    if not isinstance(modalities, Mapping) or not modalities:
        raise ValueError("modalities must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in modalities):
        raise ValueError("modality names must be non-empty strings")
    unknown_sources = set(sources or ()) - set(modalities)
    unknown_features = set(features or ()) - set(modalities)
    if unknown_sources:
        raise KeyError(
            f"sources contain unknown modalities: {sorted(unknown_sources)}"
        )
    if unknown_features:
        raise KeyError(
            f"features contain unknown modalities: {sorted(unknown_features)}"
        )

    results: dict[str, ad.AnnData] = {}
    reports: dict[str, ExecutionReport] = {}
    for name, target in modalities.items():
        task = AggregateTask(
            name=name,
            groupby=groupby,
            source="X" if sources is None else sources.get(name, "X"),
            features=None if features is None else features.get(name),
            metrics=metrics,
            where=where,
            params=params,
        )
        run = target.aggregate_many([task], batch_size=batch_size)
        results[name] = run.results[name]
        reports[name] = run.report
    return MultiModalAggregationRun(results=results, reports=reports)


__all__ = [
    "MultiModalAggregationRun",
    "SampleInputRun",
    "aggregate_modalities",
    "leave_one_out_pseudobulk",
    "prepare_communication_inputs",
]
