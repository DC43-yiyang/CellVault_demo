"""Tests for the multi-lineage workflow benchmark."""

from argparse import Namespace

import pytest

from cellvault import CellDB
from scripts.benchmark_lineage_workflow import parse_lineages, run_worker


def test_parse_lineages_supports_grouped_labels():
    assert parse_lineages(["T/NK=T cell|NK cell", "B=B cell"]) == [
        ("T/NK", ("T cell", "NK cell")),
        ("B", ("B cell",)),
    ]


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        (["invalid"], "expected"),
        (["T=T cell", "T=NK cell"], "duplicate lineage"),
        (["T=T cell", "NK=T cell"], "assigned to both"),
    ],
)
def test_parse_lineages_rejects_ambiguous_specs(specs, message):
    with pytest.raises(ValueError, match=message):
        parse_lineages(specs)


def test_worker_paths_produce_identical_subsets(tmp_path, small_adata):
    small_adata.obs["main_lineage"] = [
        "T cell" if index % 3 == 0 else "NK cell" if index % 3 == 1 else "B cell"
        for index in range(small_adata.n_obs)
    ]
    input_path = tmp_path / "input.h5ad"
    cellvault_path = tmp_path / "input.cvdb"
    small_adata.write_h5ad(input_path)
    with CellDB.from_anndata(small_adata, str(cellvault_path)):
        pass

    args = Namespace(
        input_h5ad=str(input_path),
        cellvault_path=str(cellvault_path),
        scratch_dir=str(tmp_path),
        column="main_lineage",
        lineage=["T/NK=T cell|NK cell", "B=B cell"],
        compression="lzf",
        run_analysis=False,
        n_comps=10,
        n_neighbors=5,
        resolution=1.0,
        seed=0,
        worker_method="",
    )

    runs = {}
    for method in (
        "anndata-direct",
        "anndata-saved",
        "cellvault-sql",
        "cellvault-batch",
    ):
        args.worker_method = method
        runs[method] = run_worker(args)

    expected = {
        name: record["subset"]
        for name, record in runs["anndata-direct"]["lineages"].items()
    }
    for run in runs.values():
        assert {
            name: record["subset"] for name, record in run["lineages"].items()
        } == expected

    saved = runs["anndata-saved"]
    assert saved["phase_seconds"]["save"] > 0
    assert saved["phase_seconds"]["reload"] > 0
    assert saved["artifact_bytes"] > 0
    assert runs["anndata-direct"]["artifact_bytes"] == 0
    assert runs["cellvault-sql"]["artifact_bytes"] == 0
    assert runs["cellvault-batch"]["artifact_bytes"] == 0
