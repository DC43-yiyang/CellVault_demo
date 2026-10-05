"""Tests for the aggregation attribution benchmark adapters."""

from argparse import Namespace

from cellvault import CellDB
from scripts.benchmark_aggregate_workflow import PHASES, run_worker, validate_runs


def test_attribution_workers_match_and_report_phases(tmp_path, small_adata):
    adata = small_adata.copy()
    adata.obs["sample"] = [f"sample_{index % 3}" for index in range(adata.n_obs)]
    adata.obs["cell_type"] = [
        "T" if index % 2 else "B" for index in range(adata.n_obs)
    ]
    input_path = tmp_path / "input.h5ad"
    cellvault_path = tmp_path / "input.cvdb"
    adata.write_h5ad(input_path)
    with CellDB.from_anndata(adata, str(cellvault_path)):
        pass

    args = Namespace(
        input_h5ad=str(input_path),
        cellvault_path=str(cellvault_path),
        output_json=str(tmp_path / "results.json"),
        groupby=["sample_type=sample,cell_type", "sample=sample"],
        source="X",
        layer="",
        metrics=["sum", "mean", "count_nonzero"],
        batch_size=31,
        repeats=1,
        seed=0,
        threads=1,
        rebuild_cellvault=False,
        keep_validation_arrays=True,
        worker_method="",
        worker_output="",
    )

    runs = []
    for method in (
        "independent",
        "manual-single-scan",
        "zarr-independent",
        "zarr-single-scan",
        "cellvault-joint",
    ):
        args.worker_method = method
        args.worker_output = str(tmp_path / f"{method}.json")
        run = run_worker(args)
        runs.append(run)
        assert set(run["phase_seconds"]) == set(PHASES)
        assert run["phase_seconds"]["matrix_read"] > 0
        assert run["phase_seconds"]["dispatch_aggregation"] > 0

    validation = validate_runs(runs)

    assert validation["status"] == "passed"
    assert validation["array_comparisons"] == 24
    assert runs[0]["access"]["source_scans"] == 2
    assert runs[1]["access"]["source_scans"] == 1
    assert runs[2]["access"]["source_scans"] == 2
    assert runs[3]["access"]["source_scans"] == 1
    assert runs[4]["access"]["source_scans"] == 1
    assert runs[4]["phase_seconds"]["framework_overhead"] >= 0
