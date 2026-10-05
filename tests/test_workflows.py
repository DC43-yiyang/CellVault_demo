"""Tests for thin application-level workflow adapters."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellvault import (
    AggregateTask,
    CellDB,
    aggregate_modalities,
    leave_one_out_pseudobulk,
    prepare_communication_inputs,
)


def _adata(values) -> ad.AnnData:
    return ad.AnnData(
        X=np.asarray(values, dtype=np.float32),
        obs=pd.DataFrame(
            {
                "sample": ["s1", "s1", "s2", "s2"],
                "donor": ["d1", "d1", "d2", "d2"],
                "cell_type": ["T", "B", "T", "B"],
                "condition": ["control", "control", "treated", "treated"],
            },
            index=[f"cell_{index}" for index in range(4)],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def test_prepare_communication_inputs_preserves_complete_samples(tmp_path):
    adata = _adata(
        [
            [1, 0, 2],
            [0, 3, 1],
            [4, 0, 0],
            [0, 5, 2],
        ]
    )
    with CellDB.from_anndata(adata, str(tmp_path / "communication.cvdb")) as db:
        run = prepare_communication_inputs(
            db,
            sample_column="sample",
            cell_type_column="cell_type",
            obs_columns=("condition",),
            batch_size=2,
        )

    assert list(run.results) == ["s1", "s2"]
    assert run.report.source_scan_count == 1
    assert run.report.unique_rows == adata.n_obs
    for sample, result in run.results.items():
        expected = adata[adata.obs["sample"] == sample]
        assert result.obs_names.tolist() == expected.obs_names.tolist()
        assert result.obs.columns.tolist() == ["sample", "cell_type", "condition"]
        np.testing.assert_array_equal(result.X, expected.X)


def test_prepare_communication_inputs_validates_metadata_columns(tmp_path):
    adata = _adata(np.ones((4, 3)))
    with CellDB.from_anndata(adata, str(tmp_path / "validation.cvdb")) as db:
        with pytest.raises(TypeError, match="sequence"):
            prepare_communication_inputs(
                db,
                sample_column="sample",
                cell_type_column="cell_type",
                obs_columns="condition",
            )
        with pytest.raises(KeyError, match="missing"):
            prepare_communication_inputs(
                db,
                sample_column="missing",
                cell_type_column="cell_type",
            )


def test_leave_one_out_reuses_pseudobulk_without_cell_matrix_reads(
    tmp_path, monkeypatch
):
    adata = _adata(
        [
            [1, 0, 2],
            [0, 3, 1],
            [4, 0, 0],
            [0, 5, 2],
        ]
    )
    with CellDB.from_anndata(adata, str(tmp_path / "pseudobulk.cvdb")) as db:
        pseudobulk = db.aggregate_many(
            [AggregateTask("pb", ("donor", "cell_type"), metrics=("sum",))]
        ).results["pb"]
        monkeypatch.setattr(
            db._backend,
            "read_X",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("leave-one-out reread the cell matrix")
            ),
        )
        leave_one_out = leave_one_out_pseudobulk(
            pseudobulk,
            donor_column="donor",
        )

    assert list(leave_one_out) == ["d1", "d2"]
    assert leave_one_out["d1"].obs["donor"].tolist() == ["d2", "d2"]
    assert leave_one_out["d2"].obs["donor"].tolist() == ["d1", "d1"]
    np.testing.assert_array_equal(
        leave_one_out["d1"].layers["sum"],
        pseudobulk.layers["sum"][2:],
    )


def test_aggregate_modalities_reuses_cohort_definition_across_stores(tmp_path):
    rna = _adata(
        [
            [1, 0, 2],
            [0, 3, 1],
            [4, 0, 0],
            [0, 5, 2],
        ]
    )
    adt = _adata(
        [
            [10, 0, 20],
            [0, 30, 10],
            [40, 0, 0],
            [0, 50, 20],
        ]
    )
    adt.var_names = ["CD3", "CD19", "EPCAM"]
    with (
        CellDB.from_anndata(rna, str(tmp_path / "rna.cvdb")) as rna_db,
        CellDB.from_anndata(adt, str(tmp_path / "adt.cvdb")) as adt_db,
    ):
        run = aggregate_modalities(
            {"rna": rna_db, "adt": adt_db},
            groupby=("sample", "cell_type"),
            where='"condition" = ?',
            params=("control",),
            metrics=("sum", "mean"),
            batch_size=2,
        )

    assert run.source_scan_count == 2
    assert set(run.results) == {"rna", "adt"}
    assert run.results["rna"].var_names.tolist() == ["g1", "g2", "g3"]
    assert run.results["adt"].var_names.tolist() == ["CD3", "CD19", "EPCAM"]
    np.testing.assert_array_equal(
        run.results["adt"].layers["sum"],
        run.results["rna"].layers["sum"] * 10,
    )
