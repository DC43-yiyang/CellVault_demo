"""Tests for shared-scan group-by aggregation."""

import gc
import weakref

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import cellvault.execution as execution_module
from cellvault import AggregateTask, CellDB, MaterializeTask, Task


def _aggregation_adata(*, sparse_input: bool) -> ad.AnnData:
    values = np.asarray(
        [
            [1, 0, 2, 0],
            [0, 3, 0, 4],
            [5, 0, 6, 0],
            [0, 7, 0, 8],
            [9, 0, 0, 1],
            [0, 2, 3, 0],
            [4, 0, 0, 5],
        ],
        dtype=np.int32,
    )
    matrix = sparse.csr_matrix(values) if sparse_input else values
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2", "s2", "s1", "s2", "s2"],
            "cell_type": pd.Categorical(
                ["B", "T", "B", None, "B", "T", "B"],
                categories=["T", "B", "unused"],
                ordered=True,
            ),
        },
        index=pd.Index([f"cell_{index}" for index in range(len(values))]),
    )
    var = pd.DataFrame(
        {"symbol": ["A", "B", "C", "D"]},
        index=pd.Index(["g1", "g2", "g3", "g4"], name="gene_id"),
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.layers["counts"] = matrix * 3
    return adata


def _group_mask(obs: pd.DataFrame, row: pd.Series, columns: tuple[str, ...]):
    mask = np.ones(len(obs), dtype=bool)
    for column in columns:
        value = row[column]
        if value == "<NA>":
            mask &= obs[column].isna().to_numpy()
        else:
            mask &= obs[column].eq(value).to_numpy()
    return mask


def _assert_matches_reference(
    result: ad.AnnData,
    source,
    original_obs: pd.DataFrame,
    groupby: tuple[str, ...],
) -> None:
    dense = source.toarray() if sparse.issparse(source) else np.asarray(source)
    for group_index, group in result.obs.iterrows():
        del group_index
        mask = _group_mask(original_obs, group, groupby)
        expected = dense[mask]
        np.testing.assert_array_equal(
            result.layers["sum"][group.name.split(":")[-1].__int__()],
            expected.sum(axis=0),
        )


def test_build_grouping_preserves_large_multicolumn_semantics():
    size = 120_003
    positions = np.arange(size)
    lineage_pattern = np.asarray(["B", "T", None, "B", "T", "B", None])
    sample_pattern = np.asarray(
        ["sample_2", "sample_1", "sample_2", None, "sample_1", None, "sample_2"]
    )
    donor_pattern = np.asarray([3, 1, 2, 3, 2, 1, 3])
    obs = pd.DataFrame(
        {
            "lineage": pd.Categorical(
                lineage_pattern[positions % len(lineage_pattern)],
                categories=["T", "B", "NK", "unused"],
                ordered=True,
            ),
            "sample": sample_pattern[positions % len(sample_pattern)],
            "donor": donor_pattern[positions % len(donor_pattern)],
        }
    )
    groupby = ("lineage", "sample", "donor")
    column_codes = []
    column_levels = []
    for column in groupby:
        codes, levels, _, _ = execution_module._factorize_column(obs[column])
        column_codes.append(codes)
        column_levels.append(levels)
    expected_combinations = sorted(set(zip(*column_codes, strict=True)))
    combination_to_group = {
        combination: group
        for group, combination in enumerate(expected_combinations)
    }
    expected_row_to_group = np.fromiter(
        (
            combination_to_group[combination]
            for combination in zip(*column_codes, strict=True)
        ),
        dtype=np.int64,
        count=size,
    )

    grouping = execution_module._build_grouping(obs, groupby)

    np.testing.assert_array_equal(grouping.row_to_group, expected_row_to_group)
    np.testing.assert_array_equal(
        grouping.n_cells,
        np.bincount(expected_row_to_group, minlength=len(expected_combinations)),
    )
    for column_index, column in enumerate(groupby):
        expected_values = [
            column_levels[column_index][int(combination[column_index])]
            for combination in expected_combinations
        ]
        assert grouping.obs[column].tolist() == expected_values
    assert isinstance(grouping.obs["lineage"].dtype, pd.CategoricalDtype)
    assert grouping.obs["lineage"].cat.categories.tolist() == ["T", "B", "<NA>"]
    assert grouping.obs["lineage"].cat.ordered
    assert "<NA>" in grouping.obs["sample"].tolist()
    assert len(grouping.obs) < np.prod([len(levels) for levels in column_levels])


@pytest.mark.parametrize("sparse_input", [False, True])
def test_select_matrix_rows_reuses_exact_batch(sparse_input):
    dense = np.arange(24).reshape(6, 4)
    matrix = sparse.csr_matrix(dense) if sparse_input else dense

    exact = execution_module._select_matrix_rows(matrix, np.arange(6))
    subset = execution_module._select_matrix_rows(matrix, np.asarray([1, 2, 3]))
    gathered = execution_module._select_matrix_rows(matrix, np.asarray([4, 1]))

    assert exact is matrix
    np.testing.assert_array_equal(
        subset.toarray() if sparse.issparse(subset) else subset,
        dense[1:4],
    )
    np.testing.assert_array_equal(
        gathered.toarray() if sparse.issparse(gathered) else gathered,
        dense[[4, 1]],
    )


def test_build_grouping_handles_empty_cohort():
    obs = pd.DataFrame(
        {
            "lineage": pd.Categorical([], categories=["T", "B"], ordered=True),
            "sample": pd.Series([], dtype=object),
        }
    )

    grouping = execution_module._build_grouping(obs, ("lineage", "sample"))

    assert grouping.obs.empty
    assert grouping.obs.columns.tolist() == ["lineage", "sample", "n_cells"]
    assert grouping.row_to_group.dtype == np.int64
    assert grouping.n_cells.dtype == np.int64
    assert grouping.row_to_group.size == 0
    assert grouping.n_cells.size == 0


@pytest.mark.parametrize("sparse_input", [False, True])
def test_aggregate_many_matches_reference_and_shares_scan(
    tmp_path, sparse_input, monkeypatch
):
    adata = _aggregation_adata(sparse_input=sparse_input)
    with CellDB.from_anndata(
        adata,
        str(tmp_path / f"aggregate-{sparse_input}.cvdb"),
    ) as database:
        read_count = 0
        original_read = database._backend.read_X

        def counted_read(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read(*args, **kwargs)

        monkeypatch.setattr(database._backend, "read_X", counted_read)
        run = database.aggregate_many(
            [
                AggregateTask(
                    "by_sample",
                    "sample",
                    metrics=("sum", "mean", "count_nonzero"),
                ),
                AggregateTask(
                    "by_sample_type",
                    ("sample", "cell_type"),
                    metrics=("sum", "mean", "count_nonzero"),
                ),
                AggregateTask("cell_counts", "cell_type", metrics=()),
            ],
            batch_size=3,
        )

    assert read_count == 3
    assert run.report.task_count == 3
    assert run.report.source_scan_count == 1
    assert run.report.matrix_batch_reads == 3
    assert run.report.requested_rows == 2 * adata.n_obs
    assert run.report.unique_rows == adata.n_obs
    assert run.report.reuse_ratio == 2.0
    assert run.report.total_result_bytes > 0

    for name, groupby in {
        "by_sample": ("sample",),
        "by_sample_type": ("sample", "cell_type"),
    }.items():
        result = run.results[name]
        assert result.X is None
        assert list(result.layers) == ["sum", "mean", "count_nonzero"]
        for result_position, (_, group) in enumerate(result.obs.iterrows()):
            mask = _group_mask(adata.obs, group, groupby)
            expected = np.asarray(adata.X[mask].toarray() if sparse_input else adata.X[mask])
            np.testing.assert_array_equal(
                result.layers["sum"][result_position], expected.sum(axis=0)
            )
            np.testing.assert_allclose(
                result.layers["mean"][result_position], expected.mean(axis=0)
            )
            np.testing.assert_array_equal(
                result.layers["count_nonzero"][result_position],
                np.count_nonzero(expected, axis=0),
            )
            assert result.obs.iloc[result_position]["n_cells"] == len(expected)

    metadata_result = run.results["cell_counts"]
    assert list(metadata_result.layers) == []
    assert metadata_result.obs["cell_type"].tolist() == ["T", "B", "<NA>"]
    assert metadata_result.obs["n_cells"].tolist() == [2, 4, 1]


def test_layer_feature_order_and_batch_size_are_stable(tmp_path):
    adata = _aggregation_adata(sparse_input=True)
    task = AggregateTask(
        "selected",
        ("sample", "cell_type"),
        source="layers:counts",
        metrics=("sum", "mean", "count_nonzero"),
        features=np.asarray(["g4", "g1"]),
    )
    with CellDB.from_anndata(adata, str(tmp_path / "features.cvdb")) as database:
        small_batches = database.aggregate_many([task], batch_size=2).results["selected"]
        large_batch = database.aggregate_many([task], batch_size=100).results["selected"]

    assert small_batches.var_names.tolist() == ["g4", "g1"]
    pd.testing.assert_frame_equal(small_batches.obs, large_batch.obs)
    for metric in task.metrics:
        np.testing.assert_allclose(
            small_batches.layers[metric],
            large_batch.layers[metric],
        )


def test_aggregate_many_on_cell_view_uses_only_view_rows(tmp_path):
    adata = _aggregation_adata(sparse_input=True)
    with CellDB.from_anndata(adata, str(tmp_path / "view.cvdb")) as database:
        view = database.query_obs('"sample" = ?', ["s2"], columns=["sample"])
        result = view.aggregate_many(
            [AggregateTask("types", "cell_type", metrics=("sum",))],
            batch_size=2,
        ).results["types"]

    assert result.obs["cell_type"].tolist() == ["T", "B", "<NA>"]
    assert result.obs["n_cells"].tolist() == [1, 2, 1]
    selected = np.asarray(adata[adata.obs["sample"] == "s2"].X.toarray())
    np.testing.assert_array_equal(result.layers["sum"].sum(axis=0), selected.sum(axis=0))


def test_incompatible_feature_axes_use_separate_scans(tmp_path):
    adata = _aggregation_adata(sparse_input=False)
    with CellDB.from_anndata(adata, str(tmp_path / "axes.cvdb")) as database:
        run = database.aggregate_many(
            [
                AggregateTask("one", "sample", features=("g1",)),
                AggregateTask("two", "sample", features=("g2",)),
            ],
            batch_size=4,
        )

    assert run.report.source_scan_count == 2
    assert run.report.matrix_batch_reads == 4
    assert run.report.reuse_ratio == 1.0


def test_metadata_only_task_does_not_require_or_read_x(tmp_path, monkeypatch):
    adata = _aggregation_adata(sparse_input=False)
    adata.X = None
    with CellDB.from_anndata(adata, str(tmp_path / "metadata.cvdb")) as database:
        monkeypatch.setattr(
            database._backend,
            "read_X",
            lambda *args, **kwargs: pytest.fail("metadata aggregation read X"),
        )
        run = database.aggregate_many(
            [AggregateTask("counts", "sample", metrics=())]
        )

    assert run.report.source_scan_count == 0
    assert run.report.matrix_batch_reads == 0
    assert run.report.matrix_bytes_read == 0
    assert run.results["counts"].obs["n_cells"].tolist() == [3, 4]


@pytest.mark.parametrize(
    ("task", "error", "match"),
    [
        (AggregateTask("ok", "sample"), ValueError, "unique"),
        (
            AggregateTask("missing-column", "does_not_exist"),
            KeyError,
            "obs columns",
        ),
        (
            AggregateTask("missing-feature", "sample", features=("bad",)),
            KeyError,
            "features not found",
        ),
        (
            AggregateTask("missing-layer", "sample", source="layers:bad"),
            KeyError,
            "layer 'bad'",
        ),
    ],
)
def test_aggregate_many_validates_run_inputs(tmp_path, task, error, match):
    adata = _aggregation_adata(sparse_input=True)
    with CellDB.from_anndata(adata, str(tmp_path / f"invalid-{task.name}.cvdb")) as database:
        tasks = [task, task] if task.name == "ok" else [task]
        with pytest.raises(error, match=match):
            database.aggregate_many(tasks)


def test_aggregate_task_validates_definition():
    with pytest.raises(ValueError, match="non-empty"):
        AggregateTask("", "sample")
    with pytest.raises(ValueError, match="duplicates"):
        AggregateTask("bad-groups", ("sample", "sample"))
    with pytest.raises(ValueError, match="Unknown aggregate metrics"):
        AggregateTask("bad-metric", "sample", metrics=("median",))
    with pytest.raises(ValueError, match="must not be empty"):
        AggregateTask("bad-features", "sample", features=())
    with pytest.raises(ValueError, match="must not contain duplicates"):
        AggregateTask("duplicate-features", "sample", features=("g1", "g1"))


def test_missing_value_label_collision_is_rejected(tmp_path):
    adata = _aggregation_adata(sparse_input=False)
    adata.obs["group"] = ["<NA>", None, "x", "x", "x", "x", "x"]
    with CellDB.from_anndata(adata, str(tmp_path / "missing.cvdb")) as database:
        with pytest.raises(ValueError, match="reserved missing-group"):
            database.aggregate_many([AggregateTask("groups", "group")])


def test_overlapping_sql_cohorts_share_union_reads(tmp_path, monkeypatch):
    adata = _aggregation_adata(sparse_input=True)
    adata.obs["treatment"] = ["control", "A", "control", "B", "A", "control", "B"]
    with CellDB.from_anndata(adata, str(tmp_path / "cohorts.cvdb")) as database:
        read_count = 0
        original_read = database._backend.read_X

        def counted_read(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read(*args, **kwargs)

        monkeypatch.setattr(database._backend, "read_X", counted_read)
        run = database.aggregate_many(
            [
                AggregateTask(
                    "drug_a",
                    "treatment",
                    where='"treatment" IN (?, ?)',
                    params=("control", "A"),
                ),
                AggregateTask(
                    "drug_b",
                    "treatment",
                    where='"treatment" IN (?, ?)',
                    params=("control", "B"),
                ),
            ],
            batch_size=2,
        )

    assert read_count == 4
    assert run.report.source_scan_count == 1
    assert run.report.requested_rows == 10
    assert run.report.unique_rows == 7
    assert run.report.overlap_rows == 3
    assert run.report.task_rows == {"drug_a": 5, "drug_b": 5}
    for task_name, labels in {"drug_a": ["control", "A"], "drug_b": ["control", "B"]}.items():
        result = run.results[task_name]
        assert set(result.obs["treatment"]) == set(labels)
        for position, label in enumerate(result.obs["treatment"]):
            mask = adata.obs["treatment"].eq(label).to_numpy()
            np.testing.assert_array_equal(
                result.layers["sum"][position],
                adata.X[mask].toarray().sum(axis=0),
            )


def test_task_cohort_is_intersected_with_parent_view(tmp_path):
    adata = _aggregation_adata(sparse_input=False)
    adata.obs["treatment"] = ["control", "A", "control", "B", "A", "control", "B"]
    with CellDB.from_anndata(adata, str(tmp_path / "view-cohort.cvdb")) as database:
        sample_two = database.query_obs('"sample" = ?', ["s2"])
        result = sample_two.aggregate_many(
            [
                AggregateTask(
                    "control",
                    "cell_type",
                    where='"treatment" = ?',
                    params=("control",),
                )
            ]
        ).results["control"]

    expected_mask = (
        adata.obs["sample"].eq("s2") & adata.obs["treatment"].eq("control")
    ).to_numpy()
    np.testing.assert_array_equal(
        result.layers["sum"].sum(axis=0),
        np.asarray(adata.X)[expected_mask].sum(axis=0),
    )
    assert result.obs["n_cells"].sum() == int(expected_mask.sum())


def test_aggregate_task_validates_cohort_definition():
    with pytest.raises(ValueError, match="non-empty SQL predicate"):
        AggregateTask("empty-where", "sample", where="")
    with pytest.raises(TypeError, match="params must be a sequence"):
        AggregateTask("bad-params", "sample", params="s1")


def test_overlapping_cohorts_can_share_full_matrix_materialization(
    tmp_path, monkeypatch
):
    adata = _aggregation_adata(sparse_input=True)
    adata.obs["treatment"] = ["control", "A", "control", "B", "A", "control", "B"]
    with CellDB.from_anndata(adata, str(tmp_path / "shared-control.cvdb")) as database:
        views = {
            "drug_a": database.query_obs(
                '"treatment" IN (?, ?)', ["control", "A"]
            ),
            "drug_b": database.query_obs(
                '"treatment" IN (?, ?)', ["control", "B"]
            ),
        }
        read_count = 0
        original_read = database._backend.read_X_many

        def counted_read(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read(*args, **kwargs)

        monkeypatch.setattr(database._backend, "read_X_many", counted_read)
        results = database.materialize_many(views, slots={"X", "obs", "var"})

    assert read_count == 1
    for name, labels in {"drug_a": ["control", "A"], "drug_b": ["control", "B"]}.items():
        expected = adata[adata.obs["treatment"].isin(labels)]
        np.testing.assert_array_equal(results[name].X.toarray(), expected.X.toarray())


def test_explicit_membership_preserves_overlap_but_deduplicates_reads(
    tmp_path, monkeypatch
):
    adata = _aggregation_adata(sparse_input=True)
    membership = pd.DataFrame(
        {
            "cell_id": ["cell_0", "cell_0", "cell_1", "cell_3", "cell_6"],
            "roi": ["A", "B", "A", "B", "B"],
        }
    )
    with CellDB.from_anndata(adata, str(tmp_path / "membership.cvdb")) as database:
        read_count = 0
        original_read = database._backend.read_X

        def counted_read(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read(*args, **kwargs)

        monkeypatch.setattr(database._backend, "read_X", counted_read)
        run = database.aggregate_many(
            [
                AggregateTask(
                    "roi_cell_type",
                    ("roi", "cell_type"),
                    membership=membership,
                )
            ],
            batch_size=2,
        )

    result = run.results["roi_cell_type"]
    assert read_count == 2
    assert run.report.task_rows == {"roi_cell_type": 4}
    assert run.report.task_memberships == {"roi_cell_type": 5}
    assert result.obs["n_cells"].sum() == 5
    source = adata.X.toarray()
    expected = {
        ("A", "B"): source[[0]].sum(axis=0),
        ("A", "T"): source[[1]].sum(axis=0),
        ("B", "B"): source[[0, 6]].sum(axis=0),
        ("B", "<NA>"): source[[3]].sum(axis=0),
    }
    for position, row in enumerate(result.obs.itertuples(index=False)):
        np.testing.assert_array_equal(
            result.layers["sum"][position],
            expected[(row.roi, row.cell_type)],
        )


def test_binary_membership_column_filters_false_edges(tmp_path):
    adata = _aggregation_adata(sparse_input=False)
    membership = pd.DataFrame(
        {
            "cell_id": ["cell_0", "cell_0", "cell_1"],
            "roi": ["A", "B", "A"],
            "membership": [1, 0, True],
        }
    )
    task = AggregateTask("roi", "roi", membership=membership, metrics=())
    with CellDB.from_anndata(adata, str(tmp_path / "binary-membership.cvdb")) as database:
        run = database.aggregate_many([task])

    assert run.results["roi"].obs["roi"].tolist() == ["A"]
    assert run.results["roi"].obs["n_cells"].tolist() == [2]
    assert run.report.task_rows == {"roi": 2}
    assert run.report.task_memberships == {"roi": 2}


@pytest.mark.parametrize(
    ("membership", "error", "match"),
    [
        (
            pd.DataFrame({"cell_id": ["missing"], "roi": ["A"]}),
            KeyError,
            "outside the target cohort",
        ),
        (
            pd.DataFrame(
                {"cell_id": ["cell_0", "cell_0"], "roi": ["A", "A"]}
            ),
            ValueError,
            "duplicate cell/group edges",
        ),
        (
            pd.DataFrame(
                {"cell_id": ["cell_0"], "roi": ["A"], "membership": [0.5]}
            ),
            ValueError,
            "weighted memberships are not supported",
        ),
    ],
)
def test_membership_validation(tmp_path, membership, error, match):
    adata = _aggregation_adata(sparse_input=True)
    if "outside" in match:
        task = AggregateTask("roi", "roi", membership=membership)
        with CellDB.from_anndata(
            adata, str(tmp_path / "invalid-membership.cvdb")
        ) as database:
            with pytest.raises(error, match=match):
                database.aggregate_many([task])
    else:
        with pytest.raises(error, match=match):
            AggregateTask("roi", "roi", membership=membership)


def test_mixed_execution_shares_scan_between_aggregate_and_materialize(
    tmp_path, monkeypatch
):
    adata = _aggregation_adata(sparse_input=True)
    with CellDB.from_anndata(adata, str(tmp_path / "mixed.cvdb")) as database:
        read_count = 0
        original_read = database._backend.read_X

        def counted_read(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read(*args, **kwargs)

        monkeypatch.setattr(database._backend, "read_X", counted_read)
        run = database.execute_tasks(
            [
                AggregateTask("sample_sum", "sample", metrics=("sum",)),
                MaterializeTask(
                    "sample_one",
                    where='"sample" = ?',
                    params=("s1",),
                ),
            ],
            batch_size=3,
            memory_budget_bytes=1_000_000,
        )

    assert read_count == 3
    assert run.report.source_scan_count == 1
    assert run.report.execution_waves == 1
    assert run.report.matrix_batch_reads == 3
    assert run.report.requested_rows == adata.n_obs + 3
    assert run.report.unique_rows == adata.n_obs
    assert run.report.peak_buffer_bytes > 0
    assert run.report.degradation_reasons == ()
    assert len(run.report.scan_plan) == 1
    plan = run.report.scan_plan[0]
    assert {
        key: plan[key]
        for key in (
            "wave",
            "source",
            "features",
            "n_features",
            "rows",
            "repeated_rows",
            "batch_count",
            "aggregate_tasks",
            "materialize_tasks",
        )
    } == {
        "wave": 1,
        "source": "X",
        "features": None,
        "n_features": adata.n_vars,
        "rows": adata.n_obs,
        "repeated_rows": 0,
        "batch_count": 3,
        "aggregate_tasks": ("sample_sum",),
        "materialize_tasks": ("sample_one",),
    }
    assert plan["peak_batch_bytes"] > 0
    assert plan["peak_accumulator_bytes"] > 0
    assert plan["peak_consumer_bytes"] > 0
    assert plan["peak_buffer_bytes"] == run.report.peak_buffer_bytes
    np.testing.assert_array_equal(
        run.results["sample_one"].X.toarray(),
        adata[adata.obs["sample"] == "s1"].X.toarray(),
    )
    np.testing.assert_array_equal(
        run.results["sample_sum"].layers["sum"].sum(axis=0),
        adata.X.toarray().sum(axis=0),
    )


def test_materialize_task_supports_layer_features_and_overlapping_cell_ids(tmp_path):
    adata = _aggregation_adata(sparse_input=True)
    with CellDB.from_anndata(
        adata, str(tmp_path / "materialize-task.cvdb")
    ) as database:
        run = database.execute_tasks(
            [
                MaterializeTask(
                    "roi_a",
                    source="layers:counts",
                    features=("g4", "g1"),
                    obs_columns=("sample",),
                    cell_ids=("cell_0", "cell_1", "cell_3"),
                ),
                MaterializeTask(
                    "roi_b",
                    source="layers:counts",
                    features=("g4", "g1"),
                    obs_columns=("sample",),
                    cell_ids=("cell_0", "cell_6"),
                ),
            ],
            batch_size=2,
        )

    assert run.report.source_scan_count == 1
    assert run.report.requested_rows == 5
    assert run.report.unique_rows == 4
    assert run.results["roi_a"].var_names.tolist() == ["g4", "g1"]
    np.testing.assert_array_equal(
        run.results["roi_a"].X.toarray(),
        adata.layers["counts"][[0, 1, 3]][:, [3, 0]].toarray(),
    )
    np.testing.assert_array_equal(
        run.results["roi_b"].X.toarray(),
        adata.layers["counts"][[0, 6]][:, [3, 0]].toarray(),
    )


def test_mixed_execution_enforces_memory_budget(tmp_path):
    adata = _aggregation_adata(sparse_input=False)
    with CellDB.from_anndata(adata, str(tmp_path / "budget.cvdb")) as database:
        with pytest.raises(MemoryError, match="memory_budget_bytes"):
            database.execute_tasks(
                [
                    AggregateTask("sample_sum", "sample"),
                    MaterializeTask("all_cells"),
                ],
                batch_size=4,
                memory_budget_bytes=32,
            )


def test_memory_budget_keeps_compatible_materializers_in_one_wave_when_they_fit(
    tmp_path,
):
    adata = _aggregation_adata(sparse_input=False)
    consumed = []

    def consume(adata):
        consumed.append(adata.obs_names.tolist())
        return {"n_obs": adata.n_obs}

    with CellDB.from_anndata(adata, str(tmp_path / "waves.cvdb")) as database:
        run = database.execute_tasks(
            [
                AggregateTask("sample_sum", "sample"),
                MaterializeTask(
                    "roi_a",
                    cell_ids=("cell_0", "cell_1", "cell_3"),
                    consumer=consume,
                ),
                MaterializeTask(
                    "roi_b",
                    cell_ids=("cell_0", "cell_6"),
                    consumer=consume,
                ),
            ],
            batch_size=2,
            memory_budget_bytes=10_000,
        )

    assert run.report.execution_waves == 1
    assert run.report.source_scan_count == 1
    assert run.report.degradation_reasons == ()
    assert run.report.peak_buffer_bytes <= run.report.memory_budget_bytes
    assert len(run.report.scan_plan) == 1
    plan = run.report.scan_plan[0]
    assert plan["aggregate_tasks"] == ("sample_sum",)
    assert plan["materialize_tasks"] == ("roi_a", "roi_b")
    assert plan["estimated_peak_buffer_bytes"] <= run.report.memory_budget_bytes
    assert run.report.redundant_rows == 0
    assert run.results["roi_a"] == {"n_obs": 3}
    assert run.results["roi_b"] == {"n_obs": 2}
    assert consumed == [
        ["cell_0", "cell_1", "cell_3"],
        ["cell_0", "cell_6"],
    ]


def test_memory_budget_splits_materializers_and_releases_consumed_inputs(
    tmp_path, monkeypatch
):
    adata = _aggregation_adata(sparse_input=False)
    consumed = []
    input_references = []
    released_before_later_reads = []

    def consume(materialized):
        consumed.append(materialized.obs_names.tolist())
        input_references.append(weakref.ref(materialized))
        return {"n_obs": materialized.n_obs}

    with CellDB.from_anndata(adata, str(tmp_path / "waves.cvdb")) as database:
        original_read = database._backend.read_X

        def checked_read(*args, **kwargs):
            if input_references:
                gc.collect()
                released_before_later_reads.append(input_references[-1]() is None)
            return original_read(*args, **kwargs)

        monkeypatch.setattr(database._backend, "read_X", checked_read)
        run = database.execute_tasks(
            [
                AggregateTask("sample_sum", "sample"),
                MaterializeTask(
                    "roi_a",
                    cell_ids=("cell_0", "cell_1", "cell_3"),
                    consumer=consume,
                ),
                MaterializeTask(
                    "roi_b",
                    cell_ids=("cell_0", "cell_6"),
                    consumer=consume,
                ),
            ],
            batch_size=2,
            memory_budget_bytes=160,
        )

    assert run.report.execution_waves == 2
    assert run.report.source_scan_count == 2
    assert len(run.report.degradation_reasons) == 1
    assert "memory_budget_bytes split 2 compatible materialization tasks" in (
        run.report.degradation_reasons[0]
    )
    assert [plan["wave"] for plan in run.report.scan_plan] == [1, 2]
    assert [plan["source"] for plan in run.report.scan_plan] == ["X", "X"]
    assert [plan["rows"] for plan in run.report.scan_plan] == [7, 2]
    assert [plan["repeated_rows"] for plan in run.report.scan_plan] == [0, 2]
    assert [plan["batch_count"] for plan in run.report.scan_plan] == [4, 1]
    assert run.report.scan_plan[0]["aggregate_tasks"] == ("sample_sum",)
    assert run.report.scan_plan[1]["aggregate_tasks"] == ()
    assert [plan["materialize_tasks"] for plan in run.report.scan_plan] == [
        ("roi_a",),
        ("roi_b",),
    ]
    assert all(plan["peak_accumulator_bytes"] > 0 for plan in run.report.scan_plan)
    assert all(plan["peak_consumer_bytes"] > 0 for plan in run.report.scan_plan)
    assert all(
        plan["estimated_peak_buffer_bytes"] <= run.report.memory_budget_bytes
        for plan in run.report.scan_plan
    )
    assert run.report.peak_buffer_bytes <= run.report.memory_budget_bytes
    assert run.report.redundant_rows == 2
    serialized = run.report.to_dict()
    assert serialized["scan_plan"][1]["repeated_rows"] == 2
    assert serialized["degradation_reasons"] == list(
        run.report.degradation_reasons
    )
    assert run.results["roi_a"] == {"n_obs": 3}
    assert run.results["roi_b"] == {"n_obs": 2}
    assert consumed == [
        ["cell_0", "cell_1", "cell_3"],
        ["cell_0", "cell_6"],
    ]
    assert released_before_later_reads
    assert all(released_before_later_reads)
    gc.collect()
    assert all(reference() is None for reference in input_references)


def test_memory_budget_rejects_large_retained_consumer_output(tmp_path):
    adata = _aggregation_adata(sparse_input=False)

    def consume(_materialized):
        return np.zeros(512, dtype=np.uint8)

    with CellDB.from_anndata(
        adata,
        str(tmp_path / "consumer-output-budget.cvdb"),
    ) as database:
        with pytest.raises(MemoryError, match="retained consumer output"):
            database.execute_tasks(
                [
                    MaterializeTask(
                        "one_cell",
                        cell_ids=("cell_0",),
                        consumer=consume,
                    )
                ],
                batch_size=1,
                memory_budget_bytes=128,
            )


def test_memory_budget_includes_aggregate_mean_finalization(tmp_path):
    adata = _aggregation_adata(sparse_input=False)
    with CellDB.from_anndata(
        adata,
        str(tmp_path / "aggregate-finalization-budget.cvdb"),
    ) as database:
        with pytest.raises(MemoryError, match="aggregate result finalization"):
            database.execute_tasks(
                [
                    AggregateTask(
                        "sample_mean",
                        "sample",
                        metrics=("mean", "count_nonzero"),
                    )
                ],
                batch_size=1,
                memory_budget_bytes=160,
            )

        run = database.execute_tasks(
            [
                AggregateTask(
                    "sample_mean",
                    "sample",
                    metrics=("mean", "count_nonzero"),
                )
            ],
            batch_size=1,
            memory_budget_bytes=192,
        )

    assert run.report.peak_buffer_bytes == 192
    assert run.report.scan_plan[0]["peak_buffer_bytes"] == 192


def test_materialize_task_validates_definition():
    with pytest.raises(ValueError, match="non-empty"):
        MaterializeTask("")
    with pytest.raises(ValueError, match="must not contain duplicates"):
        MaterializeTask("features", features=("g1", "g1"))
    with pytest.raises(ValueError, match="must not contain duplicates"):
        MaterializeTask("cells", cell_ids=("cell_1", "cell_1"))
    with pytest.raises(TypeError, match="must be a sequence"):
        MaterializeTask("columns", obs_columns="sample")
    with pytest.raises(TypeError, match="must be callable"):
        MaterializeTask("consumer", consumer="not-callable")


def test_concrete_tasks_implement_common_task_protocol():
    assert isinstance(AggregateTask("aggregate", "sample"), Task)
    assert isinstance(MaterializeTask("materialize"), Task)
