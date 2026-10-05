"""Tests for SQL-driven, row-filtered CellView objects."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellvault import CellDB, CellView


@pytest.fixture
def view_dataset(tmp_path, small_adata):
    adata = small_adata.copy()
    adata.obs["cell_type"] = np.where(
        np.arange(adata.n_obs) % 4 == 0,
        "T cell",
        "other",
    )
    adata.obsm["X_pca"] = np.arange(adata.n_obs * 3).reshape(adata.n_obs, 3)
    graph = sparse.diags(
        [np.ones(adata.n_obs - 1), np.ones(adata.n_obs), np.ones(adata.n_obs - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    adata.obsp["connectivities"] = graph
    cdb = CellDB.from_anndata(adata, str(tmp_path / "view.cvdb"))
    yield adata, cdb
    cdb.close()


def test_parameterized_query_preserves_matrix_alignment(view_dataset):
    adata, cdb = view_dataset
    expected_positions = np.flatnonzero(adata.obs["cell_type"].to_numpy() == "T cell")

    view = cdb.query_obs('"cell_type" = ?', ["T cell"])

    assert isinstance(view, CellView)
    assert view.shape == (len(expected_positions), adata.n_vars)
    assert len(view) == len(expected_positions)
    np.testing.assert_array_equal(view.obs_positions, expected_positions)
    assert view.obs_names.tolist() == adata.obs_names[expected_positions].tolist()
    np.testing.assert_array_equal(
        view.X.toarray(), adata.X[expected_positions].toarray()
    )


def test_query_supports_projection_and_numpy_parameters(view_dataset):
    adata, cdb = view_dataset
    params = np.asarray(["T cell"], dtype=object)

    view = cdb.query_obs('"cell_type" = ?', params, columns=["cell_type"])

    assert view.obs.columns.tolist() == ["cell_type"]
    assert view.obs_names.tolist() == adata.obs_names[::4].tolist()


def test_query_preserves_categorical_schema_and_index_name(view_dataset):
    adata, cdb = view_dataset
    adata.obs.index.name = "cell_id"
    adata.obs["batch"] = adata.obs["batch"].cat.reorder_categories(
        ["C", "B", "A"],
        ordered=True,
    )
    cdb.obs = adata.obs

    view = cdb.query_obs('"cell_type" = ?', ["T cell"])
    materialized = view.to_anndata(slots={"X", "obs", "var"})

    assert view.obs.index.name == "cell_id"
    assert isinstance(view.obs["batch"].dtype, pd.CategoricalDtype)
    assert view.obs["batch"].cat.categories.tolist() == ["C", "B", "A"]
    assert view.obs["batch"].cat.ordered
    assert materialized.obs.index.name == "cell_id"


def test_query_parameter_is_not_executed_as_sql(view_dataset):
    _, cdb = view_dataset
    view = cdb.query_obs('"cell_type" = ?', ["T cell' OR TRUE --"])
    assert view.n_obs == 0


def test_empty_query_materializes_valid_anndata(view_dataset):
    adata, cdb = view_dataset
    view = cdb.query_obs('"cell_type" = ?', ["not-present"])

    selected = view.to_anndata()

    assert selected.shape == (0, adata.n_vars)
    assert sparse.issparse(selected.X)
    assert selected.obsm["X_pca"].shape == (0, 3)
    assert selected.obsp["connectivities"].shape == (0, 0)


def test_view_subsets_obsm_and_induces_obsp_graph(view_dataset):
    adata, cdb = view_dataset
    positions = np.arange(0, adata.n_obs, 4)
    view = cdb.query_obs('"cell_type" = ? AND "n_counts" >= 0', ["T cell"])

    np.testing.assert_array_equal(view.obsm["X_pca"], adata.obsm["X_pca"][positions])
    expected_graph = adata.obsp["connectivities"][positions][:, positions]
    np.testing.assert_array_equal(
        view.obsp["connectivities"].toarray(),
        expected_graph.toarray(),
    )


def test_dense_view_matches_anndata_slice(tmp_path, dense_adata):
    dense_adata.obs["group"] = [
        "keep" if index % 3 == 0 else "drop" for index in range(dense_adata.n_obs)
    ]
    positions = np.arange(0, dense_adata.n_obs, 3)
    with CellDB.from_anndata(dense_adata, str(tmp_path / "dense.cvdb")) as cdb:
        view = cdb.query_obs('"group" = ?', ["keep"])
        np.testing.assert_array_equal(view.X, dense_adata.X[positions])


def test_materialize_creates_independent_database(tmp_path, view_dataset):
    adata, cdb = view_dataset
    view = cdb.query_obs('"cell_type" = ?', ["T cell"])
    target = str(tmp_path / "t_cells.cvdb")

    subset_db = view.materialize(target)
    try:
        assert subset_db.shape == view.shape
        np.testing.assert_array_equal(subset_db.obs_names, adata.obs_names[::4])
        np.testing.assert_array_equal(subset_db.X.toarray(), adata.X[::4].toarray())
    finally:
        subset_db.close()


def test_view_obs_is_a_read_only_copy(view_dataset):
    _, cdb = view_dataset
    view = cdb.query_obs('"cell_type" = ?', ["T cell"])
    changed = view.obs
    changed["cell_type"] = "changed"
    assert set(view.obs["cell_type"]) == {"T cell"}
    assert set(cdb.obs["cell_type"]) == {"T cell", "other"}


def test_query_rejects_multiple_statements(view_dataset):
    _, cdb = view_dataset
    with pytest.raises(ValueError, match="single SQL predicate"):
        cdb.query_obs("TRUE; DROP TABLE obs")


def test_query_rejects_scalar_parameter_and_column_strings(view_dataset):
    _, cdb = view_dataset
    with pytest.raises(TypeError, match="params"):
        cdb.query_obs('"cell_type" = ?', "T cell")
    with pytest.raises(TypeError, match="columns"):
        cdb.query_obs(columns="cell_type")


@pytest.mark.parametrize(
    ("columns", "error", "message"),
    [
        (["missing"], KeyError, "not found"),
        (["cell_type", "cell_type"], ValueError, "duplicates"),
    ],
)
def test_query_validates_projected_columns(view_dataset, columns, error, message):
    _, cdb = view_dataset
    with pytest.raises(error, match=message):
        cdb.query_obs(columns=columns)


def test_add_obs_column_public_api(view_dataset):
    _, cdb = view_dataset
    cdb.add_obs_column("reviewed", np.zeros(cdb.n_obs, dtype=bool))
    assert "reviewed" in cdb.obs.columns


def test_update_obs_where_public_api(view_dataset):
    _, cdb = view_dataset
    cdb.add_obs_column("review", [""] * cdb.n_obs)
    updated = cdb.update_obs_where(
        "review",
        "reviewed",
        '"cell_type" = ?',
        ["T cell"],
    )
    obs = cdb.obs
    assert updated == 25
    assert set(obs.loc[obs["cell_type"] == "T cell", "review"]) == {"reviewed"}
    assert set(obs.loc[obs["cell_type"] != "T cell", "review"]) == {""}
