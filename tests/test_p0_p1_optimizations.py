"""Regression tests for the lineage fan-out P0/P1 improvements."""

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy import sparse

from cellvault import CellDB, tools
from cellvault.backend import DuckDBZarrBackend


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_read_x_many_preserves_order_duplicates_and_empty(tmp_path, kind):
    dense = np.arange(48).reshape(8, 6)
    matrix = dense if kind == "dense" else sparse.csr_matrix(dense)
    with DuckDBZarrBackend(str(tmp_path / f"{kind}.cvdb")) as backend:
        backend.write_X(matrix)
        results = backend.read_X_many(
            {"first": [6, 1, 6, 0], "empty": [], "second": [7, 2]},
            column_indices=[4, 1],
        )

    for key, rows in {
        "first": [6, 1, 6, 0],
        "empty": [],
        "second": [7, 2],
    }.items():
        actual = results[key]
        if sparse.issparse(actual):
            actual = actual.toarray()
        np.testing.assert_array_equal(actual, dense[np.ix_(rows, [4, 1])])


def test_sparse_storage_compacts_safe_indices(tmp_path):
    matrix = sparse.csr_matrix(np.eye(20, dtype=np.float32))
    path = tmp_path / "compact.cvdb"
    with DuckDBZarrBackend(str(path)) as backend:
        backend.write_X(matrix)

    root = zarr.open_group(
        zarr.storage.LocalStore(str(path / "X.zarr")),
        mode="r",
    )
    assert root["indices"].dtype == np.dtype(np.int32)
    assert root["indptr"].dtype == np.dtype(np.int32)


@pytest.fixture
def rich_adata():
    obs = pd.DataFrame(
        {
            "lineage": pd.Categorical(
                ["T", "B", "T", "Myeloid", "B", "T"],
                categories=["B", "Myeloid", "T"],
                ordered=True,
            ),
            "score": [1, 2, 3, 4, 5, 6],
        },
        index=pd.Index([f"cell_{index}" for index in range(6)], name="cell_id"),
    )
    var = pd.DataFrame(
        {"kind": pd.Categorical(["coding", "other", "coding", "other"])},
        index=pd.Index([f"gene_{index}" for index in range(4)], name="gene_id"),
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.arange(24).reshape(6, 4)),
        obs=obs,
        var=var,
    )
    adata.layers["counts"] = sparse.csr_matrix(np.arange(24, 48).reshape(6, 4))
    adata.varm["loadings"] = np.arange(8).reshape(4, 2)
    adata.varp["correlations"] = np.eye(4)

    raw = ad.AnnData(
        X=sparse.csr_matrix(np.arange(30).reshape(6, 5)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"raw_gene_{index}" for index in range(5)]),
    )
    raw.varm["loadings"] = np.ones((5, 2))
    adata.raw = raw
    return adata


def test_full_anndata_slots_and_categories_roundtrip(tmp_path, rich_adata):
    with CellDB.from_anndata(rich_adata, str(tmp_path / "rich.cvdb")) as database:
        restored = database.to_anndata()

    np.testing.assert_array_equal(restored.X.toarray(), rich_adata.X.toarray())
    np.testing.assert_array_equal(
        restored.layers["counts"].toarray(),
        rich_adata.layers["counts"].toarray(),
    )
    np.testing.assert_array_equal(
        restored.varm["loadings"], rich_adata.varm["loadings"]
    )
    np.testing.assert_array_equal(
        restored.varp["correlations"], rich_adata.varp["correlations"]
    )
    assert restored.raw is not None
    np.testing.assert_array_equal(restored.raw.X.toarray(), rich_adata.raw.X.toarray())
    np.testing.assert_array_equal(
        restored.raw.varm["loadings"], rich_adata.raw.varm["loadings"]
    )
    assert isinstance(restored.obs["lineage"].dtype, pd.CategoricalDtype)
    assert restored.obs["lineage"].cat.ordered
    assert restored.obs["lineage"].cat.categories.tolist() == ["B", "Myeloid", "T"]
    assert restored.obs.index.name == "cell_id"
    assert restored.var.index.name == "gene_id"


def test_selective_new_slots(tmp_path, rich_adata):
    with CellDB.from_anndata(rich_adata, str(tmp_path / "selective.cvdb")) as database:
        restored = database.to_anndata(
            slots={"obs", "var", "layers", "raw"},
            layer_keys=["counts"],
        )

    assert restored.X is None
    assert [key for key in restored.layers if key is not None] == ["counts"]
    assert restored.raw is not None
    assert len(restored.varm) == 0
    assert len(restored.varp) == 0


def test_partition_materialize_many_and_recursive_views(tmp_path, rich_adata):
    with CellDB.from_anndata(rich_adata, str(tmp_path / "partition.cvdb")) as database:
        views = database.partition_obs(
            "lineage",
            {
                "T": np.asarray(["T"]),
                "immune_other": np.asarray(["B", "Myeloid"]),
            },
            require_complete=True,
        )
        subsets = database.materialize_many(
            views,
            slots={"X", "obs", "var", "layers", "raw"},
        )
        child = views["T"].query_obs('"score" > ?', [2])

    assert views["T"].obs_names.tolist() == ["cell_0", "cell_2", "cell_5"]
    assert child.obs_names.tolist() == ["cell_2", "cell_5"]
    np.testing.assert_array_equal(
        subsets["T"].X.toarray(),
        rich_adata[[0, 2, 5]].X.toarray(),
    )
    np.testing.assert_array_equal(
        subsets["T"].layers["counts"].toarray(),
        rich_adata[[0, 2, 5]].layers["counts"].toarray(),
    )
    np.testing.assert_array_equal(
        subsets["T"].raw.X.toarray(),
        rich_adata[[0, 2, 5]].raw.X.toarray(),
    )


def test_partition_rejects_overlap_and_incomplete_coverage(tmp_path, rich_adata):
    with CellDB.from_anndata(rich_adata, str(tmp_path / "invalid.cvdb")) as database:
        with pytest.raises(TypeError, match="columns"):
            database.partition_obs("lineage", {"T": ["T"]}, columns="score")
        with pytest.raises(ValueError, match="assigned to both"):
            database.partition_obs("lineage", {"one": ["T"], "two": ["T"]})
        with pytest.raises(ValueError, match="does not cover"):
            database.partition_obs(
                "lineage",
                {"only_t": ["T"]},
                require_complete=True,
            )


def test_named_view_and_view_writeback(tmp_path, rich_adata):
    path = str(tmp_path / "named.cvdb")
    with CellDB.from_anndata(rich_adata, path) as database:
        view = database.query_obs('"lineage" = ?', ["T"], name="t_cells")
        child = view.query_obs('"score" > ?', [2], name="activated_t_cells")
        view.update_obs("fine_label", ["T1", "T2", "T3"], create=True)
        assert database.named_views == ["activated_t_cells", "t_cells"]
        loaded = database.load_view("t_cells")
        loaded_child = database.load_view("activated_t_cells")
        assert loaded.name == "t_cells"
        assert loaded.obs_names.tolist() == view.obs_names.tolist()
        assert loaded_child.name == "activated_t_cells"
        assert loaded_child.obs_names.tolist() == child.obs_names.tolist()
        assert database.obs.loc[view.obs_names, "fine_label"].tolist() == [
            "T1",
            "T2",
            "T3",
        ]

    definitions = json.loads((tmp_path / "named.cvdb" / "views.json").read_text())
    assert definitions["t_cells"]["where"] == '"lineage" = ?'
    assert definitions["activated_t_cells"]["where"] == (
        '("lineage" = ?) AND ("score" > ?)'
    )


def test_partition_accepts_scalar_numpy_array(tmp_path, rich_adata):
    with CellDB.from_anndata(
        rich_adata, str(tmp_path / "scalar-array.cvdb")
    ) as database:
        views = database.partition_obs("lineage", {"B": np.asarray("B")})

    assert views["B"].obs_names.tolist() == ["cell_1", "cell_4"]


@pytest.mark.parametrize(
    "values",
    [
        (value for value in ["T"]),
        pd.Index(["B", "Myeloid"]),
    ],
)
def test_partition_accepts_general_iterable_values(tmp_path, rich_adata, values):
    expected_names = ["cell_0", "cell_2", "cell_5"]
    if isinstance(values, pd.Index):
        expected_names = ["cell_1", "cell_3", "cell_4"]

    with CellDB.from_anndata(rich_adata, str(tmp_path / "iterable.cvdb")) as database:
        views = database.partition_obs("lineage", {"selected": values})

    assert views["selected"].obs_names.tolist() == expected_names


def test_view_iter_x_batches_is_bounded_and_ordered(tmp_path, rich_adata):
    with CellDB.from_anndata(rich_adata, str(tmp_path / "batches.cvdb")) as database:
        view = database.query_obs('"lineage" = ?', ["T"])
        batches = list(view.iter_X_batches(batch_size=2, layer="counts"))

    assert [len(obs) for obs, _ in batches] == [2, 1]
    combined = sparse.vstack([matrix for _, matrix in batches])
    np.testing.assert_array_equal(
        combined.toarray(),
        rich_adata[[0, 2, 5]].layers["counts"].toarray(),
    )


def test_scanpy_tools_run_on_view_without_global_matrix_artifacts(
    tmp_path, small_adata
):
    adata = small_adata.copy()
    adata.obs["lineage"] = ["T" if index < 60 else "B" for index in range(adata.n_obs)]
    with CellDB.from_anndata(adata, str(tmp_path / "tools-view.cvdb")) as database:
        view = database.query_obs('"lineage" = ?', ["T"])
        tools.pca(view, n_comps=10)
        tools.neighbors(view, n_neighbors=5)
        tools.umap(view, random_state=0)
        key = tools.leiden(
            view,
            resolution=0.5,
            write_back=True,
            output_column="fine_cluster",
        )

        assert key == "fine_cluster"
        assert view.obsm["X_pca"].shape == (60, 10)
        assert view.obsm["X_umap"].shape == (60, 2)
        assert view.obsp["connectivities"].shape == (60, 60)
        assert "X_pca" not in database.obsm
        assert database.obs.loc[view.obs_names, "fine_cluster"].notna().all()
