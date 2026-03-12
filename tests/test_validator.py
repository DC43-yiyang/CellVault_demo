"""Tests for PipelineStateValidator precondition checks."""

import pytest

from cellvault.validator import PipelineStateValidator, CellVaultStateError


def _make_state(**overrides):
    """Helper to build a state dict with defaults."""
    state = {
        "X_exists": False,
        "obsm_keys": [],
        "obsp_keys": [],
        "obs_columns": [],
        "uns_keys": [],
    }
    state.update(overrides)
    return state


class TestPCAValidation:
    def test_pca_requires_X(self):
        with pytest.raises(CellVaultStateError, match="expression matrix"):
            PipelineStateValidator.validate("pca", _make_state(X_exists=False))

    def test_pca_passes_with_X(self):
        PipelineStateValidator.validate("pca", _make_state(X_exists=True))


class TestNeighborsValidation:
    def test_neighbors_requires_pca(self):
        with pytest.raises(CellVaultStateError, match="obsm"):
            PipelineStateValidator.validate(
                "neighbors", _make_state(X_exists=True, obsm_keys=[])
            )

    def test_neighbors_passes_with_pca(self):
        PipelineStateValidator.validate(
            "neighbors", _make_state(X_exists=True, obsm_keys=["X_pca"])
        )

    def test_neighbors_passes_with_integration_pca(self):
        PipelineStateValidator.validate(
            "neighbors", _make_state(X_exists=True, obsm_keys=["X_pca_harmony"])
        )


class TestUMAPValidation:
    def test_umap_requires_neighbors(self):
        with pytest.raises(CellVaultStateError, match="obsp"):
            PipelineStateValidator.validate(
                "umap", _make_state(obsm_keys=["X_pca"], obsp_keys=[])
            )

    def test_umap_passes_with_graph(self):
        PipelineStateValidator.validate(
            "umap",
            _make_state(obsp_keys=["connectivities", "distances"]),
        )


class TestLeidenValidation:
    def test_leiden_requires_neighbors(self):
        with pytest.raises(CellVaultStateError, match="obsp"):
            PipelineStateValidator.validate("leiden", _make_state(obsp_keys=[]))

    def test_leiden_passes_with_graph(self):
        PipelineStateValidator.validate(
            "leiden",
            _make_state(obsp_keys=["connectivities", "distances"]),
        )


class TestUnknownOperation:
    def test_unknown_operation_passes(self):
        """Operations without preconditions should pass silently."""
        PipelineStateValidator.validate("unknown_op", _make_state())


class TestListOperations:
    def test_list_operations(self):
        ops = PipelineStateValidator.list_operations()
        assert "pca" in ops
        assert "neighbors" in ops
        assert "umap" in ops
        assert "leiden" in ops

    def test_get_preconditions(self):
        rules = PipelineStateValidator.get_preconditions("pca")
        assert rules == {"X": True}


class TestCellVaultStateError:
    def test_error_attributes(self):
        err = CellVaultStateError("umap", "obsp connectivities", ["X_pca"])
        assert err.operation == "umap"
        assert err.missing == "obsp connectivities"
        assert err.available == ["X_pca"]
        assert "umap" in str(err)
