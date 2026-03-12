"""Tests for NameRegistry canonical name resolution."""

import pytest

from cellvault.registry import NameRegistry


class TestNameRegistryGet:
    def test_pca_default(self):
        assert NameRegistry.get("pca") == "X_pca"

    def test_pca_harmony(self):
        assert NameRegistry.get("pca", "harmony") == "X_pca_harmony"

    def test_umap_default(self):
        assert NameRegistry.get("umap") == "X_umap"

    def test_leiden_default(self):
        assert NameRegistry.get("leiden") == "leiden"

    def test_neighbors_default(self):
        assert NameRegistry.get("neighbors") == "neighbors"

    def test_connectivities(self):
        assert NameRegistry.get("connectivities") == "connectivities"

    def test_distances(self):
        assert NameRegistry.get("distances") == "distances"

    def test_case_insensitive(self):
        assert NameRegistry.get("PCA") == "X_pca"
        assert NameRegistry.get("UMAP", "Harmony") == "X_umap_harmony"

    def test_unknown_operation_raises(self):
        with pytest.raises(KeyError, match="No canonical name"):
            NameRegistry.get("nonexistent_operation")

    def test_unknown_integration_raises(self):
        with pytest.raises(KeyError, match="No canonical name"):
            NameRegistry.get("pca", "nonexistent_integration")


class TestNameRegistryLookup:
    def test_reverse_lookup(self):
        op, integration = NameRegistry.lookup("X_pca")
        assert op == "pca"
        assert integration is None

    def test_reverse_lookup_with_integration(self):
        op, integration = NameRegistry.lookup("X_pca_harmony")
        assert op == "pca"
        assert integration == "harmony"

    def test_reverse_unknown_raises(self):
        with pytest.raises(KeyError, match="not a registered canonical name"):
            NameRegistry.lookup("some_random_key")


class TestNameRegistryIsCanonical:
    def test_canonical_key(self):
        assert NameRegistry.is_canonical("X_pca") is True
        assert NameRegistry.is_canonical("X_umap") is True
        assert NameRegistry.is_canonical("leiden") is True

    def test_non_canonical_key(self):
        assert NameRegistry.is_canonical("my_custom_pca") is False
        assert NameRegistry.is_canonical("pca_v2") is False


class TestNameRegistryRegister:
    def test_register_custom(self):
        reg = NameRegistry()
        reg.register("pca", "custom_method", "X_pca_custom")
        assert NameRegistry.get("pca", "custom_method") == "X_pca_custom"
        assert NameRegistry.is_canonical("X_pca_custom") is True


class TestNameRegistryListAll:
    def test_list_all_not_empty(self):
        all_names = NameRegistry.list_all()
        assert len(all_names) > 20
        assert ("pca", None) in all_names
        assert all_names[("pca", None)] == "X_pca"
