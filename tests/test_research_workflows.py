"""Tests for the complete research-workflow result adapters."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.run_wu_research_workflow import (
    lineage_labels,
    subtype_contrast,
    summarize_local_input,
)
from scripts.run_mcfarland_response_workflow import response_summary
from scripts.run_spatial_scope_workflow import local_scope_analysis


def test_wu_lineage_mapping_is_exhaustive_for_author_labels():
    labels = pd.Series(
        [
            "T-cells",
            "B-cells",
            "Plasmablasts",
            "CAFs",
            "Endothelial",
            "PVL",
            "Cancer Epithelial",
            "Normal Epithelial",
            "Myeloid",
        ]
    )

    assert lineage_labels(labels).tolist() == [
        "T/NK",
        "B",
        "B",
        "Stromal",
        "Stromal",
        "Stromal",
        "Epithelial",
        "Epithelial",
        "Myeloid",
    ]


def test_wu_local_summary_and_subtype_contrast():
    local = ad.AnnData(
        X=sparse.csr_matrix([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]]),
        obs=pd.DataFrame(
            {"workflow_fine_label": ["T1", "T1", "T2"]},
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    summary = summarize_local_input(local)
    assert summary["fine_labels"]["T1"]["n_cells"] == 2
    assert summary["fine_labels"]["T1"]["marker_mean"] == [2.0, 1.0]

    result = ad.AnnData(
        X=None,
        obs=pd.DataFrame(
            {
                "donor_id": ["d1", "d2", "d3", "d4"],
                "subtype": ["ER+", "ER+", "TNBC", "TNBC"],
                "n_cells": [2, 3, 4, 5],
            }
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
        layers={"mean": np.asarray([[1.0, 4.0], [3.0, 2.0], [5.0, 1.0], [7.0, 3.0]])},
    )
    contrast = subtype_contrast(
        result,
        {"g1": "GENE1", "g2": "GENE2"},
        reference="ER+",
        comparison="TNBC",
        top_genes=1,
    )

    assert contrast["comparison_cells"] == 9
    assert contrast["reference_cells"] == 5
    assert contrast["top_positive"][0]["feature_name"] == "GENE1"
    assert contrast["top_positive"][0]["mean_difference"] == 4.0


def test_mcfarland_response_summary_respects_matched_strata():
    result = ad.AnnData(
        X=None,
        obs=pd.DataFrame(
            {
                "cell_line": ["A", "A", "B", "B"],
                "time": ["24", "24", "48", "48"],
                "perturbation": ["control", "drug", "control", "drug"],
                "n_cells": [2, 4, 1, 2],
            }
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
        layers={
            "sum": np.asarray(
                [
                    [2.0, 4.0],
                    [12.0, 4.0],
                    [2.0, 3.0],
                    [4.0, 8.0],
                ]
            ),
            "count_nonzero": np.ones((4, 2), dtype=np.int64),
        },
    )

    response = response_summary(result, "drug", "control", top_genes=1)

    assert len(response["matched_pairs"]) == 2
    assert response["top_increased"][0]["gene"] == "G1"
    assert response["top_increased"][0]["mean_matched_difference"] == 1.0
    assert response["top_decreased"][0]["gene"] == "G2"
    assert response["top_decreased"][0]["mean_matched_difference"] == 0.0


def test_spatial_local_analysis_preserves_scope_context():
    local = ad.AnnData(
        X=sparse.csr_matrix(
            [[1.0, 0.0, 2.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]
        ),
        obs=pd.DataFrame(
            {
                "library_id": ["fov1"] * 3,
                "donor": ["d1"] * 3,
                "Cluster": ["T", "T", "B"],
            },
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame(index=["CD3", "CD20", "KRT"]),
    )

    result = local_scope_analysis(local)

    assert result["scope_id"] == "fov1"
    assert result["donor"] == "d1"
    assert result["n_cells"] == 3
    assert result["cluster_profiles"]["T"]["n_cells"] == 2
    assert len(result["pca_variance_ratio"]) == 3
