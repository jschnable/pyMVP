"""Unit tests for FarmCPU resampling workflow"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymvp.association.farmcpu_resampling import MVP_FarmCPUResampling
from pymvp.utils.data_types import GenotypeMatrix, GenotypeMap, AssociationResults


def _build_map(n_markers: int) -> GenotypeMap:
    data = pd.DataFrame({
        'SNP': [f'SNP{i+1}' for i in range(n_markers)],
        'CHROM': ['1'] * n_markers,
        'POS': list(range(1, n_markers + 1))
    })
    return GenotypeMap(data)


def _make_association_result(pvalue_indices, n_markers):
    pvals = np.ones(n_markers)
    for idx in pvalue_indices:
        pvals[idx] = 1e-9
    effects = np.zeros(n_markers)
    ses = np.ones(n_markers)
    return AssociationResults(effects, ses, pvals)


@patch('pymvp.association.farmcpu_resampling.MVP_FarmCPU')
def test_farmcpu_resampling_without_clustering(mock_farmcpu):
    """RMIP scores should reflect per-marker stability without clustering."""

    runs = 4
    n_markers = 3
    call_plan = [
        [0, 1],  # Run 1: markers 0 and 1 significant
        [1],     # Run 2: marker 1 significant
        [0],     # Run 3: marker 0 significant
        []       # Run 4: no markers significant
    ]

    # Configure mocked FarmCPU output per run
    def side_effect(*args, **kwargs):
        idx = side_effect.counter
        side_effect.counter += 1
        return _make_association_result(call_plan[idx], n_markers)

    side_effect.counter = 0
    mock_farmcpu.side_effect = side_effect

    phe = np.column_stack([
        np.array(['ind1', 'ind2', 'ind3']),
        np.array([1.0, 2.0, 3.0])
    ])
    genotype = GenotypeMatrix(np.array([
        [0, 1, 2],
        [1, 1, 0],
        [2, 0, 1],
    ], dtype=np.int8))
    map_data = _build_map(n_markers)

    result = MVP_FarmCPUResampling(
        phe=phe,
        geno=genotype,
        map_data=map_data,
        CV=None,
        runs=runs,
        mask_proportion=0.0,
        significance_threshold=5e-8,
        cluster_markers=False,
        random_seed=42,
    )

    rmip = result.rmip_values
    assert pytest.approx(rmip.tolist()) == [0.5, 0.5]
    assert [entry.snp for entry in result.entries] == ['SNP1', 'SNP2']
    df = result.to_dataframe()
    assert list(df.columns) == ['SNP', 'Chr', 'Pos', 'RMIP', 'Trait']
    assert pytest.approx(df['RMIP'].values.tolist()) == [0.5, 0.5]
    assert not result.cluster_mode


@patch('pymvp.association.farmcpu_resampling.MVP_FarmCPU')
def test_farmcpu_resampling_with_clustering(mock_farmcpu):
    """Markers in strong LD should be collapsed into clusters with shared RMIP."""

    runs = 4
    n_markers = 3
    call_plan = [
        [0, 1],  # Run 1
        [0],     # Run 2
        [1],     # Run 3
        [2],     # Run 4
    ]

    def side_effect(*args, **kwargs):
        idx = side_effect.counter
        side_effect.counter += 1
        return _make_association_result(call_plan[idx], n_markers)

    side_effect.counter = 0
    mock_farmcpu.side_effect = side_effect

    phe = np.column_stack([
        np.array(['ind1', 'ind2', 'ind3', 'ind4']),
        np.array([1.0, 2.0, 3.0, 4.0])
    ])
    genotype = GenotypeMatrix(np.array([
        [0, 0, 2],
        [1, 1, 0],
        [2, 2, 1],
        [0, 0, 2],
    ], dtype=np.int8))
    map_data = _build_map(n_markers)

    result = MVP_FarmCPUResampling(
        phe=phe,
        geno=genotype,
        map_data=map_data,
        CV=None,
        runs=runs,
        mask_proportion=0.0,
        significance_threshold=5e-8,
        cluster_markers=True,
        ld_threshold=0.7,
        random_seed=7,
    )

    assert result.cluster_mode
    assert len(result.entries) == 2
    rmip = result.rmip_values
    assert pytest.approx(rmip.tolist()) == [0.75, 0.25]
    cluster_entry = result.entries[0]
    assert cluster_entry.cluster_size == 2
    assert cluster_entry.snp == 'SNP1'
    assert cluster_entry.cluster_members is not None
    members = cluster_entry.cluster_members
    assert pytest.approx(members['SNP1']) == 0.5
    assert pytest.approx(members['SNP2']) == 0.5

    df = result.to_dataframe()
    assert 'ClusterMembers' in df.columns
    assert df.iloc[0]['ClusterMembers'].startswith('size=2;')
