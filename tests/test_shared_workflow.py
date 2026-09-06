"""Contracts shared by the two public GWAS adapters."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from panicle.core.workflow import (
    PreparedTrait, TraitCacheKey, retained_samples, group_sample_indices,
    select_markers, association_genotype, run_method, run_trait_group,
)
from panicle.reporting.tables import association_table, flatten_results
from panicle.reporting.plots import render_analysis, render_and_close
from panicle.utils.data_types import AssociationResults, GenotypeMap, GenotypeMatrix


@pytest.fixture
def trait():
    geno = GenotypeMatrix(np.array([[0, 0, 2], [0, 1, 1], [0, 2, 0]], dtype=np.int8),
                          is_imputed=True)
    gmap = GenotypeMap(pd.DataFrame(dict(MARKER=['a', 'b', 'c'], CHROM=[1, 1, 2], POS=[1, 2, 3])))
    return PreparedTrait('height', np.column_stack([np.arange(3), [1., 2., 4.]]),
                         geno, np.ones((3, 1)), np.eye(3), np.arange(3), gmap)


def test_ordered_sample_groups_and_cache_settings():
    samples = [np.array([0, 2]), np.array([2, 0]), np.array([0, 2], dtype=np.int32)]
    assert list(group_sample_indices(samples).values()) == [[0, 2], [1]]
    key = TraitCacheKey.create(samples[0])
    assert key == TraitCacheKey.create(samples[2])
    assert key != TraitCacheKey.create(samples[1])
    for options in [dict(n_pcs=1), dict(need_kinship=True), dict(min_mac=1), dict(max_dosage=4)]:
        assert key != TraitCacheKey.create(samples[0], **options)


def test_nonfinite_samples():
    np.testing.assert_array_equal(
        retained_samples([1., np.nan, 2., np.inf], np.array([[0.], [0.], [np.inf], [0.]])),
        [True, False, False, False],
    )


def test_lazy_and_materialized_selection_agree(trait):
    lazy = select_markers(trait.genotype, trait.geno_map, min_mac=1)
    eager = select_markers(trait.genotype, trait.geno_map, min_mac=1, materialize=True)
    assert lazy.genotype is trait.genotype
    np.testing.assert_array_equal(lazy.keep_indices, [1, 2])
    np.testing.assert_array_equal(association_genotype(lazy.genotype, lazy.keep_indices).get_columns([0, 1]),
                                  eager.genotype.get_columns([0, 1]))
    pd.testing.assert_frame_equal(lazy.geno_map.to_dataframe(), eager.geno_map.to_dataframe())
    unchanged = select_markers(trait.genotype, trait.geno_map, min_mac=0)
    assert unchanged.genotype is trait.genotype and unchanged.geno_map is trait.geno_map
    assert unchanged.keep_indices is None


@pytest.mark.parametrize('method', ['GLM', 'MLM', 'MLM_LOCO', 'FARMCPU', 'BLINK', 'BAYESLOCO', 'FARMCPURESAMPLING'])
def test_dispatch_preserves_arrays_and_options(trait, method):
    runner = Mock(return_value=object())
    result = run_method(method, trait, runner=runner, options={'cpu': 2})
    kwargs = runner.call_args.kwargs
    assert kwargs['phe'] is trait.phenotype and kwargs['geno'] is trait.genotype
    assert kwargs['CV'] is trait.covariates and kwargs['cpu'] == 2
    assert ('K' in kwargs) == (method == 'MLM')
    assert ('map_data' in kwargs) == (method not in {'GLM', 'MLM'})
    assert result.result is runner.return_value and result.seconds >= 0


def test_dispatch_does_not_swallow_errors(trait):
    with pytest.raises(ValueError, match='bad input'):
        run_method('GLM', trait, runner=Mock(side_effect=ValueError('bad input')))


def test_joint_dispatch_and_legacy_trait_tuple(trait):
    second = PreparedTrait('yield', trait.phenotype * 2, trait.genotype, trait.covariates,
                           trait.kinship, trait.sample_indices, trait.geno_map)
    runner = Mock(return_value={'height': 1, 'yield': 2})
    result = run_trait_group([trait, second], trait.genotype, runner=runner)
    assert runner.call_args.kwargs['trait_names'] == ['height', 'yield']
    np.testing.assert_array_equal(runner.call_args.kwargs['phe'], [[1, 2], [2, 4], [4, 8]])
    assert result['yield'].result == 2
    assert trait.legacy_tuple()[1] is trait.genotype


def test_table_assembly_and_plot_policy(trait):
    result = AssociationResults(np.ones(3), np.ones(3), np.array([.1, .2, .3]))
    original = result.to_dataframe().copy(deep=True)
    frame = association_table(result, trait.geno_map.to_dataframe())
    assert frame['MARKER'].tolist() == ['a', 'b', 'c']
    assert frame['SNP'].tolist() == ['a', 'b', 'c']
    pd.testing.assert_frame_equal(original, result.to_dataframe())
    nested = {'height': {'GLM': result}}
    assert flatten_results(nested, single_trait=False) == {'height_GLM': result}
    renderer = Mock(return_value={'plots': {}})
    render_analysis(nested, single_trait=True, renderer=renderer, file_output=False)
    assert renderer.call_args.kwargs == {'results': {'GLM': result}, 'file_output': False}
    import matplotlib.pyplot as plt
    figure = plt.figure()
    render_and_close(renderer=Mock(return_value={'plots': {'GLM': {'qq': figure}}}))
    assert not plt.fignum_exists(figure.number)
