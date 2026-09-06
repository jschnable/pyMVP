"""Characterize existing pipeline threshold precedence, not new statistics."""
import math

import pytest

from panicle.core.thresholds import base_threshold, trait_thresholds


@pytest.mark.parametrize('options,value,source,count', [
    ({}, .0005, 'Bonferroni (markers)', 100),
    ({'n_eff': 20}, .0025, 'Bonferroni (effective tests)', 20),
    ({'estimated_me': 25.5}, .05 / 25.5, 'Bonferroni (effective tests)', 25.5),
    ({'n_eff': 20, 'estimated_me': 25}, .0025, 'Bonferroni (effective tests)', 20),
    ({'use_effective_tests': False, 'estimated_me': 25}, .0005, 'Bonferroni (markers)', 100),
    ({'use_effective_tests': False, 'n_eff': 20}, .0025, 'Bonferroni (effective tests)', 20),
    ({'n_eff': 0, 'estimated_me': 0}, .0005, 'Bonferroni (markers)', 100),
    ({'significance': 0}, 0, 'Fixed p-value', None),
    ({'significance': .01, 'n_eff': 20}, .01, 'Fixed p-value', None),
])
def test_base_precedence(options, value, source, count):
    result = base_threshold(n_markers=100, **options)
    assert result.value == value and result.source == source
    assert math.isnan(result.n_tests) if count is None else result.n_tests == count


@pytest.mark.parametrize('filtered', [False, True])
@pytest.mark.parametrize('fixed', [None, .03])
@pytest.mark.parametrize('use_estimate', [False, True])
def test_trait_mac_fixed_and_solver_precedence(filtered, fixed, use_estimate):
    base = base_threshold(n_markers=100, significance=fixed, n_eff=20,
                          estimated_me=30.9, use_effective_tests=use_estimate)
    result = trait_thresholds(base=base, methods=['GLM', 'MLM', 'BLINK', 'BAYESLOCO', 'FarmCPU', 'FarmCPUResampling'],
                              n_tested=10, mac_filtered=filtered, significance=fixed,
                              n_eff=20, estimated_me=30.9, use_effective_tests=use_estimate)
    expected = fixed if fixed is not None else .005 if filtered else .0025
    assert result.base.value == expected
    if filtered and fixed is None:
        assert result.base.source == 'Bonferroni (markers, post-MAC)'
        assert result.base.n_tests == 10
    else:
        assert result.base is base
    assert result.effective_n == (30 if use_estimate else 20)
    for method in ['GLM', 'MLM', 'BLINK', 'BAYESLOCO']:
        assert result.methods[method] is result.base
    assert result.methods['FarmCPU'].value == .01 / result.effective_n
    assert result.methods['FarmCPUResampling'].value == result.methods['FarmCPU'].value
    assert result.methods['FarmCPUResampling'].source == 'FarmCPU QTN threshold (default)'


@pytest.mark.parametrize('corrected', [False, True])
@pytest.mark.parametrize('resampling', [None, 0, .007])
def test_method_overrides_do_not_mutate_inputs(corrected, resampling):
    params = {'QTN_threshold': .02, 'QTN_threshold_is_corrected': corrected}
    if resampling is not None:
        params['resampling_significance_threshold'] = resampling
    original = params.copy()
    result = trait_thresholds(base=base_threshold(n_markers=100),
                              methods=['FarmCPU', 'FarmCPUResampling'], n_tested=10,
                              farmcpu_params=params)
    qtn = .02 if corrected else .002
    assert result.methods['FarmCPU'].value == qtn
    assert result.methods['FarmCPUResampling'].value == (qtn if resampling is None else resampling)
    assert params == original


def test_unknown_methods_do_not_get_thresholds():
    result = trait_thresholds(base=base_threshold(n_markers=100), methods=['unknown'], n_tested=10)
    assert result.methods == {}
