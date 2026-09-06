"""Named internal inputs preserve numerical references and legacy adapters."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from panicle.core.thresholds import Threshold
from panicle.core.workflow import PreparedTrait, MethodOptions, MethodRunResult
from panicle.pipelines import gwas
from panicle.reporting.models import MethodReport, TraitReport, ReportOptions
from panicle.reporting.pipeline import TraitOutputContext, write_trait_report, write_trait_results
from panicle.utils.data_types import GenotypeMatrix, GenotypeMap, AssociationResults
from panicle.association.farmcpu_resampling import FarmCPUResamplingEntry, FarmCPUResamplingResults


@pytest.fixture
def trait():
    genotype = GenotypeMatrix(np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int8), is_imputed=True)
    gmap = GenotypeMap(pd.DataFrame(dict(MARKER=['a', 'b', 'c'], CHROM=[1, 1, 2], POS=[1, 2, 3])))
    return PreparedTrait('height', np.column_stack([np.arange(3), [1., 2., 4.]]), genotype,
                         np.ones((3, 1)), np.eye(3), np.arange(3), gmap)


@pytest.mark.parametrize('method,runner_name', [
    ('GLM', 'PANICLE_GLM'), ('MLM', 'PANICLE_MLM'), ('FARMCPU', 'PANICLE_FarmCPU'),
    ('BLINK', 'PANICLE_BLINK'), ('BAYESLOCO', 'PANICLE_BayesLOCO'),
])
def test_named_and_legacy_dispatch_agree(trait, monkeypatch, method, runner_name):
    association = AssociationResults(np.ones(3), np.ones(3), np.array([.1, .2, .3]))
    runner = Mock(return_value=association)
    monkeypatch.setattr(gwas, runner_name, runner)
    config = MethodOptions(farmcpu={'QTN_threshold': .02}, blink={'maxLoop': 3},
                           bayesloco={}, max_iterations=4, n_eff=7, ncpus=2, mlm_mode='global')
    named = gwas._execute_prepared_method(method, trait, config)
    named_args = runner.call_args.kwargs
    legacy = gwas._run_single_method(
        method, trait.phenotype, trait.genotype, trait.covariates, trait.kinship,
        trait.geno_map, config.farmcpu, config.blink, config.bayesloco,
        config.max_iterations, .05, 3, config.n_eff, ncpus=2, mlm_mode='global',
    )
    assert named.error is None and legacy[4] is None
    assert named.result is association and legacy[1] is association
    assert named.lambda_gc == legacy[2]
    for key, value in named_args.items():
        other = runner.call_args.kwargs[key]
        if isinstance(value, np.ndarray) or key in {'geno', 'map_data'}:
            assert value is other
        else:
            assert value == other
    assert named_args['phe'] is trait.phenotype
    if method == 'FARMCPU':
        assert named_args['QTN_threshold'] == .02  # solver alpha is not reporting correction
        assert named_args['n_eff'] == 7


def test_named_dispatch_records_failures(trait, monkeypatch):
    monkeypatch.setattr(gwas, 'PANICLE_GLM', Mock(side_effect=ValueError('bad trait')))
    result = gwas._execute_prepared_method('GLM', trait, MethodOptions())
    assert result.error == 'bad trait' and result.result is None
    assert gwas._execute_prepared_method('unknown', trait, MethodOptions()).error == 'Unknown method unknown'


def test_named_loco_inputs(trait, monkeypatch):
    runner = Mock(return_value=AssociationResults(np.ones(3), np.ones(3), np.array([.1, .2, .3])))
    monkeypatch.setattr(gwas, 'PANICLE_MLM_LOCO', runner)
    kinship = object()
    result = gwas._execute_prepared_method('MLM', trait, MethodOptions(loco_kinship=kinship, mlm={'cpu': 3}))
    assert result.error is None
    assert runner.call_args.kwargs['loco_kinship'] is kinship
    assert runner.call_args.kwargs['map_data'] is trait.geno_map
    assert runner.call_args.kwargs['cpu'] == 3


@pytest.mark.parametrize('outputs', [[], ['qq'], ['all_marker_pvalues'], ['significant_marker_pvalues']])
def test_named_report_matches_legacy_adapter(tmp_path, trait, outputs):
    association = AssociationResults(np.array([1., 2., 3.]), np.ones(3), np.array([.01, .05, .5]))
    base = Threshold(.1, 'base', 3)
    corrected = Threshold(.05, 'custom')
    run = MethodRunResult('GLM', association, lambda_gc=1.25, lambda_gc_is_approx=True)
    report = TraitReport('height', {'GLM': MethodReport(run, corrected)}, base,
                         3, 3, 1.234, trait.genotype)
    directories = [tmp_path / 'named', tmp_path / 'legacy']
    for directory in directories:
        directory.mkdir()
    renderers = [Mock(return_value={'plots': {}}) for _ in directories]
    contexts = [TraitOutputContext(directory, trait.geno_map, trait.genotype, Mock(), renderer)
                for directory, renderer in zip(directories, renderers)]
    named_summary = write_trait_report(contexts[0], report, ReportOptions(outputs, .05, 2., True))
    legacy_summary = write_trait_results(
        contexts[1], 'height', {'GLM': association}, .1, .05, 3, 2., outputs, 'base', True,
        method_thresholds={'GLM': .05}, method_threshold_sources={'GLM': 'custom'},
        method_lambda_gc={'GLM': 1.25}, method_lambda_gc_is_approx={'GLM': True},
        n_samples=3, n_markers=3, runtime_seconds=1.234, geno_for_maf=trait.genotype,
    )
    assert named_summary == legacy_summary
    assert named_summary[0]['Threshold'] == .05 and named_summary[0]['Significant_Hits'] == 2
    assert named_summary[0]['Lambda_GC'] == 1.25
    assert sorted(p.name for p in directories[0].iterdir()) == sorted(p.name for p in directories[1].iterdir())
    for path in directories[0].iterdir():
        assert path.read_bytes() == (directories[1] / path.name).read_bytes()
    if outputs == ['qq']:
        for renderer in renderers:
            kwargs = renderer.call_args.kwargs
            assert kwargs['threshold'] == .05 and kwargs['threshold_alpha'] is None
            assert np.isnan(kwargs['threshold_n_tests'])
            assert kwargs['method_lambda_gc'] == {'GLM': 1.25}
            assert kwargs['method_lambda_gc_is_approx'] == {'GLM': True}


def test_resampling_report_preserves_rmip_policy(tmp_path, trait):
    resampling = FarmCPUResamplingResults(
        [FarmCPUResamplingEntry(0, 'a', '1', 1, .1)], 'height', 10, False,
    )
    report = TraitReport('height', {'FarmCPUResampling': MethodReport(
        MethodRunResult('FarmCPUResampling', resampling), Threshold(.001, 'resampling override'),
    )}, Threshold(.05, 'base'), 3, 3)
    context = TraitOutputContext(tmp_path, trait.geno_map, trait.genotype, Mock(), Mock(return_value={'plots': {}}))
    summary = write_trait_report(context, report, ReportOptions(['all_marker_pvalues', 'manhattan'], .05, 2.))
    assert summary[0]['Threshold'] == .1  # RMIP reporting is not the solver p-value threshold
    assert summary[0]['Significant_Hits'] == 1
    assert 'resampling override' in summary[0]['Info']
    assert (tmp_path / 'GWAS_height_FarmCPUResampling_RMIP.csv').exists()
    assert context.renderer.call_args.kwargs['results'] is resampling


def test_numpy_map_selective_rows(trait):
    selected = trait.geno_map.to_dataframe_at(np.array([0, 2]))
    assert selected['MARKER'].tolist() == ['a', 'c']
