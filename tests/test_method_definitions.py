from panicle.cli.gwas import normalize_methods
from panicle.core.methods import (
    METHODS, method_definition, ordered_pipeline_methods, ordered_report_methods,
)


def test_cli_aliases_and_unknown_names_preserve_behavior():
    assert normalize_methods(['glm', 'GLM', 'FarmCPU', 'farmcpu-resampling', 'resampling',
                              'farmcpu resampling', 'BAYESLOCO', 'unknown', '']) == [
        'GLM', 'FARMCPU', 'FarmCPUResampling', 'BAYESLOCO', 'UNKNOWN',
    ]
    assert normalize_methods(None) == []
    for definition in METHODS.values():
        for alias in (definition.identifier,) + definition.aliases:
            assert normalize_methods([alias]) == [definition.cli_name]


def test_interface_orders_remain_distinct():
    assert ordered_pipeline_methods(reversed(list(METHODS))) == ['GLM', 'MLM', 'FARMCPU', 'BLINK', 'BAYESLOCO']
    results = {'custom': None, **{item.display_name: None for item in METHODS.values()}}
    assert ordered_report_methods(results) == ['GLM', 'MLM', 'BAYESLOCO', 'FarmCPU', 'BLINK', 'FarmCPUResampling', 'custom']
    # CLI aliases are not newly accepted by the Python pipeline.
    assert ordered_pipeline_methods(['RESAMPLING', 'unknown', 'GLM', 'GLM']) == ['GLM']


def test_solver_contracts_and_internal_loco_variant():
    assert method_definition('MLM').uses_kinship
    assert not method_definition('MLM').uses_map
    assert method_definition('MLM_LOCO').uses_map
    assert not method_definition('MLM_LOCO').uses_kinship
    assert method_definition('MLM_LOCO').display_name == 'MLM'
    assert 'MLM_LOCO' not in METHODS
    assert method_definition('unknown') is None
