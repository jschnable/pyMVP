"""Method identities and input contracts, independent of solver imports.

Aliases are a CLI policy. Python interfaces retain their existing accepted names;
execution and reporting orders are deliberately separate.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Tuple


@dataclass(frozen=True)
class MethodDefinition:
    identifier: str
    display_name: str
    cli_name: str
    aliases: Tuple[str, ...] = ()
    uses_map: bool = False
    uses_kinship: bool = False


METHODS = MappingProxyType({definition.identifier: definition for definition in (
    MethodDefinition('GLM', 'GLM', 'GLM'),
    MethodDefinition('MLM', 'MLM', 'MLM', uses_kinship=True),
    MethodDefinition('FARMCPU', 'FarmCPU', 'FARMCPU', uses_map=True),
    MethodDefinition('BLINK', 'BLINK', 'BLINK', uses_map=True),
    MethodDefinition('BAYESLOCO', 'BAYESLOCO', 'BAYESLOCO', uses_map=True),
    MethodDefinition('FARMCPURESAMPLING', 'FarmCPUResampling', 'FarmCPUResampling',
                     ('FARMCPU_RESAMPLING', 'RESAMPLING'), uses_map=True),
)})
# Internal execution variant, not an additional public/CLI method.
LOCO_METHOD = MethodDefinition('MLM_LOCO', 'MLM', 'MLM_LOCO', uses_map=True)

PIPELINE_EXECUTION_ORDER = ('GLM', 'MLM', 'FARMCPU', 'BLINK', 'BAYESLOCO')
PIPELINE_REPORT_ORDER = ('GLM', 'MLM', 'BAYESLOCO', 'FARMCPU', 'BLINK', 'FARMCPURESAMPLING')


def method_definition(identifier):
    return LOCO_METHOD if identifier == 'MLM_LOCO' else METHODS.get(identifier)


def normalize_cli_methods(methods):
    aliases = {alias: definition.cli_name for definition in METHODS.values()
               for alias in (definition.identifier,) + definition.aliases}
    normalized = []
    for value in methods or ():
        key = str(value).replace('-', '_').replace(' ', '_').strip().upper()
        name = aliases.get(key, key)
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def ordered_pipeline_methods(methods):
    selected = {method.upper() for method in methods}
    return [method for method in PIPELINE_EXECUTION_ORDER if method in selected]


def ordered_report_methods(results):
    preferred = [METHODS[key].display_name for key in PIPELINE_REPORT_ORDER]
    ordered = [name for name in preferred if name in results]
    return ordered + [name for name in results if name not in ordered]
