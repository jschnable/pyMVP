"""Tests for CLI helpers and progress utilities in scripts.run_GWAS."""

from importlib import util
from pathlib import Path
from types import ModuleType

import pytest


def _load_run_gwas_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_GWAS.py"
    spec = util.spec_from_file_location("run_GWAS_cli", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_GWAS.py module for testing")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


RUN_GWAS = _load_run_gwas_module()


def test_normalize_outputs_accepts_comma_separated_values() -> None:
    outputs = RUN_GWAS.normalize_outputs(['manhattan,qq'])
    assert outputs == ['manhattan', 'qq']


def test_normalize_outputs_mixed_separators_and_deduplication() -> None:
    outputs = RUN_GWAS.normalize_outputs(['manhattan', 'qq', 'manhattan'])
    assert outputs == ['manhattan', 'qq']


def test_normalize_outputs_defaults_when_missing() -> None:
    outputs = RUN_GWAS.normalize_outputs(None)
    assert outputs == list(RUN_GWAS.OUTPUT_CHOICES)


def test_normalize_outputs_rejects_invalid_choice() -> None:
    with pytest.raises(ValueError):
        RUN_GWAS.normalize_outputs(['invalid_output'])


def test_resampling_progress_reports_initial_message(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = RUN_GWAS.FarmCPUResamplingProgressReporter('TraitA', total_runs=10)
    reporter(1, 10, 5.0)
    captured = capsys.readouterr()
    assert 'TraitA' in captured.out
    assert 'started' in captured.out


def test_resampling_progress_reports_eta_after_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = RUN_GWAS.FarmCPUResamplingProgressReporter('TraitB', total_runs=40)
    reporter(1, 40, 12.0)
    capsys.readouterr()
    reporter(6, 40, 70.0)
    captured = capsys.readouterr()
    assert 'progress' in captured.out
    assert 'ETA' in captured.out


def test_resampling_progress_reports_completion(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = RUN_GWAS.FarmCPUResamplingProgressReporter('TraitC', total_runs=5)
    reporter(5, 5, 15.0)
    captured = capsys.readouterr()
    assert 'finished' in captured.out
    assert '15' in captured.out
