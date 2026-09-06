import numpy as np
import pytest
from pathlib import Path

from panicle.data.load_genotype_vcf import _DynamicInt8MatrixWriter


def test_dynamic_int8_writer_finalize_preserves_columns():
    rows = 6
    writer = _DynamicInt8MatrixWriter(n_rows=rows, initial_capacity=16)

    columns = [
        np.array([2, 0, 0, 2, -9, 1], dtype=np.int8),
        np.array([0, 2, 0, 0, 1, -9], dtype=np.int8),
        np.array([1, 1, 1, 1, 1, 1], dtype=np.int8),
    ]

    for col in columns:
        writer.append(col)

    expected = np.column_stack(columns)
    matrix = writer.finalize()

    np.testing.assert_array_equal(matrix, expected)


def test_dynamic_int8_writer_finalize_handles_capacity_growth():
    rows = 5
    writer = _DynamicInt8MatrixWriter(n_rows=rows, initial_capacity=1)

    columns = [
        np.array([2, 0, -9, 1, 0], dtype=np.int8),
        np.array([0, 2, 0, -9, 1], dtype=np.int8),
        np.array([1, 1, 1, 1, 1], dtype=np.int8),
        np.array([2, 2, 2, 2, 2], dtype=np.int8),
    ]

    for col in columns:
        writer.append(col)

    # ensure append triggered an internal resize
    assert writer.capacity >= len(columns)

    expected = np.column_stack(columns)
    matrix = writer.finalize()

    np.testing.assert_array_equal(matrix, expected)


def test_dynamic_writer_mixed_appends_preserve_prefix_and_close_mapping():
    rng = np.random.default_rng(6)
    expected = rng.integers(-9, 3, size=(7, 79), dtype=np.int8)
    writer = _DynamicInt8MatrixWriter(7, initial_capacity=1)
    try:
        writer.append(expected[:, 0])
        old_mapping = writer.memmap._mmap
        writer.append_block(expected[:, 1:8])
        assert old_mapping.closed
        writer.append_block(expected[:, 8:8])
        writer.append(expected[:, 8])
        writer.append_block(expected[:, 9:])
        mapping = writer.memmap._mmap
        result = writer.finalize()
        np.testing.assert_array_equal(result, expected)
        assert result.flags.c_contiguous
        assert mapping.closed
        assert not Path(writer.path).exists()
    finally:
        writer.discard()


def test_dynamic_writer_remap_failure_is_cleanable(monkeypatch):
    writer = _DynamicInt8MatrixWriter(3, initial_capacity=1)
    writer.append(np.array([0, -9, 2], dtype=np.int8))
    mapping = writer.memmap._mmap

    def fail(*args, **kwargs):
        raise OSError("simulated mapping failure")

    monkeypatch.setattr(np, "memmap", fail)
    try:
        with pytest.raises(OSError, match="simulated"):
            writer.append(np.ones(3, dtype=np.int8))
        assert mapping.closed
    finally:
        writer.discard()
    assert not Path(writer.path).exists()


def test_dynamic_writer_discard_closes_mapping():
    writer = _DynamicInt8MatrixWriter(3)
    mapping = writer.memmap._mmap
    writer.discard()
    writer.discard()
    assert mapping.closed
    assert not Path(writer.path).exists()
