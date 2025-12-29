import numpy as np

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
