import numpy as np

from pymvp.data.load_genotype_vcf import _DynamicInt8MatrixWriter


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
