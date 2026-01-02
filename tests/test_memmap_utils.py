import numpy as np
import pandas as pd
import pytest

from panicle.utils.data_types import GenotypeMap, GenotypeMatrix
from panicle.utils.memmap_utils import (
    estimate_memory_usage,
    load_full_from_metadata,
    save_genotype_to_memmap,
)


def test_save_and_load_genotype_memmap_round_trip(tmp_path) -> None:
    genotype_array = np.array(
        [
            [0, 1, 2],
            [2, 1, 0],
        ],
        dtype=np.int8,
    )
    genotype = GenotypeMatrix(genotype_array)
    sample_ids = ["ind1", "ind2"]
    geno_map = GenotypeMap(
        pd.DataFrame(
            {
                "SNP": ["snp1", "snp2", "snp3"],
                "CHROM": ["1", "1", "2"],
                "POS": [100, 200, 300],
            }
        )
    )

    output_prefix = tmp_path / "cache" / "geno_cache"
    result = save_genotype_to_memmap(
        genotype,
        sample_ids=sample_ids,
        geno_map=geno_map,
        output_prefix=output_prefix,
        dtype=np.int8,
        batch_size=2,
    )

    assert result["memmap_path"].exists()
    assert result["metadata_path"].exists()
    assert result["samples_path"].exists()
    assert result["map_path"] and result["map_path"].exists()

    loaded_genotype, loaded_samples, loaded_map = load_full_from_metadata(result["metadata_path"])

    np.testing.assert_array_equal(loaded_genotype[:, :], genotype_array)
    assert loaded_genotype.shape == genotype_array.shape
    assert loaded_samples == sample_ids
    assert loaded_map is not None
    pd.testing.assert_frame_equal(loaded_map.to_dataframe(), geno_map.to_dataframe())


def test_estimate_memory_usage_matches_manual_calculation() -> None:
    shape = (2, 3)
    dtype = np.int8
    estimates = estimate_memory_usage(shape, dtype=dtype)

    bytes_per_element = np.dtype(dtype).itemsize
    genotype_bytes = shape[0] * shape[1] * bytes_per_element
    major_bytes = shape[1] * bytes_per_element
    batch_bytes = shape[0] * min(1000, shape[1]) * 8
    expected_total = genotype_bytes + major_bytes + batch_bytes

    assert estimates["genotype_mb"] == pytest.approx(genotype_bytes / (1024**2))
    assert estimates["major_alleles_mb"] == pytest.approx(major_bytes / (1024**2))
    assert estimates["batch_working_mb"] == pytest.approx(batch_bytes / (1024**2))
    assert estimates["total_mb"] == pytest.approx(expected_total / (1024**2))
    assert estimates["total_gb"] == pytest.approx(expected_total / (1024**3))
