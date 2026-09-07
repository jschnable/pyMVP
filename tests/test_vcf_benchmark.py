"""Benchmark fingerprints stay compatible when hashed in bounded chunks."""
import hashlib
import gzip

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_vcf import fingerprint
from scripts.validate_vcf_panel import validate


@pytest.mark.parametrize('markers', [0, 3, 8200])
def test_streamed_fingerprint_matches_whole_output(markers):
    genotype = np.arange(3 * markers, dtype=np.int64).reshape(3, markers).astype(np.int8)
    frame = pd.DataFrame({'MARKER': [f'm{i}' for i in range(markers)],
                          'CHROM': ['chr1'] * markers, 'POS': np.arange(markers)})
    actual = fingerprint((genotype, ['a', 'b', 'c'], frame))
    assert actual['genotype_sha256'] == hashlib.sha256(genotype.tobytes(order='C')).hexdigest()
    assert actual['map_sha256'] == hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    assert actual['ids_sha256'] == hashlib.sha256(b'a\nb\nc').hexdigest()


def test_panel_validation_and_prefix(tmp_path):
    path = tmp_path / 'panel.vcf.gz'
    header = b'##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n'
    records = [f'1\t{i}\tm{i}\tA\tG\t.\tPASS\t.\tGT\t0/1\t1|1\n'.encode() for i in range(5)]
    path.write_bytes(gzip.compress(header + b''.join(records)))
    cache_path = str(path) + '.panicle.v2.geno.npy'
    np.save(cache_path, np.array([[1] * 5, [2] * 5], dtype=np.int8))
    from pathlib import Path
    Path(str(path) + '.panicle.v2.ind.txt').write_text('a\nb\n')
    prefix = tmp_path / 'prefix.vcf.gz'
    result = validate(path, cache_path, stride=2, prefix_path=prefix, prefix_markers=3)
    assert result['records'] == 5 and result['independently_checked_markers'] == 3
    assert gzip.decompress(prefix.read_bytes()) == header + b''.join(records[:3])
    np.save(cache_path, np.zeros((2, 5), dtype=np.int8))
    with pytest.raises(AssertionError):
        validate(path, cache_path)
