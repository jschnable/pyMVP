"""Bounded-memory checks of a pre-imputed, biallelic GT-only production panel.

Checks record/sample order and sampled dosages against a freshly built PANICLE
cache. This deliberately uses independent scalar GT decoding, not loader kernels.
Run only with QC that retains every input record (the maize v2 provenance uses
min_maf=0, max_missing=1 and has no monomorphic records). Optional prefix output
provides a smaller, real-data baseline benchmark without copying the whole panel.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np


def validate(path, cache_path, *, stride=100000, prefix_path=None, prefix_markers=1000000):
    matrix = np.load(cache_path, mmap_mode='r')
    with open(str(path) + '.panicle.v2.ind.txt') as handle:
        cached_ids = [line.rstrip('\n') for line in handle]
    records = checked = 0
    formats = Counter()
    output = gzip.open(prefix_path, 'xb', compresslevel=1) if prefix_path else None
    try:
        with gzip.open(path, 'rb') as handle:
            for line in handle:
                if output is not None and (line.startswith(b'#') or records < prefix_markers):
                    output.write(line)
                if line.startswith(b'#CHROM'):
                    ids = line.rstrip(b'\r\n').decode().split('\t')[9:]
                    assert ids == cached_ids
                    assert matrix.shape[0] == len(ids)
                if line.startswith(b'#'):
                    continue
                fields = line.rstrip(b'\r\n').split(b'\t', 9)
                formats[fields[8].decode()] += 1
                assert fields[8] == b'GT' and b',' not in fields[4]
                if records % stride == 0 or records == matrix.shape[1] - 1:
                    calls = fields[9].split(b'\t')
                    assert len(calls) == matrix.shape[0]
                    dosages = []
                    for call in calls:
                        alleles = call.replace(b'|', b'/').split(b'/')
                        assert len(alleles) == 2 and all(a in (b'0', b'1') for a in alleles), call
                        dosages.append(sum(a == b'1' for a in alleles))
                    np.testing.assert_array_equal(matrix[:, records], dosages)
                    checked += 1
                records += 1
                if output is not None and records == prefix_markers:
                    output.close()
                    output = None
        assert records == matrix.shape[1], (records, matrix.shape)
    finally:
        if output is not None:
            output.close()
    return dict(records=records, samples=matrix.shape[0], independently_checked_markers=checked,
                checked_genotypes=checked * matrix.shape[0], formats=dict(formats),
                prefix_markers=min(records, prefix_markers) if prefix_path else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix-output', type=Path)
    args = parser.parse_args()
    result = validate(args.input, str(args.input) + '.panicle.v2.geno.npy', prefix_path=args.prefix_output)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
