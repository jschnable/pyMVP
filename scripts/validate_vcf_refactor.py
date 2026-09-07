"""Compare VCF behavior with another checkout selected via PYTHONPATH.

Small deterministic edge cases complement benchmark checksums. Each backend is
compared with its own baseline: backend-specific DS/ploidy behavior is not
silently standardized by this refactor. Temporary fixtures and caches are isolated.
"""
import argparse
import gzip
import json
import logging
from pathlib import Path
import tempfile

import pandas as pd

from panicle.data.load_genotype_vcf import load_genotype_vcf
from panicle.data import load_genotype_vcf as vcf_module


HEADER = ('##fileformat=VCFv4.2\n##contig=<ID=1>\n'
          '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
          '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
          '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depths">\n'
          '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n'
          '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2\ts3\ts4\ts5\n')


def record(position, calls, fmt='GT', ref='A', alt='G', marker='.'):
    return f'1\t{position}\t{marker}\t{ref}\t{alt}\t.\tPASS\t.\t{fmt}\t' + '\t'.join(calls) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--batch-markers', type=int, help='Force small batches to test resumable decoding')
    args = parser.parse_args()
    if args.batch_markers is not None:
        if args.batch_markers < 1:
            parser.error('--batch-markers must be positive')
        vcf_module._SIMPLE_BULK_BATCH_MARKERS = args.batch_markers
    logging.disable(logging.CRITICAL)  # Hundreds of intentional cache rebuilds.
    simple = record(1, ['0/0', '0/1', '1|1', './.', '0/.', '1/0'])
    fixtures = {
        'simple': simple,
        'monomorphic': record(1, ['0/0'] * 6) + record(2, ['1/1'] * 6) + record(3, ['0/1'] * 6),
        'missing': record(1, ['./.'] * 6) + simple,
        'indels': simple + record(2, ['0/0', '0/1', '1/1', '0/1', '0/0', '1/1'], ref='AT', alt='A'),
        'mixed': simple + record(2, ['0/0:8', '0/1:3', '1/1:4', './.:0', '0/0:8', '1/1:9'], 'GT:DP'),
        'ds_fallback': record(1, ['0/0:2', './.:1.8', '0/1:0', '.:0.2', '1/1:1', '.:.'], 'GT:DS'),
        'ds_only': record(1, ['0', '0.7', '2', '.', '1.2', '-1'], 'DS'),
        'reordered_format': record(1, ['2:0/0', '1:./.', '0:0/1', '0:0/0', '1:1/1', '.:./.'], 'DS:GT'),
        'multiallelic': simple + record(2, ['0/1', '0/2', '2/2', '1/2', './.', '0/0'], alt='G,T'),
        'haploid': record(1, ['0', '1', '.', '0', '1', '0']),
        'polyploid': record(1, ['0/0/0', '0/1/1', '1/1/1', '././.', '0/0/1', '0/1/0']),
        'empty': '',
        'variable_depth': record(1, ['0/0:8', '0|1:12345', '1/1:.', './.:0', '0/.:20', '1/0:7'], 'GT:DP'),
        'rich_format': record(1, ['0/0:8,0:8', '0/1:123,45:168', '1/1:0,3:3', './.:.:.',
                                  '0/.:20,0:20', '1/0:7,8:15'], 'GT:AD:DP') +
                       record(2, ['8:0/0', '12345:0|1', '.:1/1', '0:./.', '20:0/.', '7:1/0'], 'DP:GT'),
        'late_general': ''.join(record(i, ['0/0', '0/1', '1/1', './.', '0/.', '1/0']) for i in range(1, 6)) +
                        record(6, ['0/0:2', './.:1.8', '0/1:0', '.:0.2', '1/1:1', '.:.'], 'GT:DS') + simple,
        'short_gt_extra': record(1, ['0/0:8', '0/1:3', '1/1:4', '.:0', '0/.:8', '1/1:9'], 'GT:DP'),
    }
    filters = [{}, {'include_indels': False}, {'drop_monomorphic': True},
               {'max_missing': .2}, {'min_maf': .2}, {'split_multiallelic': False},
               {'drop_monomorphic': True, 'min_maf': .1, 'max_missing': .5}, {'max_missing': 1/3}]
    results = {}
    with tempfile.TemporaryDirectory(prefix='panicle-vcf-parity-') as directory:
        for name, records in fixtures.items():
            for compressed in [False, True]:
                path = Path(directory) / (name + ('.vcf.gz' if compressed else '.vcf'))
                opener = gzip.open if compressed else open
                with opener(path, 'wt') as handle:
                    handle.write(HEADER + records)
                for backend in ['builtin', 'cyvcf2']:
                    for index, options in enumerate(filters):
                        for pandas_output in [False, True]:
                            key = f'{name}/{compressed}/{backend}/{index}/{pandas_output}'
                            try:
                                genotype, ids, gmap = load_genotype_vcf(
                                    path, backend=backend, return_pandas=pandas_output,
                                    force_recache=True, **options,
                                )
                                frame = pd.DataFrame(gmap) if isinstance(gmap, list) else gmap
                                results[key] = dict(shape=list(genotype.shape), genotype=genotype.tolist(), ids=ids,
                                                    map=frame.to_dict('list'), columns=list(frame.columns),
                                                    contiguous=bool(genotype.flags.c_contiguous))
                            except Exception as exc:
                                results[key] = dict(error=type(exc).__name__, message=str(exc).replace(directory, '<tmp>'))
    if args.compare:
        before = json.loads(args.compare.read_text())
        changes = [key for key in results if results[key] != before.get(key)]
        assert not changes, f'Behavior changed: {changes[:20]}'
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    errors = sum('error' in result for result in results.values())
    print(f'{len(results)} cases recorded ({errors} expected/baseline error outcomes); comparisons passed.'
          if args.compare else f'{len(results)} baseline cases recorded ({errors} error outcomes).')


if __name__ == '__main__':
    main()
