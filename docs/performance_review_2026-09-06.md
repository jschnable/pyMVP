# PANICLE performance review — 2026-09-06

This is the baseline review at `da717f9`. Items 2–5 have since been implemented;
see the [implementation and validation follow-up](performance_changes_2026-09-06.md).

The largest opportunities are CSV serialization, repeated genotype conversion in
association scans, and BLINK's unnecessary copying and MAF work. Warm-cache runs
on this machine did not demonstrate a storage-device bottleneck. Temporary VCF
matrix growth does perform avoidable writes, and a small prototype improved that
stage without changing the resulting matrix.

Benchmarked commit: `da717f9`. Production code was not modified. The added
[stage harness](../scripts/profile_stages.py),
[isolated experiments](../scripts/benchmark_hotspots.py), and
[individual measurements](benchmarks/2026-09-06.json) make the findings reviewable.

Environment: macOS 26.2, ARM64, 10 logical CPUs; Python 3.14.7, NumPy 2.5.1 linked
to Accelerate, SciPy 1.18.0, pandas 3.0.5, Numba 0.67.0, cyvcf2 0.34.0.
Association calls used `cpu=1` except the explicit prefetch comparison. BLAS and
Numba environment variables were unset; this is not a claim that every operation
used exactly one hardware thread. A temporary virtual environment supplied missing
cyvcf2 and h5py dependencies.

The real-data test used the bundled 738-sample, 6,533-marker VCF and PlantHeight
(726 samples after phenotype missingness). Synthetic tests used 738 samples,
100,000 or 1,000,000 complete int8 markers, ten chromosomes, three covariates,
and two planted effects, with seed 20260906. These independent synthetic markers
do not reproduce real linkage disequilibrium or all iterative-model behavior.

Times below are medians of three unprofiled warm repetitions. Imports, fixture
creation, and separate cProfile runs are excluded. First-call times are retained
in the measurement file. Filesystem caches were not evicted and writes were not
fsynced; these are application timings, not durable-write throughput benchmarks.

| Stage | 100,000 markers | 1,000,000 markers |
|---|---:|---:|
| PCA, 3 components | 0.182 s | 0.467 s |
| Global kinship construction | 0.145 s | 1.436 s |
| LOCO kinship construction | 0.161 s | 1.551 s |
| GLM scan | 0.050 s | 0.504 s |
| Global MLM scan, supplied kinship | 0.304 s | 2.687 s |
| LOCO MLM scan, supplied kinship and warmed eigen cache | 0.321 s | 2.603 s |
| FarmCPU, default convergence/max 10 iterations | 0.423 s | 5.019 s |
| BLINK, default convergence/max 10 iterations | 0.461 s | 5.496 s |
| Write GLM full + significant result tables | 0.276 s | 2.595 s |
| Generate GLM Manhattan + QQ PNGs | 0.174 s | 0.249 s |

These stages are not one additive pipeline total: global and LOCO kinship are
alternatives, and the methods consume supplied structure/covariates. PCA samples
200,000 markers above its 500,000-marker threshold, explaining its sublinear
scaling. LOCO's first one-million-marker call took 3.308 s versus 2.603 s warm;
the warm figure benefits from in-process eigen caching. The first 100k LOCO call
also included JIT work and took 3.390 s. Do not extrapolate warm timings to a fresh
process or a larger sample count: kinship/transforms scale roughly with sample
count squared, and eigendecomposition with sample count cubed.

On the bundled dataset, the complete in-process GLM workflow took 0.067 s with
no requested tables/plots, 0.093 s with tables, and 0.251 s with all outputs.
The built-in VCF parser took 0.028 s when forced to reparse with warmed JIT and
filesystem caches, cyvcf2 took 0.295 s, and opening the binary cache took 0.004 s.
Opening a memmap does not mean all its genotype bytes have been physically read.
The first built-in parser invocation was 1.388 s, including first-use/JIT overhead.
These results apply to the demo's simple GT records, not BCF or complex VCF formats.

Recommended priorities, with measured evidence:

1. **Offer binary result output and avoid unnecessary full tables.** At one million
   markers, writing a single-method table takes five times as long as its GLM scan
   and produces 86.99 MB. In the output profile, CSV formatting/writing accounts
   for 1.915 of 2.534 s; another 0.395 s recomputes MAF. In a separate 100k-row,
   three-column comparison, CSV took 0.1101 s to disk and 0.1099 s to StringIO.
   This strongly implicates serialization CPU cost rather than storage latency
   for that workload. An uncompressed HDF5 prototype took 0.0040 s and 3.20 MB,
   versus CSV's 4.71 MB. This is a format microbenchmark, not a demonstrated 28x
   pipeline improvement. HDF5 is already a dependency; a production format needs
   variable-length marker IDs, complete map columns, metadata, and reader support
   (the prototype uses short fixed-width IDs). Cache MAF by sample subset, marker
   selection, and dosage, or reuse compatible sums from MAC filtering. Existing
   `--outputs significant_marker_pvalues` is useful when the full table and plots
   are unnecessary.

2. **Remove intermediate int8 copies on appropriate conversion paths.** The
   one-million-marker GLM profile spends 0.274 of 0.521 s loading/converting
   batches. [int8_to_float32](../panicle/utils/compact.py) first makes a contiguous
   int8 copy, then fills a float32 destination. A direct strided-to-C-order float32
   conversion prototype reduced the 100k GLM median from 0.0498 to 0.0460 s and
   BLINK from 0.5030 to 0.4141 s (about 8% and 18% less time), with exact equality
   of effects, standard errors, and p-values. Do not replace the parallel path
   indiscriminately: at 738 × 20,000 elements, the existing converter took
   0.00192 s versus 0.00483 s for the direct cast. Select by layout and batch size,
   preserve the large-block parallel path, and consider reusable output buffers.

3. **Preserve BLINK's input when its MAF filter keeps every marker.**
   [BLINK](../panicle/association/blink.py) unconditionally executes
   `genotype_array[:, filtered_indices]`, including the default zero-threshold
   case. This copies the entire matrix into a layout that incurs further
   contiguous conversions in every GLM iteration. At one million markers,
   contiguous copying alone accounts for 2.247 of the 5.627 s BLINK profile.
   BLINK also passes an ndarray into MAF calculation, losing the pre-imputed
   GenotypeMatrix fast path: masked MAF calculation costs another 0.755 s and
   allocates full-size masks. Keep the original matrix when nothing is removed;
   retain pre-imputation metadata and use/cache its allele-frequency calculation.
   The proposed no-filter shortcut itself has not been benchmarked here; the
   conversion prototype above addresses only part of this cost.

4. **Use the existing shared-trait APIs and evaluate prefetch for large scans.**
   Five-repetition experiments with eight actual trait columns took 0.403 vs
   0.0665 s at 100k markers and 4.197 vs 0.659 s at one million: about 6.1–6.4x
   faster jointly than eight separate GLM calls. The pipeline already groups
   compatible sample subsets, so submit traits together or use `PANICLE_GLM_MULTI`
   rather than adding another grouping implementation. Joint results matched
   separate scans within `rtol=1e-4, atol=1e-6`. Initial exploratory grouped
   measurements were superseded by this corrected, checked comparison. Separately,
   GLM `cpu=4` enabled its existing batch prefetch and reduced 100k scan time from
   0.0498 to 0.0380 s; this is not four-thread BLAS scaling. Benchmark thread
   settings on the deployment machine. Warm memmap GLM took 0.0503 s, effectively
   the same as the 0.0498 s RAM case at this size.

5. **Stop rewriting existing data when the VCF temporary file grows.**
   [_DynamicInt8MatrixWriter._grow](../panicle/data/load_genotype_vcf.py) copies
   existing mapped data to RAM, extends the file, remaps it, then copies those
   same bytes back. Extending a file preserves the existing prefix. A prototype
   retaining flush/extend/remap but omitting preservation/rewrite reduced building
   and finalizing a 738 × 100k matrix from 0.1719 to 0.1474 s, with exact matrix
   equality. With 4,096-marker appends, seven growths revisit 126,976 already
   written columns: 93.7 MB of redundant writes for a final 73.8 MB matrix, plus
   the corresponding reads/copies. Actual bulk-loader append sizes differ.
   Longer term, finalize directly into an atomic `.npy` cache with bounded
   conversion buffers; currently finalization creates a full C-order RAM copy
   before `np.save` writes the cache. Validate filesystem portability, failure
   cleanup, memory use, and filter fingerprints before implementing this.

6. **Optimize matrix transforms before scalar statistics in MLM.** At one million
   markers, global MLM's batch builder takes 1.895 of 2.650 profiled seconds,
   including 1.566 s of its own work, principally the eigenvector/genotype product.
   P-value calculation takes only 0.106 s. Preserve existing LOCO eigen reuse and
   chromosome-major shared-trait transforms. For global MLM, evaluate larger
   transform batches and a shared-trait transform path with a bounded memory
   budget. Changing to global MLM or disabling LOCO LRT refinement changes the
   analysis and should not be presented as an equivalent performance optimization.

Plots are mostly rendering and PNG encoding in these measurements. Existing
downsampling already keeps one-million-marker plot generation below 0.3 s, so
plotting changes rank below CSV and genotype layout work for large datasets.
For many tiny jobs, defer plotting and keep a process alive to amortize imports,
font setup, and JIT initialization.

The process-wide peak RSS reached 4.53 GB during the million-marker suite. This
is a cumulative high-water mark across repeated stages, not a per-method memory
requirement. `getrusage` block-I/O counters returned zero on this host; they cannot
establish physical bytes read/written. Files fit in memory, so storage-constrained
claims require a separate larger-than-RAM or controlled cold-cache run on the
target storage. BAYESLOCO, resampling, effective-test estimation, PLINK/BCF/HapMap,
large real-data parsing, and many distinct phenotype-missingness patterns were
not benchmarked. The README's 5.75M-marker timings were not independently reproduced.

Reproduce with installed project dependencies, running suites sequentially. Use
the `da717f9` checkout as `PYTHONPATH` for baseline measurements; the current
source includes the subsequent improvements:

```sh
PYTHONPATH=. MPLBACKEND=Agg python scripts/profile_stages.py --markers 100000
PYTHONPATH=. MPLBACKEND=Agg python scripts/profile_stages.py --markers 1000000
PYTHONPATH=. MPLBACKEND=Agg python scripts/benchmark_hotspots.py
```

Each command prints a fresh temporary artifact directory. Stage runs retain
individual wall/CPU times, environment details, cProfile files and summaries,
output files, and logical file sizes. Optional `--output` must name a new directory.
Original stage profiles remain in `/tmp/panicle-benchmark-main` and
`/tmp/panicle-benchmark-million`; microbenchmark artifacts are in
`/var/folders/19/5d0my4mn3694xgmjjj_nwpvw0000gn/T/panicle-hotspots-mzvr7gr6`.
The checked-in measurement JSON retains the valid comparisons if temporary
directories are later cleared. Profile filenames reference the harness's original
name, `benchmark_performance.py`, subsequently renamed `profile_stages.py`.
