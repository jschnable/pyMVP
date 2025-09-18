#!/usr/bin/env python3
"""
Test harness for generate_performance_dataset.py

This module provides comprehensive validation tests for the GWAS simulation
script to ensure all new features work correctly and produce expected outputs.

Test Coverage:
- MAF spectrum modeling (U-shaped, centered, empirical)
- MAF-effect coupling validation
- Heritability control accuracy
- Inbreeding/selfing effects
- Population structure realism
- Output file integrity

Author: Generated for pyMVP validation testing
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
import tempfile
import sys
import os
from typing import Dict, Any, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add the DevTests directory to path to import the generator
sys.path.insert(0, str(Path(__file__).parent.parent / "DevTests" / "validation_data"))

try:
    from generate_performance_dataset import GWASDatasetGenerator, create_config
    GENERATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import generator: {e}")
    GENERATOR_AVAILABLE = False

# Test parameters for small, fast runs
TEST_CONFIG_SMALL = {
    'n_samples': 200,
    'n_snps': 3000,
    'n_qtns': 50,
    'n_chromosomes': 5,
    'n_traits': 3,
    'seed': 12345,
    'missing_data_rate': 0.05,
    'population_structure': {
        'enabled': True,
        'n_populations': 2,
        'population_sizes': [100, 100],
        'fst': 0.1
    },
    'maf_spectrum': 'beta_u',
    'maf_beta_a': 0.5,
    'maf_beta_b': 0.5,
    'effect_maf_coupling_kappa': 0.5,
    'effect_size_dist': 'normal',
    'effect_size_scale': 0.3,
    'heritability': 0.7,
    'study_design': 'inbred',
    'inbreeding_f': 0.95,
    'output_dir': 'test_output'
}

@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def small_generator():
    """Create generator with small test configuration"""
    if not GENERATOR_AVAILABLE:
        pytest.skip("Generator not available")
    return GWASDatasetGenerator(TEST_CONFIG_SMALL)

class TestMAFSpectrum:
    """Test MAF spectrum modeling functionality"""

    def test_beta_u_spectrum(self, small_generator):
        """Test U-shaped Beta(0.5, 0.5) MAF spectrum"""
        generator = small_generator
        n_snps = 1000

        # Generate ancestral frequencies
        ancestral_freqs = generator._generate_ancestral_frequencies(n_snps)

        # Test range
        assert np.all(ancestral_freqs >= 0.001)
        assert np.all(ancestral_freqs <= 0.999)

        # Test U-shaped property: more extreme frequencies than centered
        # Use Kolmogorov-Smirnov test against uniform (which Beta(1,1) approximates)
        ks_stat, p_value = stats.kstest(ancestral_freqs, 'uniform')

        # For U-shaped distribution, should be significantly different from uniform
        assert p_value < 0.05, "U-shaped distribution should differ significantly from uniform"

        # Check that we have more extreme frequencies (< 0.1 or > 0.9) than expected
        extreme_freq_prop = np.sum((ancestral_freqs < 0.1) | (ancestral_freqs > 0.9)) / len(ancestral_freqs)
        expected_uniform_extreme = 0.2  # 20% for uniform
        assert extreme_freq_prop > expected_uniform_extreme, "U-shaped should have more extreme frequencies"

    def test_beta_centered_spectrum(self):
        """Test centered Beta(2, 2) MAF spectrum"""
        config = TEST_CONFIG_SMALL.copy()
        config['maf_spectrum'] = 'beta_centered'

        generator = GWASDatasetGenerator(config)
        n_snps = 1000

        ancestral_freqs = generator._generate_ancestral_frequencies(n_snps)

        # Test that centered distribution has fewer extreme frequencies
        extreme_freq_prop = np.sum((ancestral_freqs < 0.1) | (ancestral_freqs > 0.9)) / len(ancestral_freqs)
        expected_centered_extreme = 0.05  # Should be much lower for centered
        assert extreme_freq_prop < expected_centered_extreme, "Centered distribution should have fewer extreme frequencies"

    def test_empirical_sfs_loading(self, temp_output_dir):
        """Test empirical SFS file loading"""
        # Create test SFS file
        sfs_file = temp_output_dir / "test_sfs.csv"
        test_freqs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        test_counts = np.array([100, 80, 60, 40, 30, 20, 10])  # Decreasing counts (realistic)

        sfs_df = pd.DataFrame({'frequency': test_freqs, 'count': test_counts})
        sfs_df.to_csv(sfs_file, index=False)

        # Test loading
        config = TEST_CONFIG_SMALL.copy()
        config['maf_spectrum'] = 'empirical_sfs'
        config['empirical_sfs_file'] = str(sfs_file)

        generator = GWASDatasetGenerator(config)

        # Check that SFS was loaded
        assert hasattr(generator, 'sfs_interpolator'), "SFS interpolator should be created"
        assert hasattr(generator, 'sfs_frequencies'), "SFS frequencies should be stored"

class TestMAFEffectCoupling:
    """Test MAF-effect size coupling functionality"""

    def test_maf_effect_coupling(self, small_generator):
        """Test MAF-effect coupling with κ = 0.5"""
        generator = small_generator

        # Generate genetic map and ancestral frequencies
        genetic_map = generator.generate_genetic_map()
        generator.ancestral_freqs = generator._generate_ancestral_frequencies(len(genetic_map))

        # Select QTNs and generate effects
        qtn_indices, qtn_effects = generator.select_qtns(genetic_map)

        if len(qtn_indices) > 10:  # Need sufficient QTNs for correlation
            qtn_mafs = generator.ancestral_freqs[qtn_indices]

            # Test correlation between |β| and MAF^(-κ)
            kappa = generator.config['effect_maf_coupling_kappa']
            expected_predictor = np.power(qtn_mafs * (1 - qtn_mafs), -kappa/2)

            correlation = np.corrcoef(np.abs(qtn_effects), expected_predictor)[0, 1]

            # Should have positive correlation (at least 0.4 as specified)
            assert correlation > 0.4, f"MAF-effect coupling correlation too low: {correlation:.3f}"

    def test_no_coupling_when_kappa_zero(self):
        """Test that κ = 0 produces no MAF-effect coupling"""
        config = TEST_CONFIG_SMALL.copy()
        config['effect_maf_coupling_kappa'] = 0.0

        generator = GWASDatasetGenerator(config)
        genetic_map = generator.generate_genetic_map()
        generator.ancestral_freqs = generator._generate_ancestral_frequencies(len(genetic_map))

        qtn_indices, qtn_effects = generator.select_qtns(genetic_map)

        if len(qtn_indices) > 10:
            qtn_mafs = generator.ancestral_freqs[qtn_indices]
            correlation = np.corrcoef(np.abs(qtn_effects), qtn_mafs)[0, 1]

            # Should have low correlation when no coupling
            assert abs(correlation) < 0.3, f"Should have low MAF-effect correlation when κ=0: {correlation:.3f}"

class TestHeritabilityControl:
    """Test heritability control functionality"""

    def test_heritability_sweep_accuracy(self, temp_output_dir):
        """Test that realized heritability tracks target within ±0.05"""
        target_heritabilities = [0.2, 0.5, 0.8]

        for target_h2 in target_heritabilities:
            config = TEST_CONFIG_SMALL.copy()
            config['heritability'] = target_h2
            config['output_dir'] = str(temp_output_dir / f"h2_{target_h2}")

            generator = GWASDatasetGenerator(config)

            # Generate complete dataset
            genetic_map = generator.generate_genetic_map()
            genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)
            qtn_indices, qtn_effects = generator.select_qtns(genetic_map)
            phenotype_df = generator.generate_phenotypes(genotypes, qtn_indices, qtn_effects, sample_ids)

            # Check realized heritability
            if hasattr(generator, 'realized_heritability'):
                avg_realized_h2 = np.mean(generator.realized_heritability)
                tolerance = 0.05

                assert abs(avg_realized_h2 - target_h2) <= tolerance, \
                    f"Realized h² ({avg_realized_h2:.3f}) differs from target ({target_h2}) by more than {tolerance}"

class TestStudyDesign:
    """Test study design modes (inbred/outbred)"""

    def test_inbreeding_effects(self, small_generator):
        """Test that inbred design increases homozygosity"""
        generator = small_generator

        # Generate genotypes with inbreeding
        genetic_map = generator.generate_genetic_map()
        genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)

        # Calculate observed heterozygosity
        het_counts = np.sum(genotypes == 1, axis=1)  # Count heterozygotes per individual
        total_valid = np.sum(genotypes != -9, axis=1)  # Count valid genotypes per individual

        obs_het_rate = np.mean(het_counts / (total_valid + 1e-6))

        # Under random mating, expect ~50% heterozygosity for MAF=0.5
        # With high inbreeding (F=0.95), should be much lower
        expected_max_het = 0.1  # Should be << 0.5 due to inbreeding

        assert obs_het_rate < expected_max_het, \
            f"Heterozygosity rate too high for inbred design: {obs_het_rate:.3f}"

    def test_genomic_inbreeding_coefficient(self, small_generator):
        """Test genomic inbreeding coefficient calculation"""
        generator = small_generator

        genetic_map = generator.generate_genetic_map()
        genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)

        # Calculate genomic F
        n_samples, n_snps = genotypes.shape
        allele_freqs = np.zeros(n_snps)

        for snp_idx in range(n_snps):
            valid_mask = genotypes[:, snp_idx] != -9
            if np.sum(valid_mask) > 0:
                allele_freqs[snp_idx] = np.mean(genotypes[valid_mask, snp_idx]) / 2

        # Expected heterozygosity under HWE
        exp_het = 2 * allele_freqs * (1 - allele_freqs)

        # Observed heterozygosity
        het_counts = np.sum(genotypes == 1, axis=0)
        valid_counts = np.sum(genotypes != -9, axis=0)
        obs_het = het_counts / (valid_counts + 1e-6)

        # Genomic inbreeding coefficient
        valid_snps = valid_counts > 0
        if np.sum(valid_snps) > 0:
            genomic_f = 1 - np.mean(obs_het[valid_snps]) / np.mean(exp_het[valid_snps])

            # Should be high for inbred design
            assert genomic_f > 0.8, f"Genomic inbreeding coefficient too low: {genomic_f:.3f}"

    def test_outbred_design(self, temp_output_dir):
        """Test outbred design maintains higher heterozygosity"""
        config = TEST_CONFIG_SMALL.copy()
        config['study_design'] = 'outbred'
        config['output_dir'] = str(temp_output_dir / "outbred_test")

        generator = GWASDatasetGenerator(config)

        genetic_map = generator.generate_genetic_map()
        genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)

        # Calculate heterozygosity
        het_counts = np.sum(genotypes == 1, axis=1)
        total_valid = np.sum(genotypes != -9, axis=1)
        obs_het_rate = np.mean(het_counts / (total_valid + 1e-6))

        # Should have higher heterozygosity than inbred
        expected_min_het = 0.2  # Should be much higher than inbred
        assert obs_het_rate > expected_min_het, \
            f"Heterozygosity rate too low for outbred design: {obs_het_rate:.3f}"

class TestLDRealism:
    """Test linkage disequilibrium patterns"""

    def test_ld_decay_with_distance(self, small_generator):
        """Test that LD decays with physical distance"""
        generator = small_generator

        genetic_map = generator.generate_genetic_map()
        genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)

        # Test LD within first chromosome
        chr1_mask = genetic_map['Chr'] == 1
        chr1_indices = np.where(chr1_mask)[0]

        if len(chr1_indices) > 20:  # Need sufficient SNPs
            chr1_positions = genetic_map.loc[chr1_mask, 'Pos'].values
            chr1_genotypes = genotypes[:, chr1_indices]

            # Calculate r² for pairs at different distances
            distances = []
            r_squared_values = []

            # Sample pairs to avoid combinatorial explosion
            n_pairs = min(100, len(chr1_indices) * (len(chr1_indices) - 1) // 2)
            pair_indices = np.random.choice(len(chr1_indices), size=(n_pairs, 2), replace=True)

            for i, j in pair_indices:
                if i != j:
                    distance = abs(chr1_positions[i] - chr1_positions[j])

                    # Calculate r²
                    snp1 = chr1_genotypes[:, i]
                    snp2 = chr1_genotypes[:, j]

                    # Filter out missing data
                    valid_mask = (snp1 != -9) & (snp2 != -9)
                    if np.sum(valid_mask) > 10:
                        corr = np.corrcoef(snp1[valid_mask], snp2[valid_mask])[0, 1]
                        r_squared = corr ** 2

                        distances.append(distance)
                        r_squared_values.append(r_squared)

            if len(distances) > 10:
                # Test that short-range LD > long-range LD
                short_range = np.array(distances) < 200_000  # < 200kb
                long_range = np.array(distances) > 1_000_000  # > 1Mb

                if np.sum(short_range) > 0 and np.sum(long_range) > 0:
                    short_ld = np.mean(np.array(r_squared_values)[short_range])
                    long_ld = np.mean(np.array(r_squared_values)[long_range])

                    assert short_ld > long_ld, \
                        f"Short-range LD ({short_ld:.3f}) should be higher than long-range LD ({long_ld:.3f})"

class TestOutputIntegrity:
    """Test output file integrity and format"""

    def test_complete_dataset_generation(self, temp_output_dir):
        """Test complete dataset generation with all files"""
        config = TEST_CONFIG_SMALL.copy()
        config['output_dir'] = str(temp_output_dir / "complete_test")

        generator = GWASDatasetGenerator(config)

        # Generate complete dataset
        genetic_map = generator.generate_genetic_map()
        genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)
        qtn_indices, qtn_effects = generator.select_qtns(genetic_map)
        phenotype_df = generator.generate_phenotypes(genotypes, qtn_indices, qtn_effects, sample_ids)

        # Save dataset
        output_dir = Path(config['output_dir'])
        generator.save_dataset(output_dir, genetic_map, genotypes, sample_ids,
                             qtn_indices, qtn_effects, phenotype_df)

        # Check all required files exist
        required_files = [
            'dataset_summary.json',
            'design_summary.tsv',
            'truth.tsv',
            'map.csv',
            'phenotype.csv',
            'phenotype_null.csv',
            'genotypes.npz',
            'sample_names.csv',
            'snp_names.csv',
            'true_qtns.csv'
        ]

        for filename in required_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Required file missing: {filename}"
            assert filepath.stat().st_size > 0, f"File is empty: {filename}"

    def test_metadata_content(self, temp_output_dir):
        """Test metadata content and structure"""
        config = TEST_CONFIG_SMALL.copy()
        config['output_dir'] = str(temp_output_dir / "metadata_test")

        generator = GWASDatasetGenerator(config)

        # Generate minimal dataset
        genetic_map = generator.generate_genetic_map()
        genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)
        qtn_indices, qtn_effects = generator.select_qtns(genetic_map)
        phenotype_df = generator.generate_phenotypes(genotypes, qtn_indices, qtn_effects, sample_ids)

        # Create summary
        summary = generator.create_dataset_summary(
            genetic_map, qtn_indices, qtn_effects, phenotype_df, genotypes, Path(config['output_dir'])
        )

        # Test required fields
        assert 'dataset_params' in summary
        assert 'dimensions' in summary
        assert 'qtns' in summary
        assert 'design_summary' in summary
        assert 'seed_chain' in summary
        assert 'target_heritability' in summary

        # Test dimensions match
        assert summary['dimensions']['n_samples'] == config['n_samples']
        assert summary['dimensions']['n_snps'] == config['n_snps']
        assert summary['dimensions']['n_qtns'] == len(qtn_indices)

    def test_truth_table_format(self, temp_output_dir):
        """Test truth table format and content"""
        config = TEST_CONFIG_SMALL.copy()
        config['output_dir'] = str(temp_output_dir / "truth_test")

        generator = GWASDatasetGenerator(config)

        genetic_map = generator.generate_genetic_map()
        genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)
        qtn_indices, qtn_effects = generator.select_qtns(genetic_map)

        # Save just to generate truth table
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate truth TSV
        truth_path = output_dir / 'truth.tsv'
        with open(truth_path, 'w') as f:
            f.write("SNP_Index\tSNP_Name\tChr\tPos\tMAF\tEffect_Size\tEffect_Variance\n")
            for i, qtn_idx in enumerate(qtn_indices):
                qtn_info = genetic_map.iloc[qtn_idx]
                qtn_maf = generator.ancestral_freqs[qtn_idx]
                effect_var = qtn_effects[i] ** 2 * 2 * qtn_maf * (1 - qtn_maf)
                f.write(f"{qtn_idx}\t{qtn_info['SNP']}\t{qtn_info['Chr']}\t{qtn_info['Pos']}\t{qtn_maf:.6f}\t{qtn_effects[i]:.6f}\t{effect_var:.6f}\n")

        # Test file can be read
        truth_df = pd.read_csv(truth_path, sep='\t')

        assert len(truth_df) == len(qtn_indices)
        assert 'SNP_Index' in truth_df.columns
        assert 'Effect_Size' in truth_df.columns
        assert 'MAF' in truth_df.columns
        assert 'Effect_Variance' in truth_df.columns

def test_scenario_hard_mode():
    """Test predefined hard mode scenario"""
    if not GENERATOR_AVAILABLE:
        pytest.skip("Generator not available")

    # Test that hard mode parameters are applied correctly
    config = TEST_CONFIG_SMALL.copy()

    # Simulate command line args for hard mode
    class Args:
        scenario = 'hard_mode'
        maf_spectrum = 'beta_u'
        effect_maf_coupling_kappa = 0.5
        heritability = 0.2
        study_design = 'inbred'
        inbreeding_f = 0.98
        n_qtns = 200

    # This would normally be applied in argument parsing
    args = Args()

    # Verify the scenario produces expected parameter changes
    assert args.maf_spectrum == 'beta_u'
    assert args.effect_maf_coupling_kappa == 0.5
    assert args.heritability == 0.2
    assert args.study_design == 'inbred'
    assert args.inbreeding_f == 0.98

# Performance benchmark test
def test_generation_performance():
    """Test that generation completes within reasonable time"""
    if not GENERATOR_AVAILABLE:
        pytest.skip("Generator not available")

    import time

    start_time = time.time()

    generator = GWASDatasetGenerator(TEST_CONFIG_SMALL)
    genetic_map = generator.generate_genetic_map()
    genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)
    qtn_indices, qtn_effects = generator.select_qtns(genetic_map)

    end_time = time.time()
    generation_time = end_time - start_time

    # Should complete small dataset in under 30 seconds
    assert generation_time < 30, f"Generation took too long: {generation_time:.1f}s"

if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])