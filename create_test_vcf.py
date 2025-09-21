#!/usr/bin/env python3
"""
Script to create a trivial VCF file with markers for genotypes from phenotype_ordered.csv
This creates a minimal VCF for testing purposes with pyMVP.
"""

import csv
import random
import sys
from pathlib import Path

def read_genotypes_from_phenotype(phenotype_file):
    """Read genotype names from the phenotype CSV file."""
    genotypes = []
    try:
        with open(phenotype_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if row and row[0]:  # Make sure row exists and has genotype name
                    genotypes.append(row[0])
    except FileNotFoundError:
        print(f"Error: Could not find file {phenotype_file}")
        sys.exit(1)

    return genotypes

def generate_random_genotype_call():
    """Generate a random genotype call (0/0, 0/1, 1/1, or ./. for missing)."""
    calls = ["0/0", "0/1", "1/1", "./."]
    weights = [0.4, 0.3, 0.25, 0.05]  # Favor homozygous reference, some missing data
    return random.choices(calls, weights=weights)[0]

def create_vcf_file(genotypes, output_file, num_markers=12):
    """Create a VCF file with trivial markers for the given genotypes."""

    # VCF header
    vcf_lines = [
        "##fileformat=VCFv4.2",
        "##source=create_test_vcf.py",
        "##reference=test_reference",
        "##contig=<ID=1,length=300000000>",
        "##contig=<ID=2,length=250000000>",
        "##contig=<ID=3,length=200000000>",
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">'
    ]

    # Column header
    header_columns = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + genotypes
    vcf_lines.append("\t".join(header_columns))

    # Generate markers
    chromosomes = [1, 2, 3]
    ref_alleles = ["A", "T", "G", "C"]
    alt_alleles = {"A": "T", "T": "A", "G": "C", "C": "G"}

    random.seed(42)  # For reproducible results

    for i in range(num_markers):
        chrom = random.choice(chromosomes)
        pos = (i + 1) * 10000000  # Space markers 10Mb apart
        marker_id = f"SNP_{chrom}_{pos}"
        ref = random.choice(ref_alleles)
        alt = alt_alleles[ref]
        qual = "60"
        filter_val = "PASS"
        info = "DP=100"
        format_val = "GT:DP"

        # Generate genotype calls for all samples
        genotype_calls = []
        for genotype in genotypes:
            gt_call = generate_random_genotype_call()
            dp = random.randint(8, 25)  # Random depth between 8-25
            genotype_calls.append(f"{gt_call}:{dp}")

        # Create the VCF line
        vcf_line = [str(chrom), str(pos), marker_id, ref, alt, qual, filter_val, info, format_val] + genotype_calls
        vcf_lines.append("\t".join(vcf_line))

    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(vcf_lines) + "\n")

    print(f"Created VCF file: {output_file}")
    print(f"Number of samples: {len(genotypes)}")
    print(f"Number of markers: {num_markers}")

def main():
    # File paths
    phenotype_file = "tests/phenotype_ordered.csv"
    output_file = "tests/test_genotypes.vcf"

    # Check if phenotype file exists
    if not Path(phenotype_file).exists():
        print(f"Error: {phenotype_file} not found")
        sys.exit(1)

    # Read genotypes from phenotype file
    print(f"Reading genotypes from {phenotype_file}...")
    genotypes = read_genotypes_from_phenotype(phenotype_file)

    if not genotypes:
        print("Error: No genotypes found in phenotype file")
        sys.exit(1)

    print(f"Found {len(genotypes)} genotypes")

    # Create VCF file
    print(f"Creating VCF file with 12 markers...")
    create_vcf_file(genotypes, output_file, num_markers=12)

    print("Done!")

if __name__ == "__main__":
    main()