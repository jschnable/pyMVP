#!/usr/bin/env python3
"""
Create validation dataset for pyMVP implementation testing

This script creates a small, fast-loading validation dataset with known
true QTNs for testing implementation changes to GLM, MLM, and FarmCPU.

The dataset is optimized for:
- Fast loading (< 2 seconds)
- Balanced QTN vs null markers
- Realistic population structure
- Known genetic architecture for validation
"""

import sys
import subprocess
from pathlib import Path

def create_validation_dataset():
    """Create validation dataset using the performance dataset generator"""
    
    # Navigate to the generator script location
    script_dir = Path(__file__).parent.parent / "DevTests" / "validation_data"
    generator_script = script_dir / "generate_performance_dataset.py"
    
    if not generator_script.exists():
        print(f"ERROR: Generator script not found at {generator_script}")
        return False
    
    # Output directory for validation dataset
    output_dir = Path(__file__).parent / "validation_datasets" / "fast_validation_dataset"
    
    # Parameters optimized for fast validation testing
    cmd = [
        sys.executable, str(generator_script),
        "--n-samples", "500",          # Small for fast processing
        "--n-snps", "5000",           # Moderate number of markers
        "--n-qtns", "25",             # Good balance of signal/noise
        "--n-chromosomes", "5",       # Sufficient for LD structure
        "--n-traits", "1",            # Single trait for simplicity
        "--output-dir", str(output_dir),
        "--missing-data-rate", "0.03", # Realistic missing data
        "--seed", "12345",            # Reproducible results
        "--population-structure",     # Enable pop structure
        "--population-sizes", "200,200,100",  # Balanced populations
        "--fst", "0.05"               # Moderate population differentiation
    ]
    
    print("Creating validation dataset for pyMVP implementation testing...")
    print("Parameters:")
    print("  - 500 individuals")
    print("  - 5,000 markers")
    print("  - 25 true QTNs")
    print("  - 3 populations")
    print("  - Single quantitative trait")
    print("  - 3% missing data")
    
    try:
        # Run the generator
        result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Validation dataset created successfully!")
            print(f"Location: {output_dir}")
            
            # Verify files were created
            required_files = [
                "phenotype.csv", "genotype_numeric.csv", "map.csv", 
                "true_qtns.csv", "dataset_summary.json"
            ]
            
            missing_files = []
            for filename in required_files:
                filepath = output_dir / filename
                if not filepath.exists():
                    missing_files.append(filename)
            
            if missing_files:
                print(f"⚠️  Warning: Some files missing: {missing_files}")
                return False
            else:
                print("All required files created successfully")
                return True
                
        else:
            print(f"\n❌ Dataset creation failed!")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n❌ Dataset creation failed with exception: {e}")
        return False

def main():
    print("pyMVP Validation Dataset Creator")
    print("=" * 40)
    
    # Check if dataset already exists
    output_dir = Path(__file__).parent / "validation_datasets" / "fast_validation_dataset"
    
    if output_dir.exists() and (output_dir / "dataset_summary.json").exists():
        response = input(f"\nValidation dataset already exists at {output_dir}\nRecreate it? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Using existing validation dataset")
            return True
    
    success = create_validation_dataset()
    
    if success:
        print(f"\n🎉 Validation dataset ready!")
        print(f"\nUsage examples:")
        print(f"  python tests/quick_validation_test.py GLM association/glm_ultra_fast.py")
        print(f"  python tests/validate_implementation_changes.py")
    else:
        print(f"\n💥 Failed to create validation dataset")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)