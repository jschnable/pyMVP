#!/usr/bin/env python3
"""
Quick validation test script for pyMVP implementation changes

This is a lightweight wrapper around the main validation framework,
designed for rapid testing during development.

Usage:
    python tests/quick_validation_test.py [method] [implementation_file]

Examples:
    python tests/quick_validation_test.py GLM association/glm_optimized.py
    python tests/quick_validation_test.py MLM association/mlm_fast.py  
    python tests/quick_validation_test.py FarmCPU association/farmcpu_improved.py

The implementation file should contain a function with the same signature
as the original (MVP_GLM, MVP_MLM, or MVP_FarmCPU).
"""

import sys
import importlib.util
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

sys.path.insert(0, str(Path(__file__).parent))  # Add tests directory
from validate_implementation_changes import ValidationConfig, ValidationDataset, ImplementationValidator

def load_implementation_from_file(file_path: str, method: str):
    """Load implementation function from a Python file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        # Try relative to pyMVP root
        root_dir = Path(__file__).parent.parent
        file_path = root_dir / file_path
        
    if not file_path.exists():
        raise FileNotFoundError(f"Implementation file not found: {file_path}")
    
    # Load module from file
    spec = importlib.util.spec_from_file_location("test_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the appropriate function
    function_names = {
        'GLM': ['MVP_GLM_ultra', 'MVP_GLM_ultrafast', 'MVP_GLM_optimized', 'MVP_GLM'],
        'MLM': ['MVP_MLM_ultra', 'MVP_MLM_optimized', 'MVP_MLM_fast', 'MVP_MLM'],
        'FarmCPU': ['MVP_FarmCPU_ultra', 'MVP_FarmCPU_optimized', 'MVP_FarmCPU_fast', 'MVP_FarmCPU']
    }
    
    for func_name in function_names[method]:
        if hasattr(module, func_name):
            return getattr(module, func_name), func_name
    
    raise AttributeError(f"No suitable {method} function found in {file_path}. "
                        f"Expected one of: {function_names[method]}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python quick_validation_test.py [method] [implementation_file]")
        print("Methods: GLM, MLM, FarmCPU")
        print("Example: python quick_validation_test.py GLM association/glm_ultra_fast.py")
        sys.exit(1)
    
    method = sys.argv[1].upper()
    implementation_file = sys.argv[2]
    
    if method not in ['GLM', 'MLM', 'FARMCPU']:
        print(f"Error: Method must be GLM, MLM, or FarmCPU, got: {method}")
        sys.exit(1)
    
    print(f"Quick Validation Test: {method}")
    print("=" * 40)
    
    # Initialize validation system
    config = ValidationConfig()
    if not config.validate_dataset_exists():
        print("ERROR: Validation dataset not found.")
        print("Run: python tests/create_validation_dataset.py")
        sys.exit(1)
    
    # Load dataset
    dataset = ValidationDataset(config)
    if not dataset.load_dataset():
        print("ERROR: Failed to load validation dataset")
        sys.exit(1)
    
    # Load implementation to test
    try:
        test_function, func_name = load_implementation_from_file(implementation_file, method)
        print(f"Testing function: {func_name}")
    except Exception as e:
        print(f"ERROR loading implementation: {e}")
        sys.exit(1)
    
    # Run validation
    validator = ImplementationValidator(config, dataset)
    
    if method == 'GLM':
        results = validator.validate_glm_implementation(test_function, func_name)
    elif method == 'MLM':
        results = validator.validate_mlm_implementation(test_function, func_name)
    elif method == 'FARMCPU':
        results = validator.validate_farmcpu_implementation(test_function, func_name)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"QUICK VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    if results['validation_passed']:
        print(f"✅ PASSED: {func_name}")
        print(f"   Performance: {results['speedup']:.2f}x faster")
        print(f"   Effect correlation: {results['correlations']['effects']:.6f}")
        print(f"   Log p-value correlation: {results['correlations']['log_pvalues']:.6f}")
        print(f"\n🎉 Implementation is ready for integration!")
    else:
        print(f"❌ FAILED: {func_name}")
        print(f"   Status: {results['status']}")
        if 'error' in results:
            print(f"   Error: {results['error']}")
        else:
            print(f"   Effect correlation: {results['correlations']['effects']:.6f}")
            print(f"   Log p-value correlation: {results['correlations']['log_pvalues']:.6f}")
        print(f"\n⚠️  Implementation needs more work before integration")
    
    # Save detailed report
    validator.generate_validation_report()

if __name__ == "__main__":
    main()