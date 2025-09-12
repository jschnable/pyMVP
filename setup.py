"""Setup configuration for pyMVP package"""

from setuptools import setup, find_packages

setup(
    name="pymvp",
    version="0.1.0",
    author="pyMVP Development Team",
    description="Pure Python implementation of rMVP for Genome Wide Association Studies with Numba JIT acceleration",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "h5py>=3.0.0",
        "tables>=3.6.0",
        "statsmodels>=0.12.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "numba>=0.50.0",
    ],
    extras_require={
        # Faster VCF parsing and required for .bcf files
        "vcf": [
            "cyvcf2>=0.30.0",
        ],
        # PLINK .bed support
        "plink": [
            "bed-reader>=1.0.0",
        ],
        # Convenience extra to pull in all optional loaders
        "all": [
            "cyvcf2>=0.30.0",
            "bed-reader>=1.0.0",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
