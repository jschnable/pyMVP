"""
pyMVP: Python implementation of rMVP for Genome Wide Association Studies

A Memory-efficient, Visualization-enhanced, and Parallel-accelerated 
genome-wide association study (GWAS) tool.
"""

import os
import warnings

# Suppress OpenMP deprecation warnings that occur with Numba parallel processing
# This is a known issue with newer OpenMP versions and Numba's parallel features
# The warning is cosmetic and does not affect functionality
os.environ.setdefault('KMP_WARNINGS', 'off')

# Filter out the specific OpenMP deprecation warning
warnings.filterwarnings('ignore', message='.*omp_set_nested.*deprecated.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*omp_set_nested.*')

__version__ = "0.1.0"
__author__ = "pyMVP Development Team"

from .core.mvp import MVP
from .data.converters import MVP_Data
from .matrix.kinship import MVP_K_VanRaden, MVP_K_IBS
from .matrix.pca import MVP_PCA
from .association.glm import MVP_GLM
from .association.mlm import MVP_MLM
from .association.farmcpu import MVP_FarmCPU
from .association.farmcpu_resampling import MVP_FarmCPUResampling
from .visualization.manhattan import MVP_Report

__all__ = [
    'MVP',
    'MVP_Data', 
    'MVP_K_VanRaden',
    'MVP_K_IBS',
    'MVP_PCA',
    'MVP_GLM',
    'MVP_MLM',
    'MVP_FarmCPU',
    'MVP_FarmCPUResampling',
    'MVP_Report'
]
