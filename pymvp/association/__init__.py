"""
Association testing methods for GWAS analysis
"""

from .glm import MVP_GLM
from .mlm import MVP_MLM  
from .farmcpu import MVP_FarmCPU

__all__ = ['MVP_GLM', 'MVP_MLM', 'MVP_FarmCPU']
