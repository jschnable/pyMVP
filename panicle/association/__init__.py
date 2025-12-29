"""
Association testing methods for GWAS analysis
"""

from .glm import PANICLE_GLM
from .mlm import PANICLE_MLM
from .farmcpu import PANICLE_FarmCPU
from .blink import PANICLE_BLINK

__all__ = ['PANICLE_GLM', 'PANICLE_MLM', 'PANICLE_FarmCPU', 'PANICLE_BLINK']
