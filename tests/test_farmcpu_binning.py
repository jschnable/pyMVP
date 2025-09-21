import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from pymvp.association.farmcpu import select_qtns_static_binning_rmvp
from pymvp.utils.data_types import GenotypeMap


def test_static_binning_handles_string_chromosomes():
    map_df = pd.DataFrame(
        {
            "SNP": ["s1", "s2", "s3", "s4"],
            "CHROM": ["Chr01", "Chr01", "Chr02", "Chr10"],
            "POS": [101, 202, 303, 404],
        }
    )
    map_data = GenotypeMap(map_df)
    pvals = np.array([0.02, 0.03, 0.01, 0.5])

    indices = select_qtns_static_binning_rmvp(
        pvalues=pvals,
        map_data=map_data,
        bin_size=1_000,
        top_k=None,
        verbose=False,
    )

    assert indices[0] == 2
    assert set(indices).issubset({0, 1, 2, 3})
