import pandas as pd
import numpy as np
from pathlib import Path
from panicle.utils.data_types import AssociationResults, GenotypeMap
from panicle.visualization.manhattan import PANICLE_Report

results_path = Path('gwas_output_big_glm_no_compression/GWAS_DaysToFlower_all_results.csv')
if not results_path.exists():
    raise SystemExit(f"Missing {results_path}")

usecols = ['SNP', 'CHROM', 'POS', 'GLM_P', 'GLM_Effect']
df = pd.read_csv(results_path, usecols=usecols)
map_data = GenotypeMap(df[['SNP', 'CHROM', 'POS']].copy())

effects = df['GLM_Effect'].to_numpy()
se = np.full_like(effects, np.nan, dtype=float)
pvals = df['GLM_P'].to_numpy()
res = AssociationResults(effects, se, pvals)

PANICLE_Report(
    results={'GLM': res},
    map_data=map_data,
    plot_types=['manhattan'],
    output_prefix='plot_timing/GLM',
    verbose=False,
    save_plots=True,
)
