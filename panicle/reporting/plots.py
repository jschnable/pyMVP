"""Plot orchestration separated from numerical computation and table writing."""
from .tables import flatten_results


def render_analysis(results, *, single_trait, renderer, **options):
    # Retain public PANICLE behavior: callers receive figures even if not saved.
    return renderer(results=flatten_results(results, single_trait=single_trait), **options)


def render_and_close(*, renderer, **options):
    report = renderer(**options)
    import matplotlib.pyplot as plt
    for plots in report.get('plots', {}).values():
        for figure in plots.values():
            plt.close(figure)
    return report
