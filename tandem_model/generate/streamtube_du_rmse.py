"""
Compute streamtube-averaged wake deficit RMSE vs x/D in [5, 15] for the SBL
LES cases, comparing all wake models.

For each SBL case (see `generate.streamtube_sbl.SBL_DIRNAMES`) and each model
in MODELS, evaluates the streamtube-averaged velocity deficit du_avg(x) over
x/D in [5, 15] and computes the RMSE against the LES reference at the same
x locations, aggregated over x. Caches the resulting (model, case, Cr,
du_avg_rmse) table so the plot script only needs to draw the bar chart.

Kirby Heck
2026
"""

from pathlib import Path

import numpy as np
import polars as pl
import padeopsIO as pio

from tandem_model import caching as cache, constants
from tandem_model.benchmarking import Benchmark
from tandem_model.generate.mixing_length import parse_cooling_rate
from tandem_model.generate.streamtube_sbl import SBL_DIRNAMES
from tandem_model.models import curled_kwargs

MODELS = ["gauss", "varvortex", "2021", "scott", "kl-hub", "tandem"]

XAX = np.linspace(5, 15)


def compute_streamtube_du_rmse(dirnames=SBL_DIRNAMES, models=MODELS, xax=XAX, runid=5):
    """
    Computes streamtube-averaged wake deficit RMSE (aggregated over x/D in
    xax) for each SBL LES case and each wake model, returning one long-format
    DataFrame (columns: case, Cr, model, du_avg_rmse).
    """
    dirnames = [Path(d) for d in dirnames]
    sims = [
        pio.BudgetIO(d, padeops=True, runid=runid, normalize_origin="turb") for d in dirnames
    ]
    ids = [d.name for d in dirnames]

    bm = Benchmark(
        sims,
        ids=ids,
        models=list(models),
        model_kwargs=[curled_kwargs(key) for key in models],
        normalize=True,
    )

    df = bm.compute_streamtube_rmse(xax=xax)
    return df.with_columns(
        pl.col("case").map_elements(parse_cooling_rate, return_dtype=pl.Float64).alias("Cr"),
    )


def streamtube_du_rmse(dirnames=SBL_DIRNAMES, models=MODELS, xax=XAX, regenerate=False):
    """
    Computes (or loads from cache) streamtube-averaged deficit RMSE vs SBL
    case for all wake models. Cached at data/sbl/streamtube_du_rmse.csv.
    """
    cache_file = constants.DATA_PATH / "sbl" / "streamtube_du_rmse.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_streamtube_du_rmse(dirnames=dirnames, models=models, xax=xax)

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    print(streamtube_du_rmse(regenerate=True))
