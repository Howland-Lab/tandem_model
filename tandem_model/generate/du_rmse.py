"""
Compute wake deficit RMSE vs x/D in [5, 15] for the SBL LES cases, comparing
all wake models, for each metric in `generate.streamtube_sbl`'s cached table
(du_avg, du_min, du_centerline, du_rews).

Rather than re-solving the wake models (`generate.streamtube_sbl` already
does this), this just loads that module's cached long-format table and
aggregates each metric column into a per-(case, model) RMSE against the LES
reference over x/D in [5, 15]. Caches the resulting (model, case, Cr,
du_avg_rmse, du_min_rmse, du_centerline_rmse, du_rews_rmse) table so the plot
script only needs to draw the bar chart.

Kirby Heck
2026
"""

import polars as pl

from tandem_model import caching as cache, constants
from tandem_model.generate.streamtube_sbl import streamtube_sbl, MODELS

METRICS = ["du_avg", "du_min", "du_centerline", "du_rews"]
XLIM = (5, 15)


def _aggregate_rmse(df, key, xlim=XLIM):
    """
    Aggregates one metric column (see METRICS) into per-(case, source) RMSE
    against the "LES" source, over x/D in xlim.
    """
    sub = df.filter(
        pl.col("x").is_between(*xlim), pl.col(key).is_not_null(), pl.col(key).is_not_nan()
    )
    ref = (
        sub.filter(pl.col("source") == "LES")
        .select(["case", "x", key])
        .rename({key: f"{key}_ref"})
    )
    return (
        sub.filter(pl.col("source") != "LES")
        .join(ref, on=["case", "x"])
        .with_columns((pl.col(key) - pl.col(f"{key}_ref")).abs().alias(f"{key}_err"))
        .group_by(["source", "case"])
        .agg(pl.col(f"{key}_err").pow(2).mean().sqrt().alias(f"{key}_rmse"))
    )


def compute_du_rmse(models=MODELS, xlim=XLIM, regenerate_sbl=False):
    """
    Computes RMSE (aggregated over x/D in xlim) for each of du_avg, du_min,
    du_centerline, du_rews against LES, for each SBL case and wake model in
    `generate.streamtube_sbl`'s cached table. Returns one DataFrame (columns:
    case, Cr, model, du_avg_rmse, du_min_rmse, du_centerline_rmse,
    du_rews_rmse).
    """
    df = streamtube_sbl(models=models, regenerate=regenerate_sbl)
    cr = df.select(["case", "Cr"]).unique()

    rmse = None
    for key in METRICS:
        metric_rmse = _aggregate_rmse(df, key, xlim=xlim)
        rmse = metric_rmse if rmse is None else rmse.join(metric_rmse, on=["source", "case"])

    return rmse.join(cr, on="case").rename({"source": "model"}).sort(["case", "model"])


def du_rmse(models=MODELS, xlim=XLIM, regenerate=False, regenerate_sbl=False):
    """
    Computes (or loads from cache) wake deficit RMSE (du_avg, du_min,
    du_centerline, du_rews) vs SBL case for all wake models. Cached at
    data/sbl/streamtube_du_rmse.csv.
    """
    cache_file = constants.DATA_PATH / "sbl" / "du_rmse.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_du_rmse(models=models, xlim=xlim, regenerate_sbl=regenerate_sbl)

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    print(du_rmse(regenerate=True, regenerate_sbl=True))
