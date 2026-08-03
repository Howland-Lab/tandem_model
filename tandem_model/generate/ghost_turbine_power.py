"""
Compute ghost-turbine power error (proportional to REWS^3) over a grid of
downwind locations for the SBL LES cases, comparing all wake models.

For each SBL case (see `generate.streamtube_sbl.SBL_DIRNAMES`) and each model
in MODELS, evaluates the rotor-averaged wind speed at ghost-turbine locations
on a grid of x/D in [5, 15] (step 1) by y/D in [-0.5, -0.25, 0, 0.25, 0.5],
converts it to a normalized power (REWS^3, freestream = 1), and joins against
the LES reference at the same locations to get per-location power error.
Caches the joined long-format error table so the plot script only needs to
aggregate/draw.

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

XLINE = np.arange(5, 16)  # x/D = 5, 6, ..., 15
YLINE = np.array([-0.5, -0.25, 0, 0.25, 0.5])  # y/D


def compute_ghost_turbine_power(
    dirnames=SBL_DIRNAMES, models=MODELS, xline=XLINE, yline=YLINE, runid=5
):
    """
    Computes ghost-turbine power (REWS^3) at every (x, y) grid point for each
    SBL LES case and each wake model (+ the LES reference), returning one
    long-format DataFrame (columns: case, Cr, model, x, y, power). `model`
    holds solver keys (see `tandem_model.models.DISPLAY_NAMES`) or "LES" for
    the reference.
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

    xx, yy = np.meshgrid(xline, yline, indexing="ij")
    df = bm.compute_ghost_turbine_rews(xx.flatten(), yline=yy.flatten())

    # `rews` here is the rotor-averaged normalized velocity *deficit* (du),
    # negative in the wake; freestream-normalized wind speed is 1 + du, so
    # normalized power is (1 + du)^3.
    return (
        df.with_columns(
            (1 + pl.col("rews")).pow(3).alias("power"),
            pl.col("model").replace({"ref": "LES"}).alias("model"),
            pl.col("case").map_elements(parse_cooling_rate, return_dtype=pl.Float64).alias("Cr"),
        )
        .drop("rews")
    )


def compute_ghost_turbine_error(df):
    """
    Joins model ghost-turbine power against the LES reference at matching
    (case, x, y) and returns per-location power error (columns: case, Cr,
    model, x, y, power, power_les, power_err, power_err_rel).
    """
    ref = df.filter(pl.col("model") == "LES").select(
        ["case", "x", "y", pl.col("power").alias("power_les")]
    )
    return (
        df.filter(pl.col("model") != "LES")
        .join(ref, on=["case", "x", "y"])
        .with_columns(
            (pl.col("power") - pl.col("power_les")).abs().alias("power_err"),
            ((pl.col("power") - pl.col("power_les")).abs() / pl.col("power_les")).alias(
                "power_err_rel"
            ),
        )
    )


def ghost_turbine_power(
    dirnames=SBL_DIRNAMES, models=MODELS, xline=XLINE, yline=YLINE, regenerate=False
):
    """
    Computes (or loads from cache) ghost-turbine power error vs (x, y) for
    SBL cases. Cached at data/sbl/ghost_turbine_power.csv.
    """
    cache_file = constants.DATA_PATH / "sbl" / "ghost_turbine_power.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        df = compute_ghost_turbine_power(
            dirnames=dirnames, models=models, xline=xline, yline=yline
        )
        return compute_ghost_turbine_error(df)

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    print(ghost_turbine_power(regenerate=True))
