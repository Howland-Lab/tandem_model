"""
Generate wake deficit metrics vs x for SBL LES cases at fixed z0, comparing
the "varvortex" (skewed Gaussian) and "tandem" wake models: the
streamtube-averaged deficit (du_avg), the minimum deficit (du_min), the
centerline deficit (du_centerline), and the REWS of a line of ghost turbines
directly downwind at y=0 (du_rews).

Reproduces the "SBL cases" section of `kl_model/notebooks_s3/2026_nawea.ipynb`:
wraps each SBL LES case and its wake models in a `benchmarking.Benchmark` and
caches all four deficit metrics in one table, so the plot script only needs
to draw lines and `generate.streamtube_du_rmse` only needs to aggregate this
table's columns into RMSE, rather than re-solving the wake models. Model
kwargs are shared with `generate.wake_shapes` via `tandem_model.models`.

Kirby Heck
2026
"""

from pathlib import Path
import padeopsIO as pio
import polars as pl
import numpy as np

from tandem_model import caching as cache, constants, utils
from tandem_model.benchmarking import Benchmark
from tandem_model.generate.mixing_length import parse_cooling_rate
from tandem_model.models import K_KWARGS, curled_kwargs

ABL_DIRNAMES = [
    constants.SCRATCH_ROOT / "sbl" / f"G_01_z0_02_dTsurf_dt_{i:02d}" for i in range(6)
]
SBL_DIRNAMES = ABL_DIRNAMES[1:]
CTP_DIRNAMES = [constants.SCRATCH_ROOT / "oneturbine" / f"yaw_00_ct_{i:02d}" for i in np.arange(1, 20, 2)]

MODELS = ["gauss", "varvortex", "2021", "scott", "kl-hub", "tandem"]


def _compute_du_rews(bm: Benchmark, yline=0, Nr=20, Nt=18, R=0.5):
    """
    Computes the REWS of a line of ghost turbines directly downwind at
    y=yline, one per (case, model). Unlike `Benchmark.compute_du_min`/
    `compute_streamtube_du`/`compute_du_centerline` (which default `xax` to
    each case's own native LES x-grid), `Benchmark.compute_ghost_turbine_rews`
    takes a single `xline` shared across all cases, so it's called here once
    per case with that case's own native x-grid -- matching the other three
    metrics' default x-locations exactly.
    """
    records = []
    for benchmark in bm.cases:
        xax = benchmark.ref.grid.x.sel(x=slice(0, None)).to_numpy()
        for name, model in benchmark.model_cache_ref.items():
            rews = utils.line_of_ghost_turbines(model.du, xax, yline=yline, Nr=Nr, Nt=Nt, R=R)
            records.append(
                pl.DataFrame({"case": benchmark.name, "model": name, "x": xax, "du_rews": rews})
            )
    return pl.concat(records)


def compute_streamtube_sbl(dirnames=SBL_DIRNAMES, models=MODELS, runid=5):
    """
    Computes streamtube-averaged (du_avg), minimum (du_min), centerline
    (du_centerline), and ghost-turbine-REWS (du_rews) wake deficit vs x for
    each SBL LES case in dirnames and each named wake model, returning one
    long-format DataFrame (columns: case, Cr, source, x, du_avg, du_min,
    du_centerline, du_rews). All four metrics share the same x-locations per
    case (see `_compute_du_rews`). `source` holds solver keys (see
    `tandem_model.models.DISPLAY_NAMES`) or "LES" for the reference.
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

    on = ["case", "model", "x"]
    
    df = (
        bm.compute_streamtube_du()
        .join(bm.compute_du_min(), on=on)
        .join(bm.compute_du_centerline(), on=on)
        .join(_compute_du_rews(bm), on=on)
        .join(bm.compute_dk_max(), on=on)
    )
    df = df.with_columns(
        pl.col("model").replace({"ref": "LES"}).alias("source"),
    ).drop("model")

    # Null out the trailing x-sample of du_avg for curled (parabolized RANS)
    # models: their native x-grid stops just short of the LES reference's, so
    # benchmarking.streamtube_avg's interp_like(..., fill_value=0) plants a
    # spurious du_avg=0 at the LES's final x before that source's true
    # extent. du_min/du_centerline/du_rews don't have this artifact (they
    # interpolate to NaN, not 0, past a model's native extent), so only
    # du_avg needs correcting -- and only that cell, not the whole row.
    curled_keys = [key for key in models if key in K_KWARGS]
    is_last = pl.col("x") == pl.col("x").max().over(["case", "source"])
    return df.with_columns(
        pl.when(pl.col("source").is_in(curled_keys) & is_last)
        .then(None)
        .otherwise(pl.col("du_avg"))
        .alias("du_avg")
    )


def streamtube_sbl(dirnames=SBL_DIRNAMES, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) streamtube-averaged, minimum, centerline,
    and ghost-turbine-REWS deficit vs x for SBL cases. Cached at
    data/sbl/streamtube_du.csv.
    """
    cache_file = constants.DATA_PATH / "sbl" / "streamtube_du.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_streamtube_sbl(dirnames=dirnames, models=models)

    df = _generate(regenerate=regenerate).with_columns(
        pl.col("case")
        .map_elements(parse_cooling_rate, return_dtype=pl.Float64)
        .alias("Cr"),
    )
    return df


def thrust_wakes(dirnames=CTP_DIRNAMES, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) streamtube-averaged, minimum, centerline,
    and ghost-turbine-REWS deficit vs x for SBL cases. Cached at
    data/sbl/streamtube_du.csv.
    """
    cache_file = constants.DATA_PATH / "oneturbine" / "streamtube_du.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_streamtube_sbl(dirnames=dirnames, models=models)

    df = _generate(regenerate=regenerate).with_columns(
        pl.col("case")
        .map_elements(
            lambda x: float(x.split("ct_")[1]) * 0.2 + 0.2, return_dtype=float
        )
        .round(1)
        .alias("Ctprime"),
    )
    return df


if __name__ == "__main__":
    print(streamtube_sbl(regenerate=True))
    print(thrust_wakes(regenerate=True))
