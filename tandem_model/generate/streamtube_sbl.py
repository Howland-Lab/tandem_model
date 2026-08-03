"""
Generate streamtube-averaged wake deficit du vs x for SBL LES cases at fixed
z0, comparing the "varvortex" (skewed Gaussian) and "tandem" wake models.

Reproduces the "SBL cases" section of `kl_model/notebooks_s3/2026_nawea.ipynb`:
wraps each SBL LES case and its wake models in a `benchmarking.Benchmark` and
caches the streamtube-averaged deficit, so the plot script only needs to draw
lines. Model kwargs are shared with `generate.wake_shapes` via
`tandem_model.models`.

Kirby Heck
2026
"""

from pathlib import Path
import padeopsIO as pio
import polars as pl

from tandem_model import caching as cache, constants
from tandem_model.benchmarking import Benchmark
from tandem_model.generate.mixing_length import parse_cooling_rate
from tandem_model.models import K_KWARGS, curled_kwargs

SBL_DIRNAMES = [
    constants.SCRATCH_ROOT / "sbl" / f"G_01_z0_02_dTsurf_dt_{i:02d}" for i in range(1, 6)
]

MODELS = ["varvortex", "tandem"]


def compute_streamtube_sbl(dirnames=SBL_DIRNAMES, models=MODELS, runid=5):
    """
    Computes streamtube-averaged wake deficit du vs x for each SBL LES case in
    dirnames and each named wake model, returning one long-format DataFrame
    (columns: case, Cr, source, x, du_avg). `source` holds solver keys (see
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
    df = bm.compute_streamtube_du()
    df = df.with_columns(
        pl.col("model").replace({"ref": "LES"}).alias("source"),
        pl.col("case").map_elements(parse_cooling_rate, return_dtype=pl.Float64).alias("Cr"),
    ).drop("model")

    # Drop the trailing x-sample for curled (parabolized RANS) models: their
    # native x-grid stops just short of the LES reference's, so
    # benchmarking.streamtube_avg's interp_like(..., fill_value=0) plants a
    # spurious du_avg=0 at the LES's final x before that source's true extent.
    curled_keys = [key for key in models if key in K_KWARGS]
    is_last = pl.col("x") == pl.col("x").max().over(["case", "source"])
    return df.filter(~(pl.col("source").is_in(curled_keys) & is_last))


def streamtube_sbl(dirnames=SBL_DIRNAMES, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) streamtube-averaged du vs x for SBL cases.
    Cached at data/sbl/streamtube_du.csv.
    """
    cache_file = constants.DATA_PATH / "sbl" / "streamtube_du.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_streamtube_sbl(dirnames=dirnames, models=models)

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    print(streamtube_sbl(regenerate=True))
