"""
Generate wake cross-sections (du) at fixed x-locations for SBL LES cases,
comparing LES to the Gaussian, Vortex, Scott, k-l, and TANDEM wake models.

Reproduces the "SBL cases" wake-contour figure of
`kl_model/notebooks_s3/2026_nawea.ipynb`: for each SBL case and model, du is
sliced at a handful of x-locations and cached, so the plot script only needs
to draw contours. Model kwargs are shared with `generate.wake_shapes` and
`generate.streamtube_sbl` via `tandem_model.models`.

Kirby Heck
2026
"""

from pathlib import Path
import numpy as np
import padeopsIO as pio
import polars as pl

from tandem_model import caching as cache, constants
from tandem_model.benchmarking import Benchmark
from tandem_model.generate.mixing_length import parse_cooling_rate
from tandem_model.models import curled_kwargs

SBL_DIRNAMES = [
    constants.SCRATCH_ROOT / "sbl" / f"G_01_z0_02_dTsurf_dt_{i:02d}" for i in range(1, 6)
]

XVALS = [5, 10, 15]

# solver keys, in the desired figure order (see tandem_model.models.DISPLAY_NAMES
# for display names)
MODELS = ["gauss", "varvortex", "scott", "kl-hub", "tandem"]

XLIM_XY = [-2, 15]
YLIM_XY = [-3, 3]

YLIM_YZ = [-3.3, 3.3]
ZLIM_YZ = [-15 / 24, 2]


def _slice_yz_to_df(wakefield, source, x, yg, zg):
    """Slices a WakeField's du at x, reindexed onto the common yg/zg grid
    with 0 outside the wakefield's native domain, returning a long-format
    DataFrame with columns source, x, y, z, du."""
    s2d = wakefield.du.slice(xlim=x).interp(y=yg, z=zg, kwargs=dict(fill_value=0))
    s2d_k = wakefield.dk.slice(xlim=x).interp(y=yg, z=zg, kwargs=dict(fill_value=0))
    y, z = np.meshgrid(yg, zg, indexing="ij")
    return pl.DataFrame(
        {
            "source": source,
            "x": x,
            "y": y.flatten(),
            "z": z.flatten(),
            "du": s2d.to_numpy().flatten(),
            "dk": s2d_k.to_numpy().flatten(),
        }
    )


def compute_wake_contours_sbl(
    dirnames=SBL_DIRNAMES, xs=XVALS, models=MODELS, ylim=YLIM_YZ, zlim=ZLIM_YZ, runid=5
):
    """
    Computes du cross-sections at each x in xs for the LES reference and each
    named wake model, for every SBL LES case in dirnames. Each model's slice
    is reindexed onto a common y/z grid spanning ylim/zlim (taken from the
    LES reference at that x), with points outside that model's native
    domain filled with 0, so the plot script gets a dense field rather than
    gaps. Returns one long-format DataFrame (columns: case, Cr, source, x,
    y, z, du).
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

    df_ls = []
    for bmcase in bm.cases:
        cr = parse_cooling_rate(bmcase.name)
        for x in xs:
            ref_s2d = bmcase.ref.du.slice(xlim=x, ylim=ylim, zlim=zlim)
            yg, zg = ref_s2d.y.to_numpy(), ref_s2d.z.to_numpy()

            for key, wakefield in bmcase.model_cache_ref.items():
                source = "LES" if key == "ref" else key
                df = _slice_yz_to_df(wakefield, source, x, yg, zg).with_columns(
                    pl.lit(bmcase.name).alias("case"), pl.lit(cr).alias("Cr")
                )
                df_ls.append(df)

    return pl.concat(df_ls)


def wake_contours_sbl(dirnames=SBL_DIRNAMES, xs=XVALS, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) du cross-sections vs. x for SBL cases.
    Cached at data/sbl/wake_contours.csv.
    """
    cache_file = constants.DATA_PATH / "sbl" / "wake_contours.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_wake_contours_sbl(dirnames=dirnames, xs=xs, models=models)

    return _generate(regenerate=regenerate)


def _slice_xy_to_df(wakefield, source, xg, yg):
    """Slices a WakeField's du at hub height (z=0), reindexed onto the
    common xg/yg grid with 0 outside the wakefield's native domain, returning
    a long-format DataFrame with columns source, x, y, du."""
    s2d = wakefield.du.slice(zlim=0).interp(x=xg, y=yg, kwargs=dict(fill_value=0))
    s2d_k = wakefield.dk.slice(zlim=0).interp(x=xg, y=yg, kwargs=dict(fill_value=0))
    x, y = np.meshgrid(xg, yg, indexing="ij")
    return pl.DataFrame(
        {
            "source": source,
            "x": x.flatten(),
            "y": y.flatten(),
            "du": s2d.to_numpy().flatten(),
            "dk": s2d_k.to_numpy().flatten(),
        }
    )


def compute_wake_contours_xy(
    dirnames=SBL_DIRNAMES, models=MODELS, xlim=XLIM_XY, ylim=YLIM_XY, runid=5
):
    """
    Computes the xy-plane wake deficit du field at hub height (z=0) for the
    LES reference and each named wake model, for every SBL LES case in
    dirnames. Each model's field is reindexed onto a common grid spanning
    xlim/ylim (taken from the LES reference), with points outside that
    model's native (marched) domain filled with 0, so the plot script gets
    a dense field rather than gaps. Returns one long-format DataFrame
    (columns: case, Cr, source, x, y, du).
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

    df_ls = []
    for bmcase in bm.cases:
        cr = parse_cooling_rate(bmcase.name)
        ref_s2d = bmcase.ref.du.slice(xlim=xlim, ylim=ylim, zlim=0)
        xg, yg = ref_s2d.x.to_numpy(), ref_s2d.y.to_numpy()

        for key, wakefield in bmcase.model_cache_ref.items():
            source = "LES" if key == "ref" else key
            df = _slice_xy_to_df(wakefield, source, xg, yg).with_columns(
                pl.lit(bmcase.name).alias("case"), pl.lit(cr).alias("Cr")
            )
            df_ls.append(df)

    return pl.concat(df_ls)


def wake_contours_xy(
    dirnames=SBL_DIRNAMES, models=MODELS, xlim=XLIM_XY, ylim=YLIM_XY, regenerate=False
):
    """
    Computes (or loads from cache) the xy-plane du field at hub height for
    SBL cases. Cached at data/sbl/wake_contours_xy.csv.
    """
    cache_file = constants.DATA_PATH / "sbl" / "wake_contours_xy.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_wake_contours_xy(dirnames=dirnames, models=models, xlim=xlim, ylim=ylim)

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    print(wake_contours_sbl(regenerate=True))
    print(wake_contours_xy(regenerate=True))
