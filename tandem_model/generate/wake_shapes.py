"""
Generate wake cross-sections (du) at fixed x-locations, comparing LES to wake
models under a veered inflow.

Reproduces the "centerline velocity and wake shapes" section of
`kl_model/notebooks_s3/2026_nawea.ipynb`: for each model, du is sliced at a
handful of x-locations and cached, so the plot script only needs to draw
contours rather than re-solve every wake model.

Kirby Heck
2026
"""

from pathlib import Path
import time
import numpy as np
import polars as pl
import padeopsIO as pio

from tandem_model import caching as cache, constants, utils
from tandem_model.models import curled_kwargs

XVALS = [4, 7, 10, 13, 16]
MODELS = ["gauss", "varvortex", "scott", "tandem"]


def _slice_to_df(wakefield, source, xs):
    """Slices a WakeField's du at each x in xs, returning a long-format
    DataFrame with columns source, x, y, z, du."""
    df_ls = []
    for x in xs:
        s2d = wakefield.du.slice(xlim=x)
        y, z = np.meshgrid(s2d.y.values, s2d.z.values, indexing="ij")
        df_ls.append(
            pl.DataFrame(
                {
                    "source": source,
                    "x": x,
                    "y": y.flatten(),
                    "z": z.flatten(),
                    "du": s2d.to_numpy().flatten(),
                }
            )
        )
    return pl.concat(df_ls)


def compute_wake_shapes(
    dirname, xs=XVALS, models=("gauss", "varvortex", "tandem"), runid=1, precursor_runid=2
):
    """
    Computes du cross-sections at each x in xs for the LES reference and each
    named wake model, returning one long-format DataFrame (columns: source,
    x, y, z, du).
    """
    dirname = Path(dirname)
    sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
    pre = pio.BudgetIO(
        dirname, padeops=True, runid=precursor_runid, normalize_origin=sim.origin
    )

    ref = utils.LESWakeField(sim, normalize=True)
    df_ls = [_slice_to_df(ref, "LES", xs)]

    rotor_flow = utils.inflow_LES(pre, xlim=[-0.5, 0.5], normalize=True)
    fw = sim.ta[0].filterwidth / np.sqrt(12)

    for key in models:
        t_st = time.time()
        kw = curled_kwargs(key)
        if kw:  # parabolized (curled) RANS turbulence closure
            kw["model_kwargs"] = {**kw["model_kwargs"], "smooth_fact": fw}
            wake = utils.solve_windfarm_LES(sim, wakemodel=key, inflow=rotor_flow, **kw)
        else:
            wake = utils.solve_windfarm_LES(sim, wakemodel=key, inflow=rotor_flow)
        df_ls.append(_slice_to_df(wake, key, xs))
        print("Finished {} in {:.1f} s".format(key, time.time() - t_st))

    return pl.concat(df_ls)


def wake_shapes(dirname=None, xs=XVALS, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) du cross-sections vs. x for the LES
    reference and each wake model in `models`. Cached at
    data/<family>/<case>_wake_shapes.csv (see `caching.case_cache_key`).
    """
    dirname = (
        Path(dirname) if dirname is not None else constants.SCRATCH_ROOT / "nowall" / "veer_03"
    )
    family, case = cache.case_cache_key(dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_wake_shapes.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_wake_shapes(dirname, xs=xs, models=models)

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    print(wake_shapes(regenerate=True))
