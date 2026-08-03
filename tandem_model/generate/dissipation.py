"""
Generate TKE dissipation vs. delta k^{3/2} data for LES cases: an integrated
(vs. x) version and a pointwise version.

Reproduces the "Integrated dissipation vs tke" and "Pointwise dissipation vs
tke" cells of `kl_model/notebooks_s3/00_revisit_dissipation.ipynb`. In both
cases, delta is the deficit relative to the precursor (no-turbine) flow.
Data is already nondimensionalized by the LES's velocity scale U_inf and
length scale D (U_inf = 1 in these simulations; see `BudgetIO.get_uhub`).

Cases are addressed by directory, not a fixed registry. See
`caching.case_cache_key` for how the cache path is derived.

Kirby Heck
2026
"""

from pathlib import Path
import numpy as np
import padeopsIO as pio
import polars as pl

from tandem_model import caching as cache, constants, utils

TERMS = ["uu", "vv", "ww", "TKE_dissipation"]


def _get_sims(dirname, runid=None, precursor_runid=None):
    dirname = Path(dirname)
    if runid is None or precursor_runid is None:
        if (dirname / "input_interact.dat").exists():  # synthetic inflow cases
            runid = 1
            precursor_runid = 2
        else:  # concurrent-precursor cases
            runid = 5
            precursor_runid = 4

    sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
    pre = pio.BudgetIO(dirname, padeops=True, runid=precursor_runid, normalize_origin=sim.origin)
    return sim, pre


def compute_dissipation(dirname, xlim=None, ylim=(-3, 3), zlim=(-3, 3), runid=None, precursor_runid=None):
    """
    Computes integrated -delta(TKE_dissipation) and delta(k)^{3/2} as a function
    of x for a single LES simulation, returning a polars DataFrame with columns:
    dirname, name, x, tke_32, diss.
    """
    sim, pre = _get_sims(dirname, runid=runid, precursor_runid=precursor_runid)

    if xlim is None:
        xlim = [0, utils.xmax_LES(sim)]

    ds = sim.slice(budget_terms=TERMS, xlim=list(xlim), ylim=list(ylim), zlim=list(zlim))
    ds_pre = pre.slice(budget_terms=TERMS, xlim=list(xlim), ylim=list(ylim), zlim=list(zlim))
    if len(ds.grid.x) != len(ds_pre.grid.x):
        ds_pre = ds_pre.mean(("x", "y"))
    diff = ds - ds_pre

    diss = -diff["TKE_dissipation"].integrate(("y", "z"))
    tke = 0.5 * (diff["uu"] + diff["vv"] + diff["ww"])
    tke_32 = (np.maximum(tke, 0) ** 1.5).integrate(("y", "z"))

    return pl.DataFrame(
        {
            "dirname": str(sim.dirname),
            "name": sim.filename,
            "x": diss.x.to_numpy(),
            "tke_32": tke_32.to_numpy(),
            "diss": diss.to_numpy(),
        }
    )


def compute_dissipation_pointwise(dirname, xlim=None, ylim=(-1, 1), zlim=(-1, 1), runid=None, precursor_runid=None):
    """
    Computes pointwise -delta(TKE_dissipation) vs. delta(k) over the LES grid
    for a single simulation, returning a polars DataFrame with columns:
    x, y, z, dk, diss. Unmasked and unraised to the 3/2 power -- the plotting
    script is responsible for masking non-positive values and computing
    dk^{3/2}, so that this cached data can be reused for other analyses.
    """
    sim, pre = _get_sims(dirname, runid=runid, precursor_runid=precursor_runid)

    if xlim is None:
        xlim = [0, utils.xmax_LES(sim)]

    s = sim.slice(budget_terms=TERMS, xlim=list(xlim), ylim=list(ylim), zlim=list(zlim))
    s["k"] = 0.5 * (s["uu"] + s["vv"] + s["ww"])
    s_pre = pre.slice(budget_terms=TERMS, xlim=list(xlim), ylim=list(ylim), zlim=list(zlim)).mean(("x", "y"))
    s_pre["k"] = 0.5 * (s_pre["uu"] + s_pre["vv"] + s_pre["ww"])

    dk = s["k"] - s_pre["k"]
    diss = -(s["TKE_dissipation"] - s_pre["TKE_dissipation"])

    xx, yy, zz = np.meshgrid(dk.x.values, dk.y.values, dk.z.values, indexing="ij")

    return pl.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
            "dk": dk.values.ravel(),
            "diss": diss.values.ravel(),
        }
    )


def dissipation(dirname, regenerate=False, **kwargs):
    """
    Computes (or loads from cache) integrated dissipation and delta k^{3/2} vs x
    for a case directory. Cached at data/<family>/<case>_dissipation.csv
    (see `caching.case_cache_key`).
    """
    dirname = Path(dirname)
    family, case = cache.case_cache_key(dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_dissipation.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_dissipation(dirname=dirname, **kwargs)

    return _generate(regenerate=regenerate)


def dissipation_pointwise(dirname, regenerate=False, **kwargs):
    """
    Computes (or loads from cache) pointwise dissipation and delta k^{3/2} data
    for a case directory. Cached at data/<family>/<case>_dissipation_pointwise.csv
    (see `caching.case_cache_key`).
    """
    dirname = Path(dirname)
    family, case = cache.case_cache_key(dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_dissipation_pointwise.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_dissipation_pointwise(dirname=dirname, **kwargs)

    return _generate(regenerate=regenerate)


def dissipation_list(dirnames, regenerate=False, **kwargs):
    """Computes/loads dissipation vs x for a list of case directories, concatenated
    into one DataFrame tagged by a `case` column (the case directory's name)."""
    return pl.concat(
        dissipation(dirname, regenerate=regenerate, **kwargs).with_columns(
            pl.lit(Path(dirname).name).alias("case")
        )
        for dirname in dirnames
    )


if __name__ == "__main__":
    df = dissipation_pointwise(constants.SCRATCH_ROOT / "nowall" / "veer_00", regenerate=True)
    print(df)
