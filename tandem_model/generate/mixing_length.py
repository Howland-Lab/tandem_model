"""
Generate mixing length data (l_md, l) vs x for single-turbine LES cases.

Reproduces the "Single-turbine calibration case/demo" section of
`kl_model/notebooks_s3/2026_nawea.ipynb`: l_md is the minimum-dissipation
mixing length surrogate, l = nu_T/sqrt(k) is a regression-based mixing length
computed from the resolved Reynolds stresses.

Cases are addressed by directory, not a fixed registry. If the directory
lives under `constants.SCRATCH_ROOT`'s symlink farm
(SCRATCH_ROOT/<family>/<case> -> real scratch case dir), the cache lands at
data/<family>/<case>_ell_md.csv; otherwise it falls back to the case's parent
directory name as the family. See `caching.case_cache_key`.

Kirby Heck
2026
"""

from pathlib import Path
import re
import numpy as np
import padeopsIO as pio
import polars as pl

from mitwindfarm.tandem import phi_m

from tandem_model import caching as cache, constants, utils

TERMS = ["ubar", "uu", "uv", "uw", "vv", "ww"]


def compute_ell_md(
    dirname,
    xlim=(0, 15),
    ylim=(-5, 5),
    zlim=(-1, 1.5),
    runid=None,
    precursor_runid=None,
    z_h=0.625,
    kappa=0.4,
):
    """
    Computes l_md (integral form) and l = nu_T/sqrt(k) as a function of x for a single
    LES simulation, returning a polars DataFrame with columns: dirname, name, x, l, l_md,
    L_obu, l_w. L_obu is the (constant) time-averaged Obukhov length for the case, and
    l_w = kappa * z_h / phi_m(z_h / L_obu) is the corresponding wall mixing length
    at hub height z_h (see `mitwindfarm.tandem.TurbulenceModel_tandem`).
    """
    if runid is None or precursor_runid is None:
        if (dirname / "input_interact.dat").exists():  # synthetic inflow cases
            runid = 1
            precursor_runid = 2
        else:  # concurrent-precusor cases
            runid = 5
            precursor_runid = 4

    sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
    pre = pio.BudgetIO(
        dirname, padeops=True, runid=precursor_runid, normalize_origin=sim.origin
    )

    ds = sim.slice(budget_terms=TERMS, xlim=list(xlim), ylim=ylim, zlim=zlim)
    ds_pre = pre.slice(budget_terms=TERMS, xlim=list(xlim), ylim=ylim, zlim=zlim)

    l_md = utils.lmix_md_1d(ds, ds_pre)
    l = utils.lmix_x(ds, ds_pre)

    L_obu = utils.get_obukhov_length(sim)
    l_w = kappa * z_h / phi_m(np.array([z_h / L_obu]))[0]

    return pl.DataFrame(
        {
            "name": sim.filename,
            "x": ds.x.values,
            "l": l.values,
            "l_md": l_md.values,
            "L_obu": L_obu,
            "l_w": l_w,
        }
    )


def ell_md(dirname, regenerate=False, **kwargs):
    """
    Computes (or loads from cache) l_md and l vs x for a case directory.
    Cached at data/<family>/<case>_ell_md.csv (see `caching.case_cache_key`).
    """
    dirname = Path(dirname)
    family, case = cache.case_cache_key(dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_ell_md.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_ell_md(dirname=dirname, **kwargs)

    return _generate(regenerate=regenerate)


def parse_cooling_rate(case):
    """
    Parses the surface cooling rate C_r (K/hr) from a case name, e.g.
    "G_01_z0_02_dTsurf_dt_02" -> 0.2. Returns None if the case name has no
    `dTsurf_dt_<NN>` tag, i.e. C_r = 0.1 * NN.
    """
    match = re.search(r"dTsurf_dt_(\d+)", case)
    if match is None:
        return None
    return round(int(match.group(1)) * 0.1, 2)


def ell_md_list(dirnames, regenerate=False, **kwargs):
    """
    Computes/loads l_md and l vs x for a list of case directories, concatenated
    into one DataFrame tagged by the "name"
    Adds C_r (cooling rate column) as well.
    """
    return pl.concat(
        ell_md(dirname, regenerate=regenerate, **kwargs).with_columns(
            pl.lit(Path(dirname).name).alias("case"),
            pl.lit(parse_cooling_rate(Path(dirname).name)).alias("Cr"),
        )
        for dirname in dirnames
    )


if __name__ == "__main__":
    pass
