"""
Generate per-turbine normalized power (P/P_1) for LES and wake-model
predictions across the CNBL 4x1 wind-farm wind-direction sweep.

Reproduces the "4-turbine wind farms" section of
`kl_model/notebooks_s3/00_wakemodel_testing.ipynb` (figure
CNBL_4x1_allwd_Pnorm_TANDEMfixed.png): for each wind-direction case, solves
each wake model against the LES layout/setpoints, normalizing each source's
Cp by its own leading-turbine (x=0) Cp.

Cases are addressed by directory, not a fixed registry. See
`caching.case_cache_key` for how the cache path is derived.

Kirby Heck
2026
"""

from pathlib import Path
import numpy as np
import polars as pl
import padeopsIO as pio
import mitwindfarm as mitwf

from tandem_model import caching as cache, constants, utils
from tandem_model.models import curled_kwargs

MODELS = ("gauss", "kl-hub", "tandem")

# CNBL_4x1_wd{:03d}: 4-row wind farms at 0, 2.5, 5, 10 degrees wind direction
# (directory suffix = wind direction in degrees x10).
CASES = ["CNBL_4x1_wd000", "CNBL_4x1_wd025", "CNBL_4x1_wd050", "CNBL_4x1_wd100"]


def add_pnorm(df, cp_normfact=None):
    """Normalizes Cp by the leading-turbine (x=0) Cp, adding a P_norm column."""
    if cp_normfact is None:
        cp_normfact = df.filter(pl.col("x") == 0).select(pl.col("Cp")).item()
    return df.with_columns((pl.col("Cp") / cp_normfact).alias("P_norm"))


def compute_power(dirname, models=MODELS, runid=5):
    """
    Solves each named wake model (solver keys, see
    `tandem_model.models.DISPLAY_NAMES`) against an LES case's
    layout/setpoints, returning a long-format DataFrame (LES + each model)
    with columns including x, Cp, P_norm, model.
    """
    dirname = Path(dirname)
    sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
    fw = sim.ta[0].filterwidth / np.sqrt(12)
    rotor_model = mitwf.UnifiedAD_veer(rotor_grid=mitwf.Area())

    df_list = [add_pnorm(utils.to_polars(sim)).with_columns(pl.lit("LES").alias("model"))]
    for key in models:
        kw = curled_kwargs(key)
        if kw:  # parabolized (curled) RANS turbulence closure
            kw["model_kwargs"] = {**kw["model_kwargs"], "smooth_fact": fw}
            wake = utils.solve_windfarm_LES(sim, rotor_model=rotor_model, wakemodel=key, **kw)
        else:
            wake = utils.solve_windfarm_LES(sim, rotor_model=rotor_model, wakemodel=key)
        df = add_pnorm(utils.to_polars(wake.sol)).with_columns(pl.lit(key).alias("model"))
        df_list.append(df)

    out = pl.concat(df_list, how="diagonal_relaxed")
    # row number (1-indexed): turbines ordered by ascending x, within each model
    return out.with_columns(pl.col("x").rank(method="ordinal").over("model").alias("row"))


def power(dirname, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) normalized per-turbine power for LES and
    each wake model in `models`, for a single case directory. Cached at
    data/<family>/<case>_power.csv (see `caching.case_cache_key`).
    """
    dirname = Path(dirname)
    family, case = cache.case_cache_key(dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_power.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_power(dirname, models=models)

    return _generate(regenerate=regenerate)


def power_4x1(cases=CASES, models=MODELS, regenerate=False):
    """
    Computes/loads normalized power vs. x for each CNBL 4x1 wind-direction
    case in `constants.SCRATCH_ROOT / "superposition"`, concatenated into one
    DataFrame tagged by a `case` column (the case directory's name).
    """
    parent = constants.SCRATCH_ROOT / "superposition"
    df_list = []
    for case in cases:
        df = power(parent / case, models=models, regenerate=regenerate)
        # try:
        # except AttributeError as e:  # e.g. a case still running with no budgets written yet
        #     print(f"Skipping {case}: {e}")
        #     continue
        df_list.append(df.with_columns(pl.lit(case).alias("case")))
    return pl.concat(df_list, how="diagonal_relaxed")


if __name__ == "__main__":
    print(power_4x1(regenerate=True))
