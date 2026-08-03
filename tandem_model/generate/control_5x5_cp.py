"""
Generate per-turbine Cp for LES and wake-model predictions for the CNBL 5x5
wind-farm control comparison (greedy vs. yaw control).

Reproduces the "Compare models for a given case" section of
`kl_model/notebooks_s3/00_wakemodel_testing.ipynb`: for each control case,
solves each wake model against the LES layout/setpoints. LES and model Cp
share the same convention (P / (pi/8 * uhub^3), see
`utils.padeops_to_polars`), so no extra normalization is needed to compare
them directly.

Cases are addressed by directory, not a fixed registry. See
`caching.case_cache_key` for how the cache path is derived.

Kirby Heck
2026
"""

from pathlib import Path
import time
import numpy as np
import polars as pl
import padeopsIO as pio
import mitwindfarm as mitwf

from tandem_model import caching as cache, constants, utils
from tandem_model.models import curled_kwargs

MODELS = ("gauss", "varvortex", "kl-hub", "tandem")

CASES = ["nocontrol", "yawcontrol"]


def compute_cp(dirname, models=MODELS, runid=5):
    """
    Solves each named wake model (solver keys, see
    `tandem_model.models.DISPLAY_NAMES`) against an LES case's
    layout/setpoints, returning a long-format DataFrame (LES + each model)
    with columns including turbine, x, y, Cp, model.
    """
    dirname = Path(dirname)
    sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")

    print("Computing Cp for 5x5 wind farm: ", dirname.name)
    df_les = utils.to_polars(sim).with_columns(pl.lit("LES").alias("model"))
    df_list = [df_les]
    # Cp_normfact_model = Momentum.UnifiedMomentum()(2.0, 0).Cp  # normalize by Betz limit
    # Cp_normfact_les = 0.6186  # df_les["Cp"].max()  # normalize by leading Cp in the LES no control
    # df_les = df_les.with_columns((pl.col("Cp") / Cp_normfact_les).alias("Pnorm"))

    # modeling stuff: 
    fw = sim.ta[0].filterwidth / np.sqrt(12)
    rotor_model = mitwf.UnifiedAD_veer(rotor_grid=mitwf.Area())

    for key in models:
        t_st = time.time()
        kw = curled_kwargs(key)
        if kw:  # parabolized (curled) RANS turbulence closure
            xmax = np.max([t.xloc for t in sim.ta])
            kw["model_kwargs"] = {**kw["model_kwargs"], "smooth_fact": fw}
            kw["xmax"] = xmax
            sol = utils.solve_windfarm_LES(sim, return_wakefield=False, rotor_model=rotor_model, wakemodel=key, **kw)
        else:
            sol = utils.solve_windfarm_LES(sim, return_wakefield=False, rotor_model=rotor_model, wakemodel=key)

        # to polars dataframe
        df = utils.to_polars(sol).with_columns(
            pl.lit(key).alias("model"),
            # (pl.col("Cp") / Cp_normfact_model).alias("Pnorm"),
        )
        print(f"  Done solving {key} wake model in {time.time() - t_st:.1f} s")
        df_list.append(df)

    return pl.concat(df_list, how="diagonal_relaxed")


def compute_cp_offline(family, case, models=MODELS):
    """
    Same as `compute_cp`, but sources data purely from the cached
    inflow/layout/meta files under data/<family>/ (see
    `utils.load_cached_case`) rather than opening a BudgetIO object - usable
    on a machine without access to the LES data.

    Run `compute_cp`/`cp` once on a machine with access to the LES data to
    populate the cache for a given case, then copy `data/<family>/` to run
    this offline. Returns Cp per turbine per model (no LES comparison,
    since that needs the actual LES data).
    """
    print(f"Computing Cp for 5x5 wind farm (offline): {family}/{case}")
    df_list = []
    for key in models:
        t_st = time.time()
        kw = curled_kwargs(key)
        sol = utils.solve_windfarm_offline(family, case, wakemodel=key, **kw)
        df = utils.to_polars(sol).with_columns(pl.lit(key).alias("model"))
        print(f"  Done solving {key} wake model in {time.time() - t_st:.1f} s")
        df_list.append(df)

    return pl.concat(df_list, how="diagonal_relaxed")


def cp(dirname, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) per-turbine Cp for LES and each wake model
    in `models`, for a single case directory. Cached at
    data/<family>/<case>_cp.csv (see `caching.case_cache_key`).
    """
    dirname = Path(dirname)
    family, case = cache.case_cache_key(dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_cp.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_cp(dirname, models=models)

    return _generate(regenerate=regenerate)


def cp_5x5(cases=CASES, models=MODELS, regenerate=False):
    """
    Computes/loads per-turbine Cp for each CNBL 5x5 control case in
    `constants.SCRATCH_ROOT / "control_5x5"`, concatenated into one DataFrame
    tagged by a `case` column (the case directory's name).
    """
    parent = constants.SCRATCH_ROOT / "control_5x5"
    df_list = []
    for case in cases:
        df = cp(parent / case, models=models, regenerate=regenerate)
        df_list.append(df.with_columns(pl.lit(case).alias("case")))
    return pl.concat(df_list, how="diagonal_relaxed")


if __name__ == "__main__":
    print(cp_5x5(regenerate=True))
