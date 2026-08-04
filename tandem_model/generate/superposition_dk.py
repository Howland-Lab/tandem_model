"""
Generate yz-integrated wake-added turbulence (dk) vs downstream position x
for LES and the curled wake models (kl-hub, tandem), across the CNBL 4x1
wind-direction sweep and the CNBL 10x1 case, along with the downstream
turbine locations for each case.

Reproduces the yz-integrated dk sketch of
`notebooks/00_superposition_fields.ipynb`: for each case, solves each curled
wake model against the LES layout/setpoints, integrating dk (normalized by
uhub^2) over (y, z) at each x. "gauss" has no dk field, so only the curled
(TKE-transport) models are included here. Turbine locations are appended to
the same cached table as source="turbine" rows (x set, dk_int null) so the
plot script only needs this one cache.

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
from tandem_model.generate.superposition_power import CASES, CASE10

MODELS = ("kl-hub", "tandem")


def compute_dk(dirname, models=MODELS, runid=5):
    """
    Solves each named curled wake model (solver keys, see
    `tandem_model.models.DISPLAY_NAMES`) against an LES case's
    layout/setpoints, returning a long-format DataFrame (LES + each model,
    plus "turbine" marker rows) with columns source, x, dk_int.
    """
    dirname = Path(dirname)
    sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
    rotor_model = mitwf.UnifiedAD_veer(rotor_grid=mitwf.Area())

    les = utils.LESWakeField(sim)
    xax = les.grid.x.to_numpy()

    fields = {"LES": les}
    for key in models:
        kw = curled_kwargs(key)
        sol = utils.solve_windfarm_LES(
            sim, wakemodel=key, rotor_model=rotor_model, return_wakefield=False, **kw
        )
        # analytical (non-curled) models default to a short x-domain
        # (ModelWakeField's 0..20D), so extend it to match the LES/curled
        # domain - otherwise e.g. "gauss" cuts off partway through a deep
        # array.
        fields[key] = utils.ModelWakeField(sol) if kw else utils.ModelWakeField(sol, x=xax)

    df_list = []
    for name, field in fields.items():
        dk_int = np.asarray(field.dk.integrate(("y", "z")))
        df_list.append(
            pl.DataFrame({"source": name, "x": np.asarray(field.grid.x), "dk_int": dk_int})
        )

    turbine_x = [t.xloc - sim.origin[0] for t in les.sim.ta]
    df_list.append(pl.DataFrame({"source": "turbine", "x": turbine_x, "dk_int": None}))

    return pl.concat(df_list, how="diagonal_relaxed")


def dk(dirname, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) yz-integrated dk vs x for LES and each
    curled wake model in `models`, plus downstream turbine locations, for a
    single case directory. Cached at data/<family>/<case>_dk.csv (see
    `caching.case_cache_key`).
    """
    dirname = Path(dirname)
    family, case = cache.case_cache_key(dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_dk.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_dk(dirname, models=models)

    return _generate(regenerate=regenerate)


def dk_4x1(cases=CASES, models=MODELS, regenerate=False):
    """
    Computes/loads yz-integrated dk vs x (plus turbine locations) for each
    CNBL 4x1 wind-direction case in `constants.SCRATCH_ROOT / "superposition"`,
    concatenated into one DataFrame tagged by a `case` column (the case
    directory's name).
    """
    parent = constants.SCRATCH_ROOT / "superposition"
    df_list = []
    for case in cases:
        df = dk(parent / case, models=models, regenerate=regenerate)
        df_list.append(df.with_columns(pl.lit(case).alias("case")))
    return pl.concat(df_list, how="diagonal_relaxed")


def dk_10x1(case=CASE10, models=MODELS, regenerate=False):
    """
    Computes/loads yz-integrated dk vs x (plus turbine locations) for the
    CNBL 10x1 case in `constants.SCRATCH_ROOT / "superposition"`.
    """
    parent = constants.SCRATCH_ROOT / "superposition"
    return dk(parent / case, models=models, regenerate=regenerate).with_columns(
        pl.lit(case).alias("case")
    )


if __name__ == "__main__":
    print(dk_4x1(regenerate=True))
    print(dk_10x1(regenerate=True))
