"""
Generate centerline wake deficit du vs x for LES and the gauss/kl-hub/tandem
wake models, for the CNBL 10x1 "deep array" wind farm.

Mirrors `generate.superposition_dk.compute_dk`'s solve pattern (rather than
`benchmarking.Benchmark.compute_du_centerline`, which offers no way to
override the analytical-model x-domain) so the "gauss" centerline can be
extended to match the LES/curled-model domain - otherwise it cuts off
partway through the deep array.

Kirby Heck
2026
"""

from pathlib import Path
import padeopsIO as pio
import polars as pl
import mitwindfarm as mitwf

from tandem_model import caching as cache, constants, utils
from tandem_model.models import curled_kwargs
from tandem_model.generate.superposition_power import CASE10

MODELS = ["gauss", "kl-hub", "tandem"]


def compute_du_centerline(dirname, models=MODELS, runid=5):
    """
    Solves each named wake model (solver keys, see
    `tandem_model.models.DISPLAY_NAMES`) against an LES case's
    layout/setpoints, returning a long-format DataFrame (LES + each model)
    of centerline wake deficit du vs x (columns: source, x, du_centerline).
    """
    dirname = Path(dirname)
    sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
    rotor_model = mitwf.UnifiedAD_veer(rotor_grid=mitwf.Area())

    les = utils.LESWakeField(sim, normalize=True)
    xax = les.grid.x.to_numpy()

    fields = {"LES": les}
    for key in models:
        kw = curled_kwargs(key)
        sol = utils.solve_windfarm_LES(
            sim, wakemodel=key, rotor_model=rotor_model, return_wakefield=False, normalize=True, **kw
        )
        # analytical (non-curled) models default to a short x-domain
        # (ModelWakeField's 0..20D), so extend it to match the LES/curled
        # domain - see generate.superposition_dk.compute_dk.
        fields[key] = utils.ModelWakeField(sol) if kw else utils.ModelWakeField(sol, x=xax)

    df_list = []
    for name, field in fields.items():
        du_centerline = field.du.interp(y=0, z=0, x=xax)
        df_list.append(
            pl.DataFrame({"source": name, "x": xax, "du_centerline": du_centerline.to_numpy()})
        )
    return pl.concat(df_list, how="diagonal_relaxed")


def du_centerline(dirname, models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) centerline du vs x for LES and each wake
    model in `models`, for a single case directory. Cached at
    data/<family>/<case>_du_centerline.csv (see `caching.case_cache_key`).
    """
    dirname = Path(dirname)
    family, case = cache.case_cache_key(dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_du_centerline.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_du_centerline(dirname, models=models)

    return _generate(regenerate=regenerate)


def du_centerline_10x1(case=CASE10, models=MODELS, regenerate=False):
    """
    Computes/loads centerline du vs x for the CNBL 10x1 case in
    `constants.SCRATCH_ROOT / "superposition"`.
    """
    parent = constants.SCRATCH_ROOT / "superposition"
    return du_centerline(parent / case, models=models, regenerate=regenerate).with_columns(
        pl.lit(case).alias("case")
    )


if __name__ == "__main__":
    print(du_centerline_10x1(regenerate=True))
