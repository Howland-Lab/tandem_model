"""
Generate centerline wake deficit du vs x for the veer LES sweep, comparing
the "gauss" (Gaussian) and "tandem" wake models.

Adapts `generate.streamtube_sbl`'s pattern (see the "Check veer cases"
section of `kl_model/notebooks_s3/2026_nawea.ipynb`) to the centerline
deficit (`benchmarking.Benchmark.compute_du_centerline`) instead of the
streamtube-averaged deficit, for the shear_01/Ro_f_02 veer sweep at
SCRATCH_ROOT/veer_wakes (veer_00..veer_05, i.e. -10..40 deg in 10 deg steps;
veer_01 is the 0 deg baseline).

Only TI_00 (low TI) is wired up by default. TI_01 cases exist on scratch
under the same naming (swap `ti_tag="TI_01"`) and can be generated the same
way once needed.

Kirby Heck
2026
"""

from pathlib import Path
import re

import padeopsIO as pio
import polars as pl

from tandem_model import caching as cache, constants
from tandem_model.benchmarking import Benchmark
from tandem_model.models import curled_kwargs

VEER_ROOT = constants.SCRATCH_ROOT / "veer_wakes"
VEER_IDS = range(1, 6)  # veer_01 .. veer_05

MODELS = ["gauss", "varvortex", "tandem"]


def veer_dirnames(ti_tag="TI_00", shear_tag="shear_01", ro_f_tag="Ro_f_02"):
    """
    Case directories for the veer sweep at fixed shear/Ro_f and a given TI
    tag ("TI_00": low TI, "TI_01": higher TI), one per veer angle
    (veer_01..veer_05).
    """
    return [
        VEER_ROOT / f"{shear_tag}_veer_{i:02d}_{ti_tag}_{ro_f_tag}" for i in VEER_IDS
    ]


def parse_veer_deg(case):
    """
    Parses the imposed veer angle (deg) from a case name, e.g.
    "shear_01_veer_02_TI_00_Ro_f_02" -> 10.0. veer_01 is the 0 deg
    (no imposed veer) baseline; each step is 10 deg.
    """
    match = re.search(r"veer_(\d+)", case)
    if match is None:
        return None
    return (int(match.group(1)) - 1) * 10.0


def compute_veer_wakes_centerline(dirnames, models=MODELS, runid=1):
    """
    Computes centerline wake deficit du vs x for each veer LES case in
    dirnames and each named wake model, returning one long-format DataFrame
    (columns: case, veer_deg, source, x, du_centerline). `source` holds
    solver keys (see `tandem_model.models.DISPLAY_NAMES`) or "LES" for the
    reference.
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
    df = bm.compute_du_centerline()
    return df.with_columns(
        pl.col("model").replace({"ref": "LES"}).alias("source"),
        pl.col("case").map_elements(parse_veer_deg, return_dtype=pl.Float64).alias("veer_deg"),
    ).drop("model")


def veer_wakes_centerline(ti_tag="TI_00", models=MODELS, regenerate=False):
    """
    Computes (or loads from cache) centerline du vs x for the veer sweep at a
    given TI tag. Cached at data/veer_wakes/centerline_du_<ti_tag>.csv.
    """
    dirnames = veer_dirnames(ti_tag=ti_tag)
    cache_file = constants.DATA_PATH / "veer_wakes" / f"centerline_du_{ti_tag}.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_veer_wakes_centerline(dirnames=dirnames, models=models)

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    print(veer_wakes_centerline(ti_tag="TI_00", regenerate=True))
    print(veer_wakes_centerline(ti_tag="TI_01", regenerate=True))
