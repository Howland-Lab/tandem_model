"""
Generate a table of inflow properties for the SBL LES cases at
`constants.SCRATCH_ROOT / "sbl"` (z0_02 and z0_01 roughness families).

For each case, gleans/computes:
- Cr: surface cooling rate (K/hr), from the input namelist's `dTsurf_dt`
  (falls back to parsing the directory name, see `mixing_length.parse_cooling_rate`,
  if the namelist lookup fails).
- Uhub: hub-height wind speed, dimensionalized to G = 12 m/s.
- alpha: power-law shear exponent across the rotor (zhub/D = 0.625 = zref).
- veer_deg: wind-direction veer across the rotor (length scale = D), in degrees.
- ustar: friction velocity from the logfile "u_star" diagnostic, dimensionalized
  to G = 12 m/s.
- L_obu: Obukhov length from the logfile "Inv. Ob." diagnostic (already part of
  `utils.case_meta_LES`), dimensionalized to D = 240 m.
- TI: turbulence intensity at hub height (%), sqrt(2k/3) / Uhub.
- L0x, L0y: integral length scales of streamwise velocity fluctuations u',
  in x and y respectively (see `utils.compute_Lturb_z`; ported from
  `veer_wakes.utils.compute_Lturb_z`).

Inflow quantities (Uhub, alpha, veer, TI) are read from the concurrent
precursor run, xy-averaged over `xlim=[-1, 1]`, `zlim=[-1, 1]` around the
rotor plane -- matching `veer_wakes.generate.inflow_table_sbl.inflow_from_df`'s
averaging window exactly (verified bit-for-bit against that script's cached
table for a shared case), rather than `utils.inflow_LES`'s narrower
`xlim=[-0.5, 0.5]` window (tuned for wake-model solves, not table-generation
precision -- using it here made Uhub/TI/veer differ from the reference table
by a part in 1e4-1e5). Cr/ustar/L_obu come straight from the logfile, and
L0x/L0y come from a single instantaneous field snapshot of the precursor run
(the only field data these cases retain on disk).

Kirby Heck
2026
"""

import re
import numpy as np
import padeopsIO as pio
import polars as pl

from tandem_model import caching as cache, constants, utils
from tandem_model.generate.mixing_length import parse_cooling_rate

Z0_FAMILIES = ("z0_02", "z0_01")
RUNID, PRECURSOR_RUNID = 5, 4
ZHUB, R = 0.0, 0.5  # sim coordinates are already normalized to hub=0, D=1
G = 12.0  # m/s, reference geostrophic wind speed used to dimensionalize
D = constants.D  # m


def sbl_dirnames(families=Z0_FAMILIES):
    """
    Lists all SBL case directories under `constants.SCRATCH_ROOT / "sbl"`
    matching any of the given z0 family tags (e.g. "z0_02" -> all
    `*z0_02*` cases), sorted by name.
    """
    root = constants.SCRATCH_ROOT / "sbl"
    dirnames = set()
    for family in families:
        dirnames.update(root.glob(f"*{family}*"))
    return sorted(dirnames)


def _parse_dTsurf_dt(sim: pio.BudgetIO, case: str) -> float:
    """
    Surface cooling rate Cr (K/hr), read from the input namelist's
    `dTsurf_dt` (negated, since a cooling surface has dTsurf_dt < 0); falls
    back to parsing the case directory name if the namelist lookup fails.
    """
    try:
        dTsurf_dt = pio.key_search_r(sim.input_nml, "dTsurf_dt")
        return -float(dTsurf_dt)
    except (KeyError, TypeError):
        return parse_cooling_rate(case)


def compute_powerlaw(ds_inflow, zhub=ZHUB, R=R, zground=None):
    """Best-fit power-law shear exponent to U(z) across the rotor [-R, R] + zhub."""
    z = ds_inflow.z
    if zground is None:
        dz = z.diff("z").mean().item()
        zground = z.min().item() - dz / 2
    z_abs = z - zground

    Uhub = ds_inflow["U"].interp(z=zhub).item()
    log_u = np.log(ds_inflow["U"].sel(z=slice(-R + zhub, R + zhub)) / Uhub)
    log_z = np.log(z_abs.sel(z=slice(-R + zhub, R + zhub)) / (zhub - zground))
    alpha = (log_u * log_z).sum("z") / (log_z * log_z).sum("z")
    return alpha.item()


def compute_veer_deg(ds_inflow, zhub=ZHUB, R=R):
    """Wind-direction veer across the rotor [-R, R] + zhub, in degrees."""
    wd = np.arctan2(ds_inflow["vbar"], ds_inflow["ubar"])
    veer = -wd.interp(z=[-R + zhub, R + zhub]).diff("z").item()
    return veer * 180 / np.pi


def compute_inflow_row(dirname, runid=RUNID, precursor_runid=PRECURSOR_RUNID):
    """Computes one row (dict) of inflow properties for a single SBL case directory."""
    case = dirname.name
    print("Computing inflow properties for: ", case)
    sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
    pre = pio.BudgetIO(dirname, padeops=True, runid=precursor_runid, normalize_origin=sim.origin)

    # xy-average the precursor over the rotor-plane window -- see module docstring
    # for why this (not `utils.inflow_LES`) matches the reference table exactly.
    ds = pre.xy_avg(budget_terms=["ubar", "vbar", "uu", "vv", "ww"], zlim=[-1, 1], xlim=[-1, 1])
    ds["U"] = np.sqrt(ds["ubar"] ** 2 + ds["vbar"] ** 2)
    Uhub = ds["U"].interp(z=ZHUB).item()
    k_hub = (0.5 * (ds["uu"] + ds["vv"] + ds["ww"])).interp(z=ZHUB).item()

    L0x = utils.compute_Lturb_z(pre, axis=0, zlim=ZHUB).item()
    L0y = utils.compute_Lturb_z(pre, axis=1, zlim=ZHUB).item()

    return {
        "case": case,
        "z0_family": re.search(r"z0_\d+", case).group(0),
        "Cr": _parse_dTsurf_dt(sim, case),
        "Uhub": Uhub * G,
        "alpha": compute_powerlaw(ds),
        "veer_deg": compute_veer_deg(ds),
        "phi_hub": np.arctan2(ds.vbar, ds.ubar).interp(z=ZHUB).item() * 180 / np.pi,
        "ustar": utils.get_ustar(sim) * G,
        "L_obu": utils.get_obukhov_length(sim) * D,
        "TI": 100 * np.sqrt(2 / 3 * k_hub) / Uhub,
        "L0x": L0x * D,
        "L0y": L0y * D,
    }


def compute_inflow_table_sbl(families=Z0_FAMILIES):
    """
    Computes the inflow property table for all SBL cases matching `families`
    (e.g. "z0_02" -> all `*z0_02*` cases), one row per case.
    """
    return pl.from_dicts([compute_inflow_row(d) for d in sbl_dirnames(families)])


def inflow_table_sbl(families=Z0_FAMILIES, regenerate=False):
    """
    Computes (or loads from cache) the SBL inflow property table. Cached at
    data/sbl/inflow_table.csv.
    """
    cache_file = constants.DATA_PATH / "sbl" / "inflow_table.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return compute_inflow_table_sbl(families=families)

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=3):
        print(inflow_table_sbl(regenerate=True))
