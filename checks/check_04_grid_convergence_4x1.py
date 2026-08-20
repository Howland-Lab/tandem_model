"""
Check (4): grid convergence of the TANDEM turbulence closure on a canonical
(non-LES) aligned 4x1 wind farm -- no reference LES data needed here, this
is purely a model self-consistency check.

Geometry/inflow, loosely modeled on the CNBL 4x1 cases
(`generate.superposition_power`) but using a clean synthetic setup rather
than reading any LES case: 6D turbine spacing, hub height zhub = 1D, a
power-law shear profile (`mitwindfarm.PowerLaw`), constant ambient TI, and a
small linear veer -- ground at z=0, matching `mitwindfarm`'s own
`PowerLaw`/`CurledWindfarm` convention (see
`analysis_code/MITWindfarm/examples/example_08_curled_windfarm.py`), rather
than the turbine-centered z=0 frame `utils.solve_windfarm_LES` uses for real
LES cases.

Two set-point cases are swept (`YAW_CASES`):
  - "aligned": Ctprime = 4/3, yaw = 0 for all four turbines.
  - "yawed30": Ctprime = 4/3 for all turbines, but the leading turbine is
    yawed 30 degrees (the rest unyawed) -- exercises the curled-wake skew
    terms that the aligned case doesn't.

For each set-point case, two grid-refinement studies are run over the same
dy = dz sweep (`H_VALUES`), varying only how the near-wake initial-condition
smoothing/mixing length are tied to the grid:

  1) Fixed sigma_IC: `smooth_fact` (this is sigma_IC; see
     `CurledWake.CurledWakeWindfield`'s docstring and `ic_stencil_Rd.py`) is
     pinned to `SMOOTH_FACT_FIXED`, independent of dy/dz. Since the TANDEM
     closure's near-wake mixing length `l_nw` defaults to
     `curledwake.smooth_fact` whenever it isn't set explicitly (see
     `mitwindfarm.tandem.TurbulenceModel_tandem_md.__init__`), pinning
     smooth_fact this way also pins l_nw -- so this case is a "clean" test
     of grid convergence with every other length scale in the problem held
     fixed.
  2) Variable sigma_IC (mitwindfarm's own default/best-practice behavior,
     `smooth_fact=None`): sigma_IC = dy/2 and (since l_nw again follows
     smooth_fact by default) l_nw = dy/2 as well, so both shrink with the
     grid. This is what every other `tandem_model` figure/generate script
     actually uses (`models.CURLED_MODEL_KWARGS` never sets smooth_fact),
     so it's the mode that matters for the manuscript's own results.

Reports per-turbine Cp and total array Cp at each resolution, plus the
relative change from the next-coarser grid and from the finest grid tested,
so both the convergence *rate* and how close the production dy=dz=0.1 grid
(`models.CURLED_MODEL_KWARGS`) is to the asymptotic result are visible.

Neither `mitwindfarm` nor any cached/generated data is touched by this
script -- it only calls `utils.solve_windfarm_setpoints` with a synthetic
inflow/layout/meta.

Run with the repo's project venv, e.g.:
    /work2/08445/tg877441/stampede3/claude_projects/tandem/.venv/bin/python \\
        checks/check_04_grid_convergence_4x1.py

Kirby Heck
2026
"""

import numpy as np
import polars as pl
import mitwindfarm as mitwf

from tandem_model import utils
from tandem_model.models import K_KWARGS, CURLED_MODEL_KWARGS

K_MODEL = "tandem"

# --- canonical 4x1 geometry/inflow (non-dim by D = 1) ---
ZHUB = 1.0  # hub height, diameters (ground at z=0, mitwindfarm's PowerLaw convention)
SPACING = 6.0  # turbine spacing, diameters
N_TURB = 4
ALPHA = 0.11  # power-law shear exponent
TIAMB = 0.06  # ambient turbulence intensity
VEER = np.radians(2.0)  # small linear veer, rad/D (mild Ekman-like turning)
CTPRIME = 4 / 3
YAW_LEAD_DEG = 30.0

YAW_CASES = {
    "aligned (yaw = 0)": 0.0,
    "leading turbine yawed 30 deg": YAW_LEAD_DEG,
}

# sigma_IC = smooth_fact, held fixed across the whole dy/dz sweep in the
# "fixed" study -- matches the production dy=0.1 grid's own dy/2 (see
# models.CURLED_MODEL_KWARGS), so the "fixed" and "variable" studies agree
# exactly at dy=dz=0.1.
SMOOTH_FACT_FIXED = 0.05

# dy = dz sweep, diameters, coarsest to finest
H_VALUES = [0.20, 0.15, 0.10, 0.075, 0.05, 0.0375, 0.025]


def build_case(yaw_lead_deg=0.0):
    """Returns (inflow, layout, setpoints, meta) for the canonical 4x1 case."""
    inflow = mitwf.PowerLaw(Uref=1.0, zref=ZHUB, exp=ALPHA, TIamb=TIAMB, veer=VEER)

    xs = np.arange(N_TURB) * SPACING
    ys = np.zeros(N_TURB)
    zs = np.full(N_TURB, ZHUB)
    layout = mitwf.Layout(xs, ys, zs)

    yaws = np.zeros(N_TURB)
    yaws[0] = np.radians(yaw_lead_deg)
    setpoints = [(CTPRIME, yaw) for yaw in yaws]

    meta = dict(xmax=xs[-1], L_obu=np.inf, zwall=0.0, Ro=1e10)
    return inflow, layout, setpoints, meta


def solve_grid(h, smooth_fact, yaw_lead_deg):
    """Solves the canonical 4x1 case at dy=dz=h, returning per-turbine Cp."""
    inflow, layout, setpoints, meta = build_case(yaw_lead_deg)

    model_kwargs = {**CURLED_MODEL_KWARGS, "dy": h, "dz": h}
    if smooth_fact is not None:
        model_kwargs["smooth_fact"] = smooth_fact
    else:
        model_kwargs.pop("smooth_fact", None)  # let CurledWakeWindfield default to dy/2

    sol = utils.solve_windfarm_setpoints(
        inflow,
        layout,
        setpoints,
        K_MODEL,
        meta,
        rotor_model=mitwf.UnifiedAD_veer(rotor_grid=mitwf.Area()),
        k_kwargs=dict(K_KWARGS[K_MODEL]),
        model_kwargs=model_kwargs,
    )
    return np.array([r.Cp for r in sol.rotors])


def run_sweep(smooth_fact, yaw_lead_deg, h_values=H_VALUES):
    """Solves the grid sweep, returning a polars DataFrame (one row per h)."""
    rows = []
    for h in h_values:
        Cp = solve_grid(h, smooth_fact, yaw_lead_deg)
        rows.append(
            {"h": h, **{f"Cp_{i + 1}": c for i, c in enumerate(Cp)}, "Cp_sum": Cp.sum()}
        )
    return pl.DataFrame(rows)


def report(df: pl.DataFrame, title: str):
    """Prints Cp_sum vs. h, with relative changes from the previous (coarser)
    grid and from the finest grid tested."""
    cp_sum = df["Cp_sum"].to_numpy()
    finest = cp_sum[-1]
    d_prev = np.r_[np.nan, np.diff(cp_sum) / cp_sum[:-1] * 100]
    d_finest = (cp_sum - finest) / finest * 100

    print(f"  {title}")
    print(f"  {'dy=dz':>8}{'Cp_sum':>12}{'d(prev), %':>14}{'d(finest), %':>16}")
    for h, cp, dp, df_ in zip(df["h"].to_numpy(), cp_sum, d_prev, d_finest):
        dp_str = f"{dp:14.4f}" if not np.isnan(dp) else " " * 14
        print(f"  {h:8.4f}{cp:12.5f}{dp_str}{df_:16.4f}")
    print()


def main(h_values=H_VALUES):
    for case_label, yaw_lead_deg in YAW_CASES.items():
        print("=" * 70)
        print(f"Set points: Ctprime = 4/3, {case_label}")
        print("=" * 70)

        df_fixed = run_sweep(SMOOTH_FACT_FIXED, yaw_lead_deg, h_values)
        report(df_fixed, f"(1) fixed sigma_IC = {SMOOTH_FACT_FIXED} (grid-independent)")

        df_var = run_sweep(None, yaw_lead_deg, h_values)
        report(df_var, "(2) variable sigma_IC = dy/2 (mitwindfarm default, l_nw follows)")


if __name__ == "__main__":
    main()
