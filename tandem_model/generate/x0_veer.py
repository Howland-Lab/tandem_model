"""
Near-wake length x0 vs. veer: 
LES data and the (extended) unified momentum model's prediction.

LES data is postprocessed from the shear-veer sweep (see Heck and Howland, 
Flow (2026), figure 8c): x0 is the near-wake length (location of minimum 
wake deficit), veer_0 is the local (streamtube-averaged) veer at the 
turbine in degrees, and TI is in percent.

Cached at data/veer_wakes/x0_veer.csv (copied from the veer_wakes repo
postprocessed output; not regenerated here).

The model prediction comes directly from MITWindfarm's
`UnifiedMomentum_veer`.

Kirby Heck
2026
"""

import numpy as np
import polars as pl

from mitwindfarm.Rotor import UnifiedMomentum_veer

from tandem_model.constants import DATA_PATH, params

CTPRIME = params["ctp"]


def x0_veer_les():
    """
    Loads near-wake length x0 vs. veer LES data. Columns: TI (%), veer (deg,
    inflow/precursor value), veer_0 (deg, local value at the turbine), x0
    (near-wake length, x0/D), power (fitted far-wake power-law exponent).

    Note: this file is not generated in this repository. Data are included from 
    the veer_wakes repo (postprocessed output) and cached here. The source data
    for these results are published in: 
    Heck, K. and Howland, M. "Unravelling the effects of atmospheric dynamics on 
    wakes with a controlled synthetic inflow methodology" Flow, 6 2026,E24.
    """
    return pl.read_csv(DATA_PATH / "veer_wakes" / "x0_veer.csv")


def x0_model(ti, veer_deg, ctprime=CTPRIME, yaw=0.0):
    """
    Computes the near-wake length x0/D from the unified momentum model with
    veer (`mitwindfarm.Rotor.UnifiedMomentum_veer`) over a grid of turbulence
    intensities `ti` (fraction, e.g. 0.08) and veer values `veer_deg`
    (degrees). Returns a polars DataFrame with columns TI (%), veer (deg), x0.
    """
    model = UnifiedMomentum_veer(alpha=2)
    TI, V = np.meshgrid(np.atleast_1d(ti), np.radians(veer_deg), indexing="ij")
    sol = model(ctprime, yaw, TI=TI.flatten(), veer=V.flatten())

    return pl.DataFrame(
        {
            "TI": TI.flatten() * 100,
            "veer": np.degrees(V.flatten()),
            "x0": sol.x0,
        }
    )


if __name__ == "__main__":
    print(x0_veer_les())
    print(x0_model([0.03, 0.08], np.arange(-10, 50, 1)))
