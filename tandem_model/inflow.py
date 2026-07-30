"""
Fit shear, veer, Uhub, etc. to inflow properties
"""

import numpy as np
import polars as pl
from pathlib import Path
from foreach import foreach

from tandem_model import caching as cache, io, constants


def compute_inflow_quantities(precursor):
    """
    Compute inflow quantities from xy-averaged precursor simulation.

    Variables:
    - shear : linear slope dU/dz, in 1/L
    - veer : linear slope -d(alpha)/dz, in radians/L
    - veer_deg : same as veer, but in degrees/L
    - tke_h : turbulence kinetic energy at hub height (normalized by Uhub^2)
    - TI_h : turbulence intensity at hub height
    - Uhub : mean wind speed at hub height
    - phi_h : mean wind direction at hub height

    All quantities are fit within the rotor region z=[-0.5, 0.5]
    """
    s = precursor.xy_avg(budget_terms=["ubar", "vbar", "wbar", "uu", "vv", "ww"], zlim=[-0.5, 0.5])
    s["ws"] = np.sqrt(s["ubar"]**2 + s["vbar"]**2)
    s["wd"] = np.arctan2(s["vbar"], s["ubar"])
    s["tke"] = 0.5 * (s["uu"] + s["vv"] + s["ww"])
    Uhub = s["ws"].interp(z=0).item()
    phi_h = s["wd"].interp(z=0).item()
    tke_h = s["tke"].interp(z=0).item() / (Uhub**2)
    TI_h = np.sqrt(2.0 / 3.0 * tke_h)

    # fit shear and veer
    veer, _ = np.polyfit(s["z"], s["wd"], 1)
    shear, _ = np.polyfit(s["z"], s["ws"], 1)

    # Extract hub height
    z_hub = precursor.origin[2]
    z_abs = s["z"].to_numpy() + z_hub

    # Fit: log(U/Uhub) = alpha * log(z / zhub)
    alpha, _ = np.polyfit(np.log(np.abs(z_abs)), np.log(s["ws"].to_numpy() / Uhub), 1)

    return pl.DataFrame({
        "name": precursor.filename,
        "Uhub": Uhub,
        "shear": shear,
        "shear_alpha": alpha,
        "veer": -veer,
        "veer_deg": np.degrees(-veer),
        "TI_h": TI_h,
        "tke_h": tke_h,
        "phi_h": phi_h,
    })


@cache.cache_polars(constants.DATA_PATH / f"{Path(__file__).stem}_sbl.csv")
def generate_sbl(regenerate=True):
    data = io.load_sbl_ABLs(runid=5, load_precursors=True)
    dfs = foreach(compute_inflow_quantities, data["precursor"], processes=16)
    ret = pl.concat(dfs)
    return ret


if __name__ == "__main__":
    df_sbl = generate_sbl(regenerate=True)
    print(df_sbl)
