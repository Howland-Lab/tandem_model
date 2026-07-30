"""
Compute near-wake quantities for rotors in sheared/veered inflow.

Kirby Heck
2025-2026
"""

import gc
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from foreach import foreach
from streamtube import Streamtube

from tandem_model import caching as cache, io, constants


stream_kwargs = dict(R=0.35)


def compute_nearwake_concurrent(_x):
    """Takes inflow from precursor simulation"""
    sim, precursor = _x
    inflow = precursor.slice(budget_terms=["ubar", "vbar"]).mean("y")
    return compute_nearwake(sim, inflow)


def compute_streamtube(
    sim,
    xlim=None,
    ylim=None,
    zlim=None,
    stream_kwargs=None,
):
    """
    Parameters
    ----------
    sim : pio.BudgetIO
    xlim, ylim, zlim : float or list
        Slice limits, see pio.BudgetIO.slice
    """
    s = sim.slice(
        budget_terms=["ubar", "vbar", "wbar"], xlim=xlim, ylim=ylim, zlim=zlim
    )
    stream = Streamtube(s["x"], s["y"], s["z"], s["ubar"], s["vbar"], s["wbar"])

    if stream_kwargs is None:
        stream_kwargs = dict()

    stream.compute_mask(**stream_kwargs)

    return stream


def compute_nearwake(sim, inflow):
    """
    Computes near-wake properties i.e., u4, v4

    All quantities are non-dimensionalized to rho=1 and the hub-height windspeed,
    but the rotor-equivalent wind speed (REWS) is also computed (and is also non-
    dimensionalized to Uhub).
    """
    s = sim.slice(budget_terms=["ubar", "vbar", "wbar", "pbar", "xAD"])
    stream = compute_streamtube(sim, stream_kwargs=stream_kwargs)
    s["du"] = sim.budget["ubar"] - inflow["ubar"]
    s["dv"] = sim.budget["vbar"] - inflow["vbar"]
    s["mask"] = stream.mask
    U = inflow["ubar"].interp(x=0, z=0).item()

    r = 0.5
    zids = abs(sim.grid.z) < r  # integrate within the rotor area only
    # weight based on the area of the "disk"
    A = np.sqrt(r**2 - sim.grid.z[zids] ** 2).integrate("z")
    rews = (
        inflow["ubar"].interp(x=0)[zids] * np.sqrt(r**2 - sim.grid.z[zids] ** 2)
    ).integrate("z") / A / U

    # ================ Compute near-wake quantities ================
    du_x = (s["du"] * s["mask"]).sum(("y", "z")) / s["mask"].sum(("y", "z"))
    uid = du_x.argmin().item()
    du = du_x[uid].item() / U

    x0 = sim.grid.x[uid].item()

    dv_x = (s["dv"] * s["mask"]).sum(("y", "z")) / s["mask"].sum(("y", "z"))
    vid = (abs(dv_x)).argmax().item()  # get maximum magnitude
    dv = dv_x[vid].item() / U

    p_x = (
        (s["pbar"] * s["mask"]).sum(("y", "z")) / s["mask"].sum(("y", "z")) / (U**2)
    )  # "correct/normalize" p4 to u_hub**2
    p4 = p_x[uid].item()  # location where the velocity begins to recover
    p1 = p_x[0].item()  # inlet pressure

    # ================ Compute rotor quantities ================
    ud_time = sim.read_turb_uvel(tidx="all")
    time = sim.get_time_ax()
    filt = time >= sim.input_nml["budget_time_avg"]["time_budget_start"]
    if len(time) != len(ud_time):
        ud_time = sim.read_turb_uvel(tidx="all", dup_threshold=0)[-len(time):]
    ud = np.mean(ud_time[filt]) / U

    admtype = sim.input_nml["windturbines"]["adm_type"]
    if admtype == 5:
        thrust = s["xAD"].integrate(("x", "y", "z")).item()
        CT = thrust / (-np.pi / 8 * U**2)
        Ctprime = sim.ta[0].ct
    elif admtype == 6:
        thrust = s["xAD"].integrate(("x", "y", "z")).item()
        CT = sim.ta[0].ct
        Ctprime = CT * (U / ud)**2
    else:
        raise NotImplementedError("How did we get here?")

    # compute power:
    CP = CT * ud

    # clean up memory
    del s
    del stream
    gc.collect()

    _df = pd.DataFrame(
        dict(
            name=sim.filename,
            u=U + du,
            du=du,
            dv=dv,
            p1=p1,
            p4=p4,
            x0=x0,
            Ctprime=Ctprime,
            CT=CT,
            CP=CP,
            ud=ud,
            Uhub=U,
            REWS=rews.item(),
            admtype=admtype,
        ), index=[0]
    )
    return _df


@cache.cache_polars(constants.DATA_PATH / f"{Path(__file__).stem}_sbl.csv")
def generate_sbl(regenerate=True):
    data = io.load_sbl_ABLs(runid=5, load_precursors=True)
    dfs = foreach(compute_nearwake_concurrent, zip(data["sim"], data["precursor"]), processes=8)
    ret = pl.concat([pl.from_pandas(_df) for _df in dfs])
    return ret


if __name__ == "__main__":
    df = generate_sbl(regenerate=True)
