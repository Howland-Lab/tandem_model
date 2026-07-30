"""
Query and save diurnal wind farm data from the JHTDB-Wind.
"""

import numpy as np
import polars as pl
from pathlib import Path
from tandem_model import caching as cache, constants

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData, getTurbineData


DATA_DIR = constants.DATA_PATH / "JHTDB" / "diurnal"
DATA_DIR.mkdir(parents=True, exist_ok=True)
try:
    with open(constants.DATA_PATH / "JHTDB" / "auth_token.txt", "r") as f:
        auth_token = f.read().strip()
except FileNotFoundError:
    auth_token = "edu.jhu.pha.turbulence.testing-201406"  # public token

SIM_TIME_ST = 15 * 3600  # 15:00:00 in seconds, "TRUE" start time of simulation
T_FIN = 86398.5  # end time of simulation in seconds


@cache.cache_polars(DATA_DIR / "turbine_power_1Hz.csv")
def load_power_data(regenerate=False):
    """
    Load turbine power data from the diurnal wind farm dataset.

    Takes approx. 30 seconds to retrieve.
    """
    dataset_title = "diurnal_windfarm"
    output_path = str(DATA_DIR)

    dataset = turb_dataset(
        dataset_title=dataset_title,
        output_path=output_path,
        auth_token=auth_token,
    )

    turbines = list(range(1, 9))
    turbine_variable = "power"
    # 1 Hz power data:
    turbine_times = np.arange(0, 86398, 1, dtype=np.float64)

    turbine_result = getTurbineData(
        dataset,
        turbines,
        turbine_variable,
        turbine_times,
    )

    df = pl.DataFrame(turbine_result)
    return df


def load_power_window(length=600):
    """
    Load turbine power data and compute windowed averages.
    
    Parameters
    ----------
    length : int
        Length of the averaging window in seconds. Default is 600 (10 minutes).
    """
    power_1Hz = load_power_data()
    power_window = (
        power_1Hz.with_columns(
            pl.col("time").floordiv(length).alias("window"),
        )
        .group_by("turbine", "window")
        .mean()
        .select(pl.col("*").exclude("time"))
        .sort(by=["turbine", "window"])
    )
    return power_window


@cache.cache_polars(DATA_DIR / "metmast_data_1Hz.csv")
def load_metmast_data(regenerate=False):
    """
    Load met mast data at a location 500 m upstream of the wind farm
    and save u, v, w, T at 1 Hz frequency.

    Takes approx. 2 minutes to retrieve.
    """
    # ===== define constants =====
    dataset_title = "diurnal_windfarm"
    spatial_method = "none"
    spatial_operator = "field"
    temporal_method = "none"
    dz = 5  # m

    # ===== define met mast location =====
    zax = np.arange(dz / 2, 300, dz)
    mast_location = (
        7000,
        0.0,
    )  # x, y in m... hard to tell exactly where turbines are located
    points = np.array(
        np.broadcast_arrays(
            np.full_like(zax, mast_location[0]),
            np.full_like(zax, mast_location[1]),
            zax,
        )
    ).T

    output_path = str(DATA_DIR)
    dataset = turb_dataset(
        dataset_title=dataset_title,
        output_path=output_path,
        auth_token=auth_token,
    )

    to_concat = []
    time_st = 0  # normalized start time

    # we can't query all of these points at once, need to chunk
    t_chunk = 120  # 2 minute chunks
    dt = 1  # 1 second intervals
    for _tstart in np.arange(time_st, T_FIN, t_chunk):
        time_opts = [min(_tstart + t_chunk - dt, T_FIN), dt]  # [t_final, interval]
        print(f"READING TIME INTERVAL: {_tstart:.0f}, {time_opts[0]:.0f} sec")

        results_list = []  # reset this for each chunk
        for v in ["velocity", "temperature"]:
            res, t = getData(
                dataset,
                v,
                _tstart,
                temporal_method,
                spatial_method,
                spatial_operator,
                points,
                time_opts,
                return_times=True,
            )
            results_list.append(res)

        # combine into one "long-format" dataframe
        for k, (u_df, T_df) in enumerate(zip(*results_list)):
            _res = (
                pl.from_pandas(u_df)
                .with_columns(
                    pl.lit(t[k]).alias("time"),
                    T=T_df["θ"].values + 273.15,  # convert C -> K
                    z=zax,
                )
                .rename({"ux": "u", "uy": "v", "uz": "w"})
            )
            to_concat.append(_res)

    df = pl.concat(to_concat).select(["time", "z", "u", "v", "w", "T"])
    return df


def load_metmast_window(length=600):
    """
    Compute fluxes and bin metmast data down to windowed averages.
    
    Parameters
    ----------
    length : int
        Length of the averaging window in seconds. Default is 600 (10 minutes).
    """
    metmast = load_metmast_data()
    metmast = metmast.with_columns(
        pl.col("time").floordiv(length).alias("window"),
        (pl.col("u") ** 2 + pl.col("v") ** 2).sqrt().alias("ws"),
        (np.arctan2(pl.col("v"), pl.col("u")) * 180 / np.pi).alias("wd"),
    )

    mm_window = (
        metmast.group_by("z", "window")
        .mean()
        .select(pl.col("*").exclude("time"))
        .sort(by=["window", "z"])
    )

    joined =  metmast.join(
        mm_window.select(
            pl.col("window"),
            pl.col("z"),
            *[pl.col(c).alias(c + "_win") for c in ["ws", "wd", "u", "v", "w", "T"]],
        ),
        on=["z", "window"],
    ).sort(by=["time", "z"])

    fluxes = joined.with_columns(
        (pl.col(c) - pl.col(f"{c}_win")).alias(f"{c}_fluc")
        for c in ["ws", "u", "v", "w", "T", "wd"]
    ).with_columns(
        (pl.col("u_fluc") * pl.col("w_fluc")).alias("uw"),
        (pl.col("v_fluc") * pl.col("w_fluc")).alias("vw"),
        (pl.col("T_fluc") * pl.col("w_fluc")).alias("wT"),
        (pl.col("wd_fluc") * pl.col("wd_fluc")).alias("wd_var"),
        (0.5 * pl.col("ws_fluc") ** 2).alias("tke"),
    )

    flux_window = (
        fluxes.group_by("z", "window")
        .mean()
        .select(pl.col("*").exclude("time"))
        .sort(by=["window", "z"])
    ).with_columns(
        pl.col("wd_var").sqrt().alias("wd_std")
    )

    return flux_window


if __name__ == "__main__":
    df = load_metmast_data(regenerate=False)
    print(df)
