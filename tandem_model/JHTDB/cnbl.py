"""
Pull data from the conventionally neutral ABL wind farm in the
JHTDB-Wind dataset.
"""

import numpy as np
import polars as pl
from pathlib import Path
from tandem_model import caching as cache, constants

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData, getTurbineData


DATA_DIR = constants.DATA_PATH / "JHTDB" / "cnbl"
DATA_DIR.mkdir(parents=True, exist_ok=True)
T_FIN = 3598.5  # end time of simulation in seconds
DATASET_TITLE = "nbl_windfarm"
OUTPUT_PATH = str(DATA_DIR)


try:
    with open(constants.DATA_PATH / "JHTDB" / "auth_token.txt", "r") as f:
        auth_token = f.read().strip()
except FileNotFoundError:
    auth_token = "edu.jhu.pha.turbulence.testing-201406"  # public token


@cache.cache_polars(DATA_DIR / "turbine_power_1Hz.csv")
def load_power_data(regenerate=False):
    """
    Load turbine power data from the conventionally neutral boundary layer
    wind farm dataset.

    Takes approx. 30 seconds to retrieve.
    """
    dataset = turb_dataset(
        dataset_title=DATASET_TITLE,
        output_path=OUTPUT_PATH,
        auth_token=auth_token,
    )

    turbines = list(range(1, 61))
    turbine_variable = "power"
    # 1 Hz power data:
    turbine_times = np.arange(0, T_FIN, 1, dtype=np.float64)

    turbine_result = getTurbineData(
        dataset,
        turbines,
        turbine_variable,
        turbine_times,
    )

    df = pl.DataFrame(turbine_result)
    return df


@cache.cache_polars(DATA_DIR / "precursor_xyavg.csv")
def load_precursor_xyavg(regenerate=False, n=None):
    """
    Compute precursor data from n time snapshot(s) averaged in the x, y directions

    TODO: Add averaging over n-snapshots
    """
    dataset = turb_dataset(
        dataset_title=DATASET_TITLE,
        output_path=OUTPUT_PATH,
        auth_token=auth_token,
    )

    x_points = np.linspace(0, 11000, 128)
    y_points = np.linspace(0, 3780, 48)
    z_points = np.arange(2.5, 1500, 5)

    # I can query up to 2M points... this as good as we'll do
    points = np.array(
        [
            axis.ravel()
            for axis in np.meshgrid(x_points, y_points, z_points, indexing="ij")
        ],
        dtype=np.float64,
    ).T

    results = getData(
        dataset,
        "velocity",
        timepoint_original=0,
        temporal_method="none",
        spatial_method_original="none",
        spatial_operator="field",
        points=points,
    )
    # cast points to polars dataframe and merge with query
    df = (
        pl.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]})
        .with_columns(  # rename (ux,uy,uz) to (u,v,w)
            pl.DataFrame(results[0]).rename({"ux": "u", "uy": "v", "uz": "w"})
        )
        .with_columns(  # compute U magnitude
            (pl.col("u") ** 2 + pl.col("v") ** 2 + pl.col("w") ** 2).sqrt().alias("U"),
        )
    )

    # compute first-order mean quantities
    df_mean = (
        df.group_by("z")
        .mean()
        .sort("z")
        .select(pl.col("*").exclude("x", "y"))
        .with_columns(
            np.arctan2(pl.col("v"), pl.col("u")).alias("wd")
        )
    )

    # compute fluctuations and Reynolds stresses
    df_mean = (
        df.join(df_mean, on="z", suffix="_mean")
        .with_columns(
            [
                (pl.col(c) - pl.col(f"{c}_mean")).alias(f"{c}_fluc")
                for c in ["u", "v", "w", "U"]
            ]
        )
        # compute second-order statistics
        .with_columns(
            (pl.col("U_fluc") ** 2).alias("UU"),
            (pl.col("u_fluc") ** 2).alias("uu"),
            (pl.col("v_fluc") ** 2).alias("vv"),
            (pl.col("w_fluc") ** 2).alias("ww"),
            (pl.col("u_fluc") * pl.col("v_fluc")).alias("uv"),
            (pl.col("u_fluc") * pl.col("w_fluc")).alias("uw"),
            (pl.col("v_fluc") * pl.col("w_fluc")).alias("vw"),
        )
        .with_columns(
            (0.5 * (pl.col("uu") + pl.col("vv") + pl.col("ww"))).alias("tke"),
        )
        .group_by("z")
        .mean()
        .sort("z")
        .select(
            pl.col("*").exclude("x", "y", *[c for c in df_mean.columns if "mean" in c])
        )
    )

    return df_mean


def load_dk_hub(regenerate=False, z=90):
    """
    Load hub-height TKE from primary and precursor data. 
    """
    df_precursor = load_precursor_xyavg(regenerate=regenerate)
    tke_hub = np.interp(z, df_precursor["z"], df_precursor["tke"])

    dataset = turb_dataset(
        dataset_title=DATASET_TITLE,
        output_path=OUTPUT_PATH,
        auth_token=auth_token,
    )

    x_points = np.linspace(11000, 22913, 512)
    y_points = np.linspace(0, 3780, 256)

    # I can query up to 2M points... this as good as we'll do
    points = np.array(
        [
            axis.ravel()
            for axis in np.meshgrid(x_points, y_points, [z], indexing="ij")
        ],
        dtype=np.float64,
    ).T

    results = getData(
        dataset,
        "velocity",
        timepoint_original=0,
        temporal_method="none",
        spatial_method_original="none",
        spatial_operator="field",
        points=points,
    )
    # cast points to polars dataframe and merge with query
    df = (
        pl.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]})
            .with_columns(  # rename (ux,uy,uz) to (u,v,w)
                pl.DataFrame(results[0]).rename({"ux": "u", "uy": "v", "uz": "w"})
        )
    )
    return df

if __name__ == "__main__":
    # df_power = load_power_data(regenerate=True)
    # print(df_power)
    df_precursor = load_precursor_xyavg(regenerate=True, n=1)
    print(df_precursor)
