"""
LES IO utilities for the TANDEM model project.

Kirby Heck
2026
"""

import numpy as np
import polars as pl
from pathlib import Path
import padeopsIO as pio
import warnings
import socket

if "stampede3" in socket.gethostname():
    SBL_PATH = Path(r"/scratch/08445/tg877441/tandem_model/sbl")  # Stampede3 data
else:
    SBL_PATH = None


def load_data(
    maindir,
    ignore_warnings=True,
    filter_budgets_exist=True,
    filter_fields_exist=False,
    filter_args=None,
    load_precursors=False,
    runid=1,
    precursor_runid=2,
    quiet=False,
    strict_runid=True,
):
    """
    Loads data from a .csv file or from subdirectories in a parent directory.

    Returns a polars DataFrame with one row per simulation case, a `sim` column
    holding the loaded `pio.BudgetIO` handle (and `precursor` if requested).
    """
    maindir = Path(maindir)
    try:
        df = pl.read_csv(list(maindir.glob("*.csv"))[0])
    except IndexError:
        # no csv written
        warnings.warn("No .csv file found")
        dirs = sorted(maindir.glob("*"))
        df = pl.DataFrame({"dirname": [str(name) for name in dirs if name.is_dir()]})
        df = df.with_columns(pl.Series("name", [Path(p).name for p in df["dirname"]]))

    df = df.with_columns(  # round any shear/veer sweep columns, if present
        [df[key].cast(float).round(5) for key in ["shear", "veer"] if key in df.columns]
    )

    if filter_args is not None:
        df = df.filter(*filter_args)

    sims = []
    precursors = []
    for name in df["name"]:
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter("ignore")
            if (maindir / name).is_dir():
                try:
                    sim = pio.BudgetIO(
                        maindir / name, padeops=True, runid=runid, normalize_origin="turb",
                        quiet=quiet, strict_runid=strict_runid,
                    )
                except FileNotFoundError:
                    warnings.warn(f"File not found for {name}")
                    sims.append(None)
                    if load_precursors:
                        precursors.append(None)
                    continue

                sims.append(sim)
                if load_precursors:
                    precursors.append(
                        pio.BudgetIO(
                            maindir / name, padeops=True, runid=precursor_runid,
                            normalize_origin=sims[-1].origin, quiet=quiet, strict_runid=strict_runid,
                        )
                    )
                else:
                    precursors.append(None)
            else:
                warnings.warn(f"File not found for {name}")
                sims.append(None)

    budgets = [_c.associate_budgets if _c is not None else False for _c in sims]
    df = df.with_columns(
        pl.Series("sim", sims),
        pl.Series("budgets", budgets),
    )
    if load_precursors:
        df = df.with_columns(pl.Series("precursor", precursors))
    if filter_budgets_exist:
        df = df.filter(budgets)  # only select where budgets exist

    if filter_fields_exist:
        fields = [_c.associate_fields if _c is not None else False for _c in df["sim"]]
        df = df.filter(fields)  # only select where fields exist

    return df


def load_sbl_data(load_neutral=True):
    """Load SBL data"""
    filters = (pl.col("z0") == pl.col("z0").max(), )
    if not load_neutral:
        filters += (pl.col("dTsurf_dt").ne(0), )
    sbl = load_data(SBL_PATH, load_precursors=True, runid=5, precursor_runid=4, quiet=True)
    sbl = sbl.filter(filters).with_columns(
        pl.lit("SBL").alias("Inflow"),
        pl.col("dTsurf_dt"),
    )
    return sbl


def load_sbl_ABLs(runid=2, ignore_warnings=True, strict_runid=True, load_precursors=False, precursor_runid=4):
    """Load SBL and CNBL data (rotation phase)"""
    sbl = load_data(
        SBL_PATH,
        runid=runid,
        load_precursors=load_precursors,
        precursor_runid=precursor_runid,
        ignore_warnings=ignore_warnings,
        strict_runid=strict_runid,
        quiet=True,
        filter_budgets_exist=False,
        filter_fields_exist=True,
    )
    sbl = sbl.with_columns(
        pl.lit("SBL").alias("Inflow"),
        pl.col("dTsurf_dt"),
        pl.Series("z0", [sim.input_nml['sgs_model']['z0'] for sim in sbl["sim"]]),
    )
    return sbl


def load_tnbl_data(basedir=None, ids=None, load_r0=False):
    """Load TNBL data from Coriolis paper"""
    if basedir is None:
        # may need to be updated on different system
        basedir = Path(r"/anvil/scratch/x-heck/coriolis/save_ekman")

    fname = 'ro_f_{:02d}'
    if ids is None:
        ids = [0, 1, 3, 5, 7, 9] if load_r0 else range(1, 10, 2)

    data = {'sim': [], 'precursor': []}

    for k in ids:
        sim = pio.BudgetIO(basedir / fname.format(k), padeops=True, runid=5, normalize_origin='turb')
        precursor = pio.BudgetIO(basedir / fname.format(k), padeops=True, runid=4, normalize_origin=sim.origin)

        data['sim'].append(sim)
        data['precursor'].append(precursor)

    data['name'] = [sim.filename for sim in data['sim']]
    data['Ro_f'] = [np.round(sim.Ro_f) for sim in data['sim']]
    data['Ro'] = [sim.Ro for sim in data['sim']]

    return pl.DataFrame(data=data).with_columns(pl.lit("TNBL").alias("Inflow"))
