"""
Constants for the TANDEM model project (LES input generation, paths).

Kirby Heck
2024-2026
"""

from pathlib import Path
import json
import numpy as np

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / "data"
FIGPATH = BASE / "figs"
JSON_PATH = BASE / "templates" / "laminar_defaults.json"


params = dict()
D = 240  # m
zhub = 150 / D  # non-dimensional
params["D"] = D  # m

# DEFAULT SIMULATION PARAMETERS
params["nx"], params["ny"], params["nz"] = 256, 256, 256
params["Lx"], params["Ly"], params["Lz"] = 25.6, 12.8, 12.8
params["fringe_xst"] = 0.75
params["fringe_xen"] = 0.95
params["fringe_delta_xst"] = 0.15
params["fringe_delta_xen"] = 0.05
params["lambdafact"] = 0.95
params["tstop"] = params["Lx"] * 8
params["lat"] = 43.3  # degrees
params["Ro"] = 1e10
params["sgsmodelid"] = 1
params["useconstantG"] = False
params["csgs"] = 0.8
params["runid"] = 1
params["t_datadump"] = 100
# STRATIFICATION PARAMETERS
params["isStratified"] = False
params["botBC_Temp"] = 0
params["Fr"] = 1e10
# BUDGETS AND RESTARTS
params["do_budgets"] = True
params["restart_budgets"] = False
params["do_deficit_budgets"] = False
params["restart_deficit_budgets"] = False
params["time_budget_start"] = params["Lx"] * 4
params["use_restartfile"] = False
params["restart_tid"] = 0
params["restart_budget_counter"] = 0
params["budget_type"] = 3
# HIT PARAMS:
params["userestart_hit"] = True
params["dirname_hit"] = r"/scratch/08445/tg877441/hit_ad/hit_spinup/cube64"
params["restart_hit"] = 1836  # for HIT simulations
params["restart_rid_hit"] = 1
params["freeze_hit"] = False
params["nx_hit"] = params["ny"]  # cube by default, not params['nx']
params["ny_hit"] = params["ny"]
params["nz_hit"] = params["nz"]
params["Lx_hit"] = np.pi * 2
params["Ly_hit"] = np.pi * 2
params["Lz_hit"] = np.pi * 2
params["k_bandpass_left"] = 4.0
params["k_bandpass_right"] = 128.0
params["KIinv_TI"] = 2.0  # inv integral gain for TI controller (higher is slower)
# ADM properties:
params["xturb"] = 5.0
params["yturb"] = params["Ly"] / 2
params["zturb"] = params["Lz"] / 2
params["filterwidth"] = 0.3062
params["usecorrection"] = True
params["admtype"] = 5
params["ctp"] = 4 / 3.0
params["yaw"] = 0.0
params["usewindturbines"] = True
params["num_turbines"] = 1
# Inflow properties
params["inflowthick"] = 5.0
params["inflowprofileamplit"] = 0.0
params["inflowprofiletype"] = 0
params["uinflow"], params["vinflow"] = 1.0, 0
params["yaw_inflow"] = 0.0
params["zmid"] = -1
params["TI"] = -1
params["TI_fact"] = -1
params["TI_xloc"] = 0
params["time_stop_TIcont"] = -1
params["fname_inflow"] = "null"


def check_default_exists(path=JSON_PATH, remake=False):
    """Checks to make sure the defaults file exists"""
    file = Path(path)
    if not file.is_file() or remake:
        write_default_inputs(path)


def write_default_inputs(path=JSON_PATH):
    """Writes a .json file of default input parameters"""
    inputs = params.copy()

    with open(path, "w") as dst:
        json.dump(inputs, dst, indent=2)
    print("Written input file default to", path)


def get_inputs(dirname, turbine_dir=None, remake_defaults=False, **kwargs):
    """
    Returns a dictionary of LES inputs for writing input files.

    Parameters
    ----------
    dirname : path-like
        Directory
    turbine_dir : path-like
        Turbine directory
    """
    if turbine_dir is None:
        turbine_dir = Path(dirname) / "turb"

    check_default_exists(JSON_PATH, remake=remake_defaults)
    with open(JSON_PATH, "r") as src:
        inputs = json.load(src)

    inputs.update({**kwargs, **dict(dirname=dirname, turbine_dir=turbine_dir)})

    return inputs


def get_h(inputs):
    dx = inputs["Lx"] / inputs["nx"]
    dy = inputs["Ly"] / inputs["ny"]
    dz = inputs["Lz"] / inputs["nz"]
    return np.sqrt(dx**2 + dy**2 + dz**2)


if __name__ == "__main__":
    """Test retrieving input files. Writes a defaults file if none exists."""
    print(get_inputs(BASE / "tmp", remake_defaults=True))
