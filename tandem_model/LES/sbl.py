"""
TNBL (Ekman layer) re-run of Coriolis simulations, with (without)
shifted fringe targets to mitigate streak development

Kirby Heck
2025 September 02
"""

import numpy as np
import polars as pl
from pathlib import Path
import jinja2
import socket

from tandem_model import input_writer as writer

if "anvil" in socket.gethostname():
    SBL_PATH = Path(r"/anvil/scratch/x-heck/veer_WES25/sbl_0")  # Anvil sweep with low Cr
elif "stampede3" in socket.gethostname():
    SBL_PATH = Path(r"/scratch/08445/tg877441/veer_WES25/sbl_0")  # Stampede3 data
else:
    SBL_PATH = None


D = 240  # meters
dx, dy, dz = 20 / D, 12.5 / D, 6.25 / D
h = np.sqrt(dx**2 + dy**2 + dz**2)
nx, ny, nz = 384, 256, 384
t_rotation = 200  # non-dim
ft_spinup = 15  # 3 * np.pi      # end time for spin-up
ft_concurrent = 5 * np.pi  # end time for concurrent, UNUSED

PARAMS = dict(
    nx=nx,
    ny=ny,
    nz=nz,
    Pr=0.5,
    lat=45,
    z0=1e-1 / D,
    Lx=nx * dx,
    Ly=ny * dy,
    Lz=nz * dz,
    xturb=5,
    yturb=ny * dy / 2,
    zturb=150/D,
    filterwidth=2.5*h,
    usecorrection=True,
    ctp=4/3,
    yaw=0,
    Tref=300,
    Tsurf0=300,
    dTsurf_dt=0,
    time_budget_start=0,
    dTdz=0, #0.003 * D,
    usecontrol=False,
    z_ref=24,
    userestartfile=False,
    restart_rid=0,
    restart_tid=0,
    t_datadump=2000,
    t_restartdump=2000,
    budget_type=3,
    admtype=5,
    # NEEDED: dirname, do_budgets, tstop, time_budget_start, Ro, Fr, runid
)

def sweep(
    parentdir,
    write_func,
    n_hrs=4,
    node_min=1,
    extra=None,
    write_key=False,
    ignore_errors=False,
    **kwargs,
):
    """
    Writes sweep of SBL cases
    """

    # write combinations to dataframe
    cases = writer.iter_to_df(**kwargs).with_columns(
        (pl.col("G") / (D * 7.29e-5)).alias("Ro"),
        (pl.col("G") / np.sqrt(D * 9.81)).alias("Fr"),
    )

    extra = extra or dict()

    # iterate through cases and write each one
    for row in cases.iter_rows(named=True):
        _extra = extra.copy()
        _extra.update(row)
        _dir = parentdir / row["name"]

        inputs = PARAMS.copy()
        inputs.update(dirname=_dir, restart_dir=_dir, **_extra)

        try: 
            write_func(inputs, n_hrs=n_hrs, node_min=node_min)
        except Exception as e:
            if ignore_errors:
                print(f"Error writing case {row['name']}: {e}")
                continue
            else: 
                raise e

    if write_key:
        cases.write_csv(parentdir / "key.csv", float_precision=4)
    print("Done")


def sbl_fix_args(inputs, t_stop_sec):
    # inputs["tstop"] = ft_stop * inputs["Ro"] / 2 * np.sin(np.radians(inputs["lat"]))
    inputs["tstop"] = t_stop_sec * inputs["Ro"] * 7.29e-5  # convert to non-dim.
    if inputs["dTsurf_dt"] < -0.001:
        # reduce domain height
        inputs["nz"] = 256
        inputs["Lz"] = inputs["nz"] * dz
    else:
        # this is the neutral case; we need the longer spin-up. Use 15/f_c
        inputs["tstop"] -= (3600 * 11) * inputs["Ro"] * 7.29e-5  # reduce tstop by 11 hours
        inputs["tstop"] += ft_spinup * inputs["Ro"] / (2 * np.sin(np.radians(inputs["lat"])))
        inputs["dTdz"] = 0.003 * D  # add background stratification for neutral case
    return  # updates dictionary and returns None


def write_stable(inputs, n_hrs=24, node_min=1, quiet=False):
    # adding custom kwargs
    sbl_fix_args(inputs, t_stop_sec=11*3600)

    if inputs["dTsurf_dt"] == 0:
        writer.write_neutral(inputs, n_hrs=n_hrs, node_min=node_min, quiet=quiet)
        return  # write neutral PBL case instead

    # make output directory
    OUTPUT = writer.safe_mkdir(inputs, quiet=quiet, dst=None)

    # load spinup template and write template:
    with open(writer.TEMPLATE_SBL, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / "input_stable.dat", "w") as f:
        f.write(out)

    # make submit.sh file
    with open(OUTPUT / "submit.sh", "w") as f:
        f.write(
            writer.sbatch_write_file(
                inputs,
                "input_stable.dat",
                problem_name="stable_pbl",
                n_hrs=n_hrs,
                node_min=node_min,
            )
        )

    if not quiet:
        print("\tDone writing spinups files")


def wrap_stable_rotate(inputs, n_hrs=6, node_min=1, quiet=False):
    """Wrap SBL rotation function"""
    sbl_fix_args(inputs, t_stop_sec=11*3600) #ft_stop=3 * np.pi)
    inputs["tstop"] += t_rotation  # rotation phase run time

    if inputs["dTsurf_dt"] == 0:
        writer.write_rotate(
            inputs,
            n_hrs=n_hrs,
            node_min=node_min,
            quiet=quiet,
            problem_name="neutral_pbl",
        )
    else:
        writer.write_rotate(
            inputs,
            n_hrs=n_hrs,
            node_min=node_min,
            quiet=quiet,
            problem_name="stable_pbl",
        )


def wrap_stable_upsample(
    inputs, quiet=False, node_min=1, n_hrs=None
):
    """Wrap SBL upsampling function"""
    sbl_fix_args(inputs, t_stop_sec=0)
    writer.write_upsample(
        inputs, quiet=quiet, inputfile_name="input_upsample.dat"
    )
# Hans, Paul, Skylar, Anne, Abi, Jevan, Kirby

def wrap_stable_concurrent(
        inputs, quiet=False, node_min=1, n_hrs=None, ft_stop=ft_concurrent
): 
    """Wrap SBL concurrent"""
    sbl_fix_args(inputs, t_stop_sec=20*3600)
    # start budgets 5 flow-thru times after rotation phase ends
    # inputs['time_budget_start'] = 11*3600 / inputs["Ro"] * 2 * np.sin(np.radians(inputs["lat"])) + t_rotation + inputs['Lx'] * 5
    inputs['time_budget_start'] = 11*3600 * inputs["Ro"] * 7.29e-5 + t_rotation + inputs['Lx'] * 5
    inputs['turbine_dir'] = inputs['dirname'] / 'turb'

    if inputs["dTsurf_dt"] == 0:
        writer.write_concurrent(
            inputs,
            n_hrs=n_hrs,
            node_min=node_min,
            quiet=quiet,
            problem_name="neutral_pbl_concurrent",
        )
    else:
        writer.write_concurrent(
            inputs,
            n_hrs=n_hrs,
            node_min=node_min,
            quiet=quiet,
            problem_name="stable_pbl_concurrent",
        )


# !!!!!!!!!!!!!!!!!!!!!!!!!! RUN PRODUCTION SWEEP  !!!!!!!!!!!!!!!!!!!!!!!!!!!! #


def get_sweep_kws():
    return dict(
        G=[4, 12],
        z0=[1e-3 / D, 1e-2 / D, 1e-1 / D],
        # dTsurf_dt=[0, -0.25, -0.5, -0.75, -1],  # in K/hr
        dTsurf_dt=[0, -0.1, -0.2, -0.3, -0.4, -0.5],  # in K/hr
    )


def sweep_spinup(parentdir=SBL_PATH):
    sweep_kws = get_sweep_kws()
    extra = dict(
        nx=PARAMS["nx"] // 2,
        ny=PARAMS["ny"] // 2,
        do_budgets=False,
        runid=1,
    )
    sweep(
        parentdir,
        write_func=write_stable,
        n_hrs=24,
        extra=extra,
        **sweep_kws,
        write_key=True,
    )


def sweep_rotation(parentdir=SBL_PATH):
    sweep_kws = get_sweep_kws()
    extra = dict(
        do_budgets=False,
        restart_rid=1,
        runid=2,
        userestartfile=True,
        usecontrol=True,
        t_datadump=100,
    )

    # write upsample input files
    sweep(
        parentdir,
        write_func=wrap_stable_upsample,
        extra=extra,
        ignore_errors=True,
        **sweep_kws,
    )
    # update restart file RID and write rotation phase
    extra.update(restart_rid=2)
    sweep(
        parentdir,
        write_func=wrap_stable_rotate,
        n_hrs=6,
        extra=extra,
        node_min=2,
        ignore_errors=True,
        **sweep_kws,
    )


def sweep_concurrent(parentdir=SBL_PATH):
    sweep_kws = get_sweep_kws()
    extra = dict(
        do_budgets=True,
        restart_rid=2,
        userestartfile=True,
        t_datadump=1000,
        fringe_xst=0.75,
        fringe_xen=0.95,
        lambdafact=0.5,
    )

    sweep(
        parentdir,
        write_func=wrap_stable_concurrent,
        n_hrs=36,
        extra=extra,
        node_min=12,
        ignore_errors=True,
        **sweep_kws,
    )

if __name__ == "__main__":
    sweep_spinup(SBL_PATH)
    # sweep_rotation(SBL_0)
    # sweep_concurrent(SBL_0)
