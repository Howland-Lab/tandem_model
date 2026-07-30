"""
Generate two-turbine data usign MITWindfarm. 

This will take a bit of debugging, probably. 

Kirby Heck
2024 Nov 12
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from itertools import product
from tqdm import tqdm
from pathlib import Path

from mitwindfarm import (
    GridLayout,
    Windfarm,
    VariableKwGaussianWakeModel,
    VortexWakeModel,
    CurledWindfarm,
    Uniform,
    Area,
)
import mitwindfarm as mitwf
from UnifiedMomentumModel.Momentum import LimitedHeck
from mitwindfarm.Rotor import AD, UnifiedAD, UnifiedAD_veer
from mitwindfarm import tandem as mitwf_extras
from tandem_model import caching as cache
from tandem_model.constants import DATA_PATH
TIAMB = 0.056


class CosineAD(mitwf.Rotor.Rotor):
    """
    Simple Cosine model rotor. Uses Shapiro lifting line model for v4.

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self, rotor_grid: mitwf.RotorGrid = None, Pp: float = 3.0):
        """
        Initialize the UnifiedAD rotor model with the given axial induction factor.

        Parameters:
        - beta (float): Axial induction factor (default is 0.1403).
        """
        if rotor_grid is None:
            self.rotor_grid = mitwf.Area()
        else:
            self.rotor_grid = rotor_grid
        self._model = LimitedHeck()
        self.Pp = Pp

    def __call__(
        self, x: float, y: float, z: float, windfield: mitwf.Windfield, Ctprime, yaw
    ) -> mitwf.RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        sol: mitwf.MomentumSolution = self._model(Ctprime, yaw)
        sol_1d = self._model(Ctprime, yaw * 0)  # 1d solution

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)

        REWS = self.rotor_grid.average(Us)
        RETI = np.sqrt(self.rotor_grid.average(TIs**2))

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return mitwf.RotorSolution(
            yaw,
            sol_1d.Cp * np.cos(sol.yaw) ** self.Pp * REWS**3,
            sol.Ct * REWS**2,
            sol.Ctprime,
            sol.an * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            TI=RETI,
            extra=sol,
        )


rotor_lookup = dict(unified=UnifiedAD_veer(rotor_grid=Area()), jfm=AD(), cosine=CosineAD(rotor_grid=Area(), Pp=1.7))


def get_windfarm(rotormodel, wakemodel, **kwargs):
    if wakemodel == "curl":
        _solver_kwargs = dict(
            u_model="default",
            # k_model="k-l",
            k_model="kl-interp",
            k_kwargs=dict(C_nu=0.32, C_k1=1, C_w=3, l_eps=1),
            integrator="scipy_rk45",
            auto_expand=True,
            bottom_wall_z=-1,
            zero_at_boundaries=True,
            use_r4=None,  # apply momentum conservation
            smooth_fact=0.1,
        )
        if "solver_kwargs" in kwargs.keys():
            solver_kwargs = kwargs.pop("solver_kwargs")
            _solver_kwargs.update(solver_kwargs)

        _kwargs = dict(
            rotor_model=rotor_lookup[rotormodel],
            base_windfield=Uniform(TIamb=TIAMB),
            TIamb=TIAMB,
            solver_kwargs=_solver_kwargs,
        )
        _kwargs.update(kwargs)
        return CurledWindfarm(**_kwargs)

    elif wakemodel == "gauss":
        _kwargs = dict(a=0.636, b=0, c=0)
        _kwargs.update(kwargs)
        wake_model = VariableKwGaussianWakeModel(**_kwargs)

    elif wakemodel == "vortex":
        _kwargs = dict(kw=0.05, z_wall=-1)
        _kwargs.update(kwargs)
        wake_model = VortexWakeModel(**_kwargs)

    else:
        raise ValueError("Unknown wakemodel:", wakemodel)

    return Windfarm(
        rotor_model=rotor_lookup[rotormodel],
        wake_model=wake_model,
        TIamb=TIAMB,
    )


def run(rotormodel, wakemodel, **kwargs):
    print("Generating sweep for rotor:", rotormodel, ", wake model: ", wakemodel)
    layout = GridLayout(6.0133, 0.0, 2, 1).rotate(3.8)

    # Get windfarm model... 
    wf = get_windfarm(rotormodel, wakemodel, **kwargs)

    # now we need to sweep over all the set points
    sp2 = (2, 0)  # waked turbine is always at Betz
    solutions = []
    yaws = np.deg2rad(np.linspace(0, 45, 10))
    ctps = np.arange(0.4, 4.5, 0.4)
    for sp1 in tqdm(product(ctps, yaws)): 
        sol = wf(layout, [sp1, sp2])
        solutions.append(sol)

    # now make this into a dataframe
    df = pl.DataFrame({
        'ctp': [sol.rotors[0].Ctprime for sol in solutions], 
        'yaw': np.rad2deg([sol.rotors[0].yaw for sol in solutions]), 
        'Cp_1': [sol.rotors[0].Cp for sol in solutions], 
        'Cp_2': [sol.rotors[1].Cp for sol in solutions], 
        'Cp': [sol.Cp for sol in solutions], 
    })
    return df


def run_with_rotation(rotormodel, wakemodel, df=None, **kwargs):
    """Same as `run` but adds """
    print("Generating sweep with angle adjustment for rotor:", rotormodel, ", wake model: ", wakemodel)
    layout = GridLayout(6.0133, 0.0, 2, 1).rotate(3.81)
    df = les() if df is None else df

    # Get windfarm model... 
    wf = get_windfarm(rotormodel, wakemodel, **kwargs)

    # now we need to sweep over all the set points
    sp2 = (2, 0)  # waked turbine is always at Betz
    solutions = []
    for (yaw, ctp, rotate) in tqdm(df.select("yaw", "ctp", "phi_hub").iter_rows()):
        _layout = layout.rotate(-np.degrees(rotate))
        sp1 = (ctp, np.deg2rad(yaw))
        sol = wf(_layout, [sp1, sp2])
        solutions.append(sol)

    # now make this into a dataframe
    df = pl.DataFrame({
        'ctp': [sol.rotors[0].Ctprime for sol in solutions], 
        'yaw': np.rad2deg([sol.rotors[0].yaw for sol in solutions]), 
        'Cp_1': [sol.rotors[0].Cp for sol in solutions], 
        'Cp_2': [sol.rotors[1].Cp for sol in solutions], 
        'Cp': [sol.Cp for sol in solutions], 
    })
    return df


@cache.cache_polars(Path(DATA_PATH) / "twoturbine_mitwindfarm_unified_gauss.csv")
def unified_gauss(regenerate=False, run_func=run, **kwargs):
    return run_func("unified", "gauss", **kwargs)


@cache.cache_polars(Path(DATA_PATH) / "twoturbine_mitwindfarm_unified_vortex.csv")
def unified_vortex(regenerate=False, run_func=run, **kwargs):
    return run_func("unified", "vortex", **kwargs)


@cache.cache_polars(Path(DATA_PATH) / "twoturbine_mitwindfarm_unified_curl.csv")
def unified_curl(regenerate=False, run_func=run, **kwargs):
    return run_func("unified", "curl", **kwargs)

@cache.cache_polars(Path(DATA_PATH) / "twoturbine_mitwindfarm_cosine_gauss.csv")
def cosine_gauss(regenerate=False, run_func=run, **kwargs):
    return run_func("cosine", "gauss", **kwargs)

def les():
    return pl.read_csv(DATA_PATH / "twoturbine_alldata_LES.csv")


if __name__ == "__main__":
    pass
    # run("unified", "gauss")
    # run("unified", "vortex")
    # run("unified", "curl")