"""
Check (2): what is the power-law shear exponent of the inflow in the CNBL
superposition 4x1 simulations (`generate.superposition_power.CASES`)?

Follows the same fit convention as `tandem_model.inflow.compute_inflow_quantities`
(fit within the rotor-swept region z_rel in [-0.5, 0.5], relative to hub
height): converts to absolute z-coordinates (z_abs = z_rel + z_h, with
z_h = D = 1) and fits

    log(U(z) / U(z_h)) = alpha * log(z / z_h)

by linear regression (np.polyfit) over that region, using z_h = 1 as the
power-law reference height (as instructed), rather than the hub-height
speed's own best-fit intercept.

Run with the repo's project venv, e.g.:
    /work2/08445/tg877441/stampede3/claude_projects/tandem/.venv/bin/python \\
        checks/check_02_power_law_superposition_4x1.py

Kirby Heck
2026
"""

import numpy as np
import padeopsIO as pio

from tandem_model import constants, utils
from tandem_model.generate.superposition_power import CASES

Z_HUB = 1.0  # absolute z-coordinate of hub height (D = 1)
R_D = 0.5  # rotor radius / D, sets the fit region z_rel in [-R_D, +R_D]


def fit_power_law_exponent(sim: pio.BudgetIO):
    """
    Returns (alpha, Uhub) from a log(U/Uhub) = alpha*log(z_abs/z_h) fit over
    the rotor-swept region, using the cached (or freshly computed) xy/time-
    averaged inflow profile.
    """
    inflow = utils.inflow_LES(sim, normalize=False)
    z_rel = np.asarray(inflow.z)
    U_z = np.asarray(inflow.U_z)

    Uhub = np.interp(0.0, z_rel, U_z)  # z_rel=0 is hub height (turbine-origin frame)

    mask = (z_rel >= -R_D) & (z_rel <= R_D)
    z_abs = z_rel[mask] + Z_HUB
    U = U_z[mask]

    alpha, _ = np.polyfit(np.log(z_abs / Z_HUB), np.log(U / Uhub), 1)
    return alpha, Uhub


def main(cases=CASES, runid=5):
    print(f"{'case':<20}{'alpha':>10}{'Uhub':>12}")
    print("-" * 42)
    alphas = []
    for case in cases:
        dirname = constants.SCRATCH_ROOT / "superposition" / case
        sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
        alpha, Uhub = fit_power_law_exponent(sim)
        alphas.append(alpha)
        print(f"{case:<20}{alpha:10.4f}{Uhub:12.4f}")

    print("-" * 42)
    print(f"{'mean over cases':<20}{np.mean(alphas):10.4f}")

if __name__ == "__main__":
    main()
