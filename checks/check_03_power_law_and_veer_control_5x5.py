"""
Check (3): same fit as check_02, but for the wake-steering CNBL data
`control_5x5` (the yaw-controlled case, `yawcontrol` -- the ambient inflow
is shared with the `nocontrol` baseline, so both are reported for a
consistency check). Also reports the wind veer across the rotor extent, in
degrees.

Power-law exponent: same convention as check_02 / `tandem_model.inflow`
(fit within the rotor-swept region z_rel in [-0.5, 0.5]; z_abs = z_rel +
z_h, z_h = D = 1; fit log(U(z)/U(z_h)) = alpha*log(z_abs/z_h) by linear
regression, reference height z_h).

Veer across the rotor: total change in wind direction from the bottom tip
(z_rel=-0.5) to the top tip (z_rel=+0.5) of the rotor, in degrees
(veer_deg = wdir(top) - wdir(bottom), unwrapped) -- distinct from
`tandem_model.inflow.compute_inflow_quantities`'s linear veer slope, since
the user asked for the total degrees swept across the rotor, not a rate.

Run with the repo's project venv, e.g.:
    /work2/08445/tg877441/stampede3/claude_projects/tandem/.venv/bin/python \\
        checks/check_03_power_law_and_veer_control_5x5.py

Kirby Heck
2026
"""

import numpy as np
import padeopsIO as pio

from tandem_model import constants, utils
from tandem_model.generate.control_5x5_cp import CASES  # ["nocontrol", "yawcontrol"]

Z_HUB = 1.0  # absolute z-coordinate of hub height (D = 1)
R_D = 0.5  # rotor radius / D, sets the fit/veer region z_rel in [-R_D, +R_D]


def fit_power_law_exponent(sim: pio.BudgetIO):
    """
    Same fit as check_02_power_law_superposition_4x1.fit_power_law_exponent:
    log(U/Uhub) = alpha*log(z_abs/z_h) over the rotor-swept region.
    """
    inflow = utils.inflow_LES(sim, normalize=False)
    z_rel = np.asarray(inflow.z)
    U_z = np.asarray(inflow.U_z)

    Uhub = np.interp(0.0, z_rel, U_z)

    mask = (z_rel >= -R_D) & (z_rel <= R_D)
    z_abs = z_rel[mask] + Z_HUB
    U = U_z[mask]

    alpha, _ = np.polyfit(np.log(z_abs / Z_HUB), np.log(U / Uhub), 1)
    return alpha, Uhub


def veer_across_rotor(sim: pio.BudgetIO):
    """Returns the wind-direction change (degrees) from z_rel=-R_D to z_rel=+R_D."""
    inflow = utils.inflow_LES(sim, normalize=False)
    z_rel = np.asarray(inflow.z)
    wdir = np.unwrap(np.asarray(inflow.wdir_z))  # radians, unwrap vs z to avoid +-pi jumps

    wdir_bottom = np.interp(-R_D, z_rel, wdir)
    wdir_top = np.interp(R_D, z_rel, wdir)
    return np.degrees(wdir_top - wdir_bottom)


def main(cases=CASES, runid=5):
    print(f"{'case':<14}{'alpha':>10}{'Uhub':>10}{'veer (deg)':>14}")
    print("-" * 48)
    for case in cases:
        dirname = constants.SCRATCH_ROOT / "control_5x5" / case
        sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
        alpha, Uhub = fit_power_law_exponent(sim)
        veer_deg = veer_across_rotor(sim)
        print(f"{case:<14}{alpha:10.4f}{Uhub:10.4f}{veer_deg:14.3f}")

    print()


if __name__ == "__main__":
    main()
