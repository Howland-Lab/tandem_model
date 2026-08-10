"""
Check (1): for the `streamtube_sbl` figure (minimum wake velocity comparisons
between LES and various models), is the full ambient TKE profile k^B(z)
actually needed by the TANDEM closure, or would a height-independent
k^B(z) = k^B(z_h) (i.e. just the hub-height value, held constant across z)
give essentially the same wake deficit?

Method: for each SBL LES case (`generate.streamtube_sbl.SBL_DIRNAMES`), solve
the TANDEM closure twice with `utils.solve_curl_wakefield_LES`'s machinery,
swapping only the ambient windfield:
  - "full":     the LES's actual k^B(z) profile (as normally used)
  - "hub-only": k^B(z) replaced by its hub-height value k^B(z_h), constant
                across z (U(z), wdir(z) untouched -- only the ambient TKE
                closure input is flattened)
and compares the resulting centerline deficit du_centerline(x) (y=0, z=0)
between the two. Reports the max absolute error and MAE, per case and
pooled over all cases/x-stations.

Kirby Heck
2026
"""

import numpy as np
import padeopsIO as pio
import mitwindfarm as mitwf
from mitwindfarm.utils.integrate import IntegrationException

from tandem_model import utils
from tandem_model.generate.streamtube_sbl import SBL_DIRNAMES
from tandem_model.models import K_KWARGS, CURLED_MODEL_KWARGS, CURLED_YLIM, CURLED_ZLIM

K_MODEL = "tandem"


def _flatten_to_hub_height_kb(inflow: mitwf.ArbitraryZWindfield) -> mitwf.ArbitraryZWindfield:
    """
    Returns a copy of `inflow` with its ambient TKE profile replaced by a
    height-independent value equal to the TKE at z=0 (hub height). Since
    ArbitraryZWindfield stores ambient turbulence as TI_z = sqrt(2k/3)/U(z),
    not TKE directly, the hub-height TKE is first recovered from
    (TI(z), U(z)) at z=0, then reapplied through the (unchanged) shear
    profile U(z) at every z -- i.e. only k^B(z) is flattened, not the shear.
    """
    z, U_z, TI_z = inflow.z, inflow.U_z, inflow.TIamb_z
    tke_z = 1.5 * (TI_z * U_z) ** 2
    tke_hub = np.interp(0.0, z, tke_z)
    TI_z_hub_only = np.sqrt(2.0 / 3.0 * tke_hub) / U_z
    return mitwf.ArbitraryZWindfield(
        z=z, U_z=U_z, wdir_z=inflow.wdir_z, TIamb_z=TI_z_hub_only,
    )


def _solve_tandem(sim: pio.BudgetIO, inflow, meta, xmax):
    """Mirrors `utils.solve_curl_wakefield_LES`'s non-LES-IC branch, but with a custom inflow."""
    k_kwargs = dict(L_obu=meta["L_obu"], **K_KWARGS[K_MODEL])
    model_kwargs = {**CURLED_MODEL_KWARGS, "bottom_wall_z":meta["zwall"]}
    solver_kwargs = dict(k_model=K_MODEL, k_kwargs=k_kwargs, **model_kwargs)

    rotor_model = mitwf.UnifiedAD_veer(rotor_grid=mitwf.Area())
    layout, setpoints = utils.layout_LES(sim)
    wf = mitwf.CurledWindfarm(
        rotor_model=rotor_model, base_windfield=inflow, solver_kwargs=solver_kwargs
    )
    sol = wf(layout, setpoints)
    try:
        sol.windfield.march_to(x=xmax, y=0, z=0)
    except IntegrationException as e:
        print(f"  Warning: final march_to({xmax}) failed with error: {e}")

    return utils.ModelWakeField(sol, ylim=CURLED_YLIM, zlim=CURLED_ZLIM)


def main(dirnames=SBL_DIRNAMES, runid=5):
    all_abs_err = []
    print(f"{'case':<28}{'max |err|':>12}{'MAE':>12}{'N pts':>8}")
    print("-" * 60)

    for dirname in dirnames:
        sim = pio.BudgetIO(dirname, padeops=True, runid=runid, normalize_origin="turb")
        meta = utils.case_meta_LES(sim)
        inflow_full = utils.inflow_LES(sim, normalize=True)
        inflow_hub = _flatten_to_hub_height_kb(inflow_full)

        wake_full = _solve_tandem(sim, inflow_full, meta, meta["xmax"])
        wake_hub = _solve_tandem(sim, inflow_hub, meta, meta["xmax"])

        # shared native x-grid (both solves march the same rotor-diameter grid)
        x_full = wake_full.du["x"].to_numpy()
        x_hub = wake_hub.du["x"].to_numpy()
        xax = x_full if len(x_full) <= len(x_hub) else x_hub

        du_full = wake_full.du.interp(y=0, z=0, x=xax).to_numpy()
        du_hub = wake_hub.du.interp(y=0, z=0, x=xax).to_numpy()

        err = du_hub - du_full
        mask = np.isfinite(err)
        err = err[mask]

        max_err = np.max(np.abs(err))
        mae = np.mean(np.abs(err))
        all_abs_err.append(err)

        print(f"{dirname.name:<28}{max_err:12.5f}{mae:12.5f}{len(err):8d}")

    pooled = np.concatenate(all_abs_err)
    print("-" * 60)
    print(f"{'POOLED (all cases)':<28}{np.max(np.abs(pooled)):12.5f}{np.mean(np.abs(pooled)):12.5f}{len(pooled):8d}")
    print()
    print(
        "Interpretation: max |err| and MAE above are in du_centerline/U_hub "
        "units, comparing the TANDEM closure driven by the full k^B(z) "
        "profile vs. a height-independent k^B(z_h)."
    )


if __name__ == "__main__":
    main()
