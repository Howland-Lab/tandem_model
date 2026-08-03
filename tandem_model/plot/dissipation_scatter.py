"""
Hexbin plot (log-scaled counts) of pointwise -delta(TKE_dissipation) vs.
delta(k)^{3/2} for the uniform-inflow veer_00 case.

Reproduces the "Pointwise dissipation vs tke" figure of
`kl_model/notebooks_s3/00_revisit_dissipation.ipynb`. Axes are already
nondimensionalized by the LES's velocity scale U_inf (= 1) and length scale D
(see `BudgetIO.get_uhub`); only the axis labels reflect this.

Kirby Heck
2026
"""

import matplotlib.pyplot as plt
import polars as pl

from tandem_model import figuresettings  # noqa: F401
from tandem_model.constants import FIGPATH, SCRATCH_ROOT
from tandem_model.generate.dissipation import dissipation_pointwise

FIGPATH.mkdir(exist_ok=True, parents=True)

dirname = SCRATCH_ROOT / "nowall" / "veer_00"

# L_EPS = 1.1  # reference slope D / l_eps = 1


def main(regenerate=False):
    df = dissipation_pointwise(dirname, regenerate=regenerate)
    df = df.filter((pl.col("dk") > 0) & (pl.col("diss") > 0))
    dk32 = df["dk"] ** 1.5

    fig, ax = plt.subplots(figsize=(3, 2.25))
    hb = ax.hexbin(dk32, df["diss"], gridsize=50, bins="log", cmap="Blues")
    cb = fig.colorbar(hb, ax=ax, label="$N$")
    cb.minorticks_off()

    xmax = dk32.max()
    L_EPS = (df["diss"] * dk32).sum() / (dk32**2).sum()  # slope of best-fit line through origin
    ax.plot([0, xmax], [0, xmax * L_EPS], color="k", ls="--", label=f"Fit $D/\\ell_\\varepsilon = {L_EPS:.1f}$")
    ax.set_xlabel(r"$\Delta k^{3/2} / u_\infty^3$")
    ax.set_ylabel(r"$\Delta \varepsilon \, D / u_\infty^3$")
    ax.legend(fontsize=8)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
