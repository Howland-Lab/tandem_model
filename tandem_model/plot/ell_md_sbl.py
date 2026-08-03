"""
Plot mixing length l_md vs x for the single-turbine calibration case.

Kirby Heck
2026
"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings  # noqa: F401
from tandem_model.constants import FIGPATH, SCRATCH_ROOT
from tandem_model.generate.mixing_length import ell_md, ell_md_list

FIGPATH.mkdir(exist_ok=True, parents=True)
C_NU = 0.35
C_w = 3

dirnames = [SCRATCH_ROOT / "sbl" / f"G_01_z0_02_dTsurf_dt_{i:02d}" for i in range(0, 6)]
# + [
#     # r"/scratch/08445/tg877441/veer_WES25/uniform/run2",
#     SCRATCH_ROOT / "synthetic_veer_ti" / "veer_00_TI_02_k_bandpass_left_00",
# ]


def main(regenerate=False):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    df = ell_md_list(
        dirnames, regenerate=regenerate, zlim=[-1, 1], xlim=[0, 15]
    ).with_columns(
        (pl.col("l") / C_NU).alias("l_scaled"),
    )
    palette = sns.color_palette("crest", n_colors=df["Cr"].n_unique())
    ax.plot([], [], color="w", label="$C_r$ (K~hr$^{-1}$)")  # dummy for legend title
    sns.lineplot(
        df, x="l_md", y="l_scaled", hue="Cr", ax=ax, sort=False, ls="none", marker="o", ms=4, markevery=5, alpha=0.8, palette=palette
    )

    # plot l_w/D lines
    ax.plot([], [], color="w", label="$\\ell_w/D$")  # dummy for legend title
    for color, (_, group) in zip(palette, df.group_by("Cr", maintain_order=True)):
        yval = group["l_w"][0] * C_w
        ax.plot(
            [0.33, 0.5],
            [yval, yval],
            color=color,
            ls="--",
            lw=1.5,
            alpha=0.8,
            label=f"${yval:.2f}$",
        )

    ax.plot([0, 0.5], [0, 0.5], color="k", ls="--", lw=0.6)
    ax.set_xlabel(r"$\ell_\mathrm{md} / D$")
    ax.set_ylabel(r"$\ell/(C_\nu D) = \nu_T / (C_\nu D \sqrt{k})$")
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 0.5])
    plt.subplots_adjust(right=0.75)
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), fontsize=8)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(True)
