#!/usr/bin/python

# Hannah Johlas, 2019
# Revised Kirby Heck, 2024

import inspect
import matplotlib
import seaborn as sns
from pathlib import Path
from .constants import BASE

# Plot formatting
# matplotlib.rcParams['figure.figsize']     = [16.0/2.54, 6.0/2.54]               # figure dimensions, in inches
matplotlib.rcParams['savefig.format']     = 'png'                               # image file type (pdf, png)
matplotlib.rcParams['savefig.dpi']        = 600                                 # figure resolution (makes a little better .png)
matplotlib.rcParams['savefig.pad_inches'] = 0.10                                # remove extra whitespace
matplotlib.rcParams['savefig.bbox']       = 'tight'                           # remove extra whitespace = 'tight'
matplotlib.rcParams['lines.linewidth']    = 1.
matplotlib.rcParams['legend.handlelength']  = 1.5
matplotlib.rcParams['font.family']        = 'serif'
matplotlib.rcParams['axes.linewidth'] = 0.5 # set the value globally
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['text.usetex']           = True
matplotlib.rcParams['figure.figsize']        = [4,3]
matplotlib.rcParams['figure.dpi']            = 150

FIGPATH = BASE / "figs"

# Shared per-model linespec, reused across any figure comparing wake models to
# LES (wake shapes, aspect ratio, streamtube deficit, ...). Keyed by solver
# key (models.DISPLAY_NAMES), not display name, so a display-name rename
# doesn't require touching this. Add new models here (commented-out slots
# below are models not currently plotted, kept as a reminder of the
# color/dash/marker already reserved for them).
MODEL_COLORS = {
    "LES": "k",
    "gauss": "tab:red",
    "varvortex": "tab:orange",
    "2021": sns.color_palette("mako", n_colors=5)[4],
    "scott": "#289b60",
    "tandem": sns.color_palette("mako", n_colors=5)[2],
    "kl-hub": "tab:purple",
    # "reasonable variation" probes on the Gaussian model (models.py); kept
    # in the same red family as "gauss" since they share its wake shape.
    "gauss-quad": "lightcoral",
    "gauss-lin": "firebrick",
    "gauss-noti": "darksalmon",
}

MODEL_DASHES = {
    "LES": (1, 0),
    "gauss": (1, 1, 3, 1),
    "varvortex": (2, 2),
    "2021": (3, 3),
    "scott": (3, 1, 3),
    "tandem": (4, 1),
    "kl-hub": (1, 1),
    "gauss-quad": (1, 1, 3, 1), #(1, 1),
    "gauss-lin": (1, 1, 3, 1), #(5, 1),
    "gauss-noti": (1, 1, 3, 1), #(2, 1, 1, 1),
}

MODEL_MARKERS = {
    "LES": ",",
    "gauss": "s",
    "varvortex": "X",
    "2021": "^",
    "scott": "v",
    "kl-hub": "^",
    "tandem": "o",
    "gauss-quad": "*",
    "gauss-lin": "P",
    "gauss-noti": "D",
}


# Pastel yellow used to flag the truly-neutral boundary layer / Ekman case
# (Cr = 0) in SBL cooling-rate sweeps, so it reads as distinct from the
# "crest" gradient used for the stable (Cr > 0) cases rather than sitting at
# one end of it.
NBL_COLOR = "#D9DC84"


def cr_palette(cr_values):
    """Color palette for a set of SBL surface cooling rates Cr (K/hr).

    The neutral case (Cr == 0, TNBL/Ekman) is mapped to NBL_COLOR; the
    remaining Cr > 0 stable cases are mapped to the "crest" colormap, spread
    across just those stable values (so adding/removing the neutral case
    doesn't shift the stable cases' colors).

    Returns a dict {cr: color}, so it's a drop-in replacement for
    `dict(zip(cr_values, sns.color_palette("crest", n_colors=len(cr_values))))`.
    """
    cr_values = sorted(set(cr_values))
    stable = [cr for cr in cr_values if cr != 0]
    palette = dict(zip(stable, sns.color_palette("crest", n_colors=len(stable))))
    if 0 in cr_values:
        palette[0] = NBL_COLOR
    return palette


def save(stem=None):
    """Save the current figure to the FIGPATH directory with a filename based on the caller's name."""
    if stem is None:
        caller_frame = inspect.stack()[1]
        stem = Path(caller_frame.filename).stem
    matplotlib.pyplot.savefig(FIGPATH / stem)
    print(f"Saved figure to {FIGPATH / stem}.{matplotlib.rcParams['savefig.format']}")
