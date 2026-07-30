#!/usr/bin/python

# Hannah Johlas, 2019
# Revised Kirby Heck, 2024

import matplotlib

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
