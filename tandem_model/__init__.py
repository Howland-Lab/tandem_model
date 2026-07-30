"""
TANDEM (Turbulence ANd DEficit Momentum) model — paper-ready analysis code.

Core model closures live in the sibling `mitwindfarm` package
(see `mitwindfarm.tandem`); this package handles LES data loading,
caching, and figure generation for the TANDEM manuscript.
"""

import mitwindfarm.tandem  # noqa: F401 - registers TANDEM closures (kl-md, scott, ...) on import
