"""
Default orbital parameters for CTSM.

This file contains standard orbital parameter values used in CTSM simulations.
Values represent present-day Earth orbital characteristics.

Reference:
    clm_varorb.F90, lines 1-21
"""

import jax.numpy as jnp

# Present-day Earth orbital parameters
# Reference: Berger (1978), Long-Term Variations of Daily Insolation
DEFAULT_ECCEN = 0.016715  # Orbital eccentricity (dimensionless)
DEFAULT_OBLIQR = 0.409214  # Obliquity in radians (~23.44 degrees)
DEFAULT_LAMBM0 = 4.936813  # Mean longitude of perihelion at vernal equinox (radians)
DEFAULT_MVELPP = 1.796257  # Moving vernal equinox longitude + pi (radians)

# Historical orbital parameters for sensitivity studies
# Last Glacial Maximum (~21,000 years ago)
LGM_ECCEN = 0.018994
LGM_OBLIQR = 0.397789  # ~22.8 degrees
LGM_LAMBM0 = 0.872665
LGM_MVELPP = 1.570796

# Mid-Holocene (~6,000 years ago)
MID_HOLOCENE_ECCEN = 0.018682
MID_HOLOCENE_OBLIQR = 0.418879  # ~24.0 degrees
MID_HOLOCENE_LAMBM0 = 0.872665
MID_HOLOCENE_MVELPP = 1.570796

# Validation ranges
ECCEN_MIN = 0.0
ECCEN_MAX = 0.067  # Maximum Earth eccentricity over past 5 Myr
OBLIQR_MIN = 0.349066  # ~20 degrees
OBLIQR_MAX = 0.436332  # ~25 degrees