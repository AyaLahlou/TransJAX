"""
Parameters for lnd_comp_mct module.

This file contains constants and configuration parameters used by the
land component MCT interface.

Fortran source: lnd_comp_mct.F90
"""

import jax.numpy as jnp

# =============================================================================
# Precision Constants
# =============================================================================

# Double precision floating point (corresponds to r8 => shr_kind_r8)
R8 = jnp.float64

# Single precision floating point (for completeness)
R4 = jnp.float32

# Integer kinds
I4 = jnp.int32
I8 = jnp.int64


# =============================================================================
# Module Configuration
# =============================================================================

# Module version information
MODULE_VERSION = "1.0.0-jax"
MODULE_NAME = "lnd_comp_mct"

# Default bounds for testing/validation
DEFAULT_BOUNDS = {
    "begg": 1,
    "endg": 100,
    "begl": 1,
    "endl": 150,
    "begc": 1,
    "endc": 200,
    "begp": 1,
    "endp": 300,
}


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "R8",
    "R4",
    "I4",
    "I8",
    "MODULE_VERSION",
    "MODULE_NAME",
    "DEFAULT_BOUNDS",
]