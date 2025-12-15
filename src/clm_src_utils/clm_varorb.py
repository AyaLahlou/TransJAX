"""
JAX translation of clm_varorb module.

This module contains orbital parameters used in CTSM calculations for Earth's
orbital mechanics. These parameters are essential for solar radiation calculations
and seasonal variations in the climate model.

Translated from: clm_varorb.F90, lines 1-21

Key Components:
    - OrbitalParams: NamedTuple containing orbital parameters
    - create_orbital_params: Factory function for initialization
    - update_orbital_params: Immutable update function

Reference:
    Original Fortran module: clm_varorb.F90
"""

from typing import NamedTuple
import jax.numpy as jnp
from jax import Array

__all__ = [
    'OrbitalParams',
    'create_orbital_params',
    'update_orbital_params',
]


# ============================================================================
# Type Definitions
# ============================================================================

class OrbitalParams(NamedTuple):
    """
    Orbital parameters for Earth's orbit calculations.
    
    These parameters define Earth's orbital characteristics used in solar
    radiation and seasonal calculations within CTSM.
    
    Attributes:
        eccen: Orbital eccentricity factor (dimensionless).
               Range: [0, 1), where 0 is circular orbit.
               Input to orbit_params calculations. (Line 13)
        obliqr: Earth's obliquity in radians.
                The tilt of Earth's axis relative to orbital plane.
                Output from orbit_params. (Line 17)
        lambm0: Mean longitude of perihelion at the vernal equinox (radians).
                Angular position of perihelion at spring equinox.
                Output from orbit_params. (Line 18)
        mvelpp: Earth's moving vernal equinox longitude of perihelion 
                plus pi (radians).
                Adjusted perihelion longitude accounting for precession.
                Output from orbit_params. (Line 19)
    
    Notes:
        - All angular quantities are in radians
        - These parameters vary on Milankovitch timescales (10^4-10^5 years)
        - Used in conjunction with solar declination calculations
    
    Reference:
        Fortran source: clm_varorb.F90, lines 1-21
    """
    eccen: Array   # float64 scalar
    obliqr: Array  # float64 scalar
    lambm0: Array  # float64 scalar
    mvelpp: Array  # float64 scalar


# ============================================================================
# Factory and Update Functions
# ============================================================================

def create_orbital_params(
    eccen: float = 0.0,
    obliqr: float = 0.0,
    lambm0: float = 0.0,
    mvelpp: float = 0.0
) -> OrbitalParams:
    """
    Create an OrbitalParams instance with default or specified values.
    
    This factory function initializes orbital parameters, converting Python
    floats to JAX arrays with appropriate dtype for JIT compilation.
    
    Args:
        eccen: Orbital eccentricity factor (default: 0.0).
               Valid range: [0, 1). Typical Earth value: ~0.0167.
        obliqr: Earth's obliquity in radians (default: 0.0).
                Typical Earth value: ~0.4091 rad (23.44°).
        lambm0: Mean longitude of perihelion at vernal equinox in radians 
                (default: 0.0).
        mvelpp: Moving vernal equinox longitude of perihelion plus pi 
                in radians (default: 0.0).
    
    Returns:
        OrbitalParams instance with the specified values as JAX arrays.
    
    Example:
        >>> # Create with default values (all zeros)
        >>> params = create_orbital_params()
        >>> 
        >>> # Create with current Earth orbital parameters
        >>> params = create_orbital_params(
        ...     eccen=0.0167,
        ...     obliqr=0.4091,
        ...     lambm0=4.9368,
        ...     mvelpp=1.7965
        ... )
    
    Reference:
        Fortran source: clm_varorb.F90, lines 1-21
    """
    return OrbitalParams(
        eccen=jnp.asarray(eccen, dtype=jnp.float64),
        obliqr=jnp.asarray(obliqr, dtype=jnp.float64),
        lambm0=jnp.asarray(lambm0, dtype=jnp.float64),
        mvelpp=jnp.asarray(mvelpp, dtype=jnp.float64)
    )


def update_orbital_params(
    params: OrbitalParams,
    eccen: float | None = None,
    obliqr: float | None = None,
    lambm0: float | None = None,
    mvelpp: float | None = None
) -> OrbitalParams:
    """
    Update orbital parameters with new values (immutable).
    
    Creates a new OrbitalParams instance with updated values while preserving
    immutability. Only specified parameters are updated; others retain their
    original values.
    
    Args:
        params: Current OrbitalParams instance to update.
        eccen: New orbital eccentricity factor (optional).
               If None, retains current value.
        obliqr: New Earth's obliquity in radians (optional).
                If None, retains current value.
        lambm0: New mean longitude of perihelion at vernal equinox (optional).
                If None, retains current value.
        mvelpp: New moving vernal equinox longitude of perihelion plus pi 
                (optional). If None, retains current value.
    
    Returns:
        New OrbitalParams instance with updated values.
    
    Example:
        >>> # Create initial parameters
        >>> params = create_orbital_params(eccen=0.0167, obliqr=0.4091)
        >>> 
        >>> # Update only eccentricity
        >>> new_params = update_orbital_params(params, eccen=0.0200)
        >>> 
        >>> # Update multiple parameters
        >>> new_params = update_orbital_params(
        ...     params,
        ...     eccen=0.0200,
        ...     lambm0=5.0
        ... )
    
    Notes:
        - This function maintains immutability for JAX transformations
        - Original params instance is not modified
        - Useful for parameter sensitivity studies or time-varying orbits
    
    Reference:
        Fortran source: clm_varorb.F90, lines 1-21
    """
    return OrbitalParams(
        eccen=jnp.asarray(eccen, dtype=jnp.float64) if eccen is not None else params.eccen,
        obliqr=jnp.asarray(obliqr, dtype=jnp.float64) if obliqr is not None else params.obliqr,
        lambm0=jnp.asarray(lambm0, dtype=jnp.float64) if lambm0 is not None else params.lambm0,
        mvelpp=jnp.asarray(mvelpp, dtype=jnp.float64) if mvelpp is not None else params.mvelpp
    )


# ============================================================================
# Validation Utilities (Optional but Recommended)
# ============================================================================

def validate_orbital_params(params: OrbitalParams) -> bool:
    """
    Validate orbital parameters are within physically reasonable ranges.
    
    Args:
        params: OrbitalParams instance to validate.
    
    Returns:
        True if all parameters are valid, False otherwise.
    
    Notes:
        - Eccentricity must be in [0, 1)
        - Obliquity typically in [0, π/2] for Earth-like planets
        - Angular parameters should be finite
    """
    eccen_valid = jnp.logical_and(params.eccen >= 0.0, params.eccen < 1.0)
    obliqr_valid = jnp.logical_and(params.obliqr >= 0.0, params.obliqr <= jnp.pi)
    all_finite = jnp.all(jnp.isfinite(jnp.array([
        params.eccen, params.obliqr, params.lambm0, params.mvelpp
    ])))
    
    return jnp.logical_and(jnp.logical_and(eccen_valid, obliqr_valid), all_finite)