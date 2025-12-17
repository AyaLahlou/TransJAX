"""
Water Vapor Calculations for Multilayer Canopy Model.

Translated from CTSM's MLWaterVaporMod.F90

This module provides functions for calculating fundamental thermodynamic
properties needed for energy balance and water vapor flux calculations in
the multilayer canopy model:

- Saturation vapor pressure and its temperature derivative
- Latent heat of vaporization/sublimation

Key equations:
    Saturation vapor pressure (Clausius-Clapeyron relation):
        For T >= 0°C (water): es = polynomial fit (Flatau et al. 1992)
        For T < 0°C (ice): es = polynomial fit (Flatau et al. 1992)
    
    Latent heat:
        For T > Tfrz: λ = hvap * mmh2o
        For T ≤ Tfrz: λ = hsub * mmh2o

References:
    Flatau et al. (1992) "Polynomial fits to saturation vapor pressure",
    Journal of Applied Meteorology 31:1507-1513.

Fortran source: MLWaterVaporMod.F90, lines 1-140
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================

class WaterVaporConstants(NamedTuple):
    """Physical constants for water vapor calculations.
    
    Attributes:
        tfrz: Freezing point of water [K]
        hvap: Latent heat of vaporization [J/kg]
        hsub: Latent heat of sublimation [J/kg]
        mmh2o: Molecular mass of water [kg/mol]
    """
    tfrz: float = 273.15      # Freezing point [K]
    hvap: float = 2.501e6     # Latent heat of vaporization [J/kg]
    hsub: float = 2.834e6     # Latent heat of sublimation [J/kg]
    mmh2o: float = 0.018015   # Molecular mass of water [kg/mol]


# Default constants instance
DEFAULT_CONSTANTS = WaterVaporConstants()


# =============================================================================
# Saturation Vapor Pressure Calculations
# =============================================================================

def sat_vap(
    t: jnp.ndarray,
    tfrz: float = 273.15,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate saturation vapor pressure and its temperature derivative.
    
    Uses polynomial approximations from Flatau et al. (1992) for both
    water vapor (T >= 0°C) and ice (T < 0°C) phases. The polynomial
    coefficients provide accurate fits over the ranges:
    - Water vapor: 0°C to 100°C
    - Ice: -75°C to 0°C
    
    The function automatically selects the appropriate polynomial based
    on temperature and clamps input temperatures to valid ranges.
    
    Fortran source: MLWaterVaporMod.F90, lines 23-111
    
    Args:
        t: Temperature [K] [arbitrary shape]
        tfrz: Freezing point of water [K] (default: 273.15)
        
    Returns:
        Tuple of:
            es: Saturation vapor pressure [Pa] [same shape as t]
            desdt: Temperature derivative of es [Pa/K] [same shape as t]
            
    Note:
        - Temperature is clamped to [-75°C, 100°C] before applying polynomials
        - Results are converted from mb to Pa by multiplying by 100
        - The transition between water and ice formulations is sharp at 0°C
        
    Example:
        >>> t = jnp.array([273.15, 283.15, 293.15])  # 0°C, 10°C, 20°C
        >>> es, desdt = sat_vap(t)
        >>> # es ≈ [611, 1228, 2339] Pa
        >>> # desdt ≈ [44, 82, 145] Pa/K
    """
    # Polynomial coefficients for water vapor (0°C to 100°C)
    # Flatau et al. (1992), Table 1
    # Lines 26-34
    a0 = 6.11213476
    a1 = 0.444007856
    a2 = 0.143064234e-01
    a3 = 0.264461437e-03
    a4 = 0.305903558e-05
    a5 = 0.196237241e-07
    a6 = 0.892344772e-10
    a7 = -0.373208410e-12
    a8 = 0.209339997e-15
    
    # Derivative coefficients for water vapor
    # Lines 38-46
    b0 = 0.444017302
    b1 = 0.286064092e-01
    b2 = 0.794683137e-03
    b3 = 0.121211669e-04
    b4 = 0.103354611e-06
    b5 = 0.404125005e-09
    b6 = -0.788037859e-12
    b7 = -0.114596802e-13
    b8 = 0.381294516e-16
    
    # Polynomial coefficients for ice (-75°C to 0°C)
    # Flatau et al. (1992), Table 2
    # Lines 50-58
    c0 = 6.11123516
    c1 = 0.503109514
    c2 = 0.188369801e-01
    c3 = 0.420547422e-03
    c4 = 0.614396778e-05
    c5 = 0.602780717e-07
    c6 = 0.387940929e-09
    c7 = 0.149436277e-11
    c8 = 0.262655803e-14
    
    # Derivative coefficients for ice
    # Lines 62-70
    d0 = 0.503277922
    d1 = 0.377289173e-01
    d2 = 0.126801703e-02
    d3 = 0.249468427e-04
    d4 = 0.313703411e-06
    d5 = 0.257180651e-08
    d6 = 0.133268878e-10
    d7 = 0.394116744e-13
    d8 = 0.498070196e-16
    
    # Convert to Celsius and clamp to valid range
    # Lines 72-74
    tc = t - tfrz
    tc = jnp.clip(tc, -75.0, 100.0)
    
    # Compute saturation vapor pressure using appropriate polynomial
    # Horner's method for efficient polynomial evaluation
    # Lines 76-86
    
    # For water (tc >= 0): nested polynomial evaluation
    es_water = (a0 + tc * (a1 + tc * (a2 + tc * (a3 + tc * (a4 
               + tc * (a5 + tc * (a6 + tc * (a7 + tc * a8))))))))
    desdt_water = (b0 + tc * (b1 + tc * (b2 + tc * (b3 + tc * (b4 
                  + tc * (b5 + tc * (b6 + tc * (b7 + tc * b8))))))))
    
    # For ice (tc < 0): nested polynomial evaluation
    es_ice = (c0 + tc * (c1 + tc * (c2 + tc * (c3 + tc * (c4 
             + tc * (c5 + tc * (c6 + tc * (c7 + tc * c8))))))))
    desdt_ice = (d0 + tc * (d1 + tc * (d2 + tc * (d3 + tc * (d4 
                + tc * (d5 + tc * (d6 + tc * (d7 + tc * d8))))))))
    
    # Select based on temperature (sharp transition at 0°C)
    # Lines 88-89
    es = jnp.where(tc >= 0.0, es_water, es_ice)
    desdt = jnp.where(tc >= 0.0, desdt_water, desdt_ice)
    
    # Convert from mb to Pa
    # Lines 88-89
    es = es * 100.0
    desdt = desdt * 100.0
    
    return es, desdt


# =============================================================================
# Latent Heat Calculations
# =============================================================================

def lat_vap(
    t: jnp.ndarray,
    constants: WaterVaporConstants = DEFAULT_CONSTANTS,
) -> jnp.ndarray:
    """Calculate molar latent heat of vaporization.
    
    Computes latent heat as a function of temperature, switching between
    vaporization (liquid to vapor) and sublimation (ice to vapor) at the
    freezing point. The latent heat is returned in molar units [J/mol]
    by multiplying the mass-specific values by the molecular mass of water.
    
    Fortran source: MLWaterVaporMod.F90, lines 114-138
    
    Args:
        t: Temperature [K] [arbitrary shape]
        constants: Physical constants (default: DEFAULT_CONSTANTS)
        
    Returns:
        Molar latent heat of vaporization [J/mol] [same shape as t]
        
    Note:
        - Above freezing (T > Tfrz): uses hvap (liquid water evaporation)
        - Below freezing (T ≤ Tfrz): uses hsub (ice sublimation)
        - The transition is sharp at tfrz (no smoothing)
        - Typical values:
            * At 20°C: ~44 kJ/mol (vaporization)
            * At -10°C: ~51 kJ/mol (sublimation)
            
    Example:
        >>> t = jnp.array([263.15, 273.15, 283.15])  # -10°C, 0°C, 10°C
        >>> lambda_mol = lat_vap(t)
        >>> # lambda_mol ≈ [51060, 51060, 45050] J/mol
        >>> # Note: at exactly 0°C, uses sublimation (T ≤ Tfrz)
    """
    # Line 131-135: Select latent heat based on temperature
    # if (t > tfrz) then
    #    lambda = hvap
    # else
    #    lambda = hsub
    # end if
    lambda_mass = jnp.where(
        t > constants.tfrz,
        constants.hvap,
        constants.hsub
    )
    
    # Line 136: Convert from J/kg to J/mol
    # lambda = lambda * mmh2o
    lambda_molar = lambda_mass * constants.mmh2o
    
    return lambda_molar


# =============================================================================
# Convenience Functions
# =============================================================================

def sat_vap_with_constants(
    t: jnp.ndarray,
    constants: WaterVaporConstants = DEFAULT_CONSTANTS,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate saturation vapor pressure using constants from NamedTuple.
    
    Convenience wrapper around sat_vap that extracts tfrz from constants.
    
    Args:
        t: Temperature [K] [arbitrary shape]
        constants: Physical constants (default: DEFAULT_CONSTANTS)
        
    Returns:
        Tuple of (es, desdt) as in sat_vap()
    """
    return sat_vap(t, tfrz=constants.tfrz)


def vapor_pressure_deficit(
    t: jnp.ndarray,
    rh: jnp.ndarray,
    constants: WaterVaporConstants = DEFAULT_CONSTANTS,
) -> jnp.ndarray:
    """Calculate vapor pressure deficit from temperature and relative humidity.
    
    VPD = es(T) * (1 - RH/100)
    
    Args:
        t: Temperature [K] [arbitrary shape]
        rh: Relative humidity [%] [same shape as t]
        constants: Physical constants (default: DEFAULT_CONSTANTS)
        
    Returns:
        Vapor pressure deficit [Pa] [same shape as t]
        
    Example:
        >>> t = jnp.array([293.15])  # 20°C
        >>> rh = jnp.array([50.0])   # 50% RH
        >>> vpd = vapor_pressure_deficit(t, rh)
        >>> # vpd ≈ 1170 Pa (half of saturation pressure)
    """
    es, _ = sat_vap(t, tfrz=constants.tfrz)
    vpd = es * (1.0 - rh / 100.0)
    return vpd


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'WaterVaporConstants',
    'DEFAULT_CONSTANTS',
    'sat_vap',
    'lat_vap',
    'sat_vap_with_constants',
    'vapor_pressure_deficit',
]