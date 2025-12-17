"""
Leaf Heat Capacity Module.

Translated from CTSM's MLLeafHeatCapacityMod.F90

This module provides functions for calculating leaf heat capacity in the
multilayer canopy model. Leaf heat capacity determines how much energy is
required to change leaf temperature and affects the thermal inertia of
the canopy.

Key physics:
    - Heat capacity depends on leaf water content and dry matter
    - Affects leaf temperature response to radiative forcing
    - Important for diurnal temperature cycles

Key equations (lines 64-69):
    lma = 1 / slatop * 0.001                    [kg C / m2]
    dry_weight = lma / fcarbon                  [kg DM / m2]
    fresh_weight = dry_weight / (1 - fwater)    [kg FM / m2]
    leaf_water = fwater * fresh_weight          [kg H2O / m2]
    cpleaf = cpbio * dry_weight + cpliq * leaf_water  [J/K/m2 leaf]

Where:
    - slatop: Specific leaf area at canopy top [m2/gC]
    - fcarbon: Carbon fraction of dry biomass (0.5)
    - fwater: Water fraction of fresh biomass (0.7)
    - cpbio: Heat capacity of dry biomass [J/kg/K]
    - cpliq: Heat capacity of liquid water [J/kg/K]

Reference:
    MLLeafHeatCapacityMod.F90, lines 1-83
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================


class LeafHeatCapacityInput(NamedTuple):
    """Input data for leaf heat capacity calculation.
    
    Attributes:
        slatop: Specific leaf area at top of canopy [m2/gC] [n_patches]
        ncan: Number of aboveground layers [n_patches]
        dpai: Canopy layer plant area index [m2/m2] [n_patches, n_levels]
        cpbio: Heat capacity of dry biomass [J/kg/K] (scalar)
        cpliq: Heat capacity of liquid water [J/kg/K] (scalar)
        fcarbon: Carbon fraction of dry biomass (0.5) (scalar)
        fwater: Water fraction of fresh biomass (0.7) (scalar)
    """
    slatop: jnp.ndarray  # [n_patches]
    ncan: jnp.ndarray  # [n_patches]
    dpai: jnp.ndarray  # [n_patches, n_levels]
    cpbio: float
    cpliq: float
    fcarbon: float
    fwater: float


class LeafHeatCapacityParams(NamedTuple):
    """Parameters for leaf heat capacity calculations.
    
    These are typically constants from CLM physics modules.
    
    Attributes:
        cpbio: Heat capacity of dry biomass [J/kg/K]
               Typical value: 1470.0 (from MLclm_varcon)
        cpliq: Heat capacity of liquid water [J/kg/K]
               Typical value: 4188.0 (from clm_varcon)
        fcarbon: Carbon fraction of dry biomass [-]
                 Typical value: 0.5 (from MLclm_varcon)
        fwater: Water fraction of fresh biomass [-]
                Typical value: 0.7 (from MLclm_varcon)
    """
    cpbio: float = 1470.0
    cpliq: float = 4188.0
    fcarbon: float = 0.5
    fwater: float = 0.7


# =============================================================================
# Main Functions
# =============================================================================


def leaf_heat_capacity(inputs: LeafHeatCapacityInput) -> jnp.ndarray:
    """Calculate leaf heat capacity for multilayer canopy.
    
    Converts specific leaf area to leaf mass per area, then calculates heat
    capacity accounting for both dry biomass and water content. The calculation
    follows these steps (lines 47-54):
    
    1. Convert specific leaf area (m2/gC) to leaf carbon mass (kg C/m2)
    2. Convert carbon mass to dry weight (assume carbon is 50% of dry biomass)
    3. Convert dry weight to fresh weight (assume 70% of fresh biomass is water)
    4. Calculate leaf water content
    5. Calculate total heat capacity from dry biomass and water
    
    The heat capacity is only calculated for layers with plant area index > 0.
    
    Args:
        inputs: LeafHeatCapacityInput containing all required fields
        
    Returns:
        Leaf heat capacity [J/m2 leaf/K] [n_patches, n_levels]
        
    Reference:
        MLLeafHeatCapacityMod.F90, lines 22-81
        
    Note:
        - Returns 0 for layers where dpai <= 0 (line 63)
        - All calculations are vectorized for JIT compilation
        - Preserves exact physics from Fortran implementation
    """
    # Extract inputs
    slatop = inputs.slatop
    dpai = inputs.dpai
    cpbio = inputs.cpbio
    cpliq = inputs.cpliq
    fcarbon = inputs.fcarbon
    fwater = inputs.fwater
    
    # Broadcast slatop to match dpai shape for vectorized calculation
    # slatop: [n_patches] -> [n_patches, 1]
    slatop_expanded = slatop[:, jnp.newaxis]
    
    # Line 64: Convert specific leaf area to leaf carbon mass per area
    # slatop is in m2/gC, convert to kg C/m2
    # lma = 1 / slatop * 0.001
    lma = 1.0 / slatop_expanded * 0.001
    
    # Line 65: Convert carbon mass to dry weight
    # Assume carbon is 50% of dry biomass (fcarbon = 0.5)
    # kg C/m2 -> kg DM/m2
    dry_weight = lma / fcarbon
    
    # Line 66: Convert dry weight to fresh weight
    # Assume 70% of fresh biomass is water (fwater = 0.7)
    # kg DM/m2 -> kg FM/m2
    fresh_weight = dry_weight / (1.0 - fwater)
    
    # Line 67: Calculate leaf water content
    # kg H2O/m2 leaf
    leaf_water = fwater * fresh_weight
    
    # Line 68: Calculate heat capacity
    # Heat capacity = (dry biomass heat capacity * dry weight) + 
    #                 (water heat capacity * water content)
    # J/K/m2 leaf
    cpleaf_calc = cpbio * dry_weight + cpliq * leaf_water
    
    # Line 63: Only calculate where dpai > 0, otherwise set to 0
    # Lines 69-70: if (dpai(p,iv) > 0._r8) then ... else cpleaf(p,iv) = 0._r8
    cpleaf = jnp.where(dpai > 0.0, cpleaf_calc, 0.0)
    
    return cpleaf


def leaf_heat_capacity_simple(
    slatop: jnp.ndarray,
    dpai: jnp.ndarray,
    params: LeafHeatCapacityParams = LeafHeatCapacityParams(),
) -> jnp.ndarray:
    """Simplified interface for leaf heat capacity calculation.
    
    This is a convenience function that wraps the main calculation with
    default parameters. Useful when parameters are constant across the
    simulation.
    
    Args:
        slatop: Specific leaf area at top of canopy [m2/gC] [n_patches]
        dpai: Canopy layer plant area index [m2/m2] [n_patches, n_levels]
        params: Physical parameters (uses defaults if not provided)
        
    Returns:
        Leaf heat capacity [J/m2 leaf/K] [n_patches, n_levels]
        
    Example:
        >>> slatop = jnp.array([0.01, 0.015])  # m2/gC
        >>> dpai = jnp.array([[1.0, 0.5, 0.0], [1.2, 0.8, 0.3]])
        >>> cpleaf = leaf_heat_capacity_simple(slatop, dpai)
    """
    # Create input structure
    n_patches = slatop.shape[0]
    ncan = jnp.full(n_patches, dpai.shape[1], dtype=jnp.int32)
    
    inputs = LeafHeatCapacityInput(
        slatop=slatop,
        ncan=ncan,
        dpai=dpai,
        cpbio=params.cpbio,
        cpliq=params.cpliq,
        fcarbon=params.fcarbon,
        fwater=params.fwater,
    )
    
    return leaf_heat_capacity(inputs)


# =============================================================================
# JIT-compiled versions
# =============================================================================


# Pre-compile the main function for performance
leaf_heat_capacity_jit = jax.jit(leaf_heat_capacity)
leaf_heat_capacity_simple_jit = jax.jit(leaf_heat_capacity_simple)


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    'LeafHeatCapacityInput',
    'LeafHeatCapacityParams',
    'leaf_heat_capacity',
    'leaf_heat_capacity_simple',
    'leaf_heat_capacity_jit',
    'leaf_heat_capacity_simple_jit',
]