"""
Leaf Boundary Layer Conductance Module.

Translated from CTSM's MLLeafBoundaryLayerMod.F90

This module calculates leaf boundary layer conductance for heat and water vapor
transfer between the leaf surface and the canopy air. The boundary layer
conductance depends on leaf size, wind speed, and atmospheric properties.

Key physics:
    - Forced convection from wind (laminar and turbulent flow)
    - Free convection from temperature differences (buoyancy-driven)
    - Leaf dimension effects on boundary layer thickness
    - Temperature and pressure corrections to molecular diffusivities
    
The boundary layer conductance is a critical component of leaf energy balance
and transpiration calculations in multilayer canopy models.

Key equations:
    gbh = (dh * Nu / dleaf) * rhomol  [mol/m2/s]
    gbv = (dv * Shv / dleaf) * rhomol [mol/m2/s]
    gbc = (dc * Shc / dleaf) * rhomol [mol/m2/s]
    
Where:
    - dh, dv, dc: Diffusivities for heat, water vapor, CO2 [m2/s]
    - Nu, Shv, Shc: Nusselt/Sherwood numbers [dimensionless]
    - dleaf: Leaf characteristic dimension [m]
    - rhomol: Molar density of air [mol/m3]

Dimensionless numbers:
    - Reynolds: Re = wind * dleaf / visc
    - Prandtl: Pr = visc / dh
    - Schmidt: Sc = visc / d (for vapor or CO2)
    - Grashof: Gr = grav * dleaf^3 * dT / (T * visc^2)
    
Flow regimes (controlled by gb_type):
    - gb_type = 0: Simplified CLM5 formulation
    - gb_type = 1: Laminar flow only
    - gb_type = 2: Maximum of laminar and turbulent
    - gb_type = 3: Maximum of laminar/turbulent plus free convection

Reference: MLLeafBoundaryLayerMod.F90 (lines 1-206)
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================

class BoundaryLayerParams(NamedTuple):
    """Parameters for leaf boundary layer calculations.
    
    Attributes:
        gb_type: Boundary layer calculation type [0-3] [-]
                 0: Simplified CLM5 formulation
                 1: Laminar flow only
                 2: Maximum of laminar and turbulent
                 3: Maximum of laminar/turbulent plus free convection
        gb_factor: Boundary layer factor for forced convection [-]
        visc0: Reference kinematic viscosity at 0°C, 1 atm [m2/s]
        dh0: Reference thermal diffusivity at 0°C, 1 atm [m2/s]
        dv0: Reference water vapor diffusivity at 0°C, 1 atm [m2/s]
        dc0: Reference CO2 diffusivity at 0°C, 1 atm [m2/s]
        tfrz: Freezing temperature [K]
        grav: Gravitational acceleration [m/s2]
    """
    gb_type: int
    gb_factor: float
    visc0: float
    dh0: float
    dv0: float
    dc0: float
    tfrz: float
    grav: float


class BoundaryLayerIntermediates(NamedTuple):
    """Intermediate values for boundary layer calculations.
    
    All arrays have shape [n_patches, n_canopy_layers] unless noted.
    
    Attributes:
        visc: Kinematic viscosity adjusted for T and P [m2/s] [n_patches]
        dh: Thermal diffusivity adjusted for T and P [m2/s] [n_patches]
        dv: Water vapor diffusivity adjusted for T and P [m2/s] [n_patches]
        dc: CO2 diffusivity adjusted for T and P [m2/s] [n_patches]
        re: Reynolds number [-]
        pr: Prandtl number [-]
        scv: Schmidt number for water vapor [-]
        scc: Schmidt number for CO2 [-]
        gr: Grashof number [-]
        nu_lam: Nusselt number (laminar) [-]
        shv_lam: Sherwood number for H2O (laminar) [-]
        shc_lam: Sherwood number for CO2 (laminar) [-]
        nu_turb: Nusselt number (turbulent) [-]
        shv_turb: Sherwood number for H2O (turbulent) [-]
        shc_turb: Sherwood number for CO2 (turbulent) [-]
        nu_free: Nusselt number (free convection) [-]
        shv_free: Sherwood number for H2O (free) [-]
        shc_free: Sherwood number for CO2 (free) [-]
    """
    visc: jnp.ndarray
    dh: jnp.ndarray
    dv: jnp.ndarray
    dc: jnp.ndarray
    re: jnp.ndarray
    pr: jnp.ndarray
    scv: jnp.ndarray
    scc: jnp.ndarray
    gr: jnp.ndarray
    nu_lam: jnp.ndarray
    shv_lam: jnp.ndarray
    shc_lam: jnp.ndarray
    nu_turb: jnp.ndarray
    shv_turb: jnp.ndarray
    shc_turb: jnp.ndarray
    nu_free: jnp.ndarray
    shv_free: jnp.ndarray
    shc_free: jnp.ndarray


class LeafBoundaryLayerOutputs(NamedTuple):
    """Outputs from leaf boundary layer conductance calculations.
    
    All arrays have shape [n_patches, n_canopy_layers].
    
    Attributes:
        gbh: Heat conductance [mol/m2/s]
        gbv: Water vapor conductance [mol/m2/s]
        gbc: CO2 conductance [mol/m2/s]
    """
    gbh: jnp.ndarray
    gbv: jnp.ndarray
    gbc: jnp.ndarray


# =============================================================================
# Default Parameters
# =============================================================================

def get_default_params() -> BoundaryLayerParams:
    """Get default boundary layer parameters.
    
    Returns:
        BoundaryLayerParams with standard values from CTSM
        
    Note:
        These values are from MLclm_varcon.F90 and clm_varcon.F90
    """
    return BoundaryLayerParams(
        gb_type=2,  # Maximum of laminar and turbulent
        gb_factor=1.0,  # No adjustment factor
        visc0=13.3e-6,  # m2/s at 0°C, 1 atm
        dh0=18.9e-6,  # m2/s at 0°C, 1 atm
        dv0=21.8e-6,  # m2/s at 0°C, 1 atm
        dc0=13.8e-6,  # m2/s at 0°C, 1 atm
        tfrz=273.15,  # K
        grav=9.80616,  # m/s2
    )


# =============================================================================
# Core Physics Functions
# =============================================================================

def calculate_simple_conductance(
    wind: jnp.ndarray,
    dleaf: jnp.ndarray,
    dpai: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate simplified boundary layer conductance (gb_type = 0).
    
    Uses CLM5 simplification from lines 103-106.
    
    Args:
        wind: Canopy layer wind speed [m/s] [n_patches, n_canopy_layers]
        dleaf: Leaf dimension [m] [n_patches]
        dpai: Canopy layer plant area index [m2/m2] [n_patches, n_canopy_layers]
        
    Returns:
        Tuple of (gbh, gbv, gbc) in [m/s] [n_patches, n_canopy_layers]
        
    Note:
        Lines 103-106 in MLLeafBoundaryLayerMod.F90
        Units are m/s (will be converted to mol/m2/s later)
    """
    dleaf_bc = dleaf[:, jnp.newaxis]
    
    # gbh = 0.005 * sqrt(wind / dleaf) (line 103)
    gbh = 0.005 * jnp.sqrt(wind / dleaf_bc)
    
    # gbv = gbh (line 104)
    gbv = gbh
    
    # gbc = gbv / 1.4 (line 105)
    gbc = gbv / 1.4
    
    # Only apply where dpai > 0
    gbh = jnp.where(dpai > 0.0, gbh, 0.0)
    gbv = jnp.where(dpai > 0.0, gbv, 0.0)
    gbc = jnp.where(dpai > 0.0, gbc, 0.0)
    
    return gbh, gbv, gbc


def calculate_boundary_layer_intermediates(
    dleaf: jnp.ndarray,
    tref: jnp.ndarray,
    pref: jnp.ndarray,
    wind: jnp.ndarray,
    tair: jnp.ndarray,
    tleaf: jnp.ndarray,
    params: BoundaryLayerParams,
) -> BoundaryLayerIntermediates:
    """Calculate intermediate values for boundary layer conductance.
    
    This function computes dimensionless numbers and Nusselt/Sherwood numbers
    needed for boundary layer conductance calculations. It handles the physics
    from lines 72-148 of the original Fortran code.
    
    Args:
        dleaf: Leaf dimension [m] [n_patches]
        tref: Air temperature at reference height [K] [n_patches]
        pref: Air pressure at reference height [Pa] [n_patches]
        wind: Canopy layer wind speed [m/s] [n_patches, n_canopy_layers]
        tair: Canopy layer air temperature [K] [n_patches, n_canopy_layers]
        tleaf: Leaf temperature [K] [n_patches, n_canopy_layers]
        params: Boundary layer parameters
        
    Returns:
        BoundaryLayerIntermediates containing all intermediate calculations
        
    Note:
        - Lines 72-148 in MLLeafBoundaryLayerMod.F90
        - Diffusivities are adjusted for temperature and pressure (line 89-93)
        - Reynolds, Prandtl, Schmidt, and Grashof numbers (lines 109-113)
        - Nusselt and Sherwood numbers for laminar, turbulent, and free convection (lines 117-135)
    """
    # Adjust diffusivity for temperature and pressure (lines 89-93)
    # fac = 101325 / pref * (tref / tfrz)^1.81
    fac = 101325.0 / pref * jnp.power(tref / params.tfrz, 1.81)
    visc = params.visc0 * fac
    dh = params.dh0 * fac
    dv = params.dv0 * fac
    dc = params.dc0 * fac
    
    # Broadcast patch-level values to [n_patches, n_canopy_layers]
    visc_bc = visc[:, jnp.newaxis]
    dh_bc = dh[:, jnp.newaxis]
    dv_bc = dv[:, jnp.newaxis]
    dc_bc = dc[:, jnp.newaxis]
    dleaf_bc = dleaf[:, jnp.newaxis]
    
    # Calculate dimensionless numbers (lines 109-113)
    # Reynolds number: re = wind * dleaf / visc
    re = wind * dleaf_bc / visc_bc
    
    # Prandtl number: pr = visc / dh
    pr = visc_bc / dh_bc
    
    # Schmidt numbers: scv = visc / dv, scc = visc / dc
    scv = visc_bc / dv_bc
    scc = visc_bc / dc_bc
    
    # Grashof number: gr = grav * dleaf^3 * max(tleaf - tair, 0) / (tair * visc^2)
    # (line 113)
    temp_diff = jnp.maximum(tleaf - tair, 0.0)
    gr = (params.grav * jnp.power(dleaf_bc, 3.0) * temp_diff / 
          (tair * jnp.power(visc_bc, 2.0)))
    
    # Nusselt and Sherwood numbers for forced convection - laminar flow (lines 119-121)
    # nu_lam = gb_factor * 0.66 * pr^0.33 * re^0.5
    nu_lam = params.gb_factor * 0.66 * jnp.power(pr, 0.33) * jnp.power(re, 0.5)
    shv_lam = params.gb_factor * 0.66 * jnp.power(scv, 0.33) * jnp.power(re, 0.5)
    shc_lam = params.gb_factor * 0.66 * jnp.power(scc, 0.33) * jnp.power(re, 0.5)
    
    # Nusselt and Sherwood numbers for forced convection - turbulent flow (lines 125-127)
    # nu_turb = gb_factor * 0.036 * pr^0.33 * re^0.8
    nu_turb = params.gb_factor * 0.036 * jnp.power(pr, 0.33) * jnp.power(re, 0.8)
    shv_turb = params.gb_factor * 0.036 * jnp.power(scv, 0.33) * jnp.power(re, 0.8)
    shc_turb = params.gb_factor * 0.036 * jnp.power(scc, 0.33) * jnp.power(re, 0.8)
    
    # Nusselt and Sherwood numbers for free convection (lines 131-133)
    # nu_free = 0.54 * pr^0.25 * gr^0.25
    nu_free = 0.54 * jnp.power(pr, 0.25) * jnp.power(gr, 0.25)
    shv_free = 0.54 * jnp.power(scv, 0.25) * jnp.power(gr, 0.25)
    shc_free = 0.54 * jnp.power(scc, 0.25) * jnp.power(gr, 0.25)
    
    return BoundaryLayerIntermediates(
        visc=visc,
        dh=dh,
        dv=dv,
        dc=dc,
        re=re,
        pr=pr,
        scv=scv,
        scc=scc,
        gr=gr,
        nu_lam=nu_lam,
        shv_lam=shv_lam,
        shc_lam=shc_lam,
        nu_turb=nu_turb,
        shv_turb=shv_turb,
        shc_turb=shc_turb,
        nu_free=nu_free,
        shv_free=shv_free,
        shc_free=shc_free,
    )


def finalize_boundary_layer_conductances(
    intermediates: BoundaryLayerIntermediates,
    dleaf: jnp.ndarray,
    rhomol: jnp.ndarray,
    dpai: jnp.ndarray,
    params: BoundaryLayerParams,
) -> LeafBoundaryLayerOutputs:
    """Finalize boundary layer conductances based on flow regime.
    
    Selects appropriate Nusselt/Sherwood numbers based on gb_type,
    computes conductances, and converts units.
    
    Args:
        intermediates: Intermediate calculations from previous step
        dleaf: Leaf characteristic dimension [m] [n_patches]
        rhomol: Molar density of air [mol/m3] [n_patches, n_canopy_layers]
        dpai: Canopy layer plant area index [m2/m2] [n_patches, n_canopy_layers]
        params: Boundary layer parameters
        
    Returns:
        LeafBoundaryLayerOutputs containing gbh, gbv, gbc [mol/m2/s]
        
    Reference:
        Lines 149-198 in MLLeafBoundaryLayerMod.F90
    """
    # Select Nusselt/Sherwood numbers based on gb_type (lines 149-169)
    # Case 1: Laminar flow only
    nu_case1 = intermediates.nu_lam
    shv_case1 = intermediates.shv_lam
    shc_case1 = intermediates.shc_lam
    
    # Case 2: Maximum of laminar and turbulent
    nu_case2 = jnp.maximum(intermediates.nu_lam, intermediates.nu_turb)
    shv_case2 = jnp.maximum(intermediates.shv_lam, intermediates.shv_turb)
    shc_case2 = jnp.maximum(intermediates.shc_lam, intermediates.shc_turb)
    
    # Case 3: Maximum of laminar/turbulent plus free convection
    nu_case3 = jnp.maximum(intermediates.nu_lam, intermediates.nu_turb) + intermediates.nu_free
    shv_case3 = jnp.maximum(intermediates.shv_lam, intermediates.shv_turb) + intermediates.shv_free
    shc_case3 = jnp.maximum(intermediates.shc_lam, intermediates.shc_turb) + intermediates.shc_free
    
    # Select based on gb_type
    nu = jnp.where(params.gb_type == 1, nu_case1,
                   jnp.where(params.gb_type == 2, nu_case2, nu_case3))
    shv = jnp.where(params.gb_type == 1, shv_case1,
                    jnp.where(params.gb_type == 2, shv_case2, shv_case3))
    shc = jnp.where(params.gb_type == 1, shc_case1,
                    jnp.where(params.gb_type == 2, shc_case2, shc_case3))
    
    # Expand dleaf to match shape [n_patches, n_canopy_layers]
    dleaf_expanded = dleaf[:, jnp.newaxis]
    
    # Broadcast diffusivities to [n_patches, n_canopy_layers]
    dh_bc = intermediates.dh[:, jnp.newaxis]
    dv_bc = intermediates.dv[:, jnp.newaxis]
    dc_bc = intermediates.dc[:, jnp.newaxis]
    
    # Compute boundary layer conductances in m/s (lines 171-174)
    gbh_ms = dh_bc * nu / dleaf_expanded
    gbv_ms = dv_bc * shv / dleaf_expanded
    gbc_ms = dc_bc * shc / dleaf_expanded
    
    # Convert conductances from m/s to mol/m2/s (lines 182-184)
    gbh = gbh_ms * rhomol
    gbv = gbv_ms * rhomol
    gbc = gbc_ms * rhomol
    
    # Set to zero where dpai <= 0 (lines 186-191)
    compute_conductance = dpai > 0.0
    gbh = jnp.where(compute_conductance, gbh, 0.0)
    gbv = jnp.where(compute_conductance, gbv, 0.0)
    gbc = jnp.where(compute_conductance, gbc, 0.0)
    
    return LeafBoundaryLayerOutputs(
        gbh=gbh,
        gbv=gbv,
        gbc=gbc,
    )


# =============================================================================
# Main Public Interface
# =============================================================================

def leaf_boundary_layer(
    dleaf: jnp.ndarray,
    tref: jnp.ndarray,
    pref: jnp.ndarray,
    wind: jnp.ndarray,
    tair: jnp.ndarray,
    tleaf: jnp.ndarray,
    rhomol: jnp.ndarray,
    dpai: jnp.ndarray,
    params: BoundaryLayerParams | None = None,
) -> LeafBoundaryLayerOutputs:
    """Calculate leaf boundary layer conductances.
    
    This is the main entry point for computing boundary layer conductances
    for heat (gbh), water vapor (gbv), and CO2 (gbc) transfer between leaves
    and the canopy air. The calculation accounts for:
    - Leaf characteristic dimension
    - Wind speed in the canopy
    - Molecular diffusivities (temperature dependent)
    - Forced vs. free convection
    
    Translated from MLLeafBoundaryLayerMod.F90:22-204
    
    Args:
        dleaf: Leaf dimension [m] [n_patches]
        tref: Air temperature at reference height [K] [n_patches]
        pref: Air pressure at reference height [Pa] [n_patches]
        wind: Canopy layer wind speed [m/s] [n_patches, n_canopy_layers]
        tair: Canopy layer air temperature [K] [n_patches, n_canopy_layers]
        tleaf: Leaf temperature [K] [n_patches, n_canopy_layers]
        rhomol: Molar density of air [mol/m3] [n_patches, n_canopy_layers]
        dpai: Canopy layer plant area index [m2/m2] [n_patches, n_canopy_layers]
        params: Boundary layer parameters (uses defaults if None)
        
    Returns:
        LeafBoundaryLayerOutputs with gbh, gbv, gbc [mol/m2/s]
        
    Note:
        If gb_type = 0, uses simplified CLM5 formulation.
        Otherwise, uses full boundary layer theory with dimensionless numbers.
    """
    if params is None:
        params = get_default_params()
    
    # Handle simplified case (gb_type = 0)
    if params.gb_type == 0:
        gbh_ms, gbv_ms, gbc_ms = calculate_simple_conductance(wind, dleaf, dpai)
        
        # Convert from m/s to mol/m2/s
        gbh = gbh_ms * rhomol
        gbv = gbv_ms * rhomol
        gbc = gbc_ms * rhomol
        
        return LeafBoundaryLayerOutputs(gbh=gbh, gbv=gbv, gbc=gbc)
    
    # Full boundary layer calculation (gb_type = 1, 2, or 3)
    intermediates = calculate_boundary_layer_intermediates(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf,
        params=params,
    )
    
    outputs = finalize_boundary_layer_conductances(
        intermediates=intermediates,
        dleaf=dleaf,
        rhomol=rhomol,
        dpai=dpai,
        params=params,
    )
    
    return outputs


# =============================================================================
# Convenience Functions
# =============================================================================

def leaf_boundary_layer_sunlit_shaded(
    dleaf: jnp.ndarray,
    tref: jnp.ndarray,
    pref: jnp.ndarray,
    wind: jnp.ndarray,
    tair: jnp.ndarray,
    tleaf_sun: jnp.ndarray,
    tleaf_sha: jnp.ndarray,
    rhomol: jnp.ndarray,
    dpai: jnp.ndarray,
    params: BoundaryLayerParams | None = None,
) -> Tuple[LeafBoundaryLayerOutputs, LeafBoundaryLayerOutputs]:
    """Calculate boundary layer conductances for sunlit and shaded leaves.
    
    Convenience function that calls leaf_boundary_layer twice, once for
    sunlit leaves and once for shaded leaves.
    
    Args:
        dleaf: Leaf dimension [m] [n_patches]
        tref: Air temperature at reference height [K] [n_patches]
        pref: Air pressure at reference height [Pa] [n_patches]
        wind: Canopy layer wind speed [m/s] [n_patches, n_canopy_layers]
        tair: Canopy layer air temperature [K] [n_patches, n_canopy_layers]
        tleaf_sun: Sunlit leaf temperature [K] [n_patches, n_canopy_layers]
        tleaf_sha: Shaded leaf temperature [K] [n_patches, n_canopy_layers]
        rhomol: Molar density of air [mol/m3] [n_patches, n_canopy_layers]
        dpai: Canopy layer plant area index [m2/m2] [n_patches, n_canopy_layers]
        params: Boundary layer parameters (uses defaults if None)
        
    Returns:
        Tuple of (sunlit_outputs, shaded_outputs)
    """
    sunlit_outputs = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf_sun,
        rhomol=rhomol,
        dpai=dpai,
        params=params,
    )
    
    shaded_outputs = leaf_boundary_layer(
        dleaf=dleaf,
        tref=tref,
        pref=pref,
        wind=wind,
        tair=tair,
        tleaf=tleaf_sha,
        rhomol=rhomol,
        dpai=dpai,
        params=params,
    )
    
    return sunlit_outputs, shaded_outputs