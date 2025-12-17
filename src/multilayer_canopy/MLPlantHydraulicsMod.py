"""
Plant Hydraulics Module.

Translated from CTSM's MLPlantHydraulicsMod.F90

This module calculates plant hydraulic properties including:
- Whole-plant hydraulic resistance
- Soil-to-root resistance and water uptake
- Leaf water potential

The hydraulic model represents water flow through the soil-plant-atmosphere
continuum, accounting for resistances in soil, roots, stems, and leaves.

Key physics:
    Water flow follows Darcy's law analogy:
    Q = (Ψ_source - Ψ_sink) / R
    
Where:
    - Q: Water flux [kg/m²/s]
    - Ψ: Water potential [MPa]
    - R: Hydraulic resistance [MPa·s·m²/kg]

The model includes:
1. Plant resistance: Aboveground hydraulic resistance from stem to leaf
2. Soil resistance: Belowground resistance from soil to root
3. Leaf water potential: Dynamic leaf water status with capacitance

References:
    Original Fortran: MLPlantHydraulicsMod.F90 (lines 1-322)
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp


# =============================================================================
# Physical Constants
# =============================================================================

# Water properties
DENH2O = 1000.0  # Water density [kg/m³]
GRAV = 9.80616   # Gravitational acceleration [m/s²]
MMOL_H2O = 18.0  # Molar mass of water [g/mol]

# Mathematical constants
PI = jnp.pi

# Conversion factors
HEAD = DENH2O * GRAV * 1.0e-6  # Converts mm to MPa


# =============================================================================
# Type Definitions
# =============================================================================

class PlantHydraulicsParams(NamedTuple):
    """Parameters for plant hydraulics calculations.
    
    Attributes:
        gplant_SPA: Stem hydraulic conductance [mmol H₂O/m²/s/MPa] [n_pfts]
        capac_SPA: Plant capacitance [mmol H₂O/m² leaf area/MPa] [n_pfts]
        root_radius: Fine root radius [m] [n_pfts]
        root_density: Fine root density [g biomass/m³ root] [n_pfts]
        root_resist: Root tissue resistivity [MPa·s·g/mmol H₂O] [n_pfts]
        minlwp_SPA: Minimum leaf water potential [MPa] (scalar, typically -2.0)
    """
    gplant_SPA: jnp.ndarray
    capac_SPA: jnp.ndarray
    root_radius: jnp.ndarray
    root_density: jnp.ndarray
    root_resist: jnp.ndarray
    minlwp_SPA: float = -2.0


class PlantResistanceInput(NamedTuple):
    """Input state for plant resistance calculation.
    
    Attributes:
        gplant_SPA: Stem hydraulic conductance [mmol H₂O/m²/s/MPa] [n_pfts]
        ncan: Number of aboveground layers [n_patches]
        rsoil: Soil hydraulic resistance [MPa·s·m²/mmol H₂O] [n_patches]
        dpai: Canopy layer plant area index [m²/m²] [n_patches, n_levcan]
        zs: Canopy layer height for scalar concentration [m] [n_patches, n_levcan]
        itype: Patch vegetation type (PFT index) [n_patches]
    """
    gplant_SPA: jnp.ndarray
    ncan: jnp.ndarray
    rsoil: jnp.ndarray
    dpai: jnp.ndarray
    zs: jnp.ndarray
    itype: jnp.ndarray


class PlantResistanceOutput(NamedTuple):
    """Output state for plant resistance calculation.
    
    Attributes:
        lsc: Leaf-specific conductance (soil-to-leaf) [mmol H₂O/m² leaf/s/MPa] 
             [n_patches, n_levcan]
    """
    lsc: jnp.ndarray


class SoilResistanceInputs(NamedTuple):
    """Input state for soil resistance calculations.
    
    Attributes:
        root_radius: Fine root radius [m] [n_pfts]
        root_density: Fine root density [g biomass/m³ root] [n_pfts]
        root_resist: Root tissue resistivity [MPa·s·g/mmol H₂O] [n_pfts]
        dz: Soil layer thickness [m] [n_columns, n_layers]
        nbedrock: Depth to bedrock index [-] [n_columns]
        smp_l: Soil matric potential [mm] [n_columns, n_layers]
        hk_l: Soil hydraulic conductivity [mm H₂O/s] [n_columns, n_layers]
        rootfr: Fraction of roots in each layer [-] [n_patches, n_layers]
        h2osoi_ice: Soil ice content [kg H₂O/m²] [n_columns, n_layers]
        root_biomass: Fine root biomass [g biomass/m²] [n_patches]
        lai: Leaf area index [m²/m²] [n_patches]
        itype: Patch vegetation type (PFT index) [n_patches]
        patch_to_column: Mapping from patch to column index [n_patches]
        minlwp_SPA: Minimum leaf water potential [MPa] (scalar)
    """
    root_radius: jnp.ndarray
    root_density: jnp.ndarray
    root_resist: jnp.ndarray
    dz: jnp.ndarray
    nbedrock: jnp.ndarray
    smp_l: jnp.ndarray
    hk_l: jnp.ndarray
    rootfr: jnp.ndarray
    h2osoi_ice: jnp.ndarray
    root_biomass: jnp.ndarray
    lai: jnp.ndarray
    itype: jnp.ndarray
    patch_to_column: jnp.ndarray
    minlwp_SPA: float


class SoilResistanceOutputs(NamedTuple):
    """Output state from soil resistance calculations.
    
    Attributes:
        rsoil: Soil hydraulic resistance [MPa·s·m² leaf/mmol H₂O] [n_patches]
        psis: Weighted soil water potential [MPa] [n_patches]
        soil_et_loss: Fractional uptake from each layer [-] [n_patches, n_layers]
    """
    rsoil: jnp.ndarray
    psis: jnp.ndarray
    soil_et_loss: jnp.ndarray


class LeafWaterPotentialInputs(NamedTuple):
    """Inputs for leaf water potential calculation.
    
    Attributes:
        capac_SPA: Plant capacitance [mmol H₂O/m² leaf area/MPa] [n_pfts]
        ncan: Number of aboveground layers [n_patches]
        psis: Weighted soil water potential [MPa] [n_patches]
        dpai: Canopy layer plant area index [m²/m²] [n_patches, n_canopy_layers]
        zs: Canopy layer height for scalar concentration [m] [n_patches, n_canopy_layers]
        lsc: Canopy layer leaf-specific conductance [mmol H₂O/m² leaf/s/MPa] 
             [n_patches, n_canopy_layers]
        trleaf: Leaf transpiration flux [mol H₂O/m² leaf/s] 
                [n_patches, n_canopy_layers, 2]
        lwp: Leaf water potential [MPa] [n_patches, n_canopy_layers, 2]
        itype: Patch vegetation type [n_patches]
        dtime_substep: Model time step [s]
    """
    capac_SPA: jnp.ndarray
    ncan: jnp.ndarray
    psis: jnp.ndarray
    dpai: jnp.ndarray
    zs: jnp.ndarray
    lsc: jnp.ndarray
    trleaf: jnp.ndarray
    lwp: jnp.ndarray
    itype: jnp.ndarray
    dtime_substep: float


class LeafWaterPotentialOutputs(NamedTuple):
    """Outputs from leaf water potential calculation.
    
    Attributes:
        lwp: Updated leaf water potential [MPa] [n_patches, n_canopy_layers, 2]
    """
    lwp: jnp.ndarray


# =============================================================================
# Plant Resistance Functions
# =============================================================================

def plant_resistance(
    inputs: PlantResistanceInput,
) -> PlantResistanceOutput:
    """Calculate whole-plant leaf-specific conductance (soil-to-leaf).
    
    This function computes the hydraulic conductance from soil to leaf for each
    canopy layer. The conductance is the inverse of total resistance, which is
    the sum of soil resistance and aboveground plant resistance.
    
    Key equations (lines 62-66):
        rplant = 1 / gplant_SPA                    [MPa·s·m²/mmol H₂O]
        lsc = 1 / (rsoil + rplant)                 [mmol H₂O/m² leaf/s/MPa]
    
    Fortran reference: MLPlantHydraulicsMod.F90, lines 23-82
    
    Args:
        inputs: Input state containing hydraulic parameters and canopy structure
        
    Returns:
        Output state containing leaf-specific conductance for each canopy layer
        
    Note:
        - Conductance is only calculated for layers with dpai > 0 (line 58)
        - For layers without vegetation (dpai = 0), lsc is set to 0 (line 72)
        - The aboveground resistance uses gplant_SPA as conductance, not conductivity
    """
    n_patches = inputs.dpai.shape[0]
    n_levcan = inputs.dpai.shape[1]
    
    # Get PFT-specific conductance for each patch (line 63)
    gplant_patch = inputs.gplant_SPA[inputs.itype]  # [n_patches]
    
    # Broadcast rsoil and gplant_patch to match canopy layer dimensions
    rsoil_broadcast = inputs.rsoil[:, jnp.newaxis]  # [n_patches, 1]
    gplant_broadcast = gplant_patch[:, jnp.newaxis]  # [n_patches, 1]
    
    # Calculate aboveground plant resistance (line 63)
    # rplant = 1 / gplant_SPA [MPa·s·m²/mmol H₂O]
    rplant = 1.0 / gplant_broadcast
    
    # Calculate leaf-specific conductance (line 66)
    # lsc = 1 / (rsoil + rplant) [mmol H₂O/m² leaf/s/MPa]
    lsc_calculated = 1.0 / (rsoil_broadcast + rplant)
    
    # Only apply to layers with vegetation (dpai > 0) (lines 58, 72)
    has_vegetation = inputs.dpai > 0.0
    lsc = jnp.where(has_vegetation, lsc_calculated, 0.0)
    
    return PlantResistanceOutput(lsc=lsc)


# =============================================================================
# Soil Resistance Functions
# =============================================================================

def _calculate_soil_resistance_per_layer(
    root_radius: jnp.ndarray,
    root_density: jnp.ndarray,
    root_resist: jnp.ndarray,
    dz: jnp.ndarray,
    nbedrock: jnp.ndarray,
    smp_l: jnp.ndarray,
    hk_l: jnp.ndarray,
    rootfr: jnp.ndarray,
    h2osoi_ice: jnp.ndarray,
    root_biomass: jnp.ndarray,
    minlwp_SPA: float,
    n_layers: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate soil hydraulic resistance for each layer.
    
    This function implements the core loop over soil layers (lines 92-218),
    calculating hydraulic properties and maximum transpiration rates.
    
    Key equations:
        - Root length density: RLD = root_biomass / (root_density * cross_section)
        - Root distance: d = sqrt(1 / (RLD * π))
        - Soil-to-root resistance: R1 = ln(d/r) / (2π * RLD * dz * K)
        - Root-to-stem resistance: R2 = root_resist / (root_biomass_density * dz)
        - Total resistance: R = R1 + R2
        - Maximum transpiration: E = (ψ_soil - ψ_min) / R
    
    Fortran reference: MLPlantHydraulicsMod.F90, lines 92-218
    
    Args:
        root_radius: Fine root radius [m] [n_patches]
        root_density: Fine root density [g biomass/m³ root] [n_patches]
        root_resist: Root tissue resistivity [MPa·s·g/mmol H₂O] [n_patches]
        dz: Soil layer thickness [m] [n_columns, n_layers]
        nbedrock: Depth to bedrock index [-] [n_columns]
        smp_l: Soil matric potential [mm] [n_columns, n_layers]
        hk_l: Soil hydraulic conductivity [mm H₂O/s] [n_columns, n_layers]
        rootfr: Fraction of roots in each layer [-] [n_patches, n_layers]
        h2osoi_ice: Soil ice content [kg H₂O/m²] [n_columns, n_layers]
        root_biomass: Fine root biomass [g biomass/m²] [n_patches]
        minlwp_SPA: Minimum leaf water potential [MPa]
        n_layers: Maximum number of soil layers
        
    Returns:
        Tuple containing:
        - rsoil_conductance: Soil conductance sum [mmol H₂O/m² ground/s/MPa] [n_patches]
        - smp_mpa: Soil matric potential [MPa] [n_patches, n_layers]
        - evap: Potential evaporation from each layer [mmol H₂O/m² ground/s] 
                [n_patches, n_layers]
        - totevap: Total potential evaporation [mmol H₂O/m² ground/s] [n_patches]
    """
    n_patches = root_radius.shape[0]
    
    # Root cross-sectional area [m²] (line 158)
    root_cross_sec_area = PI * root_radius**2
    
    # Layer mask based on bedrock depth (line 155)
    layer_indices = jnp.arange(n_layers)
    layer_mask = layer_indices[None, :] < nbedrock[:, None]  # [n_columns, n_layers]
    
    # Convert hydraulic conductivity (lines 168-170)
    # mm/s -> m/s -> m²/s/MPa -> mmol/m/s/MPa
    hk = hk_l * (1.0e-3 / HEAD)  # mm/s -> m²/s/MPa
    hk = hk * DENH2O / MMOL_H2O * 1000.0  # m²/s/MPa -> mmol/m/s/MPa
    
    # Convert matric potential (line 171)
    # mm -> m -> MPa
    smp_mpa = smp_l * 1.0e-3 * HEAD
    
    # Root biomass density [g biomass/m³ soil] (lines 173-175)
    root_biomass_expanded = root_biomass[:, None]  # [n_patches, 1]
    root_biomass_density = root_biomass_expanded * rootfr / dz
    root_biomass_density = jnp.maximum(root_biomass_density, 1.0e-10)
    
    # Root length density [m root/m³ soil] (lines 177-178)
    root_density_expanded = root_density[:, None]
    root_cross_sec_expanded = root_cross_sec_area[:, None]
    root_length_density = root_biomass_density / (
        root_density_expanded * root_cross_sec_expanded
    )
    
    # Distance between roots [m] (lines 180-181)
    root_dist = jnp.sqrt(1.0 / (root_length_density * PI))
    
    # Soil-to-root resistance [MPa·s·m²/mmol H₂O] (lines 183-184)
    root_radius_expanded = root_radius[:, None]
    soilr1 = jnp.log(root_dist / root_radius_expanded) / (
        2.0 * PI * root_length_density * dz * hk
    )
    
    # Root-to-stem resistance [MPa·s·m²/mmol H₂O] (lines 186-187)
    root_resist_expanded = root_resist[:, None]
    soilr2 = root_resist_expanded / (root_biomass_density * dz)
    
    # Total belowground resistance per layer [MPa·s·m²/mmol H₂O] (line 189)
    soilr = soilr1 + soilr2
    
    # Sum conductances (1/resistance) across layers (lines 191-194)
    conductance = jnp.where(layer_mask, 1.0 / soilr, 0.0)
    rsoil_conductance = jnp.sum(conductance, axis=1)
    
    # Maximum transpiration per layer [mmol H₂O/m²/s] (lines 196-200)
    evap = (smp_mpa - minlwp_SPA) / soilr
    evap = jnp.maximum(evap, 0.0)  # No negative transpiration
    
    # Zero out frozen soil (line 199)
    evap = jnp.where(h2osoi_ice > 0.0, 0.0, evap)
    
    # Zero out layers below bedrock
    evap = jnp.where(layer_mask, evap, 0.0)
    
    # Total maximum transpiration (line 200)
    totevap = jnp.sum(evap, axis=1)
    
    return rsoil_conductance, smp_mpa, evap, totevap


def _finalize_soil_resistance(
    rsoil_conductance: jnp.ndarray,
    lai: jnp.ndarray,
    smp_mpa: jnp.ndarray,
    evap: jnp.ndarray,
    totevap: jnp.ndarray,
    minlwp_SPA: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Finalize soil resistance and compute weighted soil water potential.
    
    Converts soil conductance to resistance and calculates the weighted
    soil water potential based on potential evaporation from each layer.
    
    Key equations (lines 221-239):
        rsoil = LAI / conductance_sum
        psis = sum(smp_mpa[j] * evap[j]) / totevap
        soil_et_loss[j] = evap[j] / totevap
    
    Fortran reference: MLPlantHydraulicsMod.F90, lines 219-247
    
    Args:
        rsoil_conductance: Soil conductance sum [mmol H₂O/m² ground/s/MPa] [n_patches]
        lai: Leaf area index [m²/m²] [n_patches]
        smp_mpa: Soil matric potential [MPa] [n_patches, nlayers]
        evap: Potential evaporation from each layer [mmol H₂O/m² ground/s] 
              [n_patches, nlayers]
        totevap: Total potential evaporation [mmol H₂O/m² ground/s] [n_patches]
        minlwp_SPA: Minimum leaf water potential [MPa]
        
    Returns:
        Tuple containing:
        - rsoil: Soil hydraulic resistance [MPa·s·m² leaf/mmol H₂O] [n_patches]
        - psis: Weighted soil water potential [MPa] [n_patches]
        - soil_et_loss: Fractional uptake from each layer [-] [n_patches, nlayers]
    """
    nlayers = evap.shape[1]
    
    # Line 221: Belowground resistance = LAI / conductance
    rsoil = lai / rsoil_conductance
    
    # Lines 226-233: Weighted soil water potential and fractional uptake
    psis_numerator = jnp.sum(smp_mpa * evap, axis=1)
    
    # Fractional uptake from each layer
    totevap_safe = jnp.where(totevap > 0.0, totevap, 1.0)
    soil_et_loss = evap / totevap_safe[:, None]
    
    # When totevap <= 0, set uniform distribution
    uniform_fraction = 1.0 / nlayers
    soil_et_loss = jnp.where(
        totevap[:, None] > 0.0,
        soil_et_loss,
        uniform_fraction
    )
    
    # Lines 235-239: Finalize weighted soil water potential
    psis = jnp.where(
        totevap > 0.0,
        psis_numerator / totevap,
        minlwp_SPA
    )
    
    return rsoil, psis, soil_et_loss


def soil_resistance(
    inputs: SoilResistanceInputs,
) -> SoilResistanceOutputs:
    """Calculate soil hydraulic resistance and water uptake.
    
    This function orchestrates the calculation of soil resistance to water flow
    and the resulting water uptake from each soil layer. It combines the
    per-layer resistance calculations with the final weighted potential.
    
    Fortran reference: MLPlantHydraulicsMod.F90, lines 85-247
    
    Args:
        inputs: Input state containing soil properties and root distribution
        
    Returns:
        SoilResistanceOutputs containing:
            - rsoil: Soil hydraulic resistance [MPa·s·m² leaf/mmol H₂O] [n_patches]
            - psis: Weighted soil water potential [MPa] [n_patches]
            - soil_et_loss: Fractional uptake from each layer [-] [n_patches, n_layers]
    """
    n_layers = inputs.smp_l.shape[1]
    
    # Get PFT-specific parameters for each patch
    root_radius_patch = inputs.root_radius[inputs.itype]
    root_density_patch = inputs.root_density[inputs.itype]
    root_resist_patch = inputs.root_resist[inputs.itype]
    
    # Map column-level arrays to patch level
    patch_to_col = inputs.patch_to_column
    dz_patch = inputs.dz[patch_to_col]
    nbedrock_patch = inputs.nbedrock[patch_to_col]
    smp_l_patch = inputs.smp_l[patch_to_col]
    hk_l_patch = inputs.hk_l[patch_to_col]
    h2osoi_ice_patch = inputs.h2osoi_ice[patch_to_col]
    
    # Calculate per-layer resistance and potential evaporation
    rsoil_conductance, smp_mpa, evap, totevap = _calculate_soil_resistance_per_layer(
        root_radius=root_radius_patch,
        root_density=root_density_patch,
        root_resist=root_resist_patch,
        dz=dz_patch,
        nbedrock=nbedrock_patch,
        smp_l=smp_l_patch,
        hk_l=hk_l_patch,
        rootfr=inputs.rootfr,
        h2osoi_ice=h2osoi_ice_patch,
        root_biomass=inputs.root_biomass,
        minlwp_SPA=inputs.minlwp_SPA,
        n_layers=n_layers,
    )
    
    # Finalize resistance and compute weighted potential
    rsoil, psis, soil_et_loss = _finalize_soil_resistance(
        rsoil_conductance=rsoil_conductance,
        lai=inputs.lai,
        smp_mpa=smp_mpa,
        evap=evap,
        totevap=totevap,
        minlwp_SPA=inputs.minlwp_SPA,
    )
    
    return SoilResistanceOutputs(
        rsoil=rsoil,
        psis=psis,
        soil_et_loss=soil_et_loss,
    )


# =============================================================================
# Leaf Water Potential Functions
# =============================================================================

def leaf_water_potential(
    inputs: LeafWaterPotentialInputs,
    il: int,
) -> LeafWaterPotentialOutputs:
    """Calculate leaf water potential for sunlit or shaded leaves.
    
    This function implements a capacitance-based model for leaf water potential.
    The leaf acts as a capacitor that buffers changes in water potential between
    the soil water supply and transpiration demand.
    
    Key equation (lines 280-284):
        dy/dt = (a - y) / b
        
    Where:
        - y: Leaf water potential [MPa]
        - a: Equilibrium potential [MPa]
        - b: Time constant [s]
        
    The integrated solution over timestep dt is:
        dy = (a - y0) * (1 - exp(-dt/b))
    
    Fortran reference: MLPlantHydraulicsMod.F90, lines 250-320
    
    Args:
        inputs: Input data structure containing all required fields
        il: Leaf index (0 for sunlit, 1 for shaded)
        
    Returns:
        Updated leaf water potential
        
    Note:
        - For layers with no plant area (dpai=0), lwp is set to 0
        - The exponential decay prevents numerical instability
    """
    # Extract current leaf water potential for this leaf type (line 280)
    y0 = inputs.lwp[:, :, il]
    
    # Broadcast PFT-specific capacitance to patch dimension (line 282)
    capac_patch = inputs.capac_SPA[inputs.itype]  # [n_patches]
    capac_broadcast = capac_patch[:, jnp.newaxis]  # [n_patches, 1]
    
    # Calculate equilibrium potential (line 281)
    # a = soil potential - gravitational head - transpiration/conductance
    a = (inputs.psis[:, jnp.newaxis] 
         - HEAD * inputs.zs 
         - 1000.0 * inputs.trleaf[:, :, il] / inputs.lsc)
    
    # Calculate time constant (line 282)
    # b = capacitance / conductance
    b = capac_broadcast / inputs.lsc
    
    # Calculate change in leaf water potential (line 283)
    # dy = (a - y0) * (1 - exp(-dt/b))
    dy = (a - y0) * (1.0 - jnp.exp(-inputs.dtime_substep / b))
    
    # Update leaf water potential (line 284)
    lwp_updated = y0 + dy
    
    # Set to zero where no vegetation (lines 279, 286)
    has_vegetation = inputs.dpai > 0.0
    lwp_updated = jnp.where(has_vegetation, lwp_updated, 0.0)
    
    # Update the full array
    lwp_new = inputs.lwp.at[:, :, il].set(lwp_updated)
    
    return LeafWaterPotentialOutputs(lwp=lwp_new)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Parameters
    'PlantHydraulicsParams',
    
    # Input/Output types
    'PlantResistanceInput',
    'PlantResistanceOutput',
    'SoilResistanceInputs',
    'SoilResistanceOutputs',
    'LeafWaterPotentialInputs',
    'LeafWaterPotentialOutputs',
    
    # Functions
    'plant_resistance',
    'soil_resistance',
    'leaf_water_potential',
]