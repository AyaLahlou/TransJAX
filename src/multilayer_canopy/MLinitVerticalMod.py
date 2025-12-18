"""
Multilayer Canopy Vertical Structure Initialization.

Translated from CTSM's MLinitVerticalMod.F90

This module provides functions to initialize the vertical structure and profiles
for the multilayer canopy model. It sets up:
- Canopy layer heights and spacing
- Initial vertical profiles of state variables
- Canopy structural parameters

The multilayer canopy divides the canopy into discrete vertical layers to
resolve within-canopy gradients of radiation, temperature, humidity, and
turbulent fluxes.

Key functions:
    - init_vertical_structure: Defines the vertical discretization of the canopy
    - init_vertical_profiles: Initializes state variables for each canopy layer

Physics preserved from MLinitVerticalMod.F90 (lines 1-393)
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax.scipy.special import betainc


# ============================================================================
# Type Definitions
# ============================================================================


class BoundsType(NamedTuple):
    """Bounds for spatial dimensions."""
    begp: int  # Beginning patch index
    endp: int  # Ending patch index
    begc: int  # Beginning column index
    endc: int  # Ending column index
    begg: int  # Beginning gridcell index
    endg: int  # Ending gridcell index


class PatchState(NamedTuple):
    """Patch hierarchy information."""
    column: jnp.ndarray  # Column index for each patch [n_patches]
    gridcell: jnp.ndarray  # Gridcell index for each patch [n_patches]
    itype: jnp.ndarray  # PFT type index [n_patches]


class Atm2LndState(NamedTuple):
    """Atmospheric forcing state at grid cell and column level."""
    forc_u_grc: jnp.ndarray  # Atmospheric wind speed in east direction (m/s) [n_gridcells]
    forc_v_grc: jnp.ndarray  # Atmospheric wind speed in north direction (m/s) [n_gridcells]
    forc_pco2_grc: jnp.ndarray  # Atmospheric CO2 partial pressure (Pa) [n_gridcells]
    forc_t_downscaled_col: jnp.ndarray  # Atmospheric temperature (K) [n_columns]
    forc_q_downscaled_col: jnp.ndarray  # Atmospheric specific humidity (kg/kg) [n_columns]
    forc_pbot_downscaled_col: jnp.ndarray  # Atmospheric pressure (Pa) [n_columns]
    forc_hgt_u: jnp.ndarray  # Atmospheric reference height (m) [n_patches]


class CanopyStateType(NamedTuple):
    """Canopy state variables."""
    htop: jnp.ndarray  # Canopy height [m] [n_patches]
    elai: jnp.ndarray  # Leaf area index of canopy [m2/m2] [n_patches]
    esai: jnp.ndarray  # Stem area index of canopy [m2/m2] [n_patches]


class FrictionVelType(NamedTuple):
    """Friction velocity variables (placeholder for future implementation)."""
    pass


class MLCanopyType(NamedTuple):
    """Multilayer canopy variables."""
    ncan: jnp.ndarray  # Number of aboveground layers [-] [n_patches]
    ntop: jnp.ndarray  # Index for top leaf layer [-] [n_patches]
    nbot: jnp.ndarray  # Index for bottom leaf layer [-] [n_patches]
    zref: jnp.ndarray  # Atmospheric reference height [m] [n_patches]
    ztop: jnp.ndarray  # Canopy foliage top height [m] [n_patches]
    zbot: jnp.ndarray  # Canopy foliage bottom height [m] [n_patches]
    zs: jnp.ndarray  # Canopy layer height for scalar concentration [m] [n_patches, n_layers]
    dz: jnp.ndarray  # Canopy layer thickness [m] [n_patches, n_layers]
    zw: jnp.ndarray  # Heights at layer interfaces [m] [n_patches, n_layers+1]
    dlai: jnp.ndarray  # Canopy layer leaf area index [m2/m2] [n_patches, n_layers]
    dsai: jnp.ndarray  # Canopy layer stem area index [m2/m2] [n_patches, n_layers]
    dlai_frac: jnp.ndarray  # Canopy layer leaf area index (fraction of total) [-] [n_patches, n_layers]
    dsai_frac: jnp.ndarray  # Canopy layer stem area index (fraction of total) [-] [n_patches, n_layers]
    wind: jnp.ndarray  # Canopy layer wind speed [m/s] [n_patches, n_layers]
    tair: jnp.ndarray  # Canopy layer air temperature [K] [n_patches, n_layers]
    eair: jnp.ndarray  # Canopy layer vapor pressure [Pa] [n_patches, n_layers]
    cair: jnp.ndarray  # Canopy layer atmospheric CO2 [umol/mol] [n_patches, n_layers]
    h2ocan: jnp.ndarray  # Canopy layer intercepted water [kg H2O/m2] [n_patches, n_layers]
    taf: jnp.ndarray  # Air temperature at canopy top [K] [n_patches]
    qaf: jnp.ndarray  # Specific humidity at canopy top [kg/kg] [n_patches]
    tg: jnp.ndarray  # Soil surface temperature [K] [n_patches]
    tleaf: jnp.ndarray  # Leaf temperature [K] [n_patches, n_layers, 2]
    lwp: jnp.ndarray  # Leaf water potential [MPa] [n_patches, n_layers, 2]


class PFTParams(NamedTuple):
    """PFT-specific parameters."""
    pbeta_lai: jnp.ndarray  # Parameter for leaf area density beta distribution [-] [n_pfts]
    qbeta_lai: jnp.ndarray  # Parameter for leaf area density beta distribution [-] [n_pfts]
    pbeta_sai: jnp.ndarray  # Parameter for stem area density beta distribution [-] [n_pfts]
    qbeta_sai: jnp.ndarray  # Parameter for stem area density beta distribution [-] [n_pfts]


class MLCanopyParams(NamedTuple):
    """Parameters for multilayer canopy initialization."""
    # Layer spacing parameters (lines 70-119)
    dz_tall: float = 1.0  # Layer spacing for tall canopies [m]
    dz_short: float = 0.5  # Layer spacing for short canopies [m]
    dz_param: float = 10.0  # Height threshold for tall vs short spacing [m]
    nlayer_within: int = 0  # Number of layers within canopy (0 = auto)
    nlayer_above: int = 0  # Number of layers above canopy (0 = auto)
    dpai_min: float = 0.01  # Minimum PAI threshold for a layer [m2/m2]
    
    # Physical constants (lines 318-393)
    mmh2o: float = 18.016  # Molecular weight of water [g/mol]
    mmdry: float = 28.966  # Molecular weight of dry air [g/mol]
    wind_forc_min: float = 0.1  # Minimum wind speed [m/s]
    
    # Leaf indices
    isun: int = 0  # Index for sunlit leaves
    isha: int = 1  # Index for shaded leaves
    
    # Validation tolerances
    lai_tol: float = 1.0e-6  # Tolerance for LAI conservation check [m2/m2]
    sai_tol: float = 1.0e-6  # Tolerance for SAI conservation check [m2/m2]


# ============================================================================
# Helper Functions
# ============================================================================


def beta_distribution_cdf(
    pbeta: float,
    qbeta: float,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate cumulative distribution function of beta distribution.
    
    Uses the incomplete beta function to compute the CDF of a beta distribution
    with shape parameters pbeta and qbeta.
    
    Args:
        pbeta: First shape parameter of beta distribution
        qbeta: Second shape parameter of beta distribution
        x: Value at which to evaluate CDF [0-1]
        
    Returns:
        CDF value at x
        
    Note:
        When pbeta = qbeta = 1, this gives a uniform distribution.
        From MLMathToolsMod.F90 (referenced in lines 196-222)
    """
    return betainc(pbeta, qbeta, x)


# ============================================================================
# Main Functions
# ============================================================================


def calculate_vertical_structure(
    ztop: jnp.ndarray,
    zref: jnp.ndarray,
    nlayer_within: int,
    nlayer_above: int,
    dz_param: float,
    dz_tall: float,
    dz_short: float,
    nlevmlcan: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate vertical layer structure for multilayer canopy.
    
    Determines the number and spacing of canopy layers based on canopy height
    and extends layers to the reference height. Layer spacing adapts to canopy
    height using different increments for tall vs short canopies.
    
    Translated from MLinitVerticalMod.F90 (lines 70-178)
    
    Args:
        ztop: Canopy top height [m] [n_patches]
        zref: Reference height [m] [n_patches]
        nlayer_within: User-specified number of within-canopy layers (0 = auto)
        nlayer_above: User-specified number of above-canopy layers (0 = auto)
        dz_param: Height threshold for tall vs short spacing [m]
        dz_tall: Layer spacing for tall canopies [m]
        dz_short: Layer spacing for short canopies [m]
        nlevmlcan: Maximum number of canopy layers
        
    Returns:
        Tuple containing:
            - ntop: Number of within-canopy layers [n_patches]
            - ncan: Total number of canopy layers [n_patches]
            - zw: Heights at layer interfaces [m] [n_patches, nlevmlcan+1]
            
    Note:
        Layer interfaces zw are defined from ic=0 (ground) to ic=ncan (top).
        For layer ic: zw[ic-1] is bottom, zw[ic] is top.
    """
    n_patches = ztop.shape[0]
    
    # Check if using specified number of layers (lines 103-119)
    use_specified_layers = (nlayer_within > 0) & (nlayer_above > 0)
    
    if use_specified_layers:
        # Use specified number of layers (lines 105-112)
        ntop = jnp.full(n_patches, nlayer_within, dtype=jnp.int32)
        dz_within = ztop / float(nlayer_within)
        nabove = jnp.full(n_patches, nlayer_above, dtype=jnp.int32)
        ncan = ntop + nabove
        dz_above = (zref - ztop) / float(nlayer_above)
    else:
        # Determine layer spacing based on canopy height (lines 125-141)
        dz_within = jnp.where(ztop > dz_param, dz_tall, dz_short)
        
        # Calculate number of within-canopy layers (lines 129-130)
        ntop = jnp.round(ztop / dz_within).astype(jnp.int32)
        ntop = jnp.maximum(ntop, 1)  # At least one layer
        dz_within = ztop / ntop.astype(jnp.float64)
        
        # Calculate distance from canopy top to reference height
        ztop_to_zref = zref - ztop
        
        # Determine number of above-canopy layers (lines 134-141)
        dz_above = dz_within
        nabove = jnp.round(ztop_to_zref / dz_above).astype(jnp.int32)
        nabove = jnp.maximum(nabove, 1)  # At least one layer
        ncan = ntop + nabove
        dz_above = ztop_to_zref / nabove.astype(jnp.float64)
    
    # Initialize layer interface heights array
    zw = jnp.zeros((n_patches, nlevmlcan + 1))
    
    # Calculate within-canopy heights (lines 149-158)
    # Start from top and work down to ground
    for ic in range(nlevmlcan + 1):
        mask = ic <= ntop
        height = ztop - (ntop - ic).astype(jnp.float64) * dz_within
        # Ensure ground level is exactly zero
        height = jnp.where(ic == 0, 0.0, height)
        zw = zw.at[:, ic].set(jnp.where(mask, height, zw[:, ic]))
    
    # Calculate above-canopy heights (lines 162-165)
    # Start from reference height and work down to canopy top
    for ic in range(nlevmlcan + 1):
        mask = (ic > ntop) & (ic <= ncan)
        height = zref - (ncan - ic).astype(jnp.float64) * dz_above
        zw = zw.at[:, ic].set(jnp.where(mask, height, zw[:, ic]))
    
    return ntop, ncan, zw


def calculate_layer_properties(
    zw: jnp.ndarray,
    ztop: jnp.ndarray,
    elai: jnp.ndarray,
    esai: jnp.ndarray,
    ncan: jnp.ndarray,
    ntop: jnp.ndarray,
    itype: jnp.ndarray,
    pft_params: PFTParams,
    n_levels: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate layer properties and distribute plant area indices.
    
    Translated from CTSM's MLinitVerticalMod.F90 (lines 179-233)
    
    This function:
    1. Calculates layer thickness (dz) from interface heights
    2. Determines scalar concentration heights (zs) at layer centers
    3. Distributes LAI vertically using beta distribution
    4. Distributes SAI vertically using beta distribution
    
    Args:
        zw: Heights at layer interfaces [m] [n_patches, n_levels+1]
        ztop: Canopy top height [m] [n_patches]
        elai: Total leaf area index [m2/m2] [n_patches]
        esai: Total stem area index [m2/m2] [n_patches]
        ncan: Number of canopy layers [n_patches]
        ntop: Number of layers with vegetation [n_patches]
        itype: PFT type index [n_patches]
        pft_params: PFT-specific parameters
        n_levels: Maximum number of canopy levels
        
    Returns:
        Tuple containing:
            - dz: Layer thickness [m] [n_patches, n_levels]
            - zs: Scalar concentration height [m] [n_patches, n_levels]
            - dlai: Leaf area index in layer [m2/m2] [n_patches, n_levels]
            - dsai: Stem area index in layer [m2/m2] [n_patches, n_levels]
    """
    n_patches = ztop.shape[0]
    
    # Initialize output arrays
    dz = jnp.zeros((n_patches, n_levels))
    zs = jnp.zeros((n_patches, n_levels))
    dlai = jnp.zeros((n_patches, n_levels))
    dsai = jnp.zeros((n_patches, n_levels))
    
    # Layer indices
    ic_range = jnp.arange(n_levels)
    
    # Thickness of each layer (lines 181-183)
    dz_vals = jnp.where(
        ic_range[None, :] < ncan[:, None],
        zw[:, 1:n_levels+1] - zw[:, :n_levels],
        0.0
    )
    dz = dz_vals
    
    # Scalar concentration heights - physically centered in layer (lines 187-189)
    zs_vals = jnp.where(
        ic_range[None, :] < ncan[:, None],
        0.5 * (zw[:, 1:n_levels+1] + zw[:, :n_levels]),
        0.0
    )
    zs = zs_vals
    
    # Calculate leaf area index using beta distribution (lines 196-211)
    def compute_dlai_for_layer(p_idx, ic):
        """Compute LAI for a single layer of a single patch."""
        pft_idx = itype[p_idx]
        
        # Lower height at bottom of layer
        zrel_bot = jnp.minimum(zw[p_idx, ic] / ztop[p_idx], 1.0)
        beta_cdf_bot = beta_distribution_cdf(
            pft_params.pbeta_lai[pft_idx],
            pft_params.qbeta_lai[pft_idx],
            zrel_bot
        )
        
        # Upper height at top of layer
        zrel_top = jnp.minimum(zw[p_idx, ic + 1] / ztop[p_idx], 1.0)
        beta_cdf_top = beta_distribution_cdf(
            pft_params.pbeta_lai[pft_idx],
            pft_params.qbeta_lai[pft_idx],
            zrel_top
        )
        
        # Leaf area index (m2/m2)
        return (beta_cdf_top - beta_cdf_bot) * elai[p_idx]
    
    # Vectorize over patches and layers
    for p in range(n_patches):
        for ic in range(n_levels):
            if ic < ntop[p]:
                dlai = dlai.at[p, ic].set(compute_dlai_for_layer(p, ic))
    
    # Repeat for stem area index (lines 215-222)
    def compute_dsai_for_layer(p_idx, ic):
        """Compute SAI for a single layer of a single patch."""
        pft_idx = itype[p_idx]
        
        zrel_bot = jnp.minimum(zw[p_idx, ic] / ztop[p_idx], 1.0)
        beta_cdf_bot = beta_distribution_cdf(
            pft_params.pbeta_sai[pft_idx],
            pft_params.qbeta_sai[pft_idx],
            zrel_bot
        )
        
        zrel_top = jnp.minimum(zw[p_idx, ic + 1] / ztop[p_idx], 1.0)
        beta_cdf_top = beta_distribution_cdf(
            pft_params.pbeta_sai[pft_idx],
            pft_params.qbeta_sai[pft_idx],
            zrel_top
        )
        
        return (beta_cdf_top - beta_cdf_bot) * esai[p_idx]
    
    for p in range(n_patches):
        for ic in range(n_levels):
            if ic < ntop[p]:
                dsai = dsai.at[p, ic].set(compute_dsai_for_layer(p, ic))
    
    return dz, zs, dlai, dsai


def redistribute_plant_area(
    dlai: jnp.ndarray,
    dsai: jnp.ndarray,
    ntop: jnp.ndarray,
    zw: jnp.ndarray,
    elai: jnp.ndarray,
    esai: jnp.ndarray,
    dpai_min: float,
    lai_tol: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Redistribute plant area index and determine bottom canopy layer.
    
    This function removes layers with PAI below a minimum threshold and redistributes
    the missing leaf and stem area proportionally across the remaining layers. It then
    determines the lowest layer with vegetation and calculates the bottom canopy height.
    
    From MLinitVerticalMod.F90 lines 234-291.
    
    Args:
        dlai: Leaf area index per layer [m2/m2] [n_patches, n_layers]
        dsai: Stem area index per layer [m2/m2] [n_patches, n_layers]
        ntop: Top layer index [n_patches]
        zw: Layer interface heights [m] [n_patches, n_layers+1]
        elai: Total exposed LAI [m2/m2] [n_patches]
        esai: Total exposed SAI [m2/m2] [n_patches]
        dpai_min: Minimum PAI threshold for a layer [m2/m2]
        lai_tol: Tolerance for LAI conservation check [m2/m2]
        
    Returns:
        Tuple containing:
            - dlai: Redistributed LAI [m2/m2] [n_patches, n_layers]
            - dsai: Redistributed SAI [m2/m2] [n_patches, n_layers]
            - nbot: Bottom layer index [n_patches]
            - zbot: Bottom height of canopy [m] [n_patches]
    """
    n_patches, n_layers = dlai.shape
    
    # Initialize outputs
    dlai_out = dlai.copy()
    dsai_out = dsai.copy()
    nbot = jnp.zeros(n_patches, dtype=jnp.int32)
    zbot = jnp.zeros(n_patches)
    
    # Layer mask for valid layers (up to ntop)
    layer_indices = jnp.arange(n_layers)
    layer_mask = layer_indices[None, :] < ntop[:, None]
    
    # Identify layers with PAI < dpai_min (lines 239-246)
    dpai = dlai + dsai
    below_threshold = (dpai < dpai_min) & layer_mask
    
    # Accumulate missing area
    lai_miss = jnp.sum(jnp.where(below_threshold, dlai, 0.0), axis=1)
    sai_miss = jnp.sum(jnp.where(below_threshold, dsai, 0.0), axis=1)
    
    # Zero out layers below threshold
    dlai_out = jnp.where(below_threshold, 0.0, dlai_out)
    dsai_out = jnp.where(below_threshold, 0.0, dsai_out)
    
    # Redistribute missing LAI proportionally (lines 248-253)
    lai_sum = jnp.sum(dlai_out * layer_mask, axis=1)
    lai_redistrib = jnp.where(
        (lai_miss[:, None] > 0.0) & (lai_sum[:, None] > 0.0),
        lai_miss[:, None] * (dlai_out / lai_sum[:, None]),
        0.0
    )
    dlai_out = dlai_out + lai_redistrib
    
    # Redistribute missing SAI proportionally (lines 255-260)
    sai_sum = jnp.sum(dsai_out * layer_mask, axis=1)
    sai_redistrib = jnp.where(
        (sai_miss[:, None] > 0.0) & (sai_sum[:, None] > 0.0),
        sai_miss[:, None] * (dsai_out / sai_sum[:, None]),
        0.0
    )
    dsai_out = dsai_out + sai_redistrib
    
    # Find the lowest leaf/stem layer (nbot) (lines 262-270)
    dpai_out = dlai_out + dsai_out
    has_vegetation = (dpai_out > 0.0) & layer_mask
    
    # Layer indices (1-based like Fortran)
    layer_indices_1based = jnp.arange(1, n_layers + 1)
    
    # Find maximum layer index with vegetation
    nbot = jnp.max(
        jnp.where(has_vegetation, layer_indices_1based[None, :], 0),
        axis=1
    )
    
    # Bottom height of canopy is at bottom of layer nbot (line 277)
    # zw is 0-indexed, so nbot-1 gives the bottom of layer nbot
    zbot = jnp.where(
        nbot > 0,
        zw[jnp.arange(n_patches), nbot - 1],
        0.0
    )
    
    return dlai_out, dsai_out, nbot, zbot


def finalize_vertical_structure(
    dlai: jnp.ndarray,
    dsai: jnp.ndarray,
    elai: jnp.ndarray,
    esai: jnp.ndarray,
    ntop: jnp.ndarray,
    ncan: jnp.ndarray,
    tolerance: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Finalize vertical structure with validation and normalization.
    
    Validates that the redistributed SAI matches the input SAI, zeros out
    above-canopy layers, and normalizes the LAI/SAI profiles.
    
    From MLinitVerticalMod.F90 lines 292-315.
    
    Args:
        dlai: Layer leaf area index [m2/m2] [n_patches, n_can_max]
        dsai: Layer stem area index [m2/m2] [n_patches, n_can_max]
        elai: Total exposed LAI [m2/m2] [n_patches]
        esai: Total exposed SAI [m2/m2] [n_patches]
        ntop: Index of top canopy layer [1-based] [n_patches]
        ncan: Number of canopy layers [n_patches]
        tolerance: Tolerance for SAI validation [m2/m2]
        
    Returns:
        Tuple containing:
            - dlai: Finalized LAI [m2/m2] [n_patches, n_can_max]
            - dsai: Finalized SAI [m2/m2] [n_patches, n_can_max]
            - dlai_frac: Fractional LAI profile [0-1] [n_patches, n_can_max]
            - dsai_frac: Fractional SAI profile [0-1] [n_patches, n_can_max]
    """
    n_patches, n_can_max = dlai.shape
    
    # Layer indices (1-based)
    layer_indices = jnp.arange(n_can_max) + 1
    
    # Validate SAI redistribution (lines 292-295)
    mask_to_ntop = layer_indices[None, :] <= ntop[:, None]
    sai_sum = jnp.sum(dsai * mask_to_ntop, axis=1)
    sai_err = jnp.abs(sai_sum - esai)
    is_valid = sai_err <= tolerance
    
    # Zero out above-canopy layers (lines 297-301)
    mask_above_canopy = layer_indices[None, :] > ntop[:, None]
    dlai_zeroed = jnp.where(mask_above_canopy, 0.0, dlai)
    dsai_zeroed = jnp.where(mask_above_canopy, 0.0, dsai)
    
    # Normalize profiles (lines 303-307)
    # Avoid division by zero
    elai_safe = jnp.where(elai > 0.0, elai, 1.0)[:, None]
    esai_safe = jnp.where(esai > 0.0, esai, 1.0)[:, None]
    
    dlai_frac = dlai_zeroed / elai_safe
    dsai_frac = dsai_zeroed / esai_safe
    
    # Set fractions to zero where there is no LAI/SAI
    dlai_frac = jnp.where(elai[:, None] > 0.0, dlai_frac, 0.0)
    dsai_frac = jnp.where(esai[:, None] > 0.0, dsai_frac, 0.0)
    
    # Mark invalid patches with NaN
    dlai_frac = jnp.where(is_valid[:, None], dlai_frac, jnp.nan)
    dsai_frac = jnp.where(is_valid[:, None], dsai_frac, jnp.nan)
    dlai_zeroed = jnp.where(is_valid[:, None], dlai_zeroed, jnp.nan)
    dsai_zeroed = jnp.where(is_valid[:, None], dsai_zeroed, jnp.nan)
    
    return dlai_zeroed, dsai_zeroed, dlai_frac, dsai_frac


def init_vertical_structure(
    bounds: BoundsType,
    canopystate_inst: CanopyStateType,
    patch_state: PatchState,
    pft_params: PFTParams,
    params: MLCanopyParams = MLCanopyParams(),
    nlevmlcan: int = 50,
) -> MLCanopyType:
    """Define canopy layer vertical structure.
    
    Initializes the vertical discretization of the multilayer canopy model,
    including layer heights, leaf area distribution, and cumulative profiles.
    
    Translated from MLinitVerticalMod.F90, lines 24-315 (complete subroutine).
    
    Args:
        bounds: Bounds for spatial dimensions (patches, columns, etc.)
        canopystate_inst: Canopy state variables
        patch_state: Patch hierarchy information
        pft_params: PFT-specific parameters
        params: Multilayer canopy parameters
        nlevmlcan: Maximum number of canopy layers
        
    Returns:
        MLCanopyType with initialized vertical structure
        
    Note:
        This function integrates all the inner units:
        - Part 1: Basic setup and layer determination (lines 70-119)
        - Part 2: Layer height calculations (lines 120-178)
        - Part 3: Leaf area distribution (lines 179-233)
        - Part 4: PAI redistribution (lines 234-291)
        - Part 5: Finalization (lines 292-315)
    """
    n_patches = canopystate_inst.htop.shape[0]
    
    # Extract inputs
    ztop = canopystate_inst.htop
    elai = canopystate_inst.elai
    esai = canopystate_inst.esai
    zref = patch_state.forc_hgt_u  # Atmospheric reference height
    itype = patch_state.itype
    
    # Part 1 & 2: Calculate vertical structure (lines 70-178)
    ntop, ncan, zw = calculate_vertical_structure(
        ztop=ztop,
        zref=zref,
        nlayer_within=params.nlayer_within,
        nlayer_above=params.nlayer_above,
        dz_param=params.dz_param,
        dz_tall=params.dz_tall,
        dz_short=params.dz_short,
        nlevmlcan=nlevmlcan,
    )
    
    # Part 3: Calculate layer properties and distribute PAI (lines 179-233)
    dz, zs, dlai, dsai = calculate_layer_properties(
        zw=zw,
        ztop=ztop,
        elai=elai,
        esai=esai,
        ncan=ncan,
        ntop=ntop,
        itype=itype,
        pft_params=pft_params,
        n_levels=nlevmlcan,
    )
    
    # Part 4: Redistribute plant area (lines 234-291)
    dlai, dsai, nbot, zbot = redistribute_plant_area(
        dlai=dlai,
        dsai=dsai,
        ntop=ntop,
        zw=zw,
        elai=elai,
        esai=esai,
        dpai_min=params.dpai_min,
        lai_tol=params.lai_tol,
    )
    
    # Part 5: Finalize structure (lines 292-315)
    dlai, dsai, dlai_frac, dsai_frac = finalize_vertical_structure(
        dlai=dlai,
        dsai=dsai,
        elai=elai,
        esai=esai,
        ntop=ntop,
        ncan=ncan,
        tolerance=params.sai_tol,
    )
    
    # Initialize placeholder arrays for profiles (will be set by init_vertical_profiles)
    wind = jnp.zeros((n_patches, nlevmlcan))
    tair = jnp.zeros((n_patches, nlevmlcan))
    eair = jnp.zeros((n_patches, nlevmlcan))
    cair = jnp.zeros((n_patches, nlevmlcan))
    h2ocan = jnp.zeros((n_patches, nlevmlcan))
    taf = jnp.zeros(n_patches)
    qaf = jnp.zeros(n_patches)
    tg = jnp.zeros(n_patches)
    tleaf = jnp.zeros((n_patches, nlevmlcan, 2))
    lwp = jnp.zeros((n_patches, nlevmlcan, 2))
    
    return MLCanopyType(
        ncan=ncan,
        ntop=ntop,
        nbot=nbot,
        zref=zref,
        ztop=ztop,
        zbot=zbot,
        zs=zs,
        dz=dz,
        zw=zw,
        dlai=dlai,
        dsai=dsai,
        dlai_frac=dlai_frac,
        dsai_frac=dsai_frac,
        wind=wind,
        tair=tair,
        eair=eair,
        cair=cair,
        h2ocan=h2ocan,
        taf=taf,
        qaf=qaf,
        tg=tg,
        tleaf=tleaf,
        lwp=lwp,
    )


def init_vertical_profiles(
    atm2lnd_inst: Atm2LndState,
    mlcanopy_inst: MLCanopyType,
    patch_state: PatchState,
    params: MLCanopyParams = MLCanopyParams(),
) -> MLCanopyType:
    """Initialize vertical profiles and canopy states.
    
    Sets initial conditions for atmospheric profiles (wind, temperature, humidity,
    CO2) and canopy states (leaf temperature, leaf water potential, intercepted
    water) at each canopy layer. All profiles are initialized from atmospheric
    forcing at the top of the canopy.
    
    Fortran source: MLinitVerticalMod.F90, lines 318-393
    
    Args:
        atm2lnd_inst: Atmospheric forcing state
        mlcanopy_inst: Multilayer canopy state to initialize
        patch_state: Patch hierarchy information (column, gridcell indices)
        params: Physical constants and parameters
        
    Returns:
        Updated MLCanopyType with initialized profiles
        
    Note:
        - Wind speed is computed from u and v components with minimum threshold
        - Vapor pressure is computed from specific humidity using molecular weights
        - CO2 concentration is converted from partial pressure to ppm
        - Initial leaf water potential is set to -0.1 MPa
        - Intercepted water is initialized to zero
    """
    n_patches = mlcanopy_inst.ncan.shape[0]
    n_layers = mlcanopy_inst.wind.shape[1]
    
    # Get column and gridcell indices for each patch
    col_idx = patch_state.column
    grid_idx = patch_state.gridcell
    
    # Extract atmospheric forcing at patch locations
    forc_u = atm2lnd_inst.forc_u_grc[grid_idx]
    forc_v = atm2lnd_inst.forc_v_grc[grid_idx]
    forc_pco2 = atm2lnd_inst.forc_pco2_grc[grid_idx]
    forc_t = atm2lnd_inst.forc_t_downscaled_col[col_idx]
    forc_q = atm2lnd_inst.forc_q_downscaled_col[col_idx]
    forc_pbot = atm2lnd_inst.forc_pbot_downscaled_col[col_idx]
    
    # Compute wind speed from components (line 368)
    wind_speed = jnp.maximum(
        params.wind_forc_min,
        jnp.sqrt(forc_u * forc_u + forc_v * forc_v)
    )
    
    # Compute vapor pressure from specific humidity (line 370)
    mmh2o_mmdry = params.mmh2o / params.mmdry
    vapor_pressure = (
        forc_q * forc_pbot / (mmh2o_mmdry + (1.0 - mmh2o_mmdry) * forc_q)
    )
    
    # Compute CO2 concentration in ppm (line 371)
    co2_concentration = forc_pco2 / forc_pbot * 1.0e6
    
    # Initialize profiles by broadcasting to all layers (lines 368-376)
    wind = jnp.broadcast_to(wind_speed[:, None], (n_patches, n_layers))
    tair = jnp.broadcast_to(forc_t[:, None], (n_patches, n_layers))
    eair = jnp.broadcast_to(vapor_pressure[:, None], (n_patches, n_layers))
    cair = jnp.broadcast_to(co2_concentration[:, None], (n_patches, n_layers))
    
    # Initialize leaf temperatures to air temperature (line 373)
    tleaf = jnp.broadcast_to(forc_t[:, None, None], (n_patches, n_layers, 2))
    
    # Initialize leaf water potential to -0.1 MPa (line 374)
    lwp = jnp.full((n_patches, n_layers, 2), -0.1)
    
    # Initialize intercepted water to zero (line 375)
    h2ocan = jnp.zeros((n_patches, n_layers))
    
    # Initialize canopy top and soil surface conditions (lines 378-380)
    taf = forc_t
    qaf = forc_q
    tg = forc_t
    
    # Return updated state
    return mlcanopy_inst._replace(
        wind=wind,
        tair=tair,
        eair=eair,
        cair=cair,
        h2ocan=h2ocan,
        taf=taf,
        qaf=qaf,
        tg=tg,
        tleaf=tleaf,
        lwp=lwp,
    )