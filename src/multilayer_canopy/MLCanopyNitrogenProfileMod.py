"""
Canopy Nitrogen Profile Module.

Translated from CTSM's MLCanopyNitrogenProfileMod.F90

This module calculates the vertical profile of nitrogen and photosynthetic
capacity within the canopy. The nitrogen profile affects light absorption
and photosynthesis rates at different canopy layers.

Key concepts:
    - Nitrogen is distributed vertically following leaf area
    - Photosynthetic capacity scales with nitrogen content
    - Profile affects multi-layer canopy radiation and photosynthesis
    - Separate calculations for sunlit and shaded leaves

Key equations:
    Nitrogen decay coefficient:
        kn = exp(0.00963 * vcmax25top - 2.43)  [if kn_val < 0]
        kn = kn_val                             [if kn_val > 0]
    
    Temperature acclimation (if enabled):
        jmax25_to_vcmax25 = 2.59 - 0.035 * min(max(T_acclim - 273.15, 11), 35)
    
    Nitrogen profile:
        fn = exp(-kn * pai_above) * (1 - exp(-kn * dpai)) / kn
        fn_sun = (clump_fac / (kn + kb*clump_fac)) * exp(-kn * pai_above) * tbi 
                 * (1 - exp(-(kn + kb*clump_fac) * dpai))
        fn_sha = fn - fn_sun
    
    Scaling factors:
        nscale_sun = fn_sun / (fracsun * dpai)
        nscale_sha = fn_sha / ((1 - fracsun) * dpai)

References:
    Original Fortran: MLCanopyNitrogenProfileMod.F90, lines 1-184
    Kattge & Knorr (2007) for temperature acclimation
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================

class CanopyNitrogenParams(NamedTuple):
    """Parameters for canopy nitrogen profile calculations.
    
    Attributes:
        acclim_type: Acclimation type (0=none, 1=temperature) [-]
        kn_val: Nitrogen decay coefficient value (negative=calculate) [-]
        jmax25_to_vcmax25_noacclim: Jmax25/Vcmax25 ratio without acclimation [-]
        rd25_to_vcmax25_c3: Rd25/Vcmax25 ratio for C3 plants [-]
        rd25_to_vcmax25_c4: Rd25/Vcmax25 ratio for C4 plants [-]
        kp25_to_vcmax25_c4: Kp25/Vcmax25 ratio for C4 plants [-]
        tfrz: Freezing point of water [K]
    """
    acclim_type: int
    kn_val: float
    jmax25_to_vcmax25_noacclim: float
    rd25_to_vcmax25_c3: float
    rd25_to_vcmax25_c4: float
    kp25_to_vcmax25_c4: float
    tfrz: float


class CanopyTopParams(NamedTuple):
    """Photosynthetic parameters at canopy top.
    
    Attributes:
        vcmax25top: Maximum carboxylation rate at 25C [umol/m2/s]
        jmax25top: C3 maximum electron transport rate at 25C [umol/m2/s]
        rd25top: Leaf respiration rate at 25C [umol CO2/m2/s]
        kp25top: C4 initial slope of CO2 response curve at 25C [mol/m2/s]
        kn: Nitrogen decay coefficient [-]
    """
    vcmax25top: jnp.ndarray  # [n_patches]
    jmax25top: jnp.ndarray   # [n_patches]
    rd25top: jnp.ndarray     # [n_patches]
    kp25top: jnp.ndarray     # [n_patches]
    kn: jnp.ndarray          # [n_patches]


class CanopyNitrogenProfile(NamedTuple):
    """Complete canopy nitrogen profile results.
    
    Attributes:
        vcmax25_leaf_sun: Vcmax at 25C for sunlit leaves [umol/m2/s] [n_patches, n_layers]
        vcmax25_leaf_sha: Vcmax at 25C for shaded leaves [umol/m2/s] [n_patches, n_layers]
        jmax25_leaf_sun: Jmax at 25C for sunlit leaves [umol/m2/s] [n_patches, n_layers]
        jmax25_leaf_sha: Jmax at 25C for shaded leaves [umol/m2/s] [n_patches, n_layers]
        rd25_leaf_sun: Leaf respiration at 25C for sunlit leaves [umol/m2/s] [n_patches, n_layers]
        rd25_leaf_sha: Leaf respiration at 25C for shaded leaves [umol/m2/s] [n_patches, n_layers]
        kp25_leaf_sun: Kp at 25C for sunlit leaves [umol/m2/s] [n_patches, n_layers]
        kp25_leaf_sha: Kp at 25C for shaded leaves [umol/m2/s] [n_patches, n_layers]
        vcmax25_profile: Layer-weighted Vcmax at 25C [umol/m2/s] [n_patches, n_layers]
        jmax25_profile: Layer-weighted Jmax at 25C [umol/m2/s] [n_patches, n_layers]
        rd25_profile: Layer-weighted leaf respiration at 25C [umol/m2/s] [n_patches, n_layers]
        kp25_profile: Layer-weighted Kp at 25C [umol/m2/s] [n_patches, n_layers]
        kn: Nitrogen decay coefficient [-] [n_patches]
    """
    vcmax25_leaf_sun: jnp.ndarray
    vcmax25_leaf_sha: jnp.ndarray
    jmax25_leaf_sun: jnp.ndarray
    jmax25_leaf_sha: jnp.ndarray
    rd25_leaf_sun: jnp.ndarray
    rd25_leaf_sha: jnp.ndarray
    kp25_leaf_sun: jnp.ndarray
    kp25_leaf_sha: jnp.ndarray
    vcmax25_profile: jnp.ndarray
    jmax25_profile: jnp.ndarray
    rd25_profile: jnp.ndarray
    kp25_profile: jnp.ndarray
    kn: jnp.ndarray


class CanopyNitrogenValidation(NamedTuple):
    """Results of canopy nitrogen profile validation.
    
    Attributes:
        numerical: Numerically integrated vcmax25 [umol/m2/s] [n_patches]
        analytical: Analytically integrated vcmax25 [umol/m2/s] [n_patches]
        max_error: Maximum absolute error across patches [scalar]
        is_valid: Whether all patches pass validation [scalar boolean]
    """
    numerical: jnp.ndarray
    analytical: jnp.ndarray
    max_error: jnp.ndarray
    is_valid: jnp.ndarray


# =============================================================================
# Default Parameters
# =============================================================================

def get_default_params(tfrz: float = 273.15) -> CanopyNitrogenParams:
    """Get default canopy nitrogen parameters.
    
    Args:
        tfrz: Freezing point of water [K] (default 273.15)
        
    Returns:
        CanopyNitrogenParams with default values
        
    Note:
        Default values from MLclm_varcon.F90 and MLclm_varctl.F90
    """
    return CanopyNitrogenParams(
        acclim_type=0,  # No acclimation by default
        kn_val=-1.0,    # Calculate from vcmax25top
        jmax25_to_vcmax25_noacclim=2.59,
        rd25_to_vcmax25_c3=0.015,
        rd25_to_vcmax25_c4=0.025,
        kp25_to_vcmax25_c4=0.02,
        tfrz=tfrz,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_jmax25_to_vcmax25_ratio(
    tacclim: jnp.ndarray,
    acclim_type: int,
    jmax25_to_vcmax25_noacclim: float,
    tfrz: float,
) -> jnp.ndarray:
    """Calculate Jmax25/Vcmax25 ratio with optional temperature acclimation.
    
    Fortran lines: 91-99
    
    Args:
        tacclim: Average air temperature for acclimation [K] [n_patches]
        acclim_type: Acclimation type (0=none, 1=temperature) [-]
        jmax25_to_vcmax25_noacclim: Ratio without acclimation [-]
        tfrz: Freezing point of water [K]
        
    Returns:
        Jmax25/Vcmax25 ratio [-] [n_patches]
        
    Note:
        For acclim_type=1, uses equation from Kattge & Knorr (2007):
        ratio = 2.59 - 0.035 * T_acclim_C
        where T_acclim_C is clamped to [11, 35] °C
    """
    if acclim_type == 0:
        # No acclimation - use constant ratio
        return jnp.full_like(tacclim, jmax25_to_vcmax25_noacclim)
    elif acclim_type == 1:
        # Temperature acclimation
        # Convert to Celsius and clamp to [11, 35]
        tacclim_c = tacclim - tfrz
        tacclim_c_clamped = jnp.clip(tacclim_c, 11.0, 35.0)
        jmax25_to_vcmax25_acclim = 2.59 - 0.035 * tacclim_c_clamped
        return jmax25_to_vcmax25_acclim
    else:
        # Invalid acclim_type - return NaN to signal error
        return jnp.full_like(tacclim, jnp.nan)


def calculate_canopy_top_parameters(
    vcmaxpft: jnp.ndarray,
    c3psn: jnp.ndarray,
    tacclim: jnp.ndarray,
    params: CanopyNitrogenParams,
) -> CanopyTopParams:
    """Calculate photosynthetic parameters at canopy top.
    
    Fortran lines: 61-116
    
    This function initializes the photosynthetic parameters (Vcmax25, Jmax25,
    Rd25, Kp25) at the top of the canopy based on PFT-specific values and
    photosynthetic pathway (C3 vs C4).
    
    Args:
        vcmaxpft: PFT-specific Vcmax25 [umol/m2/s] [n_patches]
        c3psn: Photosynthetic pathway (1=C3, 0=C4) [-] [n_patches]
        tacclim: Average air temperature for acclimation [K] [n_patches]
        params: Canopy nitrogen parameters
        
    Returns:
        CanopyTopParams with all top-of-canopy parameters
        
    Note:
        - C3 plants have non-zero Jmax25 and Rd25, zero Kp25
        - C4 plants have zero Jmax25, non-zero Rd25 and Kp25
        - Nitrogen decay coefficient kn can be calculated or specified
    """
    # Vcmax at canopy top (same for C3 and C4)
    vcmax25top = vcmaxpft
    
    # Calculate Jmax25/Vcmax25 ratio with optional acclimation
    jmax25_to_vcmax25 = calculate_jmax25_to_vcmax25_ratio(
        tacclim,
        params.acclim_type,
        params.jmax25_to_vcmax25_noacclim,
        params.tfrz,
    )
    
    # Determine if C3 plant (c3psn == 1)
    is_c3 = jnp.round(c3psn).astype(bool)
    
    # Calculate parameters based on photosynthetic pathway
    # C3 plants: jmax25 = ratio * vcmax25, rd25 = c3_ratio * vcmax25, kp25 = 0
    # C4 plants: jmax25 = 0, rd25 = c4_ratio * vcmax25, kp25 = c4_ratio * vcmax25
    jmax25top = jnp.where(
        is_c3,
        jmax25_to_vcmax25 * vcmax25top,
        0.0,
    )
    
    rd25top = jnp.where(
        is_c3,
        params.rd25_to_vcmax25_c3 * vcmax25top,
        params.rd25_to_vcmax25_c4 * vcmax25top,
    )
    
    kp25top = jnp.where(
        is_c3,
        0.0,
        params.kp25_to_vcmax25_c4 * vcmax25top,
    )
    
    # Calculate nitrogen decay coefficient
    # If kn_val < 0: calculate from vcmax25top
    # If kn_val > 0: use specified value
    kn_calculated = jnp.exp(0.00963 * vcmax25top - 2.43)
    kn = jnp.where(
        params.kn_val < 0.0,
        kn_calculated,
        params.kn_val,
    )
    
    return CanopyTopParams(
        vcmax25top=vcmax25top,
        jmax25top=jmax25top,
        rd25top=rd25top,
        kp25top=kp25top,
        kn=kn,
    )


def calculate_nitrogen_profile(
    top_params: CanopyTopParams,
    dpai: jnp.ndarray,
    kb: jnp.ndarray,
    tbi: jnp.ndarray,
    fracsun: jnp.ndarray,
    clump_fac: jnp.ndarray,
    ncan: jnp.ndarray,
) -> CanopyNitrogenProfile:
    """Calculate vertical profile of photosynthetic parameters through canopy.
    
    This function computes the nitrogen scaling factors for each canopy layer
    and applies them to the top-of-canopy photosynthetic parameters. The
    calculation proceeds from top to bottom of the canopy (reverse order in
    the layer index), accumulating PAI as it goes.
    
    Fortran source: MLCanopyNitrogenProfileMod.F90, lines 117-169
    
    Args:
        top_params: Photosynthetic parameters at canopy top
        dpai: Layer plant area index increment [m2/m2] [n_patches, n_canopy_layers]
        kb: Direct beam extinction coefficient [-] [n_patches, n_canopy_layers]
        tbi: Transmission of direct beam through layer [-] [n_patches, n_canopy_layers]
        fracsun: Sunlit fraction of layer [-] [n_patches, n_canopy_layers]
        clump_fac: Foliage clumping factor [-] [n_patches]
        ncan: Number of canopy layers [n_patches]
        
    Returns:
        CanopyNitrogenProfile containing photosynthetic parameters
        for sunlit and shaded leaves in each layer, plus layer-weighted means.
        
    Note:
        The loop proceeds from top to bottom (ic = ncan down to 1), accumulating
        PAI above each layer. When dpai <= 0, all parameters are set to zero.
        The nitrogen scaling factors (nscale_sun, nscale_sha) are computed from
        the integrated nitrogen profile over each layer.
    """
    n_patches = top_params.vcmax25top.shape[0]
    n_canopy_layers = dpai.shape[1]
    
    # Extract kn from top_params
    kn = top_params.kn
    
    # Compute cumulative PAI above each layer (line 126)
    # pai_above for layer ic = sum of dpai for layers > ic
    # In 0-indexed: pai_above[ic] = sum(dpai[ic+1:])
    pai_above = jnp.flip(jnp.cumsum(jnp.flip(dpai, axis=1), axis=1), axis=1)
    pai_above = jnp.roll(pai_above, 1, axis=1)
    pai_above = pai_above.at[:, 0].set(0.0)
    
    # Expand kn for broadcasting
    kn_expanded = kn[:, jnp.newaxis]
    
    # Compute nitrogen factors (lines 138-143)
    # fn: total nitrogen factor for layer
    fn = jnp.exp(-kn_expanded * pai_above) * (1.0 - jnp.exp(-kn_expanded * dpai)) / kn_expanded
    
    # fn_sun: nitrogen factor for sunlit leaves
    clump_fac_expanded = clump_fac[:, jnp.newaxis]
    kb_clump = kb * clump_fac_expanded
    fn_sun = (clump_fac_expanded / (kn_expanded + kb_clump)) * \
             jnp.exp(-kn_expanded * pai_above) * tbi * \
             (1.0 - jnp.exp(-(kn_expanded + kb_clump) * dpai))
    
    # fn_sha: nitrogen factor for shaded leaves
    fn_sha = fn - fn_sun
    
    # Compute nitrogen scaling factors (lines 144-145)
    # Avoid division by zero when dpai or fracsun is zero
    safe_dpai = jnp.where(dpai > 0.0, dpai, 1.0)
    safe_fracsun = jnp.where(fracsun > 0.0, fracsun, 1.0)
    safe_fracsha = jnp.where((1.0 - fracsun) > 0.0, 1.0 - fracsun, 1.0)
    
    nscale_sun = fn_sun / (safe_fracsun * safe_dpai)
    nscale_sha = fn_sha / (safe_fracsha * safe_dpai)
    
    # Apply scaling to top-of-canopy values (lines 147-150)
    vcmax25top_expanded = top_params.vcmax25top[:, jnp.newaxis]
    jmax25top_expanded = top_params.jmax25top[:, jnp.newaxis]
    rd25top_expanded = top_params.rd25top[:, jnp.newaxis]
    kp25top_expanded = top_params.kp25top[:, jnp.newaxis]
    
    vcmax25_leaf_sun_calc = vcmax25top_expanded * nscale_sun
    vcmax25_leaf_sha_calc = vcmax25top_expanded * nscale_sha
    jmax25_leaf_sun_calc = jmax25top_expanded * nscale_sun
    jmax25_leaf_sha_calc = jmax25top_expanded * nscale_sha
    rd25_leaf_sun_calc = rd25top_expanded * nscale_sun
    rd25_leaf_sha_calc = rd25top_expanded * nscale_sha
    kp25_leaf_sun_calc = kp25top_expanded * nscale_sun
    kp25_leaf_sha_calc = kp25top_expanded * nscale_sha
    
    # Only apply values where dpai > 0 (line 136)
    has_pai = dpai > 0.0
    vcmax25_leaf_sun = jnp.where(has_pai, vcmax25_leaf_sun_calc, 0.0)
    vcmax25_leaf_sha = jnp.where(has_pai, vcmax25_leaf_sha_calc, 0.0)
    jmax25_leaf_sun = jnp.where(has_pai, jmax25_leaf_sun_calc, 0.0)
    jmax25_leaf_sha = jnp.where(has_pai, jmax25_leaf_sha_calc, 0.0)
    rd25_leaf_sun = jnp.where(has_pai, rd25_leaf_sun_calc, 0.0)
    rd25_leaf_sha = jnp.where(has_pai, rd25_leaf_sha_calc, 0.0)
    kp25_leaf_sun = jnp.where(has_pai, kp25_leaf_sun_calc, 0.0)
    kp25_leaf_sha = jnp.where(has_pai, kp25_leaf_sha_calc, 0.0)
    
    # Compute layer-weighted means (lines 160-163)
    vcmax25_profile = vcmax25_leaf_sun * fracsun + vcmax25_leaf_sha * (1.0 - fracsun)
    jmax25_profile = jmax25_leaf_sun * fracsun + jmax25_leaf_sha * (1.0 - fracsun)
    rd25_profile = rd25_leaf_sun * fracsun + rd25_leaf_sha * (1.0 - fracsun)
    kp25_profile = kp25_leaf_sun * fracsun + kp25_leaf_sha * (1.0 - fracsun)
    
    return CanopyNitrogenProfile(
        vcmax25_leaf_sun=vcmax25_leaf_sun,
        vcmax25_leaf_sha=vcmax25_leaf_sha,
        jmax25_leaf_sun=jmax25_leaf_sun,
        jmax25_leaf_sha=jmax25_leaf_sha,
        rd25_leaf_sun=rd25_leaf_sun,
        rd25_leaf_sha=rd25_leaf_sha,
        kp25_leaf_sun=kp25_leaf_sun,
        kp25_leaf_sha=kp25_leaf_sha,
        vcmax25_profile=vcmax25_profile,
        jmax25_profile=jmax25_profile,
        rd25_profile=rd25_profile,
        kp25_profile=kp25_profile,
        kn=kn,
    )


def validate_canopy_nitrogen_profile(
    vcmax25_profile: jnp.ndarray,
    dpai: jnp.ndarray,
    vcmax25top: jnp.ndarray,
    kn: jnp.ndarray,
    lai: jnp.ndarray,
    sai: jnp.ndarray,
    ncan: jnp.ndarray,
    tolerance: float = 1.0e-6,
) -> CanopyNitrogenValidation:
    """Validate canopy nitrogen profile integration.
    
    Checks that the numerical integration of vcmax25 over the canopy layers
    matches the analytical solution from the exponential nitrogen profile.
    
    From CTSM's MLCanopyNitrogenProfileMod.F90, lines 170-182:
    - Line 172: numerical = sum(vcmax25_profile(p,1:ncan(p)) * dpai(p,1:ncan(p)))
    - Line 173: analytical = vcmax25top * (1._r8 - exp(-kn*(lai(p) + sai(p)))) / kn
    - Line 174: if (abs(numerical-analytical) > 1.e-06_r8) then
    
    Args:
        vcmax25_profile: Vcmax25 for each canopy layer [umol/m2/s] [n_patches, n_layers]
        dpai: Incremental plant area index [m2/m2] [n_patches, n_layers]
        vcmax25top: Vcmax25 at canopy top [umol/m2/s] [n_patches]
        kn: Nitrogen decay coefficient [dimensionless] [n_patches]
        lai: Leaf area index [m2/m2] [n_patches]
        sai: Stem area index [m2/m2] [n_patches]
        ncan: Number of canopy layers per patch [n_patches]
        tolerance: Maximum allowed error (default 1e-6)
        
    Returns:
        CanopyNitrogenValidation containing numerical and analytical integrals,
        maximum error, and validation status
        
    Note:
        In the original Fortran, this calls endrun() if validation fails.
        In JAX, we return the validation status and let the caller decide
        how to handle failures (e.g., raise an error outside JIT context).
    """
    # Create mask for valid layers based on ncan
    n_layers = vcmax25_profile.shape[1]
    layer_indices = jnp.arange(n_layers)
    valid_mask = layer_indices[None, :] < ncan[:, None]  # [n_patches, n_layers]
    
    # Numerical integration: sum(vcmax25_profile * dpai) over valid layers
    # Line 172 from original Fortran
    weighted_vcmax = vcmax25_profile * dpai * valid_mask
    numerical = jnp.sum(weighted_vcmax, axis=1)  # [n_patches]
    
    # Analytical integration: vcmax25top * (1 - exp(-kn*(lai+sai))) / kn
    # Line 173 from original Fortran
    total_pai = lai + sai
    exp_term = jnp.exp(-kn * total_pai)
    
    # Handle kn=0 case (though unlikely in practice)
    # When kn->0, (1-exp(-kn*x))/kn -> x
    analytical = jnp.where(
        jnp.abs(kn) > 1.0e-10,
        vcmax25top * (1.0 - exp_term) / kn,
        vcmax25top * total_pai
    )
    
    # Calculate absolute error
    # Line 174 from original Fortran
    abs_error = jnp.abs(numerical - analytical)
    
    # Check if all patches pass validation
    max_error = jnp.max(abs_error)
    is_valid = max_error <= tolerance
    
    return CanopyNitrogenValidation(
        numerical=numerical,
        analytical=analytical,
        max_error=max_error,
        is_valid=is_valid,
    )


# =============================================================================
# Main Public Function
# =============================================================================

def canopy_nitrogen_profile(
    vcmaxpft: jnp.ndarray,
    c3psn: jnp.ndarray,
    tacclim: jnp.ndarray,
    dpai: jnp.ndarray,
    kb: jnp.ndarray,
    tbi: jnp.ndarray,
    fracsun: jnp.ndarray,
    clump_fac: jnp.ndarray,
    ncan: jnp.ndarray,
    lai: jnp.ndarray,
    sai: jnp.ndarray,
    params: CanopyNitrogenParams | None = None,
    validate: bool = True,
) -> tuple[CanopyNitrogenProfile, CanopyNitrogenValidation | None]:
    """Calculate canopy profile of nitrogen and photosynthetic capacity.
    
    This is the main entry point for calculating the vertical distribution
    of photosynthetic parameters through the canopy. The nitrogen profile
    follows an exponential decay with cumulative plant area index from the
    top of the canopy.
    
    Fortran source: MLCanopyNitrogenProfileMod.F90, lines 22-182
    
    Args:
        vcmaxpft: PFT-specific Vcmax25 [umol/m2/s] [n_patches]
        c3psn: Photosynthetic pathway (1=C3, 0=C4) [-] [n_patches]
        tacclim: Average air temperature for acclimation [K] [n_patches]
        dpai: Layer plant area index increment [m2/m2] [n_patches, n_layers]
        kb: Direct beam extinction coefficient [-] [n_patches, n_layers]
        tbi: Transmission of direct beam through layer [-] [n_patches, n_layers]
        fracsun: Sunlit fraction of layer [-] [n_patches, n_layers]
        clump_fac: Foliage clumping factor [-] [n_patches]
        ncan: Number of canopy layers [n_patches]
        lai: Leaf area index [m2/m2] [n_patches]
        sai: Stem area index [m2/m2] [n_patches]
        params: Canopy nitrogen parameters (uses defaults if None)
        validate: Whether to validate numerical integration (default True)
        
    Returns:
        Tuple of (CanopyNitrogenProfile, CanopyNitrogenValidation or None)
        - CanopyNitrogenProfile: Complete nitrogen and photosynthetic profiles
        - CanopyNitrogenValidation: Validation results if validate=True, else None
        
    Note:
        The nitrogen profile affects photosynthesis calculations in the
        multi-layer canopy model. Separate profiles are calculated for
        sunlit and shaded leaves based on the radiation extinction through
        the canopy.
        
    Example:
        >>> profile, validation = canopy_nitrogen_profile(
        ...     vcmaxpft=jnp.array([60.0, 40.0]),
        ...     c3psn=jnp.array([1.0, 0.0]),
        ...     tacclim=jnp.array([298.15, 303.15]),
        ...     dpai=jnp.ones((2, 10)) * 0.5,
        ...     kb=jnp.ones((2, 10)) * 0.5,
        ...     tbi=jnp.ones((2, 10)) * 0.9,
        ...     fracsun=jnp.ones((2, 10)) * 0.5,
        ...     clump_fac=jnp.array([0.85, 0.85]),
        ...     ncan=jnp.array([10, 10]),
        ...     lai=jnp.array([5.0, 4.0]),
        ...     sai=jnp.array([1.0, 1.0]),
        ... )
        >>> print(profile.vcmax25_profile.shape)
        (2, 10)
        >>> print(validation.is_valid)
        True
    """
    # Use default parameters if not provided
    if params is None:
        params = get_default_params()
    
    # Step 1: Calculate top-of-canopy parameters (lines 61-116)
    top_params = calculate_canopy_top_parameters(
        vcmaxpft=vcmaxpft,
        c3psn=c3psn,
        tacclim=tacclim,
        params=params,
    )
    
    # Step 2: Calculate nitrogen profile through canopy (lines 117-169)
    profile = calculate_nitrogen_profile(
        top_params=top_params,
        dpai=dpai,
        kb=kb,
        tbi=tbi,
        fracsun=fracsun,
        clump_fac=clump_fac,
        ncan=ncan,
    )
    
    # Step 3: Validate numerical integration (lines 170-182)
    validation = None
    if validate:
        validation = validate_canopy_nitrogen_profile(
            vcmax25_profile=profile.vcmax25_profile,
            dpai=dpai,
            vcmax25top=top_params.vcmax25top,
            kn=top_params.kn,
            lai=lai,
            sai=sai,
            ncan=ncan,
        )
    
    return profile, validation


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'CanopyNitrogenParams',
    'CanopyTopParams',
    'CanopyNitrogenProfile',
    'CanopyNitrogenValidation',
    'get_default_params',
    'canopy_nitrogen_profile',
    'validate_canopy_nitrogen_profile',
]