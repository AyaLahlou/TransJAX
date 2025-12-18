"""
Solar Radiation Transfer Through Multilayer Canopy.

Translated from CTSM's MLSolarRadiationMod.F90

This module calculates solar radiation transfer through the multilayer canopy
using either Norman radiative transfer or two-stream approximation methods.

Key physics:
    - Radiative transfer through layered canopy
    - Absorption and scattering by leaves
    - Direct and diffuse beam separation
    - Sunlit and shaded leaf fractions

Methods:
    1. Norman (1979): Tridiagonal matrix solution for diffuse radiation
    2. Two-Stream: Integrated solution with depth-varying optical properties

References:
    - Norman, J.M. (1979). Modeling the complete crop canopy.
    - Sellers, P.J. (1985). Canopy reflectance, photosynthesis and transpiration.

Fortran source: MLSolarRadiationMod.F90 (lines 1-789)
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class BoundsType(NamedTuple):
    """Domain decomposition bounds.
    
    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
        begg: Beginning gridcell index
        endg: Ending gridcell index
    """
    begp: int
    endp: int
    begg: int
    endg: int


class MLCanopyState(NamedTuple):
    """Multilayer canopy state variables.
    
    Attributes:
        dlai_profile: Layer leaf area index [m2/m2] [n_patches, nlevmlcan]
        dsai_profile: Layer stem area index [m2/m2] [n_patches, nlevmlcan]
        dpai_profile: Layer plant area index [m2/m2] [n_patches, nlevmlcan]
        ntop_canopy: Index for top leaf layer [n_patches]
        nbot_canopy: Index for bottom leaf layer [n_patches]
        ncan_canopy: Number of aboveground layers [n_patches]
    """
    dlai_profile: jnp.ndarray
    dsai_profile: jnp.ndarray
    dpai_profile: jnp.ndarray
    ntop_canopy: jnp.ndarray
    nbot_canopy: jnp.ndarray
    ncan_canopy: jnp.ndarray


class PatchState(NamedTuple):
    """Patch-level state variables.
    
    Attributes:
        itype: PFT type index [n_patches]
        cosz: Cosine of solar zenith angle [-] [n_patches]
        swskyb: Direct beam solar radiation [W/m2] [n_patches, numrad]
        swskyd: Diffuse solar radiation [W/m2] [n_patches, numrad]
        albsoib: Direct beam albedo of ground [-] [n_patches, numrad]
        albsoid: Diffuse albedo of ground [-] [n_patches, numrad]
    """
    itype: jnp.ndarray
    cosz: jnp.ndarray
    swskyb: jnp.ndarray
    swskyd: jnp.ndarray
    albsoib: jnp.ndarray
    albsoid: jnp.ndarray


class PFTParams(NamedTuple):
    """PFT-specific optical parameters.
    
    Attributes:
        rhol: Leaf reflectance [n_pfts, numrad]
        taul: Leaf transmittance [n_pfts, numrad]
        rhos: Stem reflectance [n_pfts, numrad]
        taus: Stem transmittance [n_pfts, numrad]
        xl: Leaf/stem orientation index [-] [n_pfts]
        clump_fac: Foliage clumping factor [-] [n_pfts]
    """
    rhol: jnp.ndarray
    taul: jnp.ndarray
    rhos: jnp.ndarray
    taus: jnp.ndarray
    xl: jnp.ndarray
    clump_fac: jnp.ndarray


class OpticalProperties(NamedTuple):
    """Canopy layer optical properties.
    
    Attributes:
        rho: Leaf/stem reflectance [n_patches, nlevmlcan, numrad]
        tau: Leaf/stem transmittance [n_patches, nlevmlcan, numrad]
        omega: Leaf/stem scattering coefficient [n_patches, nlevmlcan, numrad]
        kb: Direct beam extinction coefficient [n_patches, nlevmlcan]
        fracsun: Canopy layer sunlit fraction [n_patches, nlevmlcan]
        tb: Canopy layer transmittance of direct beam [n_patches, nlevmlcan]
        td: Canopy layer transmittance of diffuse [n_patches, nlevmlcan]
        tbi: Cumulative transmittance of direct beam [n_patches, nlevmlcan+1]
        avmu: Average inverse diffuse optical depth [n_patches, nlevmlcan]
        betab: Upscatter parameter for direct beam [n_patches, nlevmlcan, numrad]
        betad: Upscatter parameter for diffuse [n_patches, nlevmlcan, numrad]
    """
    rho: jnp.ndarray
    tau: jnp.ndarray
    omega: jnp.ndarray
    kb: jnp.ndarray
    fracsun: jnp.ndarray
    tb: jnp.ndarray
    td: jnp.ndarray
    tbi: jnp.ndarray
    avmu: jnp.ndarray
    betab: jnp.ndarray
    betad: jnp.ndarray


class RadiationFluxes(NamedTuple):
    """Radiation fluxes and absorption.
    
    Attributes:
        swleaf: Absorbed radiation per unit leaf area [W/m2 leaf] 
                [n_patches, nlevmlcan, 2, numrad] (2 = sunlit/shaded)
        swsoi: Solar radiation absorbed by soil [W/m2] [n_patches, numrad]
        swveg: Total canopy absorption [W/m2] [n_patches, numrad]
        swvegsun: Sunlit canopy absorption [W/m2] [n_patches, numrad]
        swvegsha: Shaded canopy absorption [W/m2] [n_patches, numrad]
        albcan: Canopy albedo [-] [n_patches, numrad]
        apar_sun: APAR for sunlit leaves [umol/m2/s] [n_patches, nlevmlcan]
        apar_shade: APAR for shaded leaves [umol/m2/s] [n_patches, nlevmlcan]
    """
    swleaf: jnp.ndarray
    swsoi: jnp.ndarray
    swveg: jnp.ndarray
    swvegsun: jnp.ndarray
    swvegsha: jnp.ndarray
    albcan: jnp.ndarray
    apar_sun: jnp.ndarray
    apar_shade: jnp.ndarray


# =============================================================================
# PARAMETERS AND CONSTANTS
# =============================================================================

# Physical constants
PI = jnp.pi

# Radiation parameters (from MLclm_varcon)
CHIL_MIN = -0.4  # Minimum leaf angle parameter
CHIL_MAX = 0.6   # Maximum leaf angle parameter
KB_MAX = 20.0    # Maximum extinction coefficient
J_TO_UMOL = 4.6  # Conversion from W/m2 to umol photons/m2/s

# Numerical tolerances
MIN_PHI = 1e-10      # Minimum value for phi functions
MIN_OMEGA = 1e-10    # Minimum scattering coefficient
MIN_OPTICAL = 1e-06  # Minimum reflectance/transmittance

# Radiation band indices
IVIS = 0  # Visible band index
INIR = 1  # Near-infrared band index
NUMRAD = 2  # Number of radiation bands

# Leaf type indices
ISUN = 0  # Sunlit leaf index
ISHA = 1  # Shaded leaf index


# =============================================================================
# OPTICAL PROPERTIES INITIALIZATION
# =============================================================================

def initialize_optical_properties(
    patch_state: PatchState,
    mlcanopy_state: MLCanopyState,
    pft_params: PFTParams,
    n_patches: int,
    nlevmlcan: int,
    numrad: int = NUMRAD,
) -> OpticalProperties:
    """Initialize canopy layer optical properties.
    
    Calculates weighted reflectance and transmittance for each canopy layer
    based on leaf and stem area indices.
    
    Fortran source: lines 36-135
    
    Args:
        patch_state: Patch-level state variables
        mlcanopy_state: Multilayer canopy state variables
        pft_params: PFT-specific parameters including rhol, taul, rhos, taus
        n_patches: Number of patches
        nlevmlcan: Number of canopy layers
        numrad: Number of radiation wavebands
        
    Returns:
        OpticalProperties containing initialized arrays for radiation transfer
    """
    # Initialize all arrays to zero
    rho = jnp.zeros((n_patches, nlevmlcan, numrad))
    tau = jnp.zeros((n_patches, nlevmlcan, numrad))
    omega = jnp.zeros((n_patches, nlevmlcan, numrad))
    kb = jnp.zeros((n_patches, nlevmlcan))
    fracsun = jnp.zeros((n_patches, nlevmlcan))
    tb = jnp.zeros((n_patches, nlevmlcan))
    td = jnp.zeros((n_patches, nlevmlcan))
    tbi = jnp.zeros((n_patches, nlevmlcan + 1))
    avmu = jnp.zeros((n_patches, nlevmlcan))
    betab = jnp.zeros((n_patches, nlevmlcan, numrad))
    betad = jnp.zeros((n_patches, nlevmlcan, numrad))
    
    # Get PFT indices for each patch
    pft_indices = patch_state.itype
    
    # Extract LAI, SAI, PAI profiles
    dlai = mlcanopy_state.dlai_profile
    dsai = mlcanopy_state.dsai_profile
    dpai = mlcanopy_state.dpai_profile
    
    # Get canopy layer indices
    ntop = mlcanopy_state.ntop_canopy
    nbot = mlcanopy_state.nbot_canopy
    
    # Avoid division by zero in dpai
    dpai_safe = jnp.where(dpai > 0.0, dpai, 1.0)
    
    # Weight by leaf and stem fractions
    wl = dlai / dpai_safe  # Leaf fraction
    ws = dsai / dpai_safe  # Stem fraction
    
    # Calculate reflectance and transmittance for each waveband
    for ib in range(numrad):
        # Get PFT-specific optical properties
        rhol_pft = pft_params.rhol[pft_indices, ib]
        taul_pft = pft_params.taul[pft_indices, ib]
        rhos_pft = pft_params.rhos[pft_indices, ib]
        taus_pft = pft_params.taus[pft_indices, ib]
        
        # Broadcast to [n_patches, nlevmlcan]
        rhol_layer = rhol_pft[:, jnp.newaxis]
        taul_layer = taul_pft[:, jnp.newaxis]
        rhos_layer = rhos_pft[:, jnp.newaxis]
        taus_layer = taus_pft[:, jnp.newaxis]
        
        # Weighted reflectance
        rho_layer = rhol_layer * wl + rhos_layer * ws
        rho_layer = jnp.maximum(rho_layer, MIN_OPTICAL)
        
        # Weighted transmittance
        tau_layer = taul_layer * wl + taus_layer * ws
        tau_layer = jnp.maximum(tau_layer, MIN_OPTICAL)
        
        # Scattering coefficient
        omega_layer = rho_layer + tau_layer
        
        # Store in arrays
        rho = rho.at[:, :, ib].set(rho_layer)
        tau = tau.at[:, :, ib].set(tau_layer)
        omega = omega.at[:, :, ib].set(omega_layer)
    
    # Mask out layers outside active canopy range
    layer_indices = jnp.arange(nlevmlcan)
    is_active = (layer_indices[jnp.newaxis, :] >= (nlevmlcan - nbot[:, jnp.newaxis])) & \
                (layer_indices[jnp.newaxis, :] <= (nlevmlcan - ntop[:, jnp.newaxis]))
    
    # Apply mask
    is_active_3d = is_active[:, :, jnp.newaxis]
    rho = jnp.where(is_active_3d, rho, 0.0)
    tau = jnp.where(is_active_3d, tau, 0.0)
    omega = jnp.where(is_active_3d, omega, 0.0)
    
    return OpticalProperties(
        rho=rho,
        tau=tau,
        omega=omega,
        kb=kb,
        fracsun=fracsun,
        tb=tb,
        td=td,
        tbi=tbi,
        avmu=avmu,
        betab=betab,
        betad=betad,
    )


# =============================================================================
# BEAM EXTINCTION AND TRANSMITTANCE
# =============================================================================

def calculate_beam_extinction_and_transmittance(
    xl: jnp.ndarray,
    solar_zen: jnp.ndarray,
    dpai: jnp.ndarray,
    clump_fac: jnp.ndarray,
    itype: jnp.ndarray,
    ntop: jnp.ndarray,
    n_layers: int,
    chil_min: float = CHIL_MIN,
    chil_max: float = CHIL_MAX,
    kb_max: float = KB_MAX,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate direct beam extinction coefficient and transmittances.
    
    Computes the extinction coefficient for direct beam radiation and various
    transmittance terms needed for radiation transfer through the canopy.
    
    Fortran source: lines 136-186
    
    Args:
        xl: Leaf/stem orientation index [-] [n_pfts]
        solar_zen: Solar zenith angle [radians] [n_patches]
        dpai: Layer plant area index [m2/m2] [n_patches, n_layers]
        clump_fac: Foliage clumping factor [-] [n_pfts]
        itype: PFT type index [n_patches]
        ntop: Index of top canopy layer [n_patches]
        n_layers: Number of canopy layers
        chil_min: Minimum leaf angle parameter
        chil_max: Maximum leaf angle parameter
        kb_max: Maximum extinction coefficient
        
    Returns:
        kb: Direct beam extinction coefficient [1/m] [n_patches, n_layers]
        tb: Direct beam transmittance through layer [-] [n_patches, n_layers]
        td: Diffuse transmittance through layer [-] [n_patches, n_layers]
        tbi: Unscattered direct beam transmittance onto layer [-] [n_patches, n_layers]
        fracsun: Sunlit fraction of layer [-] [n_patches, n_layers]
    """
    n_patches = solar_zen.shape[0]
    
    # Initialize output arrays
    kb = jnp.zeros((n_patches, n_layers))
    tb = jnp.zeros((n_patches, n_layers))
    td = jnp.zeros((n_patches, n_layers))
    tbi = jnp.zeros((n_patches, n_layers))
    fracsun = jnp.zeros((n_patches, n_layers))
    
    # Get PFT-specific parameters
    chil = xl[itype]
    clump = clump_fac[itype]
    
    # Constrain chil to valid range
    chil = jnp.clip(chil, chil_min, chil_max)
    chil = jnp.where(jnp.abs(chil) <= 0.01, 0.01, chil)
    
    # Calculate phi1 and phi2
    phi1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    
    # Process each layer
    def process_layer(carry: jnp.ndarray, ic: int) -> Tuple[jnp.ndarray, Tuple]:
        """Process a single canopy layer."""
        tbi_prev = carry
        
        # Direct beam extinction coefficient
        gdir = phi1 + phi2 * jnp.cos(solar_zen)
        kb_ic = gdir / jnp.cos(solar_zen)
        kb_ic = jnp.minimum(kb_ic, kb_max)
        
        # Direct beam transmittance through single layer
        tb_ic = jnp.exp(-kb_ic * dpai[:, ic] * clump)
        
        # Diffuse transmittance through single layer
        # Integrate over 9 sky angles from 5° to 85° in 10° increments
        angles = jnp.arange(1, 10) * 10.0 - 5.0  # [5, 15, 25, ..., 85] degrees
        angles_rad = angles * PI / 180.0
        
        def integrate_angle(angle_rad: float) -> jnp.ndarray:
            """Integrate diffuse transmittance for one sky angle."""
            gdirj = phi1 + phi2 * jnp.cos(angle_rad)
            term = jnp.exp(-gdirj / jnp.cos(angle_rad) * dpai[:, ic] * clump)
            term = term * jnp.sin(angle_rad) * jnp.cos(angle_rad)
            return term
        
        td_terms = jax.vmap(integrate_angle)(angles_rad)
        td_ic = jnp.sum(td_terms, axis=0) * 2.0 * (10.0 * PI / 180.0)
        
        # Transmittance of unscattered direct beam onto layer
        is_top = (ic == ntop)
        tbi_ic = jnp.where(
            is_top,
            1.0,
            tbi_prev * jnp.exp(-kb_ic * dpai[:, ic] * clump)
        )
        
        # Sunlit fraction of layer
        kb_dpai = kb_ic * dpai[:, ic]
        kb_dpai_safe = jnp.where(kb_dpai > 1e-10, kb_dpai, 1e-10)
        
        fracsun_ic = (tbi_ic / kb_dpai_safe) * (
            1.0 - jnp.exp(-kb_ic * clump * dpai[:, ic])
        )
        fracsun_ic = jnp.clip(fracsun_ic, 1e-10, 1.0 - 1e-10)
        
        return tbi_ic, (kb_ic, tb_ic, td_ic, tbi_ic, fracsun_ic)
    
    # Process layers from top to bottom
    tbi_init = jnp.ones(n_patches)
    layer_indices = jnp.arange(n_layers, dtype=jnp.int32)
    _, layer_results = jax.lax.scan(process_layer, tbi_init, layer_indices)
    
    kb, tb, td, tbi, fracsun = layer_results
    
    # Transpose to get [n_patches, n_layers] shape
    kb = jnp.transpose(kb)
    tb = jnp.transpose(tb)
    td = jnp.transpose(td)
    tbi = jnp.transpose(tbi)
    fracsun = jnp.transpose(fracsun)
    
    return kb, tb, td, tbi, fracsun


# =============================================================================
# TWO-STREAM PARAMETERS
# =============================================================================

def calculate_twostream_parameters(
    phi1: jnp.ndarray,
    phi2: jnp.ndarray,
    rho: jnp.ndarray,
    tau: jnp.ndarray,
    omega: jnp.ndarray,
    chil: jnp.ndarray,
    gdir: jnp.ndarray,
    solar_zen: jnp.ndarray,
    kb: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate two-stream radiative transfer parameters.
    
    Computes the special parameters needed for two-stream radiative transfer:
    - avmu: average inverse diffuse optical depth per unit leaf area
    - betad: upscatter parameter for diffuse radiation
    - betab: upscatter parameter for direct beam radiation
    
    Fortran source: lines 187-206
    
    Args:
        phi1: Ross-Goudriaan function phi1 [n_patches, n_canopy_layers]
        phi2: Ross-Goudriaan function phi2 [n_patches, n_canopy_layers]
        rho: Leaf reflectance [n_patches, n_canopy_layers, n_bands]
        tau: Leaf transmittance [n_patches, n_canopy_layers, n_bands]
        omega: Leaf scattering coefficient [n_patches, n_canopy_layers, n_bands]
        chil: Leaf angle distribution parameter [n_patches, n_canopy_layers]
        gdir: Relative projected area of leaf elements [n_patches, n_canopy_layers]
        solar_zen: Solar zenith angle [radians] [n_patches]
        kb: Direct beam extinction coefficient [n_patches, n_canopy_layers]
        
    Returns:
        avmu: Average inverse diffuse optical depth [n_patches, n_canopy_layers]
        betad: Upscatter for diffuse radiation [n_patches, n_canopy_layers, n_bands]
        betab: Upscatter for direct beam [n_patches, n_canopy_layers, n_bands]
    """
    # Avoid division by zero and log of zero
    phi1_safe = jnp.maximum(phi1, MIN_PHI)
    phi2_safe = jnp.maximum(phi2, MIN_PHI)
    
    # avmu - average inverse diffuse optical depth per unit leaf area
    log_term = jnp.log((phi1_safe + phi2_safe) / phi1_safe)
    avmu = (1.0 - phi1_safe / phi2_safe * log_term) / phi2_safe
    
    # Expand dimensions for broadcasting
    avmu_expanded = avmu[:, :, jnp.newaxis]
    kb_expanded = kb[:, :, jnp.newaxis]
    chil_expanded = chil[:, :, jnp.newaxis]
    gdir_expanded = gdir[:, :, jnp.newaxis]
    solar_zen_expanded = solar_zen[:, jnp.newaxis, jnp.newaxis]
    
    # betad - upscatter parameter for diffuse radiation
    omega_safe = jnp.maximum(omega, MIN_OMEGA)
    chil_term = ((1.0 + chil_expanded) / 2.0) ** 2
    betad = 0.5 / omega_safe * (
        rho + tau + (rho - tau) * chil_term
    )
    
    # betab - upscatter parameter for direct beam radiation
    cos_zen = jnp.cos(solar_zen_expanded)
    phi2_expanded = phi2[:, :, jnp.newaxis]
    phi1_expanded = phi1[:, :, jnp.newaxis]
    
    tmp0 = gdir_expanded + phi2_expanded * cos_zen
    tmp1 = phi1_expanded * cos_zen
    
    tmp0_safe = jnp.maximum(jnp.abs(tmp0), MIN_PHI)
    tmp1_safe = jnp.maximum(jnp.abs(tmp1), MIN_PHI)
    
    tmp2 = 1.0 - tmp1_safe / tmp0_safe * jnp.log((tmp1_safe + tmp0_safe) / tmp1_safe)
    asu = 0.5 * omega * gdir_expanded / tmp0_safe * tmp2
    
    denominator = omega_safe * avmu_expanded * kb_expanded
    denominator_safe = jnp.maximum(jnp.abs(denominator), MIN_OMEGA * MIN_PHI)
    
    betab = (1.0 + avmu_expanded * kb_expanded) / denominator_safe * asu
    
    return avmu, betad, betab


def convert_apar_to_umol(
    swleaf_sun_vis: jnp.ndarray,
    swleaf_shade_vis: jnp.ndarray,
    j_to_umol: float = J_TO_UMOL,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert absorbed PAR from W/m2 to umol/m2/s.
    
    Fortran source: lines 229-236
    
    Args:
        swleaf_sun_vis: Absorbed visible radiation for sunlit leaves [W/m2]
        swleaf_shade_vis: Absorbed visible radiation for shaded leaves [W/m2]
        j_to_umol: Conversion factor from W/m2 to umol/m2/s
        
    Returns:
        apar_sun: APAR for sunlit leaves [umol/m2/s]
        apar_shade: APAR for shaded leaves [umol/m2/s]
    """
    apar_sun = swleaf_sun_vis * j_to_umol
    apar_shade = swleaf_shade_vis * j_to_umol
    
    return apar_sun, apar_shade


# =============================================================================
# NORMAN RADIATIVE TRANSFER
# =============================================================================

def tridiag(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    n: jnp.ndarray,
) -> jnp.ndarray:
    """Solve tridiagonal system of equations.
    
    Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    
    Args:
        a: Lower diagonal [n_patches, max_equations]
        b: Main diagonal [n_patches, max_equations]
        c: Upper diagonal [n_patches, max_equations]
        d: Right-hand side [n_patches, max_equations]
        n: Number of equations per patch [n_patches]
        
    Returns:
        x: Solution vector [n_patches, max_equations]
    """
    n_patches = a.shape[0]
    max_equations = a.shape[1]
    
    # Initialize solution
    x = jnp.zeros_like(d)
    
    # Forward elimination
    cp = jnp.zeros_like(c)
    dp = jnp.zeros_like(d)
    
    # First equation
    cp = cp.at[:, 0].set(c[:, 0] / b[:, 0])
    dp = dp.at[:, 0].set(d[:, 0] / b[:, 0])
    
    # Remaining equations
    for i in range(1, max_equations):
        denom = b[:, i] - a[:, i] * cp[:, i-1]
        denom = jnp.where(jnp.abs(denom) > 1e-20, denom, 1e-20)
        
        cp = cp.at[:, i].set(c[:, i] / denom)
        dp = dp.at[:, i].set((d[:, i] - a[:, i] * dp[:, i-1]) / denom)
    
    # Back substitution
    # Last equation
    x = x.at[:, -1].set(dp[:, -1])
    
    # Remaining equations
    for i in range(max_equations - 2, -1, -1):
        x = x.at[:, i].set(dp[:, i] - cp[:, i] * x[:, i+1])
    
    return x


def norman_radiation_transfer(
    swskyb: jnp.ndarray,
    swskyd: jnp.ndarray,
    optical_props: OpticalProperties,
    mlcanopy_state: MLCanopyState,
    patch_state: PatchState,
    pft_params: PFTParams,
    nlevmlcan: int,
    numrad: int = NUMRAD,
) -> RadiationFluxes:
    """Compute solar radiation transfer using Norman (1979) method.
    
    This method solves the radiative transfer equations for diffuse radiation
    using a tridiagonal matrix system.
    
    Fortran source: lines 243-526
    
    Args:
        swskyb: Atmospheric direct beam solar radiation [W/m2]
        swskyd: Atmospheric diffuse solar radiation [W/m2]
        optical_props: Optical properties from initialization
        mlcanopy_state: Multilayer canopy state
        patch_state: Patch state
        pft_params: PFT parameters
        nlevmlcan: Number of canopy layers
        numrad: Number of radiation bands
        
    Returns:
        RadiationFluxes containing all radiation fluxes and absorption
    """
    n_patches = swskyb.shape[0]
    max_equations = 2 * (nlevmlcan + 1)
    
    # Create patch indices for advanced indexing
    patch_indices = jnp.arange(n_patches)
    
    # Initialize flux arrays
    swup = jnp.zeros((n_patches, nlevmlcan + 1, numrad))
    swdn = jnp.zeros((n_patches, nlevmlcan + 1, numrad))
    swleaf = jnp.zeros((n_patches, nlevmlcan, 2, numrad))
    
    # Initialize tridiagonal coefficient arrays
    atri = jnp.zeros((n_patches, max_equations, numrad))
    btri = jnp.zeros((n_patches, max_equations, numrad))
    ctri = jnp.zeros((n_patches, max_equations, numrad))
    dtri = jnp.zeros((n_patches, max_equations, numrad))
    
    # Extract needed variables
    rho = optical_props.rho
    tau = optical_props.tau
    tb = optical_props.tb
    td = optical_props.td
    tbi = optical_props.tbi
    dpai = mlcanopy_state.dpai_profile
    fracsun = optical_props.fracsun
    ntop = mlcanopy_state.ntop_canopy
    nbot = mlcanopy_state.nbot_canopy
    albsoib = patch_state.albsoib
    albsoid = patch_state.albsoid
    
    # Process each radiation band
    for ib in range(numrad):
        # Soil: upward flux
        m = 0
        atri = atri.at[:, m, ib].set(0.0)
        btri = btri.at[:, m, ib].set(1.0)
        ctri = ctri.at[:, m, ib].set(-albsoid[:, ib])
        dtri = dtri.at[:, m, ib].set(swskyb[:, ib] * tbi[:, 0] * albsoib[:, ib])
        
        # Soil: downward flux
        m = 1
        td_nbot = td[patch_indices, nbot]
        rho_nbot_ib = rho[patch_indices, nbot, ib]
        tau_nbot_ib = tau[patch_indices, nbot, ib]
        tb_nbot = tb[patch_indices, nbot]
        tbi_nbot = tbi[patch_indices, nbot]
        
        refld = (1.0 - td_nbot) * rho_nbot_ib
        trand = (1.0 - td_nbot) * tau_nbot_ib + td_nbot
        aic = refld - trand * trand / refld
        bic = trand / refld
        
        atri = atri.at[:, m, ib].set(-aic)
        btri = btri.at[:, m, ib].set(1.0)
        ctri = ctri.at[:, m, ib].set(-bic)
        dtri = dtri.at[:, m, ib].set(
            swskyb[:, ib] * tbi_nbot * (1.0 - tb_nbot) * 
            (tau_nbot_ib - rho_nbot_ib * bic)
        )
        
        # Leaf layers (excluding top)
        for ic_offset in range(nlevmlcan - 1):
            ic = nbot + ic_offset
            layer_exists = ic < ntop
            
            # Upward flux
            m = 2 + 2 * ic_offset
            td_ic = td[patch_indices, ic]
            rho_ic_ib = rho[patch_indices, ic, ib]
            tau_ic_ib = tau[patch_indices, ic, ib]
            tb_ic = tb[patch_indices, ic]
            tbi_ic = tbi[patch_indices, ic]
            
            refld = (1.0 - td_ic) * rho_ic_ib
            trand = (1.0 - td_ic) * tau_ic_ib + td_ic
            fic = refld - trand * trand / refld
            eic = trand / refld
            
            atri = jnp.where(
                layer_exists[:, None, None],
                atri.at[:, m, ib].set(-eic),
                atri
            )
            btri = jnp.where(
                layer_exists[:, None, None],
                btri.at[:, m, ib].set(1.0),
                btri
            )
            ctri = jnp.where(
                layer_exists[:, None, None],
                ctri.at[:, m, ib].set(-fic),
                ctri
            )
            dtri = jnp.where(
                layer_exists[:, None, None],
                dtri.at[:, m, ib].set(
                    swskyb[:, ib] * tbi_ic * (1.0 - tb_ic) * 
                    (rho_ic_ib - tau_ic_ib * eic)
                ),
                dtri
            )
            
            # Downward flux
            m = 3 + 2 * ic_offset
            ic_plus_1 = ic + 1
            td_ic_plus_1 = td[patch_indices, ic_plus_1]
            rho_ic_plus_1_ib = rho[patch_indices, ic_plus_1, ib]
            tau_ic_plus_1_ib = tau[patch_indices, ic_plus_1, ib]
            tb_ic_plus_1 = tb[patch_indices, ic_plus_1]
            tbi_ic_plus_1 = tbi[patch_indices, ic_plus_1]
            
            refld = (1.0 - td_ic_plus_1) * rho_ic_plus_1_ib
            trand = (1.0 - td_ic_plus_1) * tau_ic_plus_1_ib + td_ic_plus_1
            aic = refld - trand * trand / refld
            bic = trand / refld
            
            atri = jnp.where(
                layer_exists[:, None, None],
                atri.at[:, m, ib].set(-aic),
                atri
            )
            btri = jnp.where(
                layer_exists[:, None, None],
                btri.at[:, m, ib].set(1.0),
                btri
            )
            ctri = jnp.where(
                layer_exists[:, None, None],
                ctri.at[:, m, ib].set(-bic),
                ctri
            )
            dtri = jnp.where(
                layer_exists[:, None, None],
                dtri.at[:, m, ib].set(
                    swskyb[:, ib] * tbi_ic_plus_1 * (1.0 - tb_ic_plus_1) * 
                    (tau_ic_plus_1_ib - rho_ic_plus_1_ib * bic)
                ),
                dtri
            )
        
        # Top canopy layer
        ic = ntop
        
        # Upward flux
        m = m + 1
        td_ic = td[patch_indices, ic]
        rho_ic_ib = rho[patch_indices, ic, ib]
        tau_ic_ib = tau[patch_indices, ic, ib]
        tb_ic = tb[patch_indices, ic]
        tbi_ic = tbi[patch_indices, ic]
        
        refld = (1.0 - td_ic) * rho_ic_ib
        trand = (1.0 - td_ic) * tau_ic_ib + td_ic
        fic = refld - trand * trand / refld
        eic = trand / refld
        
        atri = atri.at[patch_indices, m, ib].set(-eic)
        btri = btri.at[patch_indices, m, ib].set(1.0)
        ctri = ctri.at[patch_indices, m, ib].set(-fic)
        
        rhs = swskyb[:, ib] * tbi_ic * (1.0 - tb_ic) * (rho_ic_ib - tau_ic_ib * eic)
        dtri = dtri.at[patch_indices, m, ib].set(rhs)
        
        # Downward flux
        m = m + 1
        atri = atri.at[patch_indices, m, ib].set(0.0)
        btri = btri.at[patch_indices, m, ib].set(1.0)
        ctri = ctri.at[patch_indices, m, ib].set(0.0)
        dtri = dtri.at[patch_indices, m, ib].set(swskyd[:, ib])
        
        # Solve tridiagonal system
        m_array = jnp.full(n_patches, m)
        utri = tridiag(atri[:, :, ib], btri[:, :, ib], ctri[:, :, ib], 
                      dtri[:, :, ib], m_array)
        
        # Extract solution to flux arrays
        m_extract = 0
        
        # Soil fluxes
        m_extract = m_extract + 1
        swup = swup.at[patch_indices, 0, ib].set(
            utri[patch_indices, m_extract]
        )
        m_extract = m_extract + 1
        swdn = swdn.at[patch_indices, 0, ib].set(
            utri[patch_indices, m_extract]
        )
        
        # Leaf layer fluxes
        for ic_layer in range(1, nlevmlcan + 1):
            valid = (ic_layer >= nbot) & (ic_layer <= ntop)
            
            m_extract = m_extract + 1
            swup_val = jnp.where(
                valid,
                utri[patch_indices, m_extract],
                swup[patch_indices, ic_layer, ib]
            )
            swup = swup.at[patch_indices, ic_layer, ib].set(swup_val)
            
            m_extract = m_extract + 1
            swdn_val = jnp.where(
                valid,
                utri[patch_indices, m_extract],
                swdn[patch_indices, ic_layer, ib]
            )
            swdn = swdn.at[patch_indices, ic_layer, ib].set(swdn_val)
    
    # Calculate absorption by soil and vegetation
    swsoi = jnp.zeros((n_patches, numrad))
    swveg = jnp.zeros((n_patches, numrad))
    swvegsun = jnp.zeros((n_patches, numrad))
    swvegsha = jnp.zeros((n_patches, numrad))
    
    for ib in range(numrad):
        # Solar radiation absorbed by ground
        swbeam_soil = tbi[:, 0] * swskyb[:, ib]
        swabsb_soil = swbeam_soil * (1.0 - albsoib[:, ib])
        swabsd_soil = swdn[:, 0, ib] * (1.0 - albsoid[:, ib])
        swsoi = swsoi.at[:, ib].set(swabsb_soil + swabsd_soil)
        
        # Loop over canopy layers
        for ic in range(nlevmlcan):
            active_mask = (ic >= nbot) & (ic <= ntop)
            
            swbeam = tbi[:, ic] * swskyb[:, ib]
            swabsb = swbeam * (1.0 - tb[:, ic]) * (1.0 - optical_props.omega[:, ic, ib])
            
            icm1 = jnp.where(ic == nbot, 0, ic - 1)
            swup_below = jnp.take_along_axis(
                swup[:, :, ib], 
                icm1[:, None], 
                axis=1
            ).squeeze(axis=1)
            swabsd = (swdn[:, ic, ib] + swup_below) * (1.0 - td[:, ic]) * \
                     (1.0 - optical_props.omega[:, ic, ib])
            
            swsha = swabsd * (1.0 - fracsun[:, ic])
            swsun = swabsd * fracsun[:, ic] + swabsb
            
            # Per unit leaf area
            eps = 1e-10
            fracsun_safe = jnp.maximum(fracsun[:, ic], eps)
            fracsha_safe = jnp.maximum(1.0 - fracsun[:, ic], eps)
            dpai_safe = jnp.maximum(dpai[:, ic], eps)
            
            swleaf_sun = swsun / (fracsun_safe * dpai_safe)
            swleaf_sha = swsha / (fracsha_safe * dpai_safe)
            
            swleaf_sun = jnp.where(active_mask, swleaf_sun, 0.0)
            swleaf_sha = jnp.where(active_mask, swleaf_sha, 0.0)
            
            swleaf = swleaf.at[:, ic, ISUN, ib].set(swleaf_sun)
            swleaf = swleaf.at[:, ic, ISHA, ib].set(swleaf_sha)
            
            # Sum vegetation absorption
            swabsb_masked = jnp.where(active_mask, swabsb, 0.0)
            swabsd_masked = jnp.where(active_mask, swabsd, 0.0)
            swsun_masked = jnp.where(active_mask, swsun, 0.0)
            swsha_masked = jnp.where(active_mask, swsha, 0.0)
            
            swveg = swveg.at[:, ib].add(swabsb_masked + swabsd_masked)
            swvegsun = swvegsun.at[:, ib].add(swsun_masked)
            swvegsha = swvegsha.at[:, ib].add(swsha_masked)
    
    # Calculate canopy albedo
    suminc = swskyb + swskyd
    swup_ntop = swup[patch_indices, ntop, :]
    sumref = swup_ntop * swskyb + swup_ntop * swskyd
    albcan = jnp.where(suminc > 0.0, sumref / suminc, 0.0)
    
    # Convert visible band to APAR
    apar_sun, apar_shade = convert_apar_to_umol(
        swleaf[:, :, ISUN, IVIS],
        swleaf[:, :, ISHA, IVIS]
    )
    
    return RadiationFluxes(
        swleaf=swleaf,
        swsoi=swsoi,
        swveg=swveg,
        swvegsun=swvegsun,
        swvegsha=swvegsha,
        albcan=albcan,
        apar_sun=apar_sun,
        apar_shade=apar_shade,
    )


# =============================================================================
# TWO-STREAM RADIATIVE TRANSFER
# =============================================================================

def twostream_radiation_transfer(
    swskyb: jnp.ndarray,
    swskyd: jnp.ndarray,
    optical_props: OpticalProperties,
    mlcanopy_state: MLCanopyState,
    patch_state: PatchState,
    pft_params: PFTParams,
    nlevmlcan: int,
    numrad: int = NUMRAD,
) -> RadiationFluxes:
    """Compute solar radiation transfer using two-stream approximation.
    
    This method provides an integrated solution over each layer with
    depth-varying optical properties.
    
    Fortran source: lines 529-789
    
    Args:
        swskyb: Atmospheric direct beam solar radiation [W/m2]
        swskyd: Atmospheric diffuse solar radiation [W/m2]
        optical_props: Optical properties from initialization
        mlcanopy_state: Multilayer canopy state
        patch_state: Patch state
        pft_params: PFT parameters
        nlevmlcan: Number of canopy layers
        numrad: Number of radiation bands
        
    Returns:
        RadiationFluxes containing all radiation fluxes and absorption
    """
    n_patches = swskyb.shape[0]
    
    # Create patch indices for advanced indexing
    patch_indices = jnp.arange(n_patches)
    
    # Extract needed variables
    omega = optical_props.omega
    betad = optical_props.betad
    betab = optical_props.betab
    avmu = optical_props.avmu
    kb = optical_props.kb
    dpai = mlcanopy_state.dpai_profile
    tbi = optical_props.tbi
    fracsun = optical_props.fracsun
    ntop = mlcanopy_state.ntop_canopy
    nbot = mlcanopy_state.nbot_canopy
    albsoib = patch_state.albsoib
    albsoid = patch_state.albsoid
    clump_fac = pft_params.clump_fac[patch_state.itype]
    
    # Initialize flux arrays
    iupwb0 = jnp.zeros((n_patches, nlevmlcan, numrad))
    iupwb = jnp.zeros((n_patches, nlevmlcan, numrad))
    idwnb = jnp.zeros((n_patches, nlevmlcan, numrad))
    iabsb = jnp.zeros((n_patches, nlevmlcan, numrad))
    iabsb_sun = jnp.zeros((n_patches, nlevmlcan, numrad))
    iabsb_sha = jnp.zeros((n_patches, nlevmlcan, numrad))
    
    iupwd0 = jnp.zeros((n_patches, nlevmlcan, numrad))
    iupwd = jnp.zeros((n_patches, nlevmlcan, numrad))
    idwnd = jnp.zeros((n_patches, nlevmlcan, numrad))
    iabsd = jnp.zeros((n_patches, nlevmlcan, numrad))
    iabsd_sun = jnp.zeros((n_patches, nlevmlcan, numrad))
    iabsd_sha = jnp.zeros((n_patches, nlevmlcan, numrad))
    
    # Initialize albedos below current layer
    albb_below = albsoib.copy()
    albd_below = albsoid.copy()
    
    # Constants for unit radiation
    unitb = 1.0
    unitd = 1.0
    
    # Process each radiation band
    for ib in range(numrad):
        # Process each layer from bottom to top
        for ic in range(nlevmlcan):
            # Common terms
            b = (1.0 - (1.0 - betad[:, ic, ib]) * omega[:, ic, ib]) / avmu[:, ic]
            c = betad[:, ic, ib] * omega[:, ic, ib] / avmu[:, ic]
            h = jnp.sqrt(b * b - c * c)
            u = (h - b - c) / (2.0 * h)
            v = (h + b + c) / (2.0 * h)
            d = omega[:, ic, ib] * kb[:, ic] * unitb / (h * h - kb[:, ic] * kb[:, ic])
            g1 = (betab[:, ic, ib] * kb[:, ic] - b * betab[:, ic, ib] - 
                  c * (1.0 - betab[:, ic, ib])) * d
            g2 = ((1.0 - betab[:, ic, ib]) * kb[:, ic] + c * betab[:, ic, ib] + 
                  b * (1.0 - betab[:, ic, ib])) * d
            s1 = jnp.exp(-h * clump_fac * dpai[:, ic])
            s2 = jnp.exp(-kb[:, ic] * clump_fac * dpai[:, ic])
            
            # Terms for direct beam radiation
            num1 = v * (g1 + g2 * albd_below[:, ib] + albb_below[:, ib] * unitb) * s2
            num2 = g2 * (u + v * albd_below[:, ib]) * s1
            den1 = v * (v + u * albd_below[:, ib]) / s1
            den2 = u * (u + v * albd_below[:, ib]) * s1
            n2b = (num1 - num2) / (den1 - den2)
            n1b = (g2 - n2b * u) / v
            
            a1b = (-g1 * (1.0 - s2 * s2) / (2.0 * kb[:, ic]) +
                   n1b * u * (1.0 - s2 * s1) / (kb[:, ic] + h) +
                   n2b * v * (1.0 - s2 / s1) / (kb[:, ic] - h))
            a2b = (g2 * (1.0 - s2 * s2) / (2.0 * kb[:, ic]) -
                   n1b * v * (1.0 - s2 * s1) / (kb[:, ic] + h) -
                   n2b * u * (1.0 - s2 / s1) / (kb[:, ic] - h))
            a1b = a1b * tbi[:, ic]
            a2b = a2b * tbi[:, ic]
            
            # Direct beam radiative fluxes
            iupwb0_ic = -g1 + n1b * u + n2b * v
            iupwb_ic = -g1 * s2 + n1b * u * s1 + n2b * v / s1
            idwnb_ic = g2 * s2 - n1b * v * s1 - n2b * u / s1
            iabsb_ic = unitb * (1.0 - s2) - iupwb0_ic + iupwb_ic - idwnb_ic
            iabsb_sun_ic = ((1.0 - omega[:, ic, ib]) * 
                            ((1.0 - s2) * unitb + 
                             clump_fac / avmu[:, ic] * (a1b + a2b)))
            iabsb_sha_ic = iabsb_ic - iabsb_sun_ic
            
            # Only update for layers within canopy
            in_canopy = (ic >= ntop) & (ic <= nbot)
            
            iupwb0 = iupwb0.at[:, ic, ib].set(
                jnp.where(in_canopy, iupwb0_ic, iupwb0[:, ic, ib])
            )
            iupwb = iupwb.at[:, ic, ib].set(
                jnp.where(in_canopy, iupwb_ic, iupwb[:, ic, ib])
            )
            idwnb = idwnb.at[:, ic, ib].set(
                jnp.where(in_canopy, idwnb_ic, idwnb[:, ic, ib])
            )
            iabsb = iabsb.at[:, ic, ib].set(
                jnp.where(in_canopy, iabsb_ic, iabsb[:, ic, ib])
            )
            iabsb_sun = iabsb_sun.at[:, ic, ib].set(
                jnp.where(in_canopy, iabsb_sun_ic, iabsb_sun[:, ic, ib])
            )
            iabsb_sha = iabsb_sha.at[:, ic, ib].set(
                jnp.where(in_canopy, iabsb_sha_ic, iabsb_sha[:, ic, ib])
            )
            
            # Terms for diffuse radiation
            num1 = unitd * (u + v * albd_below[:, ib]) * s1
            den1 = v * (v + u * albd_below[:, ib]) / s1
            den2 = u * (u + v * albd_below[:, ib]) * s1
            n2d = num1 / (den1 - den2)
            n1d = -(unitd + n2d * u) / v
            
            a1d = (n1d * u * (1.0 - s2 * s1) / (kb[:, ic] + h) +
                   n2d * v * (1.0 - s2 / s1) / (kb[:, ic] - h))
            a2d = (-n1d * v * (1.0 - s2 * s1) / (kb[:, ic] + h) -
                   n2d * u * (1.0 - s2 / s1) / (kb[:, ic] - h))
            a1d = a1d * tbi[:, ic]
            a2d = a2d * tbi[:, ic]
            
            # Diffuse radiative fluxes
            iupwd0_ic = n1d * u + n2d * v
            iupwd_ic = n1d * u * s1 + n2d * v / s1
            idwnd_ic = -n1d * v * s1 - n2d * u / s1
            iabsd_ic = unitd - iupwd0_ic + iupwd_ic - idwnd_ic
            iabsd_sun_ic = ((1.0 - omega[:, ic, ib]) * clump_fac / avmu[:, ic] * 
                            (a1d + a2d))
            iabsd_sha_ic = iabsd_ic - iabsd_sun_ic
            
            # Update diffuse fluxes
            iupwd0 = iupwd0.at[:, ic, ib].set(
                jnp.where(in_canopy, iupwd0_ic, iupwd0[:, ic, ib])
            )
            iupwd = iupwd.at[:, ic, ib].set(
                jnp.where(in_canopy, iupwd_ic, iupwd[:, ic, ib])
            )
            idwnd = idwnd.at[:, ic, ib].set(
                jnp.where(in_canopy, idwnd_ic, idwnd[:, ic, ib])
            )
            iabsd = iabsd.at[:, ic, ib].set(
                jnp.where(in_canopy, iabsd_ic, iabsd[:, ic, ib])
            )
            iabsd_sun = iabsd_sun.at[:, ic, ib].set(
                jnp.where(in_canopy, iabsd_sun_ic, iabsd_sun[:, ic, ib])
            )
            iabsd_sha = iabsd_sha.at[:, ic, ib].set(
                jnp.where(in_canopy, iabsd_sha_ic, iabsd_sha[:, ic, ib])
            )
            
            # Update albedos for next layer
            albb_below = albb_below.at[:, ib].set(
                jnp.where(in_canopy, iupwb0_ic, albb_below[:, ib])
            )
            albd_below = albd_below.at[:, ib].set(
                jnp.where(in_canopy, iupwd0_ic, albd_below[:, ib])
            )
    
    # Calculate final fluxes
    swleaf = jnp.zeros((n_patches, nlevmlcan, 2, numrad))
    
    # Initialize with incident radiation at top
    dir = swskyb
    dif = swskyd
    
    # Process each layer from top to bottom
    for ic in range(nlevmlcan - 1, -1, -1):
        for ib in range(numrad):
            # Absorption by canopy layer (W/m2 leaf)
            fracsun_dpai = fracsun[:, ic] * dpai[:, ic]
            fracsun_dpai = jnp.where(fracsun_dpai > 0, fracsun_dpai, 1.0)
            
            sun = (iabsb_sun[:, ic, ib] * dir[:, ib] + 
                   iabsd_sun[:, ic, ib] * dif[:, ib]) / fracsun_dpai
            
            fracsha_dpai = (1.0 - fracsun[:, ic]) * dpai[:, ic]
            fracsha_dpai = jnp.where(fracsha_dpai > 0, fracsha_dpai, 1.0)
            
            sha = (iabsb_sha[:, ic, ib] * dir[:, ib] + 
                   iabsd_sha[:, ic, ib] * dif[:, ib]) / fracsha_dpai
            
            swleaf = swleaf.at[:, ic, ISUN, ib].set(sun)
            swleaf = swleaf.at[:, ic, ISHA, ib].set(sha)
            
            # Update radiation for next layer
            dif_new = dir[:, ib] * idwnb[:, ic, ib] + dif[:, ib] * idwnd[:, ic, ib]
            dir_new = dir[:, ib] * jnp.exp(-kb[:, ic] * clump_fac * dpai[:, ic])
            
            dir = dir.at[:, ib].set(dir_new)
            dif = dif.at[:, ib].set(dif_new)
    
    # Solar radiation absorbed by ground
    swsoi = dir * (1.0 - albsoib) + dif * (1.0 - albsoid)
    
    # Canopy albedo
    suminc = swskyb + swskyd
    sumref = iupwb0[:, 0, :] * swskyb + iupwd0[:, 0, :] * swskyd
    albcan = jnp.where(suminc > 0.0, sumref / suminc, 0.0)
    
    # Sum canopy absorption
    swveg = jnp.zeros((n_patches, numrad))
    swvegsun = jnp.zeros((n_patches, numrad))
    swvegsha = jnp.zeros((n_patches, numrad))
    
    for ic in range(nlevmlcan):
        fracsun_ic = fracsun[:, ic:ic+1]
        dpai_ic = dpai[:, ic:ic+1]
        sun = swleaf[:, ic, ISUN, :] * fracsun_ic * dpai_ic
        sha = swleaf[:, ic, ISHA, :] * (1.0 - fracsun_ic) * dpai_ic
        swveg = swveg + sun + sha
        swvegsun = swvegsun + sun
        swvegsha = swvegsha + sha
    
    # Convert visible band to APAR
    apar_sun, apar_shade = convert_apar_to_umol(
        swleaf[:, :, ISUN, IVIS],
        swleaf[:, :, ISHA, IVIS]
    )
    
    return RadiationFluxes(
        swleaf=swleaf,
        swsoi=swsoi,
        swveg=swveg,
        swvegsun=swvegsun,
        swvegsha=swvegsha,
        albcan=albcan,
        apar_sun=apar_sun,
        apar_shade=apar_shade,
    )


# =============================================================================
# MAIN DRIVER
# =============================================================================

def solar_radiation(
    bounds: BoundsType,
    num_filter: int,
    filter_indices: jnp.ndarray,
    patch_state: PatchState,
    mlcanopy_state: MLCanopyState,
    pft_params: PFTParams,
    nlevmlcan: int,
    light_type: int = 1,
    numrad: int = NUMRAD,
) -> RadiationFluxes:
    """Calculate solar radiation transfer through canopy.
    
    Main driver for multilayer canopy solar radiation calculations.
    This function orchestrates the radiation transfer computation through
    the canopy layers, calculating absorption, transmission, and reflection
    for both direct beam and diffuse radiation.
    
    Fortran source: lines 30-240
    
    Args:
        bounds: Bounds type containing patch indices
        num_filter: Number of patches in filter [scalar]
        filter_indices: Indices of patches to process [num_filter]
        patch_state: Patch-level state variables
        mlcanopy_state: Multilayer canopy state variables
        pft_params: PFT-specific parameters
        nlevmlcan: Number of canopy layers
        light_type: Radiation transfer method (1=Norman, 2=TwoStream)
        numrad: Number of radiation bands
        
    Returns:
        RadiationFluxes containing computed radiation fields
        
    Note:
        - light_type=1: Norman (1979) tridiagonal matrix method
        - light_type=2: Two-stream approximation
    """
    n_patches = patch_state.cosz.shape[0]
    
    # Initialize optical properties
    optical_props = initialize_optical_properties(
        patch_state,
        mlcanopy_state,
        pft_params,
        n_patches,
        nlevmlcan,
        numrad,
    )
    
    # Calculate beam extinction and transmittance
    kb, tb, td, tbi, fracsun = calculate_beam_extinction_and_transmittance(
        pft_params.xl,
        jnp.arccos(patch_state.cosz),
        mlcanopy_state.dpai_profile,
        pft_params.clump_fac,
        patch_state.itype,
        mlcanopy_state.ntop_canopy,
        nlevmlcan,
    )
    
    # Update optical properties with calculated values
    optical_props = optical_props._replace(
        kb=kb,
        tb=tb,
        td=td,
        tbi=tbi,
        fracsun=fracsun,
    )
    
    # Calculate two-stream parameters
    phi1 = 0.5 - 0.633 * pft_params.xl[patch_state.itype] - \
           0.330 * pft_params.xl[patch_state.itype] ** 2
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    gdir = phi1[:, jnp.newaxis] + phi2[:, jnp.newaxis] * \
           jnp.cos(jnp.arccos(patch_state.cosz))[:, jnp.newaxis]
    chil = pft_params.xl[patch_state.itype]
    
    avmu, betad, betab = calculate_twostream_parameters(
        phi1[:, jnp.newaxis] * jnp.ones((n_patches, nlevmlcan)),
        phi2[:, jnp.newaxis] * jnp.ones((n_patches, nlevmlcan)),
        optical_props.rho,
        optical_props.tau,
        optical_props.omega,
        chil[:, jnp.newaxis] * jnp.ones((n_patches, nlevmlcan)),
        gdir,
        jnp.arccos(patch_state.cosz),
        kb,
    )
    
    # Update optical properties with two-stream parameters
    optical_props = optical_props._replace(
        avmu=avmu,
        betad=betad,
        betab=betab,
    )
    
    # Choose radiation transfer method
    if light_type == 1:
        # Norman (1979) method
        fluxes = norman_radiation_transfer(
            patch_state.swskyb,
            patch_state.swskyd,
            optical_props,
            mlcanopy_state,
            patch_state,
            pft_params,
            nlevmlcan,
            numrad,
        )
    else:
        # Two-stream approximation
        fluxes = twostream_radiation_transfer(
            patch_state.swskyb,
            patch_state.swskyd,
            optical_props,
            mlcanopy_state,
            patch_state,
            pft_params,
            nlevmlcan,
            numrad,
        )
    
    return fluxes