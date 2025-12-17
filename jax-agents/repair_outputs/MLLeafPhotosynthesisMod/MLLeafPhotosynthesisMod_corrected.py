"""
Leaf Photosynthesis and Stomatal Conductance Module.

Translated from CTSM's MLLeafPhotosynthesisMod.F90

This module calculates leaf-level photosynthesis and stomatal conductance
for multilayer canopy models. It implements biochemical photosynthesis models
(C3 and C4) coupled with stomatal optimization or empirical conductance models.

Key processes:
    - Leaf photosynthesis (An): Net CO2 assimilation
    - Stomatal conductance (gs): Water vapor and CO2 exchange
    - Temperature responses: Enzyme kinetics and inhibition
    - Optimization: Maximizing carbon gain per unit water loss
    - 13C fractionation: Isotope discrimination during photosynthesis

Physics:
    The module implements the Farquhar et al. (1980) model for C3 photosynthesis
    and the Collatz et al. (1992) model for C4 photosynthesis, with various
    options for stomatal conductance including optimization-based and empirical
    approaches.

References:
    - Farquhar, G.D., von Caemmerer, S., Berry, J.A. (1980)
    - Collatz, G.J., et al. (1992)
    - Medlyn, B.E., et al. (2011)
"""

from typing import NamedTuple, Tuple, Callable
import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

# Physical constants
TFRZ = 273.15  # Freezing point of water [K]
RGAS = 8.314  # Universal gas constant [J/K/mol]

# Photosynthesis constants (from MLclm_varcon)
KC25 = 404.9  # Michaelis-Menten constant for CO2 at 25°C [umol/mol]
KO25 = 278.4  # Michaelis-Menten constant for O2 at 25°C [mmol/mol]
CP25 = 42.75  # CO2 compensation point at 25°C [umol/mol]
KCHA = 79430.0  # Activation energy for Kc [J/mol]
KOHA = 36380.0  # Activation energy for Ko [J/mol]
CPHA = 37830.0  # Activation energy for CO2 compensation point [J/mol]

# Temperature acclimation parameters
VCMAXHA_NOACCLIM = 65330.0  # Vcmax activation energy without acclimation [J/mol]
VCMAXHA_ACCLIM = 72000.0  # Vcmax activation energy with acclimation [J/mol]
JMAXHA_NOACCLIM = 43540.0  # Jmax activation energy without acclimation [J/mol]
JMAXHA_ACCLIM = 50000.0  # Jmax activation energy with acclimation [J/mol]
VCMAXHD_NOACCLIM = 149250.0  # Vcmax deactivation energy without acclimation [J/mol]
VCMAXHD_ACCLIM = 200000.0  # Vcmax deactivation energy with acclimation [J/mol]
JMAXHD_NOACCLIM = 152040.0  # Jmax deactivation energy without acclimation [J/mol]
JMAXHD_ACCLIM = 200000.0  # Jmax deactivation energy with acclimation [J/mol]
VCMAXSE_NOACCLIM = 485.0  # Vcmax entropy term without acclimation [J/mol/K]
VCMAXSE_ACCLIM = 668.39  # Vcmax entropy term with acclimation [J/mol/K]
JMAXSE_NOACCLIM = 495.0  # Jmax entropy term without acclimation [J/mol/K]
JMAXSE_ACCLIM = 659.70  # Jmax entropy term with acclimation [J/mol/K]

# Dark respiration parameters
RDHA = 46390.0  # Rd activation energy [J/mol]
RDHD = 150650.0  # Rd deactivation energy [J/mol]
RDSE = 490.0  # Rd entropy term [J/mol/K]

# Electron transport parameters
PHI_PSII = 0.85  # Quantum yield of PSII [mol e-/mol photons]
THETA_J = 0.90  # Curvature parameter for electron transport [-]

# Stomatal conductance parameters
VPD_MIN_MED = 100.0  # Minimum VPD for Medlyn model [Pa]
RH_MIN_BB = 0.01  # Minimum relative humidity for Ball-Berry [fraction]
DH2O_TO_DCO2 = 1.6  # Ratio of H2O to CO2 diffusivity [-]

# C4 photosynthesis parameters
QE_C4 = 0.05  # C4 quantum efficiency [mol CO2/mol photons]

# Co-limitation parameters
COLIM_C3A = 0.98  # C3 co-limitation parameter for Ac-Aj [-]
COLIM_C4A = 0.80  # C4 co-limitation parameter for Ac-Aj [-]
COLIM_C4B = 0.95  # C4 co-limitation parameter for Ai-Ap [-]


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class PhotosynthesisParams(NamedTuple):
    """Parameters for leaf photosynthesis calculations."""
    # Physical constants
    tfrz: float = TFRZ
    rgas: float = RGAS
    
    # Michaelis-Menten constants
    kc25: float = KC25
    ko25: float = KO25
    cp25: float = CP25
    kcha: float = KCHA
    koha: float = KOHA
    cpha: float = CPHA
    
    # Temperature acclimation
    vcmaxha_noacclim: float = VCMAXHA_NOACCLIM
    vcmaxha_acclim: float = VCMAXHA_ACCLIM
    jmaxha_noacclim: float = JMAXHA_NOACCLIM
    jmaxha_acclim: float = JMAXHA_ACCLIM
    vcmaxhd_noacclim: float = VCMAXHD_NOACCLIM
    vcmaxhd_acclim: float = VCMAXHD_ACCLIM
    jmaxhd_noacclim: float = JMAXHD_NOACCLIM
    jmaxhd_acclim: float = JMAXHD_ACCLIM
    vcmaxse_noacclim: float = VCMAXSE_NOACCLIM
    vcmaxse_acclim: float = VCMAXSE_ACCLIM
    jmaxse_noacclim: float = JMAXSE_NOACCLIM
    jmaxse_acclim: float = JMAXSE_ACCLIM
    
    # Dark respiration
    rdha: float = RDHA
    rdhd: float = RDHD
    rdse: float = RDSE
    
    # Electron transport
    phi_psii: float = PHI_PSII
    theta_j: float = THETA_J
    
    # Stomatal conductance
    vpd_min_med: float = VPD_MIN_MED
    rh_min_bb: float = RH_MIN_BB
    dh2o_to_dco2: float = DH2O_TO_DCO2
    
    # C4 photosynthesis
    qe_c4: float = QE_C4
    
    # Co-limitation
    colim_c3a: float = COLIM_C3A
    colim_c4a: float = COLIM_C4A
    colim_c4b: float = COLIM_C4B
    
    # Model configuration
    gs_type: int = 0
    acclim_type: int = 0
    gspot_type: int = 0
    colim_type: int = 1


class LeafPhotosynthesisState(NamedTuple):
    """State variables for leaf photosynthesis."""
    # Stomatal conductance parameters
    g0: jnp.ndarray
    g1: jnp.ndarray
    btran: jnp.ndarray
    
    # Photosynthesis parameters
    kc: jnp.ndarray
    ko: jnp.ndarray
    cp: jnp.ndarray
    vcmax: jnp.ndarray
    jmax: jnp.ndarray
    je: jnp.ndarray
    kp: jnp.ndarray
    rd: jnp.ndarray
    
    # Leaf gas exchange
    ci: jnp.ndarray
    hs: jnp.ndarray
    vpd: jnp.ndarray
    ceair: jnp.ndarray
    leaf_esat: jnp.ndarray
    gspot: jnp.ndarray
    
    # Photosynthesis rates
    ac: jnp.ndarray
    aj: jnp.ndarray
    ap: jnp.ndarray
    agross: jnp.ndarray
    anet: jnp.ndarray
    cs: jnp.ndarray
    gs: jnp.ndarray
    
    # Isotope fractionation
    alphapsn: jnp.ndarray


# =============================================================================
# TEMPERATURE RESPONSE FUNCTIONS
# =============================================================================

def ft(
    tl: jnp.ndarray,
    ha: float,
    tfrz: float = TFRZ,
    rgas: float = RGAS,
) -> jnp.ndarray:
    """Calculate photosynthesis temperature response."""
    t_ref = tfrz + 25.0
    ans = jnp.exp(ha / (rgas * t_ref) * (1.0 - t_ref / tl))
    return ans


def fth(
    tl: jnp.ndarray,
    hd: float,
    se: float,
    c: float,
    rgas: float = RGAS,
) -> jnp.ndarray:
    """Calculate photosynthesis temperature inhibition factor."""
    exponent = (-hd + se * tl) / (rgas * tl)
    ans = c / (1.0 + jnp.exp(exponent))
    return ans


def fth25(
    hd: float,
    se: float,
    tfrz: float = TFRZ,
    rgas: float = RGAS,
) -> float:
    """Calculate temperature inhibition scaling factor at 25°C."""
    t_ref = tfrz + 25.0
    ans = 1.0 + jnp.exp((-hd + se * t_ref) / (rgas * t_ref))
    return ans


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def quadratic(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve quadratic equation: a*x^2 + b*x + c = 0."""
    discriminant = b * b - 4.0 * a * c
    sqrt_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    
    r1 = (-b + sqrt_discriminant) / (2.0 * a)
    r2 = (-b - sqrt_discriminant) / (2.0 * a)
    
    return r1, r2


def satvap(t: jnp.ndarray) -> jnp.ndarray:
    """Calculate saturation vapor pressure."""
    t_celsius = t - TFRZ
    esat = 611.0 * jnp.exp(17.27 * t_celsius / (t_celsius + 237.3))
    return esat


# =============================================================================
# MAIN PHOTOSYNTHESIS FUNCTION
# =============================================================================

def leaf_photosynthesis(
    # PFT constants
    c3psn: jnp.ndarray,
    g0_BB: jnp.ndarray,
    g1_BB: jnp.ndarray,
    g0_MED: jnp.ndarray,
    g1_MED: jnp.ndarray,
    psi50_gs: jnp.ndarray,
    shape_gs: jnp.ndarray,
    gsmin_SPA: jnp.ndarray,
    iota_SPA: jnp.ndarray,
    
    # Forcing and canopy structure
    tacclim: jnp.ndarray,
    ncan: jnp.ndarray,
    dpai: jnp.ndarray,
    eair: jnp.ndarray,
    o2ref: jnp.ndarray,
    pref: jnp.ndarray,
    
    # Leaf-level inputs
    cair: jnp.ndarray,
    vcmax25: jnp.ndarray,
    jmax25: jnp.ndarray,
    kp25: jnp.ndarray,
    rd25: jnp.ndarray,
    tleaf: jnp.ndarray,
    gbv: jnp.ndarray,
    gbc: jnp.ndarray,
    apar: jnp.ndarray,
    lwp: jnp.ndarray,
    
    # Parameters
    params: PhotosynthesisParams,
) -> LeafPhotosynthesisState:
    """Calculate leaf photosynthesis and stomatal conductance."""
    n_patches, n_layers, n_leaf = vcmax25.shape
    
    # Initialize btran (soil moisture stress factor)
    btran = jnp.ones((n_patches,))
    
    # =========================================================================
    # PART 1: Temperature acclimation and response
    # =========================================================================
    
    # Select acclimation parameters
    if params.acclim_type == 0:
        vcmaxha = params.vcmaxha_noacclim
        jmaxha = params.jmaxha_noacclim
        vcmaxhd = params.vcmaxhd_noacclim
        jmaxhd = params.jmaxhd_noacclim
        vcmaxse = params.vcmaxse_noacclim
        jmaxse = params.jmaxse_noacclim
    else:
        vcmaxha = params.vcmaxha_acclim
        jmaxha = params.jmaxha_acclim
        vcmaxhd = params.vcmaxhd_acclim
        jmaxhd = params.jmaxhd_acclim
        # Calculate entropy terms from acclimation temperature
        tacclim_c = jnp.mean(tacclim) - params.tfrz
        tacclim_c_clipped = jnp.clip(tacclim_c, 11.0, 35.0)
        vcmaxse = 668.39 - 1.07 * tacclim_c_clipped
        jmaxse = 659.70 - 0.75 * tacclim_c_clipped
    
    # High temperature deactivation scaling factors
    vcmaxc = fth25(vcmaxhd, vcmaxse, params.tfrz, params.rgas)
    jmaxc = fth25(jmaxhd, jmaxse, params.tfrz, params.rgas)
    rdc = fth25(params.rdhd, params.rdse, params.tfrz, params.rgas)
    
    # C3 photosynthetic temperature response
    kc = params.kc25 * ft(tleaf, params.kcha, params.tfrz, params.rgas)
    ko = params.ko25 * ft(tleaf, params.koha, params.tfrz, params.rgas)
    cp = params.cp25 * ft(tleaf, params.cpha, params.tfrz, params.rgas)
    vcmax_c3 = vcmax25 * ft(tleaf, vcmaxha, params.tfrz, params.rgas) * \
               fth(tleaf, vcmaxhd, vcmaxse, vcmaxc, params.rgas)
    jmax = jmax25 * ft(tleaf, jmaxha, params.tfrz, params.rgas) * \
           fth(tleaf, jmaxhd, jmaxse, jmaxc, params.rgas)
    rd_c3 = rd25 * ft(tleaf, params.rdha, params.tfrz, params.rgas) * \
            fth(tleaf, params.rdhd, params.rdse, rdc, params.rgas)
    
    # C4 photosynthetic temperature response
    t1 = 2.0 ** ((tleaf - (params.tfrz + 25.0)) / 10.0)
    t2 = 1.0 + jnp.exp(0.2 * ((params.tfrz + 15.0) - tleaf))
    t3 = 1.0 + jnp.exp(0.3 * (tleaf - (params.tfrz + 40.0)))
    t4 = 1.0 + jnp.exp(1.3 * (tleaf - (params.tfrz + 55.0)))
    
    vcmax_c4 = vcmax25 * t1 / (t2 * t3)
    rd_c4 = rd25 * t1 / t4
    kp_c4 = kp25 * t1
    
    # Select C3 or C4 values
    is_c3 = jnp.round(c3psn[:, None, None]) == 1.0
    vcmax = jnp.where(is_c3, vcmax_c3, vcmax_c4)
    rd = jnp.where(is_c3, rd_c3, rd_c4)
    kp = jnp.where(is_c3, 0.0, kp_c4)
    
    # Apply btran to vcmax
    vcmax = vcmax * btran[:, None, None]
    
    # Only process layers with plant area
    has_lai = dpai[:, :, None] > 0.0
    
    # =========================================================================
    # PART 2: Stomatal conductance setup and electron transport
    # =========================================================================
    
    # Select stomatal conductance parameters
    if params.gs_type == 0:  # Medlyn
        g0 = g0_MED
        g1 = g1_MED
    elif params.gs_type == 1:  # Ball-Berry
        g0 = g0_BB
        g1 = g1_BB
    else:  # WUE optimization
        g0 = gsmin_SPA
        g1 = jnp.zeros_like(gsmin_SPA)
    
    # Saturation vapor pressure at leaf temperature
    leaf_esat = satvap(tleaf)
    
    # Constrain canopy air vapor pressure
    eair_expanded = eair[:, :, None]
    ceair = jnp.minimum(eair_expanded, leaf_esat)
    
    # For Ball-Berry, ensure eair >= rh_min_BB * leaf_esat
    if params.gs_type == 1:
        ceair = jnp.maximum(ceair, params.rh_min_bb * leaf_esat)
    
    # Electron transport rate for C3 plants
    qabs = 0.5 * params.phi_psii * apar
    aquad = params.theta_j
    bquad = -(qabs + jmax)
    cquad = qabs * jmax
    r1, r2 = quadratic(aquad, bquad, cquad)
    je = jnp.minimum(r1, r2)
    
    # =========================================================================
    # PART 3: Photosynthesis calculation
    # =========================================================================
    
    # Initial Ci estimate
    ci_init = jnp.where(is_c3, 0.7 * cair[:, :, None], 0.4 * cair[:, :, None])
    
    # Expand o2ref for broadcasting
    o2ref_expanded = o2ref[:, None, None]
    
    # C3 Rubisco-limited
    ac_c3 = vcmax * jnp.maximum(ci_init - cp, 0.0) / (ci_init + kc * (1.0 + o2ref_expanded / ko))
    
    # C3 RuBP-limited
    aj_c3 = je * jnp.maximum(ci_init - cp, 0.0) / (4.0 * ci_init + 8.0 * cp)
    
    # C3 product-limited
    ap_c3 = jnp.zeros_like(ac_c3)
    
    # C4 Rubisco-limited
    ac_c4 = vcmax
    
    # C4 RuBP-limited
    aj_c4 = params.qe_c4 * apar
    
    # C4 PEP carboxylase-limited
    ap_c4 = kp * jnp.maximum(ci_init, 0.0)
    
    # Select C3 or C4
    ac = jnp.where(is_c3, ac_c3, ac_c4)
    aj = jnp.where(is_c3, aj_c3, aj_c4)
    ap = jnp.where(is_c3, ap_c3, ap_c4)
    
    # Co-limitation
    if params.colim_type == 0:
        agross_c3 = jnp.minimum(ac, aj)
        agross_c4 = jnp.minimum(jnp.minimum(ac, aj), ap)
        agross = jnp.where(is_c3, agross_c3, agross_c4)
    else:
        aquad = jnp.where(is_c3, params.colim_c3a, params.colim_c4a)
        bquad = -(ac + aj)
        cquad = ac * aj
        r1, r2 = quadratic(aquad, bquad, cquad)
        ai = jnp.minimum(r1, r2)
        
        aquad_c4 = params.colim_c4b
        bquad_c4 = -(ai + ap)
        cquad_c4 = ai * ap
        r1_c4, r2_c4 = quadratic(aquad_c4, bquad_c4, cquad_c4)
        agross_c4 = jnp.minimum(r1_c4, r2_c4)
        
        agross = jnp.where(is_c3, ai, agross_c4)
    
    # Prevent negative photosynthesis
    ac = jnp.maximum(ac, 0.0)
    aj = jnp.maximum(aj, 0.0)
    ap = jnp.maximum(ap, 0.0)
    agross = jnp.maximum(agross, 0.0)
    
    # Net photosynthesis
    anet = agross - rd
    
    # CO2 at leaf surface
    cs = cair[:, :, None] - anet / gbc
    cs = jnp.maximum(cs, 1.0)
    
    # =========================================================================
    # PART 4: Stomatal conductance
    # =========================================================================
    
    if params.gs_type == 1:  # Ball-Berry
        term = anet / cs
        aquad_gs = jnp.ones_like(anet)
        bquad_gs = gbv - g0[:, None, None] - g1[:, None, None] * term
        cquad_gs = -gbv * (g0[:, None, None] + g1[:, None, None] * term * ceair / leaf_esat)
        r1_gs, r2_gs = quadratic(aquad_gs, bquad_gs, cquad_gs)
        gs = jnp.where(anet > 0.0, jnp.maximum(r1_gs, r2_gs), g0[:, None, None])
    elif params.gs_type == 0:  # Medlyn
        vpd_term = jnp.maximum((leaf_esat - ceair), params.vpd_min_med) * 0.001
        term = params.dh2o_to_dco2 * anet / cs
        aquad_gs = jnp.ones_like(anet)
        bquad_gs = -(2.0 * (g0[:, None, None] + term) + (g1[:, None, None] * term)**2 / (gbv * vpd_term))
        cquad_gs = g0[:, None, None] * g0[:, None, None] + \
                   (2.0 * g0[:, None, None] + term * (1.0 - g1[:, None, None] * g1[:, None, None] / vpd_term)) * term
        r1_gs, r2_gs = quadratic(aquad_gs, bquad_gs, cquad_gs)
        gs = jnp.where(anet > 0.0, jnp.maximum(r1_gs, r2_gs), g0[:, None, None])
    else:  # WUE optimization
        gs = gsmin_SPA[:, None, None] * jnp.ones_like(anet)
    
    # Update Ci from diffusion equation
    gleaf = 1.0 / (1.0 / gbc + params.dh2o_to_dco2 / gs)
    ci = cair[:, :, None] - anet / gleaf
    
    # =========================================================================
    # PART 5: Water stress adjustment
    # =========================================================================
    
    # Save potential conductance
    gspot = gs
    
    # Calculate water stress factor
    if params.gspot_type == 1:
        ratio = lwp / psi50_gs[:, None, None]
        fpsi = 1.0 / (1.0 + ratio ** shape_gs[:, None, None])
    else:
        fpsi = jnp.ones_like(gs)
    
    # Apply water stress
    gs = jnp.maximum(gspot * fpsi, gsmin_SPA[:, None, None])
    
    # Recalculate photosynthesis for water-stressed gs
    gleaf_stressed = 1.0 / (1.0 / gbc + params.dh2o_to_dco2 / gs)
    
    # Recalculate photosynthesis rates with stressed conductance
    # C3 Rubisco-limited with stressed gleaf
    a0_c3 = vcmax
    b0_c3 = kc * (1.0 + o2ref_expanded / ko)
    aquad_c3 = 1.0 / gleaf_stressed
    bquad_c3 = -(cair[:, :, None] + b0_c3) - (a0_c3 - rd) / gleaf_stressed
    cquad_c3 = a0_c3 * (cair[:, :, None] - cp) - rd * (cair[:, :, None] + b0_c3)
    r1_c3, r2_c3 = quadratic(aquad_c3, bquad_c3, cquad_c3)
    ac_stressed_c3 = jnp.minimum(r1_c3, r2_c3) + rd
    
    # C3 RuBP-limited with stressed gleaf
    a0_j = je / 4.0
    b0_j = 2.0 * cp
    aquad_j = 1.0 / gleaf_stressed
    bquad_j = -(cair[:, :, None] + b0_j) - (a0_j - rd) / gleaf_stressed
    cquad_j = a0_j * (cair[:, :, None] - cp) - rd * (cair[:, :, None] + b0_j)
    r1_j, r2_j = quadratic(aquad_j, bquad_j, cquad_j)
    aj_stressed_c3 = jnp.minimum(r1_j, r2_j) + rd
    
    ap_stressed_c3 = jnp.zeros_like(ac_stressed_c3)
    
    # C4 rates remain the same for Rubisco and RuBP
    ac_stressed_c4 = vcmax
    aj_stressed_c4 = params.qe_c4 * apar
    ap_stressed_c4 = kp * (cair[:, :, None] * gleaf_stressed + rd) / (gleaf_stressed + kp)
    
    # Select C3 or C4
    ac = jnp.where(is_c3, ac_stressed_c3, ac_stressed_c4)
    aj = jnp.where(is_c3, aj_stressed_c3, aj_stressed_c4)
    ap = jnp.where(is_c3, ap_stressed_c3, ap_stressed_c4)
    
    # Co-limitation with stressed values
    if params.colim_type == 0:
        agross = jnp.where(
            is_c3,
            jnp.minimum(ac, aj),
            jnp.minimum(jnp.minimum(ac, aj), ap),
        )
    else:
        aquad = jnp.where(is_c3, params.colim_c3a, params.colim_c4a)
        bquad = -(ac + aj)
        cquad = ac * aj
        r1, r2 = quadratic(aquad, bquad, cquad)
        ai = jnp.minimum(r1, r2)
        
        aquad_c4 = params.colim_c4b
        bquad_c4 = -(ai + ap)
        cquad_c4 = ai * ap
        r1_c4, r2_c4 = quadratic(aquad_c4, bquad_c4, cquad_c4)
        agross_c4 = jnp.minimum(r1_c4, r2_c4)
        
        agross = jnp.where(is_c3, ai, agross_c4)
    
    # Net photosynthesis
    anet = agross - rd
    
    # CO2 at leaf surface
    cs = jnp.maximum(cair[:, :, None] - anet / gbc, 1.0)
    
    # Intercellular CO2
    ci = cair[:, :, None] - anet / gleaf_stressed
    
    # =========================================================================
    # PART 6: Final calculations
    # =========================================================================
    
    # Relative humidity at leaf surface
    hs = (gbv * eair_expanded + gs * leaf_esat) / ((gbv + gs) * leaf_esat)
    
    # Vapor pressure deficit
    vpd = jnp.maximum(leaf_esat - hs * leaf_esat, 0.1)
    
    # Apply masking for layers without vegetation
    ac = jnp.where(has_lai, ac, 0.0)
    aj = jnp.where(has_lai, aj, 0.0)
    ap = jnp.where(has_lai, ap, 0.0)
    agross = jnp.where(has_lai, agross, 0.0)
    anet = jnp.where(has_lai, anet, 0.0)
    cs = jnp.where(has_lai, cs, 0.0)
    gs = jnp.where(has_lai, gs, 0.0)
    ci = jnp.where(has_lai, ci, 0.0)
    hs = jnp.where(has_lai, hs, 0.0)
    vpd = jnp.where(has_lai, vpd, 0.0)
    
    # =========================================================================
    # C13 fractionation
    # =========================================================================
    
    # C3 fractionation
    alphapsn_c3 = 1.0 + (4.4 + 22.6 * ci / cair[:, :, None]) / 1000.0
    
    # C4 fractionation
    alphapsn_c4 = 1.0 + 4.4 / 1000.0
    
    # Select based on pathway
    alphapsn = jnp.where(is_c3, alphapsn_c3, alphapsn_c4)
    
    # Apply PAR and plant area conditions
    alphapsn = jnp.where(apar > 0.0, alphapsn, 1.0)
    alphapsn = jnp.where(has_lai, alphapsn, 0.0)
    
    # =========================================================================
    # Return state
    # =========================================================================
    
    return LeafPhotosynthesisState(
        g0=g0[:, None, None] * jnp.ones_like(ac),
        g1=g1[:, None, None] * jnp.ones_like(ac),
        btran=btran,
        kc=kc,
        ko=ko,
        cp=cp,
        vcmax=vcmax,
        jmax=jmax,
        je=je,
        kp=kp,
        rd=rd,
        ci=ci,
        hs=hs,
        vpd=vpd,
        ceair=ceair,
        leaf_esat=leaf_esat,
        gspot=gspot,
        ac=ac,
        aj=aj,
        ap=ap,
        agross=agross,
        anet=anet,
        cs=cs,
        gs=gs,
        alphapsn=alphapsn,
    )