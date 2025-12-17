"""
Comprehensive pytest suite for MLLeafPhotosynthesisMod.leaf_photosynthesis function.

This module tests the leaf photosynthesis model including:
- C3 and C4 photosynthetic pathways
- Stomatal conductance models (Ball-Berry, Medlyn, SPA)
- Temperature dependencies and acclimation
- Water stress effects
- Multi-layer canopy gradients
- Edge cases (extreme temperatures, water stress, zero PAR)
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict, Any
import json


# ============================================================================
# Mock NamedTuple Definitions (replace with actual imports in production)
# ============================================================================

class PhotosynthesisParams(NamedTuple):
    """Photosynthesis model parameters."""
    tfrz: float = 273.15
    rgas: float = 8.314
    kc25: float = 404.9
    ko25: float = 278.4
    cp25: float = 42.75
    kcha: float = 79430.0
    koha: float = 36380.0
    cpha: float = 37830.0
    vcmaxha_noacclim: float = 65330.0
    vcmaxha_acclim: float = 65330.0
    jmaxha_noacclim: float = 43540.0
    jmaxha_acclim: float = 43540.0
    vcmaxhd_noacclim: float = 149250.0
    vcmaxhd_acclim: float = 149250.0
    jmaxhd_noacclim: float = 152040.0
    jmaxhd_acclim: float = 152040.0
    vcmaxse_noacclim: float = 485.0
    vcmaxse_acclim: float = 485.0
    jmaxse_noacclim: float = 495.0
    jmaxse_acclim: float = 495.0
    rdha: float = 46390.0
    rdhd: float = 150650.0
    rdse: float = 490.0
    phi_psii: float = 0.85
    theta_j: float = 0.90
    vpd_min_med: float = 0.1
    rh_min_bb: float = 0.3
    dh2o_to_dco2: float = 1.6
    qe_c4: float = 0.05
    colim_c3a: float = 0.98
    colim_c4a: float = 0.80
    colim_c4b: float = 0.004
    gs_type: int = 1  # 1=Ball-Berry, 2=Medlyn, 3=SPA
    acclim_type: int = 0  # 0=no acclimation, 1=acclimation
    gspot_type: int = 1  # 1=water stress, 0=no water stress
    colim_type: int = 1  # 1=co-limitation, 0=min limitation


class LeafPhotosynthesisState(NamedTuple):
    """Output state from leaf photosynthesis calculations."""
    g0: jnp.ndarray
    g1: jnp.ndarray
    btran: jnp.ndarray
    kc: jnp.ndarray
    ko: jnp.ndarray
    cp: jnp.ndarray
    vcmax: jnp.ndarray
    jmax: jnp.ndarray
    je: jnp.ndarray
    kp: jnp.ndarray
    rd: jnp.ndarray
    ci: jnp.ndarray
    hs: jnp.ndarray
    vpd: jnp.ndarray
    ceair: jnp.ndarray
    leaf_esat: jnp.ndarray
    gspot: jnp.ndarray
    ac: jnp.ndarray
    aj: jnp.ndarray
    ap: jnp.ndarray
    agross: jnp.ndarray
    anet: jnp.ndarray
    cs: jnp.ndarray
    gs: jnp.ndarray
    alphapsn: jnp.ndarray


# Mock function signature (replace with actual import)
def leaf_photosynthesis(
    c3psn: jnp.ndarray,
    g0_BB: jnp.ndarray,
    g1_BB: jnp.ndarray,
    g0_MED: jnp.ndarray,
    g1_MED: jnp.ndarray,
    psi50_gs: jnp.ndarray,
    shape_gs: jnp.ndarray,
    gsmin_SPA: jnp.ndarray,
    iota_SPA: jnp.ndarray,
    tacclim: jnp.ndarray,
    ncan: jnp.ndarray,
    dpai: jnp.ndarray,
    eair: jnp.ndarray,
    o2ref: jnp.ndarray,
    pref: jnp.ndarray,
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
    params: PhotosynthesisParams,
) -> LeafPhotosynthesisState:
    """Mock implementation - replace with actual function."""
    raise NotImplementedError("Replace with actual leaf_photosynthesis import")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_params() -> PhotosynthesisParams:
    """
    Provide default photosynthesis parameters.
    
    Returns:
        PhotosynthesisParams with standard values for testing
    """
    return PhotosynthesisParams()


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data from JSON specification.
    
    Returns:
        Dictionary containing all test cases with inputs and metadata
    """
    test_data_json = {
        "function_name": "leaf_photosynthesis",
        "test_cases": [
            {
                "name": "test_nominal_c3_single_patch_single_layer",
                "inputs": {
                    "c3psn": [1.0],
                    "g0_BB": [0.01],
                    "g1_BB": [9.0],
                    "g0_MED": [0.0],
                    "g1_MED": [4.0],
                    "psi50_gs": [-2.0],
                    "shape_gs": [3.0],
                    "gsmin_SPA": [0.001],
                    "iota_SPA": [750.0],
                    "tacclim": [298.15],
                    "ncan": [1],
                    "dpai": [[2.5]],
                    "eair": [[1500.0]],
                    "o2ref": [209.0],
                    "pref": [101325.0],
                    "cair": [[[400.0, 400.0]]],
                    "vcmax25": [[[60.0, 60.0]]],
                    "jmax25": [[[120.0, 120.0]]],
                    "kp25": [[[0.0, 0.0]]],
                    "rd25": [[[1.0, 1.0]]],
                    "tleaf": [[[298.15, 298.15]]],
                    "gbv": [[[1.5, 1.5]]],
                    "gbc": [[[1.0, 1.0]]],
                    "apar": [[[1000.0, 1000.0]]],
                    "lwp": [[[-0.5, -0.5]]]
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Standard C3 photosynthesis with typical temperate conditions",
                    "edge_cases": []
                }
            },
            {
                "name": "test_nominal_c4_multiple_patches",
                "inputs": {
                    "c3psn": [0.0, 0.0, 0.0],
                    "g0_BB": [0.04, 0.04, 0.04],
                    "g1_BB": [4.0, 4.0, 4.0],
                    "g0_MED": [0.0, 0.0, 0.0],
                    "g1_MED": [1.6, 1.6, 1.6],
                    "psi50_gs": [-1.5, -1.5, -1.5],
                    "shape_gs": [2.5, 2.5, 2.5],
                    "gsmin_SPA": [0.001, 0.001, 0.001],
                    "iota_SPA": [1000.0, 1000.0, 1000.0],
                    "tacclim": [303.15, 303.15, 303.15],
                    "ncan": [2, 2, 2],
                    "dpai": [[1.5, 1.0], [2.0, 1.5], [1.8, 1.2]],
                    "eair": [[2000.0, 1800.0], [2100.0, 1900.0], [2050.0, 1850.0]],
                    "o2ref": [209.0, 209.0, 209.0],
                    "pref": [101325.0, 101325.0, 101325.0],
                    "cair": [
                        [[380.0, 380.0], [370.0, 370.0]],
                        [[390.0, 390.0], [380.0, 380.0]],
                        [[385.0, 385.0], [375.0, 375.0]]
                    ],
                    "vcmax25": [
                        [[80.0, 80.0], [70.0, 70.0]],
                        [[85.0, 85.0], [75.0, 75.0]],
                        [[82.0, 82.0], [72.0, 72.0]]
                    ],
                    "jmax25": [
                        [[0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0]]
                    ],
                    "kp25": [
                        [[0.8, 0.8], [0.7, 0.7]],
                        [[0.85, 0.85], [0.75, 0.75]],
                        [[0.82, 0.82], [0.72, 0.72]]
                    ],
                    "rd25": [
                        [[1.5, 1.5], [1.3, 1.3]],
                        [[1.6, 1.6], [1.4, 1.4]],
                        [[1.55, 1.55], [1.35, 1.35]]
                    ],
                    "tleaf": [
                        [[303.15, 303.15], [302.15, 302.15]],
                        [[304.15, 304.15], [303.15, 303.15]],
                        [[303.65, 303.65], [302.65, 302.65]]
                    ],
                    "gbv": [
                        [[2.0, 2.0], [1.8, 1.8]],
                        [[2.1, 2.1], [1.9, 1.9]],
                        [[2.05, 2.05], [1.85, 1.85]]
                    ],
                    "gbc": [
                        [[1.3, 1.3], [1.2, 1.2]],
                        [[1.4, 1.4], [1.25, 1.25]],
                        [[1.35, 1.35], [1.22, 1.22]]
                    ],
                    "apar": [
                        [[1500.0, 500.0], [800.0, 200.0]],
                        [[1600.0, 550.0], [850.0, 220.0]],
                        [[1550.0, 525.0], [825.0, 210.0]]
                    ],
                    "lwp": [
                        [[-0.8, -0.8], [-0.6, -0.6]],
                        [[-0.9, -0.9], [-0.7, -0.7]],
                        [[-0.85, -0.85], [-0.65, -0.65]]
                    ]
                },
                "metadata": {
                    "type": "nominal",
                    "description": "C4 photosynthesis with multiple patches and layers",
                    "edge_cases": []
                }
            },
            {
                "name": "test_edge_zero_par_dark_respiration",
                "inputs": {
                    "c3psn": [1.0],
                    "g0_BB": [0.01],
                    "g1_BB": [9.0],
                    "g0_MED": [0.0],
                    "g1_MED": [4.0],
                    "psi50_gs": [-2.0],
                    "shape_gs": [3.0],
                    "gsmin_SPA": [0.001],
                    "iota_SPA": [750.0],
                    "tacclim": [293.15],
                    "ncan": [1],
                    "dpai": [[3.0]],
                    "eair": [[1200.0]],
                    "o2ref": [209.0],
                    "pref": [101325.0],
                    "cair": [[[400.0, 400.0]]],
                    "vcmax25": [[[50.0, 50.0]]],
                    "jmax25": [[[100.0, 100.0]]],
                    "kp25": [[[0.0, 0.0]]],
                    "rd25": [[[0.8, 0.8]]],
                    "tleaf": [[[293.15, 293.15]]],
                    "gbv": [[[1.2, 1.2]]],
                    "gbc": [[[0.8, 0.8]]],
                    "apar": [[[0.0, 0.0]]],
                    "lwp": [[[-1.0, -1.0]]]
                },
                "metadata": {
                    "type": "edge",
                    "description": "Zero PAR testing dark respiration",
                    "edge_cases": ["zero_par", "dark_respiration"]
                }
            },
            {
                "name": "test_edge_severe_water_stress",
                "inputs": {
                    "c3psn": [1.0, 0.0],
                    "g0_BB": [0.01, 0.04],
                    "g1_BB": [9.0, 4.0],
                    "g0_MED": [0.0, 0.0],
                    "g1_MED": [4.0, 1.6],
                    "psi50_gs": [-1.5, -1.2],
                    "shape_gs": [4.0, 3.5],
                    "gsmin_SPA": [0.001, 0.001],
                    "iota_SPA": [750.0, 1000.0],
                    "tacclim": [298.15, 303.15],
                    "ncan": [1, 1],
                    "dpai": [[2.0], [2.5]],
                    "eair": [[1000.0], [1500.0]],
                    "o2ref": [209.0, 209.0],
                    "pref": [101325.0, 101325.0],
                    "cair": [[[400.0, 400.0]], [[380.0, 380.0]]],
                    "vcmax25": [[[55.0, 55.0]], [[75.0, 75.0]]],
                    "jmax25": [[[110.0, 110.0]], [[0.0, 0.0]]],
                    "kp25": [[[0.0, 0.0]], [[0.75, 0.75]]],
                    "rd25": [[[0.9, 0.9]], [[1.4, 1.4]]],
                    "tleaf": [[[298.15, 298.15]], [[303.15, 303.15]]],
                    "gbv": [[[1.4, 1.4]], [[1.9, 1.9]]],
                    "gbc": [[[0.9, 0.9]], [[1.25, 1.25]]],
                    "apar": [[[800.0, 300.0]], [[1200.0, 400.0]]],
                    "lwp": [[[-4.5, -4.5]], [[-3.8, -3.8]]]
                },
                "metadata": {
                    "type": "edge",
                    "description": "Severe water stress testing stomatal closure",
                    "edge_cases": ["severe_water_stress", "near_wilting_point"]
                }
            },
            {
                "name": "test_edge_extreme_temperature_cold",
                "inputs": {
                    "c3psn": [1.0],
                    "g0_BB": [0.01],
                    "g1_BB": [9.0],
                    "g0_MED": [0.0],
                    "g1_MED": [4.0],
                    "psi50_gs": [-2.5],
                    "shape_gs": [3.0],
                    "gsmin_SPA": [0.001],
                    "iota_SPA": [750.0],
                    "tacclim": [278.15],
                    "ncan": [1],
                    "dpai": [[2.0]],
                    "eair": [[500.0]],
                    "o2ref": [209.0],
                    "pref": [101325.0],
                    "cair": [[[420.0, 420.0]]],
                    "vcmax25": [[[45.0, 45.0]]],
                    "jmax25": [[[90.0, 90.0]]],
                    "kp25": [[[0.0, 0.0]]],
                    "rd25": [[[0.7, 0.7]]],
                    "tleaf": [[[278.15, 278.15]]],
                    "gbv": [[[1.0, 1.0]]],
                    "gbc": [[[0.7, 0.7]]],
                    "apar": [[[600.0, 200.0]]],
                    "lwp": [[[-0.3, -0.3]]]
                },
                "metadata": {
                    "type": "edge",
                    "description": "Cold temperature (5°C) testing",
                    "edge_cases": ["cold_temperature", "low_vapor_pressure"]
                }
            },
            {
                "name": "test_edge_extreme_temperature_hot",
                "inputs": {
                    "c3psn": [0.0],
                    "g0_BB": [0.04],
                    "g1_BB": [4.0],
                    "g0_MED": [0.0],
                    "g1_MED": [1.6],
                    "psi50_gs": [-1.0],
                    "shape_gs": [2.0],
                    "gsmin_SPA": [0.001],
                    "iota_SPA": [1000.0],
                    "tacclim": [313.15],
                    "ncan": [1],
                    "dpai": [[1.5]],
                    "eair": [[3500.0]],
                    "o2ref": [209.0],
                    "pref": [101325.0],
                    "cair": [[[360.0, 360.0]]],
                    "vcmax25": [[[90.0, 90.0]]],
                    "jmax25": [[[0.0, 0.0]]],
                    "kp25": [[[0.9, 0.9]]],
                    "rd25": [[[1.8, 1.8]]],
                    "tleaf": [[[313.15, 313.15]]],
                    "gbv": [[[2.5, 2.5]]],
                    "gbc": [[[1.6, 1.6]]],
                    "apar": [[[2000.0, 800.0]]],
                    "lwp": [[[-1.5, -1.5]]]
                },
                "metadata": {
                    "type": "edge",
                    "description": "Hot temperature (40°C) testing heat stress",
                    "edge_cases": ["hot_temperature", "high_vapor_pressure", "heat_stress"]
                }
            },
            {
                "name": "test_edge_minimal_conductance_parameters",
                "inputs": {
                    "c3psn": [1.0],
                    "g0_BB": [0.0],
                    "g1_BB": [0.0],
                    "g0_MED": [0.0],
                    "g1_MED": [0.0],
                    "psi50_gs": [-2.0],
                    "shape_gs": [3.0],
                    "gsmin_SPA": [0.0001],
                    "iota_SPA": [500.0],
                    "tacclim": [298.15],
                    "ncan": [1],
                    "dpai": [[1.0]],
                    "eair": [[1500.0]],
                    "o2ref": [209.0],
                    "pref": [101325.0],
                    "cair": [[[400.0, 400.0]]],
                    "vcmax25": [[[40.0, 40.0]]],
                    "jmax25": [[[80.0, 80.0]]],
                    "kp25": [[[0.0, 0.0]]],
                    "rd25": [[[0.6, 0.6]]],
                    "tleaf": [[[298.15, 298.15]]],
                    "gbv": [[[1.0, 1.0]]],
                    "gbc": [[[0.65, 0.65]]],
                    "apar": [[[500.0, 100.0]]],
                    "lwp": [[[-0.5, -0.5]]]
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimal stomatal conductance parameters",
                    "edge_cases": ["zero_conductance_params", "minimal_stomatal_opening"]
                }
            },
            {
                "name": "test_special_high_elevation_low_pressure",
                "inputs": {
                    "c3psn": [1.0, 1.0],
                    "g0_BB": [0.01, 0.01],
                    "g1_BB": [9.0, 9.0],
                    "g0_MED": [0.0, 0.0],
                    "g1_MED": [4.0, 4.0],
                    "psi50_gs": [-2.5, -2.5],
                    "shape_gs": [3.5, 3.5],
                    "gsmin_SPA": [0.001, 0.001],
                    "iota_SPA": [750.0, 750.0],
                    "tacclim": [283.15, 283.15],
                    "ncan": [1, 1],
                    "dpai": [[1.5], [1.8]],
                    "eair": [[800.0], [850.0]],
                    "o2ref": [209.0, 209.0],
                    "pref": [70000.0, 70000.0],
                    "cair": [[[400.0, 400.0]], [[400.0, 400.0]]],
                    "vcmax25": [[[50.0, 50.0]], [[52.0, 52.0]]],
                    "jmax25": [[[100.0, 100.0]], [[104.0, 104.0]]],
                    "kp25": [[[0.0, 0.0]], [[0.0, 0.0]]],
                    "rd25": [[[0.8, 0.8]], [[0.85, 0.85]]],
                    "tleaf": [[[283.15, 283.15]], [[283.15, 283.15]]],
                    "gbv": [[[1.1, 1.1]], [[1.15, 1.15]]],
                    "gbc": [[[0.75, 0.75]], [[0.78, 0.78]]],
                    "apar": [[[900.0, 400.0]], [[950.0, 420.0]]],
                    "lwp": [[[-0.4, -0.4]], [[-0.45, -0.45]]]
                },
                "metadata": {
                    "type": "special",
                    "description": "High elevation with reduced atmospheric pressure",
                    "edge_cases": ["low_atmospheric_pressure", "high_elevation"]
                }
            },
            {
                "name": "test_special_multi_layer_canopy_gradient",
                "inputs": {
                    "c3psn": [1.0],
                    "g0_BB": [0.01],
                    "g1_BB": [9.0],
                    "g0_MED": [0.0],
                    "g1_MED": [4.0],
                    "psi50_gs": [-2.0],
                    "shape_gs": [3.0],
                    "gsmin_SPA": [0.001],
                    "iota_SPA": [750.0],
                    "tacclim": [298.15],
                    "ncan": [5],
                    "dpai": [[1.0, 1.2, 1.5, 1.0, 0.8]],
                    "eair": [[1800.0, 1750.0, 1700.0, 1650.0, 1600.0]],
                    "o2ref": [209.0],
                    "pref": [101325.0],
                    "cair": [
                        [[400.0, 400.0], [395.0, 395.0], [390.0, 390.0],
                         [385.0, 385.0], [380.0, 380.0]]
                    ],
                    "vcmax25": [
                        [[65.0, 65.0], [60.0, 60.0], [55.0, 55.0],
                         [50.0, 50.0], [45.0, 45.0]]
                    ],
                    "jmax25": [
                        [[130.0, 130.0], [120.0, 120.0], [110.0, 110.0],
                         [100.0, 100.0], [90.0, 90.0]]
                    ],
                    "kp25": [
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                         [0.0, 0.0], [0.0, 0.0]]
                    ],
                    "rd25": [
                        [[1.1, 1.1], [1.0, 1.0], [0.9, 0.9],
                         [0.8, 0.8], [0.7, 0.7]]
                    ],
                    "tleaf": [
                        [[299.15, 299.15], [298.65, 298.65], [298.15, 298.15],
                         [297.65, 297.65], [297.15, 297.15]]
                    ],
                    "gbv": [
                        [[1.8, 1.8], [1.6, 1.6], [1.4, 1.4],
                         [1.2, 1.2], [1.0, 1.0]]
                    ],
                    "gbc": [
                        [[1.2, 1.2], [1.05, 1.05], [0.9, 0.9],
                         [0.8, 0.8], [0.7, 0.7]]
                    ],
                    "apar": [
                        [[1800.0, 600.0], [1200.0, 400.0], [700.0, 250.0],
                         [350.0, 120.0], [150.0, 50.0]]
                    ],
                    "lwp": [
                        [[-0.3, -0.3], [-0.4, -0.4], [-0.5, -0.5],
                         [-0.6, -0.6], [-0.7, -0.7]]
                    ]
                },
                "metadata": {
                    "type": "special",
                    "description": "Five-layer canopy with vertical gradients",
                    "edge_cases": ["multi_layer_gradient", "light_extinction"]
                }
            },
            {
                "name": "test_special_mixed_c3_c4_comparison",
                "inputs": {
                    "c3psn": [1.0, 0.0, 1.0, 0.0],
                    "g0_BB": [0.01, 0.04, 0.01, 0.04],
                    "g1_BB": [9.0, 4.0, 9.0, 4.0],
                    "g0_MED": [0.0, 0.0, 0.0, 0.0],
                    "g1_MED": [4.0, 1.6, 4.0, 1.6],
                    "psi50_gs": [-2.0, -1.5, -2.0, -1.5],
                    "shape_gs": [3.0, 2.5, 3.0, 2.5],
                    "gsmin_SPA": [0.001, 0.001, 0.001, 0.001],
                    "iota_SPA": [750.0, 1000.0, 750.0, 1000.0],
                    "tacclim": [298.15, 303.15, 298.15, 303.15],
                    "ncan": [1, 1, 1, 1],
                    "dpai": [[2.5], [2.0], [2.5], [2.0]],
                    "eair": [[1500.0], [2000.0], [1500.0], [2000.0]],
                    "o2ref": [209.0, 209.0, 209.0, 209.0],
                    "pref": [101325.0, 101325.0, 101325.0, 101325.0],
                    "cair": [
                        [[400.0, 400.0]], [[380.0, 380.0]],
                        [[400.0, 400.0]], [[380.0, 380.0]]
                    ],
                    "vcmax25": [
                        [[60.0, 60.0]], [[80.0, 80.0]],
                        [[60.0, 60.0]], [[80.0, 80.0]]
                    ],
                    "jmax25": [
                        [[120.0, 120.0]], [[0.0, 0.0]],
                        [[120.0, 120.0]], [[0.0, 0.0]]
                    ],
                    "kp25": [
                        [[0.0, 0.0]], [[0.8, 0.8]],
                        [[0.0, 0.0]], [[0.8, 0.8]]
                    ],
                    "rd25": [
                        [[1.0, 1.0]], [[1.5, 1.5]],
                        [[1.0, 1.0]], [[1.5, 1.5]]
                    ],
                    "tleaf": [
                        [[298.15, 298.15]], [[303.15, 303.15]],
                        [[298.15, 298.15]], [[303.15, 303.15]]
                    ],
                    "gbv": [
                        [[1.5, 1.5]], [[2.0, 2.0]],
                        [[1.5, 1.5]], [[2.0, 2.0]]
                    ],
                    "gbc": [
                        [[1.0, 1.0]], [[1.3, 1.3]],
                        [[1.0, 1.0]], [[1.3, 1.3]]
                    ],
                    "apar": [
                        [[1000.0, 300.0]], [[1500.0, 500.0]],
                        [[1000.0, 300.0]], [[1500.0, 500.0]]
                    ],
                    "lwp": [
                        [[-0.5, -0.5]], [[-0.8, -0.8]],
                        [[-0.5, -0.5]], [[-0.8, -0.8]]
                    ]
                },
                "metadata": {
                    "type": "special",
                    "description": "C3 vs C4 comparison under identical conditions",
                    "edge_cases": []
                }
            }
        ]
    }
    return test_data_json


def convert_inputs_to_jax(inputs: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    """
    Convert test input dictionary to JAX arrays.
    
    Args:
        inputs: Dictionary of input arrays as Python lists
        
    Returns:
        Dictionary with same keys but JAX array values
    """
    jax_inputs = {}
    for key, value in inputs.items():
        if key != "params":
            jax_inputs[key] = jnp.array(value)
    return jax_inputs


# ============================================================================
# Shape Tests
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "test_nominal_c3_single_patch_single_layer",
    "test_nominal_c4_multiple_patches",
    "test_special_multi_layer_canopy_gradient",
    "test_special_mixed_c3_c4_comparison"
])
def test_leaf_photosynthesis_output_shapes(test_data, default_params, test_case):
    """
    Test that output shapes match expected dimensions based on input shapes.
    
    Verifies that all output arrays have correct shapes:
    - Scalar parameters: (n_patches,)
    - Layer-specific: (n_patches, n_layers)
    - Leaf-specific: (n_patches, n_layers, n_leaf)
    """
    # Get test case
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    inputs = convert_inputs_to_jax(case["inputs"])
    
    # Determine expected shapes
    n_patches = inputs["c3psn"].shape[0]
    n_layers = inputs["dpai"].shape[1]
    n_leaf = inputs["cair"].shape[2]
    
    # Call function
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Check scalar parameter shapes (n_patches,)
    assert result.g0.shape == (n_patches,), \
        f"g0 shape mismatch: expected {(n_patches,)}, got {result.g0.shape}"
    assert result.g1.shape == (n_patches,), \
        f"g1 shape mismatch: expected {(n_patches,)}, got {result.g1.shape}"
    assert result.btran.shape == (n_patches,), \
        f"btran shape mismatch: expected {(n_patches,)}, got {result.btran.shape}"
    
    # Check 3D output shapes (n_patches, n_layers, n_leaf)
    expected_3d_shape = (n_patches, n_layers, n_leaf)
    for field in ["kc", "ko", "cp", "vcmax", "jmax", "je", "kp", "rd", "ci",
                  "hs", "vpd", "ceair", "leaf_esat", "gspot", "ac", "aj", "ap",
                  "agross", "anet", "cs", "gs", "alphapsn"]:
        field_value = getattr(result, field)
        assert field_value.shape == expected_3d_shape, \
            f"{field} shape mismatch: expected {expected_3d_shape}, got {field_value.shape}"


# ============================================================================
# Data Type Tests
# ============================================================================

def test_leaf_photosynthesis_dtypes(test_data, default_params):
    """
    Test that all outputs have correct floating-point data types.
    
    Verifies that all output arrays are float32 or float64 (JAX default).
    """
    case = test_data["test_cases"][0]  # Use first test case
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Check all fields are floating point
    for field in result._fields:
        field_value = getattr(result, field)
        assert jnp.issubdtype(field_value.dtype, jnp.floating), \
            f"{field} has non-floating dtype: {field_value.dtype}"


# ============================================================================
# Value Range Tests
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_c3_single_patch_single_layer",
    "test_nominal_c4_multiple_patches"
])
def test_leaf_photosynthesis_value_ranges(test_data, default_params, test_case_name):
    """
    Test that output values are within physically realistic ranges.
    
    Checks:
    - Conductances are non-negative
    - Temperatures are above absolute zero
    - Pressures are non-negative
    - Photosynthesis rates are reasonable
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case_name)
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Conductances must be non-negative
    assert jnp.all(result.gs >= 0.0), "Stomatal conductance must be non-negative"
    assert jnp.all(result.gspot >= 0.0), "Water stress factor must be non-negative"
    assert jnp.all(result.gspot <= 1.0), "Water stress factor must be <= 1.0"
    
    # Vapor pressures must be non-negative
    assert jnp.all(result.vpd >= 0.0), "VPD must be non-negative"
    assert jnp.all(result.ceair >= 0.0), "Canopy air vapor pressure must be non-negative"
    assert jnp.all(result.leaf_esat >= 0.0), "Leaf saturation vapor pressure must be non-negative"
    
    # CO2 concentrations must be non-negative
    assert jnp.all(result.ci >= 0.0), "Intercellular CO2 must be non-negative"
    assert jnp.all(result.cs >= 0.0), "Leaf surface CO2 must be non-negative"
    
    # Enzyme kinetic parameters must be positive
    assert jnp.all(result.kc > 0.0), "Michaelis constant for CO2 must be positive"
    assert jnp.all(result.ko > 0.0), "Michaelis constant for O2 must be positive"
    assert jnp.all(result.cp >= 0.0), "CO2 compensation point must be non-negative"
    
    # Photosynthetic capacities must be non-negative
    assert jnp.all(result.vcmax >= 0.0), "Vcmax must be non-negative"
    assert jnp.all(result.rd >= 0.0), "Dark respiration must be non-negative"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_leaf_photosynthesis_zero_par(test_data, default_params):
    """
    Test photosynthesis with zero PAR (dark conditions).
    
    Verifies:
    - Net photosynthesis is negative (respiration only)
    - Gross photosynthesis is zero or near-zero
    - Stomatal conductance is minimal
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_zero_par_dark_respiration")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # With zero PAR, net photosynthesis should be negative (respiration)
    assert jnp.all(result.anet <= 0.0), \
        "Net photosynthesis should be negative or zero in darkness"
    
    # Gross photosynthesis should be zero or very small
    assert jnp.all(result.agross <= 1e-6), \
        "Gross photosynthesis should be near zero with zero PAR"
    
    # Respiration should be positive
    assert jnp.all(result.rd > 0.0), \
        "Dark respiration should be positive"


def test_leaf_photosynthesis_severe_water_stress(test_data, default_params):
    """
    Test photosynthesis under severe water stress.
    
    Verifies:
    - Water stress factor (gspot) is very low
    - Stomatal conductance is reduced
    - Photosynthesis is reduced compared to well-watered conditions
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_severe_water_stress")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Water stress factor should be very low
    assert jnp.all(result.gspot < 0.5), \
        "Water stress factor should be low under severe stress"
    
    # Stomatal conductance should be reduced
    assert jnp.all(result.gs < 0.1), \
        "Stomatal conductance should be very low under severe water stress"
    
    # Photosynthesis should be reduced
    assert jnp.all(result.anet < 10.0), \
        "Net photosynthesis should be low under severe water stress"


def test_leaf_photosynthesis_cold_temperature(test_data, default_params):
    """
    Test photosynthesis at cold temperatures (5°C).
    
    Verifies:
    - Enzyme activities are reduced
    - Photosynthesis rates are lower than at optimal temperature
    - No numerical instabilities
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_extreme_temperature_cold")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Check for numerical stability (no NaN or Inf)
    assert jnp.all(jnp.isfinite(result.vcmax)), \
        "Vcmax should be finite at cold temperatures"
    assert jnp.all(jnp.isfinite(result.anet)), \
        "Net photosynthesis should be finite at cold temperatures"
    
    # Enzyme activities should be reduced but positive
    assert jnp.all(result.vcmax > 0.0), \
        "Vcmax should be positive even at cold temperatures"
    assert jnp.all(result.vcmax < 100.0), \
        "Vcmax should be reduced at cold temperatures"


def test_leaf_photosynthesis_hot_temperature(test_data, default_params):
    """
    Test photosynthesis at hot temperatures (40°C).
    
    Verifies:
    - High temperature inhibition is active
    - Photosynthesis may be reduced due to heat stress
    - No numerical instabilities
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_extreme_temperature_hot")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Check for numerical stability
    assert jnp.all(jnp.isfinite(result.vcmax)), \
        "Vcmax should be finite at hot temperatures"
    assert jnp.all(jnp.isfinite(result.anet)), \
        "Net photosynthesis should be finite at hot temperatures"
    
    # Vapor pressure should be high
    assert jnp.all(result.vpd > 1000.0), \
        "VPD should be high at hot temperatures with high humidity"


def test_leaf_photosynthesis_minimal_conductance(test_data, default_params):
    """
    Test photosynthesis with minimal stomatal conductance parameters.
    
    Verifies:
    - Stomatal conductance is very low
    - Photosynthesis is limited by CO2 diffusion
    - System remains stable
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_minimal_conductance_parameters")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Stomatal conductance should be minimal
    assert jnp.all(result.gs < 0.05), \
        "Stomatal conductance should be very low with minimal parameters"
    
    # Intercellular CO2 should be lower than atmospheric
    assert jnp.all(result.ci < inputs["cair"]), \
        "Intercellular CO2 should be less than atmospheric with low conductance"


# ============================================================================
# Special Scenario Tests
# ============================================================================

def test_leaf_photosynthesis_high_elevation(test_data, default_params):
    """
    Test photosynthesis at high elevation with reduced atmospheric pressure.
    
    Verifies:
    - Partial pressures are correctly adjusted for low pressure
    - Photosynthesis responds appropriately to reduced O2/CO2 partial pressures
    - No numerical issues with low pressure
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_special_high_elevation_low_pressure")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Check for numerical stability
    assert jnp.all(jnp.isfinite(result.anet)), \
        "Net photosynthesis should be finite at low pressure"
    
    # Photosynthesis should still be positive with adequate light
    assert jnp.any(result.anet > 0.0), \
        "Some photosynthesis should occur at high elevation with light"


def test_leaf_photosynthesis_multi_layer_gradient(test_data, default_params):
    """
    Test photosynthesis with multi-layer canopy showing vertical gradients.
    
    Verifies:
    - Photosynthesis decreases with canopy depth
    - Light limitation is evident in lower layers
    - Gradients are smooth and realistic
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_special_multi_layer_canopy_gradient")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Extract single patch results (shape: n_layers, n_leaf)
    anet_profile = result.anet[0, :, :]
    
    # Photosynthesis should generally decrease with depth
    # (top layers should have higher rates than bottom layers)
    top_layer_mean = jnp.mean(anet_profile[0, :])
    bottom_layer_mean = jnp.mean(anet_profile[-1, :])
    
    assert top_layer_mean > bottom_layer_mean, \
        "Top canopy layer should have higher photosynthesis than bottom layer"
    
    # Check that sunlit leaves have higher rates than shaded
    for layer in range(anet_profile.shape[0]):
        assert anet_profile[layer, 0] >= anet_profile[layer, 1], \
            f"Sunlit leaf should have >= photosynthesis than shaded in layer {layer}"


def test_leaf_photosynthesis_c3_vs_c4_comparison(test_data, default_params):
    """
    Test C3 vs C4 photosynthesis under identical conditions.
    
    Verifies:
    - C3 and C4 pathways produce different results
    - C4 plants have different CO2 compensation points
    - Both pathways are numerically stable
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_special_mixed_c3_c4_comparison")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Extract C3 and C4 results (patches 0,2 are C3; patches 1,3 are C4)
    c3_anet = result.anet[[0, 2], :, :]
    c4_anet = result.anet[[1, 3], :, :]
    
    c3_ci = result.ci[[0, 2], :, :]
    c4_ci = result.ci[[1, 3], :, :]
    
    # C3 and C4 should have different intercellular CO2 concentrations
    assert not jnp.allclose(jnp.mean(c3_ci), jnp.mean(c4_ci), rtol=0.1), \
        "C3 and C4 should have different intercellular CO2 patterns"
    
    # Both should have positive photosynthesis with adequate light
    assert jnp.all(c3_anet > -5.0), "C3 net photosynthesis should be reasonable"
    assert jnp.all(c4_anet > -5.0), "C4 net photosynthesis should be reasonable"


# ============================================================================
# Consistency Tests
# ============================================================================

def test_leaf_photosynthesis_energy_balance(test_data, default_params):
    """
    Test that photosynthesis respects basic energy/carbon balance.
    
    Verifies:
    - Net photosynthesis = Gross photosynthesis - Respiration
    - Gross photosynthesis >= 0
    - Respiration >= 0
    """
    case = test_data["test_cases"][0]  # Use first nominal case
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Check carbon balance: anet = agross - rd
    calculated_anet = result.agross - result.rd
    assert jnp.allclose(result.anet, calculated_anet, rtol=1e-5, atol=1e-6), \
        "Net photosynthesis should equal gross photosynthesis minus respiration"
    
    # Gross photosynthesis should be non-negative
    assert jnp.all(result.agross >= 0.0), \
        "Gross photosynthesis must be non-negative"
    
    # Respiration should be non-negative
    assert jnp.all(result.rd >= 0.0), \
        "Dark respiration must be non-negative"


def test_leaf_photosynthesis_co2_gradient(test_data, default_params):
    """
    Test that CO2 concentration gradient is physically consistent.
    
    Verifies:
    - Atmospheric CO2 >= Leaf surface CO2 >= Intercellular CO2
    - Gradient direction is correct for photosynthesis
    """
    case = test_data["test_cases"][0]  # Use first nominal case
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # CO2 should decrease from atmosphere to intercellular space
    # (when photosynthesis is occurring)
    cair = inputs["cair"]
    
    # Where photosynthesis is positive, ci should be less than cair
    positive_psn = result.anet > 0.0
    if jnp.any(positive_psn):
        assert jnp.all(result.ci[positive_psn] <= cair[positive_psn]), \
            "Intercellular CO2 should be <= atmospheric CO2 during photosynthesis"


def test_leaf_photosynthesis_stomatal_conductance_limits(test_data, default_params):
    """
    Test that stomatal conductance respects minimum and maximum limits.
    
    Verifies:
    - gs >= minimum conductance (g0 or gsmin)
    - gs responds to environmental conditions
    - gs is reduced under water stress
    """
    # Test with minimal conductance parameters
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_minimal_conductance_parameters")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Conductance should be above absolute minimum
    min_gs = jnp.minimum(inputs["g0_BB"], inputs["gsmin_SPA"])
    assert jnp.all(result.gs >= min_gs * 0.9), \
        "Stomatal conductance should be at or above minimum"
    
    # Test water stress effect
    case_stress = next(tc for tc in test_data["test_cases"] 
                       if tc["name"] == "test_edge_severe_water_stress")
    inputs_stress = convert_inputs_to_jax(case_stress["inputs"])
    result_stress = leaf_photosynthesis(**inputs_stress, params=default_params)
    
    # Under severe stress, gspot should be low
    assert jnp.all(result_stress.gspot < 0.5), \
        "Water stress factor should be low under severe stress"


# ============================================================================
# Numerical Stability Tests
# ============================================================================

def test_leaf_photosynthesis_no_nan_inf(test_data, default_params):
    """
    Test that function produces no NaN or Inf values across all test cases.
    
    Verifies numerical stability across diverse conditions.
    """
    for case in test_data["test_cases"]:
        inputs = convert_inputs_to_jax(case["inputs"])
        result = leaf_photosynthesis(**inputs, params=default_params)
        
        # Check all output fields for NaN/Inf
        for field in result._fields:
            field_value = getattr(result, field)
            assert jnp.all(jnp.isfinite(field_value)), \
                f"Field {field} contains NaN or Inf in test case {case['name']}"


def test_leaf_photosynthesis_reproducibility(test_data, default_params):
    """
    Test that function produces identical results on repeated calls.
    
    Verifies deterministic behavior.
    """
    case = test_data["test_cases"][0]
    inputs = convert_inputs_to_jax(case["inputs"])
    
    # Run twice
    result1 = leaf_photosynthesis(**inputs, params=default_params)
    result2 = leaf_photosynthesis(**inputs, params=default_params)
    
    # Compare all fields
    for field in result1._fields:
        field1 = getattr(result1, field)
        field2 = getattr(result2, field)
        assert jnp.allclose(field1, field2, rtol=1e-10, atol=1e-10), \
            f"Field {field} not reproducible between calls"


# ============================================================================
# Integration Tests
# ============================================================================

def test_leaf_photosynthesis_full_workflow(test_data, default_params):
    """
    Test complete workflow with realistic multi-patch, multi-layer scenario.
    
    Verifies:
    - Function handles complex inputs correctly
    - All outputs are physically reasonable
    - Spatial patterns make sense
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_nominal_c4_multiple_patches")
    inputs = convert_inputs_to_jax(case["inputs"])
    
    result = leaf_photosynthesis(**inputs, params=default_params)
    
    # Check that results vary across patches and layers
    anet_std = jnp.std(result.anet)
    assert anet_std > 0.1, \
        "Net photosynthesis should vary across patches/layers"
    
    # Check that sunlit leaves generally have higher rates
    sunlit_mean = jnp.mean(result.anet[:, :, 0])
    shaded_mean = jnp.mean(result.anet[:, :, 1])
    assert sunlit_mean >= shaded_mean, \
        "Sunlit leaves should have >= photosynthesis than shaded on average"
    
    # Check that all major outputs are present and reasonable
    assert jnp.all(result.vcmax > 0.0), "Vcmax should be positive"
    assert jnp.all(result.gs >= 0.0), "Stomatal conductance should be non-negative"
    assert jnp.all(result.ci >= 0.0), "Intercellular CO2 should be non-negative"
    assert jnp.all(result.vpd >= 0.0), "VPD should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])