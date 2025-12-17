"""
Comprehensive pytest suite for MLCanopyNitrogenProfileMod.canopy_nitrogen_profile

This module tests the canopy nitrogen profile calculation function, which computes
photosynthetic parameters (Vcmax, Jmax, Rd, Kp) throughout the canopy based on
nitrogen distribution, light extinction, and temperature acclimation.

Test coverage includes:
- Nominal cases: typical C3/C4 plants with moderate LAI
- Edge cases: zero LAI, extreme temperatures, full shade
- Special cases: uniform clumping, variable layer counts
- Physical constraints: temperature > 0K, fractions in [0,1]
- Numerical validation: analytical vs numerical integration
"""

import json
import pytest
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Import the function and related types
# Adjust import path based on actual module structure
from multilayer_canopy.MLCanopyNitrogenProfileMod import (
    canopy_nitrogen_profile,
    CanopyNitrogenParams,
    CanopyNitrogenProfile,
    CanopyNitrogenValidation,
    get_default_params,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def test_data() -> Dict[str, Any]:
    """
    Load test data from JSON specification.
    
    Returns:
        Dictionary containing all test cases with inputs and metadata
    """
    test_data_json = {
        "function_name": "canopy_nitrogen_profile",
        "test_cases": [
            {
                "name": "test_nominal_single_patch_c3_moderate_lai",
                "inputs": {
                    "vcmaxpft": [62.0],
                    "c3psn": [1],
                    "tacclim": [293.15],
                    "dpai": [[0.5, 0.4, 0.3, 0.2, 0.1]],
                    "kb": [[0.5, 0.5, 0.5, 0.5, 0.5]],
                    "tbi": [[0.95, 0.85, 0.75, 0.65, 0.55]],
                    "fracsun": [[0.8, 0.7, 0.6, 0.5, 0.4]],
                    "clump_fac": [0.85],
                    "ncan": [5],
                    "lai": [1.5],
                    "sai": [0.5],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Single C3 patch with moderate LAI, typical temperate conditions, 5 canopy layers",
                    "edge_cases": []
                }
            },
            {
                "name": "test_nominal_multiple_patches_mixed_pathways",
                "inputs": {
                    "vcmaxpft": [62.0, 40.0, 55.0],
                    "c3psn": [1, 0, 1],
                    "tacclim": [293.15, 303.15, 283.15],
                    "dpai": [
                        [0.6, 0.5, 0.4, 0.3, 0.2],
                        [0.8, 0.7, 0.6, 0.5, 0.4],
                        [0.4, 0.3, 0.2, 0.1, 0.05]
                    ],
                    "kb": [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.6, 0.6, 0.6, 0.6, 0.6],
                        [0.4, 0.4, 0.4, 0.4, 0.4]
                    ],
                    "tbi": [
                        [0.9, 0.8, 0.7, 0.6, 0.5],
                        [0.85, 0.75, 0.65, 0.55, 0.45],
                        [0.95, 0.88, 0.8, 0.72, 0.65]
                    ],
                    "fracsun": [
                        [0.75, 0.65, 0.55, 0.45, 0.35],
                        [0.7, 0.6, 0.5, 0.4, 0.3],
                        [0.8, 0.7, 0.6, 0.5, 0.4]
                    ],
                    "clump_fac": [0.85, 0.75, 0.9],
                    "ncan": [5, 5, 5],
                    "lai": [2.0, 3.0, 1.45],
                    "sai": [0.6, 0.8, 0.4],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Three patches with mixed C3/C4 pathways, varying temperatures and LAI values",
                    "edge_cases": []
                }
            },
            {
                "name": "test_nominal_high_lai_dense_canopy",
                "inputs": {
                    "vcmaxpft": [80.0, 70.0],
                    "c3psn": [1, 1],
                    "tacclim": [298.15, 295.15],
                    "dpai": [
                        [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                    ],
                    "kb": [
                        [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                    ],
                    "tbi": [
                        [0.88, 0.77, 0.67, 0.58, 0.5, 0.43, 0.37, 0.32, 0.27, 0.23],
                        [0.9, 0.8, 0.71, 0.63, 0.56, 0.5, 0.44, 0.39, 0.35, 0.31]
                    ],
                    "fracsun": [
                        [0.65, 0.55, 0.45, 0.38, 0.32, 0.27, 0.23, 0.19, 0.16, 0.13],
                        [0.7, 0.6, 0.5, 0.42, 0.35, 0.29, 0.24, 0.2, 0.17, 0.14]
                    ],
                    "clump_fac": [0.8, 0.82],
                    "ncan": [10, 10],
                    "lai": [6.6, 5.5],
                    "sai": [1.2, 1.0],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Dense canopy with high LAI (6-7), 10 layers, typical for tropical forests",
                    "edge_cases": []
                }
            },
            {
                "name": "test_edge_zero_lai_no_canopy",
                "inputs": {
                    "vcmaxpft": [50.0, 45.0],
                    "c3psn": [1, 0],
                    "tacclim": [290.15, 295.15],
                    "dpai": [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    "kb": [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5]
                    ],
                    "tbi": [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0]
                    ],
                    "fracsun": [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0]
                    ],
                    "clump_fac": [1.0, 1.0],
                    "ncan": [0, 0],
                    "lai": [0.0, 0.0],
                    "sai": [0.0, 0.0],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "edge",
                    "description": "Zero LAI/SAI representing bare ground or dormant vegetation",
                    "edge_cases": ["zero_lai", "zero_dpai", "full_transmission"]
                }
            },
            {
                "name": "test_edge_minimal_lai_single_layer",
                "inputs": {
                    "vcmaxpft": [30.0],
                    "c3psn": [1],
                    "tacclim": [288.15],
                    "dpai": [[0.05]],
                    "kb": [[0.5]],
                    "tbi": [[0.975]],
                    "fracsun": [[0.95]],
                    "clump_fac": [0.95],
                    "ncan": [1],
                    "lai": [0.05],
                    "sai": [0.01],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "edge",
                    "description": "Minimal LAI with single canopy layer, sparse vegetation",
                    "edge_cases": ["minimal_lai", "single_layer"]
                }
            },
            {
                "name": "test_edge_extreme_cold_temperature",
                "inputs": {
                    "vcmaxpft": [40.0, 35.0],
                    "c3psn": [1, 1],
                    "tacclim": [253.15, 258.15],
                    "dpai": [
                        [0.3, 0.25, 0.2, 0.15, 0.1],
                        [0.35, 0.3, 0.25, 0.2, 0.15]
                    ],
                    "kb": [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5]
                    ],
                    "tbi": [
                        [0.92, 0.84, 0.76, 0.68, 0.6],
                        [0.91, 0.82, 0.73, 0.64, 0.55]
                    ],
                    "fracsun": [
                        [0.78, 0.68, 0.58, 0.48, 0.38],
                        [0.76, 0.66, 0.56, 0.46, 0.36]
                    ],
                    "clump_fac": [0.88, 0.86],
                    "ncan": [5, 5],
                    "lai": [1.0, 1.25],
                    "sai": [0.3, 0.35],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "edge",
                    "description": "Extreme cold temperatures (-20°C to -15°C), boreal/arctic conditions",
                    "edge_cases": ["extreme_cold", "low_temperature_acclimation"]
                }
            },
            {
                "name": "test_edge_extreme_hot_temperature",
                "inputs": {
                    "vcmaxpft": [90.0, 85.0],
                    "c3psn": [0, 0],
                    "tacclim": [313.15, 318.15],
                    "dpai": [
                        [0.7, 0.6, 0.5, 0.4, 0.3],
                        [0.8, 0.7, 0.6, 0.5, 0.4]
                    ],
                    "kb": [
                        [0.6, 0.6, 0.6, 0.6, 0.6],
                        [0.65, 0.65, 0.65, 0.65, 0.65]
                    ],
                    "tbi": [
                        [0.87, 0.75, 0.64, 0.54, 0.45],
                        [0.85, 0.72, 0.6, 0.5, 0.41]
                    ],
                    "fracsun": [
                        [0.68, 0.58, 0.48, 0.38, 0.28],
                        [0.65, 0.55, 0.45, 0.35, 0.25]
                    ],
                    "clump_fac": [0.7, 0.72],
                    "ncan": [5, 5],
                    "lai": [2.5, 3.0],
                    "sai": [0.7, 0.8],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "edge",
                    "description": "Extreme hot temperatures (40-45°C), C4 plants in desert/tropical conditions",
                    "edge_cases": ["extreme_heat", "high_temperature_acclimation", "c4_pathway"]
                }
            },
            {
                "name": "test_edge_full_shade_zero_transmission",
                "inputs": {
                    "vcmaxpft": [55.0],
                    "c3psn": [1],
                    "tacclim": [295.15],
                    "dpai": [[0.8, 0.7, 0.6, 0.5, 0.4]],
                    "kb": [[0.8, 0.8, 0.8, 0.8, 0.8]],
                    "tbi": [[0.5, 0.25, 0.12, 0.06, 0.0]],
                    "fracsun": [[0.4, 0.2, 0.1, 0.05, 0.0]],
                    "clump_fac": [0.6],
                    "ncan": [5],
                    "lai": [3.0],
                    "sai": [0.8],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "edge",
                    "description": "Deep canopy with zero transmission in lower layers, fully shaded understory",
                    "edge_cases": ["zero_transmission", "zero_fracsun", "full_shade"]
                }
            },
            {
                "name": "test_special_uniform_clumping_perfect_distribution",
                "inputs": {
                    "vcmaxpft": [65.0, 58.0],
                    "c3psn": [1, 1],
                    "tacclim": [291.15, 294.15],
                    "dpai": [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.6, 0.6, 0.6, 0.6, 0.6]
                    ],
                    "kb": [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5]
                    ],
                    "tbi": [
                        [0.9, 0.81, 0.73, 0.66, 0.59],
                        [0.88, 0.77, 0.68, 0.6, 0.53]
                    ],
                    "fracsun": [
                        [0.72, 0.64, 0.57, 0.51, 0.45],
                        [0.7, 0.62, 0.54, 0.48, 0.42]
                    ],
                    "clump_fac": [1.0, 1.0],
                    "ncan": [5, 5],
                    "lai": [2.5, 3.0],
                    "sai": [0.5, 0.6],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "special",
                    "description": "Perfect uniform foliage distribution (clump_fac=1.0), equal dpai across layers",
                    "edge_cases": ["uniform_clumping", "equal_layer_distribution"]
                }
            },
            {
                "name": "test_special_variable_layer_count_asymmetric",
                "inputs": {
                    "vcmaxpft": [60.0, 55.0, 50.0, 65.0],
                    "c3psn": [1, 0, 1, 0],
                    "tacclim": [292.15, 298.15, 287.15, 303.15],
                    "dpai": [
                        [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.4, 0.35, 0.3, 0.25, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.25, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.48, 0.45, 0.42, 0.4, 0.38, 0.35, 0.32, 0.3, 0.28, 0.25, 0.22, 0.2, 0.18, 0.15]
                    ],
                    "kb": [
                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
                        [0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48],
                        [0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52]
                    ],
                    "tbi": [
                        [0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [0.92, 0.84, 0.77, 0.7, 0.64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [0.94, 0.88, 0.83, 0.78, 0.73, 0.68, 0.64, 0.6, 0.56, 0.53, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48, 0.43, 0.39, 0.35, 0.31, 0.28, 0.25, 0.23, 0.2]
                    ],
                    "fracsun": [
                        [0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.78, 0.7, 0.62, 0.54, 0.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.8, 0.73, 0.67, 0.61, 0.55, 0.5, 0.45, 0.4, 0.36, 0.32, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.72, 0.65, 0.59, 0.53, 0.48, 0.43, 0.39, 0.35, 0.31, 0.28, 0.25, 0.22, 0.2, 0.18, 0.15]
                    ],
                    "clump_fac": [0.9, 0.75, 0.85, 0.7],
                    "ncan": [1, 5, 10, 15],
                    "lai": [1.5, 1.5, 1.5, 5.58],
                    "sai": [0.3, 0.4, 0.5, 1.2],
                    "params": None,
                    "validate": True
                },
                "metadata": {
                    "type": "special",
                    "description": "Four patches with highly variable layer counts (1, 5, 10, 15), testing asymmetric canopy structures",
                    "edge_cases": ["variable_ncan", "single_layer", "many_layers"]
                }
            }
        ]
    }
    return test_data_json


@pytest.fixture
def default_params() -> CanopyNitrogenParams:
    """
    Fixture providing default canopy nitrogen parameters.
    
    Returns:
        CanopyNitrogenParams with default values
    """
    return get_default_params()


# ============================================================================
# Helper Functions
# ============================================================================

def convert_to_jax_arrays(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert test input lists to JAX arrays.
    
    Args:
        inputs: Dictionary with test inputs as Python lists
        
    Returns:
        Dictionary with inputs converted to JAX arrays
    """
    jax_inputs = {}
    for key, value in inputs.items():
        if key == "params":
            jax_inputs[key] = value  # Keep None or CanopyNitrogenParams as-is
        elif key == "validate":
            jax_inputs[key] = value  # Keep boolean as-is
        elif isinstance(value, list):
            jax_inputs[key] = jnp.array(value)
        else:
            jax_inputs[key] = value
    return jax_inputs


def get_expected_shapes(inputs: Dict[str, Any]) -> Dict[str, Tuple[int, ...]]:
    """
    Calculate expected output shapes based on input dimensions.
    
    Args:
        inputs: Dictionary with JAX array inputs
        
    Returns:
        Dictionary mapping output field names to expected shapes
    """
    n_patches = inputs["vcmaxpft"].shape[0]
    n_layers = inputs["dpai"].shape[1]
    
    return {
        # Profile arrays (n_patches, n_layers)
        "vcmax25_leaf_sun": (n_patches, n_layers),
        "vcmax25_leaf_sha": (n_patches, n_layers),
        "jmax25_leaf_sun": (n_patches, n_layers),
        "jmax25_leaf_sha": (n_patches, n_layers),
        "rd25_leaf_sun": (n_patches, n_layers),
        "rd25_leaf_sha": (n_patches, n_layers),
        "kp25_leaf_sun": (n_patches, n_layers),
        "kp25_leaf_sha": (n_patches, n_layers),
        "vcmax25_profile": (n_patches, n_layers),
        "jmax25_profile": (n_patches, n_layers),
        "rd25_profile": (n_patches, n_layers),
        "kp25_profile": (n_patches, n_layers),
        # Scalar arrays (n_patches,)
        "kn": (n_patches,),
        # Validation arrays
        "numerical": (n_patches,),
        "analytical": (n_patches,),
        "max_error": (),  # scalar
        "is_valid": (),  # scalar boolean
    }


# ============================================================================
# Test: Output Shapes
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "test_nominal_single_patch_c3_moderate_lai",
    "test_nominal_multiple_patches_mixed_pathways",
    "test_nominal_high_lai_dense_canopy",
    "test_edge_zero_lai_no_canopy",
    "test_edge_minimal_lai_single_layer",
    "test_special_variable_layer_count_asymmetric",
])
def test_canopy_nitrogen_profile_shapes(test_data: Dict[str, Any], test_case: str):
    """
    Test that canopy_nitrogen_profile returns outputs with correct shapes.
    
    Verifies that all fields in CanopyNitrogenProfile and CanopyNitrogenValidation
    have the expected dimensions based on input array shapes.
    
    Args:
        test_data: Fixture containing all test cases
        test_case: Name of the specific test case to run
    """
    # Find the test case
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    inputs = convert_to_jax_arrays(case["inputs"])
    expected_shapes = get_expected_shapes(inputs)
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check CanopyNitrogenProfile shapes
    assert profile.vcmax25_leaf_sun.shape == expected_shapes["vcmax25_leaf_sun"], \
        f"vcmax25_leaf_sun shape mismatch: {profile.vcmax25_leaf_sun.shape} != {expected_shapes['vcmax25_leaf_sun']}"
    assert profile.vcmax25_leaf_sha.shape == expected_shapes["vcmax25_leaf_sha"], \
        f"vcmax25_leaf_sha shape mismatch"
    assert profile.jmax25_leaf_sun.shape == expected_shapes["jmax25_leaf_sun"], \
        f"jmax25_leaf_sun shape mismatch"
    assert profile.jmax25_leaf_sha.shape == expected_shapes["jmax25_leaf_sha"], \
        f"jmax25_leaf_sha shape mismatch"
    assert profile.rd25_leaf_sun.shape == expected_shapes["rd25_leaf_sun"], \
        f"rd25_leaf_sun shape mismatch"
    assert profile.rd25_leaf_sha.shape == expected_shapes["rd25_leaf_sha"], \
        f"rd25_leaf_sha shape mismatch"
    assert profile.kp25_leaf_sun.shape == expected_shapes["kp25_leaf_sun"], \
        f"kp25_leaf_sun shape mismatch"
    assert profile.kp25_leaf_sha.shape == expected_shapes["kp25_leaf_sha"], \
        f"kp25_leaf_sha shape mismatch"
    assert profile.vcmax25_profile.shape == expected_shapes["vcmax25_profile"], \
        f"vcmax25_profile shape mismatch"
    assert profile.jmax25_profile.shape == expected_shapes["jmax25_profile"], \
        f"jmax25_profile shape mismatch"
    assert profile.rd25_profile.shape == expected_shapes["rd25_profile"], \
        f"rd25_profile shape mismatch"
    assert profile.kp25_profile.shape == expected_shapes["kp25_profile"], \
        f"kp25_profile shape mismatch"
    assert profile.kn.shape == expected_shapes["kn"], \
        f"kn shape mismatch"
    
    # Check CanopyNitrogenValidation shapes (if validation enabled)
    if inputs["validate"] and validation is not None:
        assert validation.numerical.shape == expected_shapes["numerical"], \
            f"numerical shape mismatch"
        assert validation.analytical.shape == expected_shapes["analytical"], \
            f"analytical shape mismatch"
        assert validation.max_error.shape == expected_shapes["max_error"], \
            f"max_error shape mismatch"
        assert validation.is_valid.shape == expected_shapes["is_valid"], \
            f"is_valid shape mismatch"


# ============================================================================
# Test: Data Types
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "test_nominal_single_patch_c3_moderate_lai",
    "test_nominal_multiple_patches_mixed_pathways",
])
def test_canopy_nitrogen_profile_dtypes(test_data: Dict[str, Any], test_case: str):
    """
    Test that canopy_nitrogen_profile returns outputs with correct data types.
    
    Verifies that all numeric outputs are floating point and boolean outputs
    are boolean type.
    
    Args:
        test_data: Fixture containing all test cases
        test_case: Name of the specific test case to run
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check CanopyNitrogenProfile dtypes (all should be float)
    assert jnp.issubdtype(profile.vcmax25_leaf_sun.dtype, jnp.floating), \
        f"vcmax25_leaf_sun should be float, got {profile.vcmax25_leaf_sun.dtype}"
    assert jnp.issubdtype(profile.vcmax25_leaf_sha.dtype, jnp.floating), \
        f"vcmax25_leaf_sha should be float"
    assert jnp.issubdtype(profile.jmax25_leaf_sun.dtype, jnp.floating), \
        f"jmax25_leaf_sun should be float"
    assert jnp.issubdtype(profile.jmax25_leaf_sha.dtype, jnp.floating), \
        f"jmax25_leaf_sha should be float"
    assert jnp.issubdtype(profile.rd25_leaf_sun.dtype, jnp.floating), \
        f"rd25_leaf_sun should be float"
    assert jnp.issubdtype(profile.rd25_leaf_sha.dtype, jnp.floating), \
        f"rd25_leaf_sha should be float"
    assert jnp.issubdtype(profile.kp25_leaf_sun.dtype, jnp.floating), \
        f"kp25_leaf_sun should be float"
    assert jnp.issubdtype(profile.kp25_leaf_sha.dtype, jnp.floating), \
        f"kp25_leaf_sha should be float"
    assert jnp.issubdtype(profile.vcmax25_profile.dtype, jnp.floating), \
        f"vcmax25_profile should be float"
    assert jnp.issubdtype(profile.jmax25_profile.dtype, jnp.floating), \
        f"jmax25_profile should be float"
    assert jnp.issubdtype(profile.rd25_profile.dtype, jnp.floating), \
        f"rd25_profile should be float"
    assert jnp.issubdtype(profile.kp25_profile.dtype, jnp.floating), \
        f"kp25_profile should be float"
    assert jnp.issubdtype(profile.kn.dtype, jnp.floating), \
        f"kn should be float"
    
    # Check CanopyNitrogenValidation dtypes
    if validation is not None:
        assert jnp.issubdtype(validation.numerical.dtype, jnp.floating), \
            f"numerical should be float"
        assert jnp.issubdtype(validation.analytical.dtype, jnp.floating), \
            f"analytical should be float"
        assert jnp.issubdtype(validation.max_error.dtype, jnp.floating), \
            f"max_error should be float"
        assert jnp.issubdtype(validation.is_valid.dtype, jnp.bool_), \
            f"is_valid should be boolean, got {validation.is_valid.dtype}"


# ============================================================================
# Test: Physical Constraints
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "test_nominal_single_patch_c3_moderate_lai",
    "test_nominal_multiple_patches_mixed_pathways",
    "test_nominal_high_lai_dense_canopy",
    "test_edge_extreme_cold_temperature",
    "test_edge_extreme_hot_temperature",
])
def test_canopy_nitrogen_profile_physical_constraints(test_data: Dict[str, Any], test_case: str):
    """
    Test that canopy_nitrogen_profile outputs satisfy physical constraints.
    
    Verifies:
    - All photosynthetic parameters are non-negative
    - Nitrogen decay coefficient (kn) is non-negative
    - No NaN or Inf values in outputs
    
    Args:
        test_data: Fixture containing all test cases
        test_case: Name of the specific test case to run
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check non-negativity of photosynthetic parameters
    assert jnp.all(profile.vcmax25_leaf_sun >= 0), \
        "vcmax25_leaf_sun contains negative values"
    assert jnp.all(profile.vcmax25_leaf_sha >= 0), \
        "vcmax25_leaf_sha contains negative values"
    assert jnp.all(profile.jmax25_leaf_sun >= 0), \
        "jmax25_leaf_sun contains negative values"
    assert jnp.all(profile.jmax25_leaf_sha >= 0), \
        "jmax25_leaf_sha contains negative values"
    assert jnp.all(profile.rd25_leaf_sun >= 0), \
        "rd25_leaf_sun contains negative values"
    assert jnp.all(profile.rd25_leaf_sha >= 0), \
        "rd25_leaf_sha contains negative values"
    assert jnp.all(profile.kp25_leaf_sun >= 0), \
        "kp25_leaf_sun contains negative values"
    assert jnp.all(profile.kp25_leaf_sha >= 0), \
        "kp25_leaf_sha contains negative values"
    assert jnp.all(profile.vcmax25_profile >= 0), \
        "vcmax25_profile contains negative values"
    assert jnp.all(profile.jmax25_profile >= 0), \
        "jmax25_profile contains negative values"
    assert jnp.all(profile.rd25_profile >= 0), \
        "rd25_profile contains negative values"
    assert jnp.all(profile.kp25_profile >= 0), \
        "kp25_profile contains negative values"
    assert jnp.all(profile.kn >= 0), \
        "kn contains negative values"
    
    # Check for NaN/Inf
    assert jnp.all(jnp.isfinite(profile.vcmax25_leaf_sun)), \
        "vcmax25_leaf_sun contains NaN or Inf"
    assert jnp.all(jnp.isfinite(profile.vcmax25_leaf_sha)), \
        "vcmax25_leaf_sha contains NaN or Inf"
    assert jnp.all(jnp.isfinite(profile.jmax25_leaf_sun)), \
        "jmax25_leaf_sun contains NaN or Inf"
    assert jnp.all(jnp.isfinite(profile.jmax25_leaf_sha)), \
        "jmax25_leaf_sha contains NaN or Inf"
    assert jnp.all(jnp.isfinite(profile.rd25_leaf_sun)), \
        "rd25_leaf_sun contains NaN or Inf"
    assert jnp.all(jnp.isfinite(profile.rd25_leaf_sha)), \
        "rd25_leaf_sha contains NaN or Inf"
    assert jnp.all(jnp.isfinite(profile.kp25_leaf_sun)), \
        "kp25_leaf_sun contains NaN or Inf"
    assert jnp.all(jnp.isfinite(profile.kp25_leaf_sha)), \
        "kp25_leaf_sha contains NaN or Inf"
    assert jnp.all(jnp.isfinite(profile.kn)), \
        "kn contains NaN or Inf"


# ============================================================================
# Test: Edge Cases - Zero LAI
# ============================================================================

def test_canopy_nitrogen_profile_zero_lai(test_data: Dict[str, Any]):
    """
    Test canopy_nitrogen_profile with zero LAI (bare ground).
    
    Verifies that the function handles zero LAI gracefully and returns
    appropriate values (likely zeros or minimal values for photosynthetic
    parameters).
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_zero_lai_no_canopy")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # With zero LAI and zero dpai, expect minimal or zero photosynthetic activity
    # The exact behavior depends on implementation, but values should be finite
    assert jnp.all(jnp.isfinite(profile.vcmax25_profile)), \
        "vcmax25_profile should be finite for zero LAI"
    assert jnp.all(jnp.isfinite(profile.kn)), \
        "kn should be finite for zero LAI"
    
    # Validation should still work
    if validation is not None:
        assert jnp.isfinite(validation.max_error), \
            "max_error should be finite for zero LAI"


# ============================================================================
# Test: Edge Cases - Single Layer
# ============================================================================

def test_canopy_nitrogen_profile_single_layer(test_data: Dict[str, Any]):
    """
    Test canopy_nitrogen_profile with single canopy layer.
    
    Verifies that the function correctly handles the minimal case of a
    single-layer canopy with very low LAI.
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_minimal_lai_single_layer")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check that outputs are reasonable for single layer
    n_patches = inputs["vcmaxpft"].shape[0]
    assert profile.vcmax25_profile.shape[0] == n_patches, \
        "Profile should have correct number of patches"
    
    # Values should be positive and finite
    assert jnp.all(profile.vcmax25_profile >= 0), \
        "vcmax25_profile should be non-negative"
    assert jnp.all(jnp.isfinite(profile.vcmax25_profile)), \
        "vcmax25_profile should be finite"


# ============================================================================
# Test: Edge Cases - Extreme Temperatures
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "test_edge_extreme_cold_temperature",
    "test_edge_extreme_hot_temperature",
])
def test_canopy_nitrogen_profile_extreme_temperatures(test_data: Dict[str, Any], test_case: str):
    """
    Test canopy_nitrogen_profile with extreme temperatures.
    
    Verifies that the function handles extreme cold (-20°C) and hot (45°C)
    temperatures without numerical issues, and that temperature acclimation
    affects the Jmax/Vcmax ratio appropriately.
    
    Args:
        test_data: Fixture containing all test cases
        test_case: Name of the specific test case to run
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check that outputs are finite despite extreme temperatures
    assert jnp.all(jnp.isfinite(profile.vcmax25_profile)), \
        f"vcmax25_profile should be finite for {test_case}"
    assert jnp.all(jnp.isfinite(profile.jmax25_profile)), \
        f"jmax25_profile should be finite for {test_case}"
    assert jnp.all(jnp.isfinite(profile.rd25_profile)), \
        f"rd25_profile should be finite for {test_case}"
    
    # For C3 plants, check that Jmax/Vcmax ratio is reasonable
    c3_mask = inputs["c3psn"] == 1
    if jnp.any(c3_mask):
        # Where vcmax25 > 0, jmax25/vcmax25 should be positive and reasonable
        vcmax_nonzero = profile.vcmax25_profile[c3_mask] > 1e-10
        if jnp.any(vcmax_nonzero):
            ratio = jnp.where(
                vcmax_nonzero,
                profile.jmax25_profile[c3_mask] / profile.vcmax25_profile[c3_mask],
                0.0
            )
            # Typical range is 1.5-3.0, but allow wider range for extreme temps
            assert jnp.all((ratio >= 0) & (ratio <= 10.0)), \
                f"Jmax/Vcmax ratio out of reasonable range for {test_case}"


# ============================================================================
# Test: Edge Cases - Full Shade
# ============================================================================

def test_canopy_nitrogen_profile_full_shade(test_data: Dict[str, Any]):
    """
    Test canopy_nitrogen_profile with zero transmission in lower layers.
    
    Verifies that the function correctly handles deep canopy conditions where
    lower layers receive no direct beam radiation (fracsun=0, tbi=0).
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_edge_full_shade_zero_transmission")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check that shaded leaf parameters are computed for all layers
    assert jnp.all(jnp.isfinite(profile.vcmax25_leaf_sha)), \
        "vcmax25_leaf_sha should be finite in full shade"
    assert jnp.all(profile.vcmax25_leaf_sha >= 0), \
        "vcmax25_leaf_sha should be non-negative in full shade"
    
    # In fully shaded layers (fracsun=0), sunlit and shaded values should be similar
    # or sunlit should be zero/minimal
    fully_shaded = inputs["fracsun"] < 1e-6
    if jnp.any(fully_shaded):
        # Shaded values should still be positive (nitrogen still present)
        assert jnp.all(profile.vcmax25_leaf_sha[fully_shaded] >= 0), \
            "Shaded leaves should have non-negative vcmax25 even in full shade"


# ============================================================================
# Test: Special Cases - Uniform Clumping
# ============================================================================

def test_canopy_nitrogen_profile_uniform_clumping(test_data: Dict[str, Any]):
    """
    Test canopy_nitrogen_profile with perfect uniform foliage distribution.
    
    Verifies that clump_fac=1.0 (uniform distribution) produces expected
    results with equal dpai across layers.
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_special_uniform_clumping_perfect_distribution")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # With uniform clumping and equal dpai, nitrogen profile should decay
    # smoothly through canopy
    for patch_idx in range(inputs["vcmaxpft"].shape[0]):
        ncan = int(inputs["ncan"][patch_idx])
        if ncan > 1:
            # Check that vcmax25 decreases monotonically through canopy
            # (from top to bottom, accounting for nitrogen decay)
            vcmax_profile = profile.vcmax25_profile[patch_idx, :ncan]
            # Allow for numerical precision issues
            diffs = vcmax_profile[:-1] - vcmax_profile[1:]
            assert jnp.all(diffs >= -1e-6), \
                f"Vcmax25 should decrease monotonically through canopy for patch {patch_idx}"


# ============================================================================
# Test: Special Cases - Variable Layer Count
# ============================================================================

def test_canopy_nitrogen_profile_variable_layers(test_data: Dict[str, Any]):
    """
    Test canopy_nitrogen_profile with highly variable layer counts.
    
    Verifies that the function correctly handles patches with different
    numbers of canopy layers (1, 5, 10, 15) in a single call.
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_special_variable_layer_count_asymmetric")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check each patch individually
    n_patches = inputs["vcmaxpft"].shape[0]
    for patch_idx in range(n_patches):
        ncan = int(inputs["ncan"][patch_idx])
        
        # Active layers should have positive values
        if ncan > 0:
            active_layers = profile.vcmax25_profile[patch_idx, :ncan]
            assert jnp.all(active_layers >= 0), \
                f"Active layers should have non-negative vcmax25 for patch {patch_idx}"
            assert jnp.all(jnp.isfinite(active_layers)), \
                f"Active layers should have finite vcmax25 for patch {patch_idx}"
        
        # Inactive layers (beyond ncan) should be zero or minimal
        if ncan < inputs["dpai"].shape[1]:
            inactive_layers = profile.vcmax25_profile[patch_idx, ncan:]
            # Inactive layers should have zero dpai, so profile values should be zero or minimal
            assert jnp.all(inactive_layers >= 0), \
                f"Inactive layers should be non-negative for patch {patch_idx}"


# ============================================================================
# Test: Validation - Numerical Integration
# ============================================================================

@pytest.mark.parametrize("test_case", [
    "test_nominal_single_patch_c3_moderate_lai",
    "test_nominal_multiple_patches_mixed_pathways",
    "test_nominal_high_lai_dense_canopy",
])
def test_canopy_nitrogen_profile_validation(test_data: Dict[str, Any], test_case: str):
    """
    Test that numerical integration validation passes for nominal cases.
    
    Verifies that the numerical integration of vcmax25 through the canopy
    matches the analytical solution within tolerance.
    
    Args:
        test_data: Fixture containing all test cases
        test_case: Name of the specific test case to run
    """
    case = next(tc for tc in test_data["test_cases"] if tc["name"] == test_case)
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function with validation enabled
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check that validation was performed
    assert validation is not None, \
        "Validation should be performed when validate=True"
    
    # Check that validation passed
    assert validation.is_valid, \
        f"Validation should pass for {test_case}: " \
        f"max_error={float(validation.max_error):.2e}, " \
        f"numerical={validation.numerical}, analytical={validation.analytical}"
    
    # Check that numerical and analytical values are close
    assert jnp.allclose(validation.numerical, validation.analytical, rtol=1e-4, atol=1e-6), \
        f"Numerical and analytical integrals should match for {test_case}"
    
    # Check that max_error is small
    assert validation.max_error < 1e-4, \
        f"Maximum error should be small for {test_case}: {float(validation.max_error):.2e}"


# ============================================================================
# Test: C3 vs C4 Pathways
# ============================================================================

def test_canopy_nitrogen_profile_c3_vs_c4(test_data: Dict[str, Any]):
    """
    Test that C3 and C4 pathways produce different photosynthetic parameters.
    
    Verifies that:
    - C3 plants have positive Jmax values
    - C4 plants have positive Kp values
    - Rd values differ between C3 and C4 (different ratios to Vcmax)
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_nominal_multiple_patches_mixed_pathways")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Identify C3 and C4 patches
    c3_mask = inputs["c3psn"] == 1
    c4_mask = inputs["c3psn"] == 0
    
    # C3 plants should have positive Jmax
    if jnp.any(c3_mask):
        c3_jmax = profile.jmax25_profile[c3_mask]
        # Where vcmax > 0, jmax should also be > 0
        c3_vcmax = profile.vcmax25_profile[c3_mask]
        active_c3 = c3_vcmax > 1e-10
        if jnp.any(active_c3):
            assert jnp.all(c3_jmax[active_c3] > 0), \
                "C3 plants should have positive Jmax where Vcmax > 0"
    
    # C4 plants should have positive Kp
    if jnp.any(c4_mask):
        c4_kp = profile.kp25_profile[c4_mask]
        # Where vcmax > 0, kp should also be > 0
        c4_vcmax = profile.vcmax25_profile[c4_mask]
        active_c4 = c4_vcmax > 1e-10
        if jnp.any(active_c4):
            assert jnp.all(c4_kp[active_c4] > 0), \
                "C4 plants should have positive Kp where Vcmax > 0"
    
    # Rd/Vcmax ratio should differ between C3 and C4
    if jnp.any(c3_mask) and jnp.any(c4_mask):
        c3_vcmax = profile.vcmax25_profile[c3_mask]
        c3_rd = profile.rd25_profile[c3_mask]
        c4_vcmax = profile.vcmax25_profile[c4_mask]
        c4_rd = profile.rd25_profile[c4_mask]
        
        # Calculate ratios where vcmax > 0
        c3_active = c3_vcmax > 1e-10
        c4_active = c4_vcmax > 1e-10
        
        if jnp.any(c3_active) and jnp.any(c4_active):
            c3_ratio = jnp.mean(c3_rd[c3_active] / c3_vcmax[c3_active])
            c4_ratio = jnp.mean(c4_rd[c4_active] / c4_vcmax[c4_active])
            
            # C4 typically has higher Rd/Vcmax ratio (0.025 vs 0.015)
            assert c4_ratio > c3_ratio, \
                f"C4 Rd/Vcmax ratio ({c4_ratio:.4f}) should be higher than C3 ({c3_ratio:.4f})"


# ============================================================================
# Test: Sunlit vs Shaded Leaves
# ============================================================================

def test_canopy_nitrogen_profile_sunlit_vs_shaded(test_data: Dict[str, Any]):
    """
    Test that sunlit and shaded leaves have appropriate parameter values.
    
    Verifies that:
    - Sunlit leaves generally have higher photosynthetic capacity
    - Both sunlit and shaded values are non-negative
    - Profile values are weighted averages of sunlit and shaded
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_nominal_single_patch_c3_moderate_lai")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check that both sunlit and shaded values are non-negative
    assert jnp.all(profile.vcmax25_leaf_sun >= 0), \
        "Sunlit vcmax25 should be non-negative"
    assert jnp.all(profile.vcmax25_leaf_sha >= 0), \
        "Shaded vcmax25 should be non-negative"
    
    # In layers with significant sunlit fraction, sunlit values should be >= shaded
    # (due to higher nitrogen allocation to sunlit leaves)
    significant_sun = inputs["fracsun"] > 0.1
    if jnp.any(significant_sun):
        sun_vs_shade = profile.vcmax25_leaf_sun[significant_sun] >= profile.vcmax25_leaf_sha[significant_sun]
        # Allow for some numerical tolerance
        assert jnp.mean(sun_vs_shade) > 0.8, \
            "Sunlit leaves should generally have higher or equal vcmax25 compared to shaded leaves"
    
    # Profile values should be between sunlit and shaded values (weighted average)
    # This is a sanity check on the weighting calculation
    for patch_idx in range(inputs["vcmaxpft"].shape[0]):
        ncan = int(inputs["ncan"][patch_idx])
        if ncan > 0:
            for layer_idx in range(ncan):
                profile_val = profile.vcmax25_profile[patch_idx, layer_idx]
                sun_val = profile.vcmax25_leaf_sun[patch_idx, layer_idx]
                sha_val = profile.vcmax25_leaf_sha[patch_idx, layer_idx]
                
                # Profile should be between min and max of sun/shade (with tolerance)
                min_val = jnp.minimum(sun_val, sha_val)
                max_val = jnp.maximum(sun_val, sha_val)
                
                assert profile_val >= min_val - 1e-6, \
                    f"Profile value should be >= min(sun, shade) for patch {patch_idx}, layer {layer_idx}"
                assert profile_val <= max_val + 1e-6, \
                    f"Profile value should be <= max(sun, shade) for patch {patch_idx}, layer {layer_idx}"


# ============================================================================
# Test: Parameter Defaults
# ============================================================================

def test_canopy_nitrogen_profile_default_params(test_data: Dict[str, Any], default_params: CanopyNitrogenParams):
    """
    Test that function works correctly with default parameters.
    
    Verifies that passing params=None uses default parameters and produces
    valid results.
    
    Args:
        test_data: Fixture containing all test cases
        default_params: Fixture providing default parameters
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_nominal_single_patch_c3_moderate_lai")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run with default params (None)
    profile_default, _ = canopy_nitrogen_profile(**inputs)
    
    # Run with explicit default params
    inputs_explicit = inputs.copy()
    inputs_explicit["params"] = default_params
    profile_explicit, _ = canopy_nitrogen_profile(**inputs_explicit)
    
    # Results should be identical
    assert jnp.allclose(profile_default.vcmax25_profile, profile_explicit.vcmax25_profile, rtol=1e-10), \
        "Default params (None) should produce same results as explicit default params"
    assert jnp.allclose(profile_default.jmax25_profile, profile_explicit.jmax25_profile, rtol=1e-10), \
        "Jmax profiles should match with default params"
    assert jnp.allclose(profile_default.kn, profile_explicit.kn, rtol=1e-10), \
        "Nitrogen decay coefficients should match with default params"


# ============================================================================
# Test: Validation Disabled
# ============================================================================

def test_canopy_nitrogen_profile_validation_disabled(test_data: Dict[str, Any]):
    """
    Test that validation can be disabled and returns None.
    
    Verifies that when validate=False, the function returns None for the
    validation output.
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_nominal_single_patch_c3_moderate_lai")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Disable validation
    inputs["validate"] = False
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Validation should be None
    assert validation is None, \
        "Validation should be None when validate=False"
    
    # Profile should still be valid
    assert profile is not None, \
        "Profile should still be returned when validate=False"
    assert jnp.all(jnp.isfinite(profile.vcmax25_profile)), \
        "Profile should be finite when validate=False"


# ============================================================================
# Test: Nitrogen Decay Coefficient
# ============================================================================

def test_canopy_nitrogen_profile_nitrogen_decay(test_data: Dict[str, Any]):
    """
    Test that nitrogen decay coefficient (kn) is calculated correctly.
    
    Verifies that:
    - kn is positive
    - kn affects the vertical profile of photosynthetic parameters
    - Higher kn leads to steeper decline through canopy
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_nominal_high_lai_dense_canopy")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Check that kn is positive
    assert jnp.all(profile.kn > 0), \
        "Nitrogen decay coefficient should be positive"
    
    # Check that vcmax25 decreases through canopy
    for patch_idx in range(inputs["vcmaxpft"].shape[0]):
        ncan = int(inputs["ncan"][patch_idx])
        if ncan > 2:  # Need at least 3 layers to check trend
            vcmax_profile = profile.vcmax25_profile[patch_idx, :ncan]
            
            # Calculate relative decline from top to bottom
            top_val = vcmax_profile[0]
            bottom_val = vcmax_profile[ncan-1]
            
            if top_val > 1e-10:  # Avoid division by zero
                relative_decline = (top_val - bottom_val) / top_val
                
                # Should have significant decline (at least 10% for dense canopy)
                assert relative_decline > 0.1, \
                    f"Vcmax25 should decline significantly through dense canopy for patch {patch_idx}"


# ============================================================================
# Test: Conservation of Total Nitrogen
# ============================================================================

def test_canopy_nitrogen_profile_nitrogen_conservation(test_data: Dict[str, Any]):
    """
    Test that total canopy nitrogen is conserved in the profile.
    
    Verifies that the numerical integration of vcmax25 through the canopy
    matches the analytical expectation based on top-of-canopy values and
    nitrogen decay.
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_nominal_multiple_patches_mixed_pathways")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run function with validation
    profile, validation = canopy_nitrogen_profile(**inputs)
    
    # Validation checks nitrogen conservation
    assert validation is not None, \
        "Validation should be enabled for this test"
    
    # Check that numerical and analytical integrals are close
    relative_error = jnp.abs(validation.numerical - validation.analytical) / (validation.analytical + 1e-10)
    
    assert jnp.all(relative_error < 1e-3), \
        f"Relative error in nitrogen conservation should be < 0.1%: {relative_error}"
    
    # Check that validation passed
    assert validation.is_valid, \
        "Nitrogen conservation validation should pass"


# ============================================================================
# Test: Consistency Across Patches
# ============================================================================

def test_canopy_nitrogen_profile_patch_independence(test_data: Dict[str, Any]):
    """
    Test that patches are processed independently.
    
    Verifies that results for each patch depend only on that patch's inputs,
    not on other patches in the batch.
    
    Args:
        test_data: Fixture containing all test cases
    """
    case = next(tc for tc in test_data["test_cases"] 
                if tc["name"] == "test_nominal_multiple_patches_mixed_pathways")
    inputs = convert_to_jax_arrays(case["inputs"])
    
    # Run with all patches
    profile_all, _ = canopy_nitrogen_profile(**inputs)
    
    # Run each patch individually and compare
    n_patches = inputs["vcmaxpft"].shape[0]
    for patch_idx in range(n_patches):
        # Create single-patch inputs
        single_inputs = {
            "vcmaxpft": inputs["vcmaxpft"][patch_idx:patch_idx+1],
            "c3psn": inputs["c3psn"][patch_idx:patch_idx+1],
            "tacclim": inputs["tacclim"][patch_idx:patch_idx+1],
            "dpai": inputs["dpai"][patch_idx:patch_idx+1],
            "kb": inputs["kb"][patch_idx:patch_idx+1],
            "tbi": inputs["tbi"][patch_idx:patch_idx+1],
            "fracsun": inputs["fracsun"][patch_idx:patch_idx+1],
            "clump_fac": inputs["clump_fac"][patch_idx:patch_idx+1],
            "ncan": inputs["ncan"][patch_idx:patch_idx+1],
            "lai": inputs["lai"][patch_idx:patch_idx+1],
            "sai": inputs["sai"][patch_idx:patch_idx+1],
            "params": inputs["params"],
            "validate": False,  # Skip validation for speed
        }
        
        # Run single patch
        profile_single, _ = canopy_nitrogen_profile(**single_inputs)
        
        # Compare results
        assert jnp.allclose(
            profile_all.vcmax25_profile[patch_idx:patch_idx+1],
            profile_single.vcmax25_profile,
            rtol=1e-10,
            atol=1e-10
        ), f"Patch {patch_idx} vcmax25_profile should be independent of other patches"
        
        assert jnp.allclose(
            profile_all.kn[patch_idx:patch_idx+1],
            profile_single.kn,
            rtol=1e-10,
            atol=1e-10
        ), f"Patch {patch_idx} kn should be independent of other patches"