"""
Comprehensive pytest suite for MLLeafBoundaryLayerMod.leaf_boundary_layer function.

This module tests the leaf boundary layer conductance calculations across various
environmental conditions, leaf sizes, and edge cases. Tests cover:
- Nominal conditions (temperate, tropical, alpine, desert)
- Edge cases (minimal wind, extreme gradients, leaf size extremes)
- Output shapes, dtypes, and physical constraints
- Multiple patches and canopy layers
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict, Any, List
import json

# Import the module under test
# Note: Adjust import path based on actual module location
try:
    from multilayer_canopy.MLLeafBoundaryLayerMod import (
        leaf_boundary_layer,
        leaf_boundary_layer_sunlit_shaded,
        get_default_params,
        BoundaryLayerParams,
        LeafBoundaryLayerOutputs,
    )
except ImportError:
    # Fallback for testing - create mock structures
    class BoundaryLayerParams(NamedTuple):
        gb_type: int = 0
        gb_factor: float = 1.0
        visc0: float = 1.326e-5
        dh0: float = 1.895e-5
        dv0: float = 2.178e-5
        dc0: float = 1.381e-5
        tfrz: float = 273.15
        grav: float = 9.80616

    class LeafBoundaryLayerOutputs(NamedTuple):
        gbh: jnp.ndarray
        gbv: jnp.ndarray
        gbc: jnp.ndarray

    def get_default_params():
        return BoundaryLayerParams()

    def leaf_boundary_layer(*args, **kwargs):
        raise NotImplementedError("Module not available for testing")


# Test data embedded from JSON
TEST_DATA = {
    "function_name": "leaf_boundary_layer",
    "test_cases": [
        {
            "name": "test_nominal_single_patch_single_layer",
            "inputs": {
                "dleaf": [0.05],
                "tref": [298.15],
                "pref": [101325.0],
                "wind": [[2.5]],
                "tair": [[295.15]],
                "tleaf": [[297.15]],
                "rhomol": [[41.5]],
                "dpai": [[1.5]],
                "params": None
            },
            "metadata": {
                "type": "nominal",
                "description": "Standard temperate conditions with single patch and canopy layer.",
                "edge_cases": []
            }
        },
        {
            "name": "test_nominal_multiple_patches_layers",
            "inputs": {
                "dleaf": [0.03, 0.08, 0.06],
                "tref": [300.15, 295.15, 303.15],
                "pref": [101325.0, 95000.0, 101325.0],
                "wind": [[1.2, 0.8, 0.5], [3.5, 2.1, 1.3], [0.9, 0.6, 0.3]],
                "tair": [[298.15, 297.15, 296.15], [293.15, 292.15, 291.15], [301.15, 300.15, 299.15]],
                "tleaf": [[299.15, 298.15, 297.15], [294.15, 293.15, 292.15], [302.15, 301.15, 300.15]],
                "rhomol": [[40.8, 41.0, 41.2], [43.5, 43.7, 43.9], [40.2, 40.4, 40.6]],
                "dpai": [[2.0, 1.5, 0.8], [3.5, 2.8, 1.2], [1.8, 1.3, 0.6]],
                "params": None
            },
            "metadata": {
                "type": "nominal",
                "description": "Multiple patches (3) with multiple canopy layers (3).",
                "edge_cases": []
            }
        },
        {
            "name": "test_nominal_tropical_conditions",
            "inputs": {
                "dleaf": [0.12, 0.15],
                "tref": [305.15, 308.15],
                "pref": [101325.0, 101325.0],
                "wind": [[4.5, 3.2, 2.1, 1.5], [5.2, 3.8, 2.6, 1.8]],
                "tair": [[303.15, 302.15, 301.15, 300.15], [306.15, 305.15, 304.15, 303.15]],
                "tleaf": [[304.65, 303.65, 302.65, 301.65], [307.65, 306.65, 305.65, 304.65]],
                "rhomol": [[39.5, 39.7, 39.9, 40.1], [38.8, 39.0, 39.2, 39.4]],
                "dpai": [[4.5, 3.8, 2.5, 1.2], [5.2, 4.3, 3.0, 1.5]],
                "params": None
            },
            "metadata": {
                "type": "nominal",
                "description": "Tropical rainforest conditions with large leaves and high LAI.",
                "edge_cases": []
            }
        },
        {
            "name": "test_nominal_cold_alpine_conditions",
            "inputs": {
                "dleaf": [0.02, 0.025],
                "tref": [278.15, 275.15],
                "pref": [80000.0, 78000.0],
                "wind": [[6.5, 4.2], [7.8, 5.1]],
                "tair": [[276.15, 275.15], [273.65, 272.65]],
                "tleaf": [[277.15, 276.15], [274.65, 273.65]],
                "rhomol": [[44.8, 45.2], [45.5, 45.9]],
                "dpai": [[0.8, 0.4], [1.2, 0.6]],
                "params": None
            },
            "metadata": {
                "type": "nominal",
                "description": "Alpine/cold climate with small leaves and low temperatures.",
                "edge_cases": []
            }
        },
        {
            "name": "test_nominal_desert_hot_conditions",
            "inputs": {
                "dleaf": [0.01, 0.015, 0.012],
                "tref": [313.15, 315.15, 311.15],
                "pref": [101325.0, 101325.0, 101325.0],
                "wind": [[8.5, 6.2, 4.1], [9.2, 6.8, 4.5], [7.8, 5.5, 3.7]],
                "tair": [[311.15, 310.15, 309.15], [313.15, 312.15, 311.15], [309.15, 308.15, 307.15]],
                "tleaf": [[315.15, 314.15, 313.15], [317.15, 316.15, 315.15], [313.15, 312.15, 311.15]],
                "rhomol": [[38.2, 38.4, 38.6], [37.8, 38.0, 38.2], [38.6, 38.8, 39.0]],
                "dpai": [[0.5, 0.3, 0.15], [0.6, 0.35, 0.18], [0.45, 0.28, 0.12]],
                "params": None
            },
            "metadata": {
                "type": "nominal",
                "description": "Hot desert conditions with very small leaves and high temperatures.",
                "edge_cases": []
            }
        },
        {
            "name": "test_edge_minimal_wind_calm_conditions",
            "inputs": {
                "dleaf": [0.05, 0.06],
                "tref": [298.15, 299.15],
                "pref": [101325.0, 101325.0],
                "wind": [[0.01, 0.005], [0.02, 0.01]],
                "tair": [[296.15, 295.15], [297.15, 296.15]],
                "tleaf": [[300.15, 299.15], [301.15, 300.15]],
                "rhomol": [[41.5, 41.7], [41.3, 41.5]],
                "dpai": [[1.5, 0.8], [1.8, 0.9]],
                "params": None
            },
            "metadata": {
                "type": "edge",
                "description": "Near-zero wind speeds to test free convection dominance.",
                "edge_cases": ["minimal_wind", "free_convection_dominant"]
            }
        },
        {
            "name": "test_edge_zero_wind_isothermal",
            "inputs": {
                "dleaf": [0.04],
                "tref": [298.15],
                "pref": [101325.0],
                "wind": [[0.0]],
                "tair": [[298.15]],
                "tleaf": [[298.15]],
                "rhomol": [[41.5]],
                "dpai": [[1.0]],
                "params": None
            },
            "metadata": {
                "type": "edge",
                "description": "Absolute zero wind and isothermal conditions.",
                "edge_cases": ["zero_wind", "isothermal", "minimal_convection"]
            }
        },
        {
            "name": "test_edge_extreme_temperature_gradient",
            "inputs": {
                "dleaf": [0.05, 0.06],
                "tref": [298.15, 300.15],
                "pref": [101325.0, 101325.0],
                "wind": [[0.5, 0.3], [0.6, 0.4]],
                "tair": [[288.15, 287.15], [290.15, 289.15]],
                "tleaf": [[318.15, 317.15], [320.15, 319.15]],
                "rhomol": [[41.5, 41.7], [41.3, 41.5]],
                "dpai": [[2.0, 1.2], [2.2, 1.4]],
                "params": None
            },
            "metadata": {
                "type": "edge",
                "description": "Extreme temperature difference (30K) between leaf and air.",
                "edge_cases": ["extreme_temperature_gradient", "high_grashof"]
            }
        },
        {
            "name": "test_edge_very_small_leaves",
            "inputs": {
                "dleaf": [0.001, 0.0005, 0.002],
                "tref": [298.15, 299.15, 297.15],
                "pref": [101325.0, 101325.0, 101325.0],
                "wind": [[2.0, 1.5], [2.5, 1.8], [1.8, 1.3]],
                "tair": [[296.15, 295.15], [297.15, 296.15], [295.15, 294.15]],
                "tleaf": [[297.15, 296.15], [298.15, 297.15], [296.15, 295.15]],
                "rhomol": [[41.5, 41.7], [41.3, 41.5], [41.7, 41.9]],
                "dpai": [[0.5, 0.3], [0.6, 0.35], [0.45, 0.28]],
                "params": None
            },
            "metadata": {
                "type": "edge",
                "description": "Very small leaf dimensions (0.5-2mm) typical of needle-like leaves.",
                "edge_cases": ["minimal_leaf_dimension", "low_reynolds"]
            }
        },
        {
            "name": "test_edge_very_large_leaves_high_wind",
            "inputs": {
                "dleaf": [0.5, 0.8],
                "tref": [298.15, 299.15],
                "pref": [101325.0, 101325.0],
                "wind": [[15.0, 12.0, 9.0], [18.0, 14.0, 10.0]],
                "tair": [[296.15, 295.15, 294.15], [297.15, 296.15, 295.15]],
                "tleaf": [[297.15, 296.15, 295.15], [298.15, 297.15, 296.15]],
                "rhomol": [[41.5, 41.7, 41.9], [41.3, 41.5, 41.7]],
                "dpai": [[3.0, 2.5, 1.8], [3.5, 2.8, 2.0]],
                "params": None
            },
            "metadata": {
                "type": "edge",
                "description": "Very large leaves (50-80cm) with high wind speeds.",
                "edge_cases": ["large_leaf_dimension", "high_reynolds", "high_wind", "turbulent_flow"]
            }
        }
    ]
}


# Fixtures
@pytest.fixture
def test_data():
    """
    Fixture providing all test case data.
    
    Returns:
        dict: Complete test data structure with all test cases
    """
    return TEST_DATA


@pytest.fixture
def default_params():
    """
    Fixture providing default boundary layer parameters.
    
    Returns:
        BoundaryLayerParams: Default parameters for boundary layer calculations
    """
    return get_default_params()


@pytest.fixture
def convert_to_jax():
    """
    Fixture providing a function to convert test inputs to JAX arrays.
    
    Returns:
        callable: Function that converts dict of lists to dict of JAX arrays
    """
    def _convert(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert test input lists to JAX arrays."""
        converted = {}
        for key, value in inputs.items():
            if key == "params":
                converted[key] = value
            elif isinstance(value, list):
                converted[key] = jnp.array(value)
            else:
                converted[key] = value
        return converted
    return _convert


# Helper functions
def get_test_case_by_name(test_data: Dict, name: str) -> Dict:
    """
    Retrieve a specific test case by name.
    
    Args:
        test_data: Complete test data structure
        name: Name of the test case to retrieve
        
    Returns:
        dict: Test case data
        
    Raises:
        ValueError: If test case name not found
    """
    for case in test_data["test_cases"]:
        if case["name"] == name:
            return case
    raise ValueError(f"Test case '{name}' not found")


def get_test_cases_by_type(test_data: Dict, case_type: str) -> List[Dict]:
    """
    Retrieve all test cases of a specific type.
    
    Args:
        test_data: Complete test data structure
        case_type: Type of test cases to retrieve ('nominal' or 'edge')
        
    Returns:
        list: List of test case dictionaries
    """
    return [case for case in test_data["test_cases"] 
            if case["metadata"]["type"] == case_type]


# Parametrized test data generators
def generate_test_ids(test_cases: List[Dict]) -> List[str]:
    """Generate readable test IDs from test case names."""
    return [case["name"].replace("test_", "") for case in test_cases]


# Shape validation tests
@pytest.mark.parametrize(
    "test_case",
    TEST_DATA["test_cases"],
    ids=generate_test_ids(TEST_DATA["test_cases"])
)
def test_leaf_boundary_layer_output_shapes(test_case, convert_to_jax):
    """
    Test that leaf_boundary_layer returns outputs with correct shapes.
    
    Verifies that:
    - Output is a LeafBoundaryLayerOutputs namedtuple
    - gbh, gbv, gbc all have shape (n_patches, n_canopy_layers)
    - Output shapes match input wind/tair/tleaf shapes
    
    Args:
        test_case: Test case dictionary with inputs and metadata
        convert_to_jax: Fixture to convert inputs to JAX arrays
    """
    inputs = convert_to_jax(test_case["inputs"])
    
    # Get expected shape from wind array
    expected_shape = inputs["wind"].shape
    n_patches = expected_shape[0]
    n_layers = expected_shape[1] if len(expected_shape) > 1 else 1
    
    # Call function
    result = leaf_boundary_layer(**inputs)
    
    # Verify output type
    assert isinstance(result, LeafBoundaryLayerOutputs), \
        f"Output should be LeafBoundaryLayerOutputs, got {type(result)}"
    
    # Verify shapes
    assert result.gbh.shape == expected_shape, \
        f"gbh shape {result.gbh.shape} != expected {expected_shape}"
    assert result.gbv.shape == expected_shape, \
        f"gbv shape {result.gbv.shape} != expected {expected_shape}"
    assert result.gbc.shape == expected_shape, \
        f"gbc shape {result.gbc.shape} != expected {expected_shape}"


@pytest.mark.parametrize(
    "test_case",
    TEST_DATA["test_cases"],
    ids=generate_test_ids(TEST_DATA["test_cases"])
)
def test_leaf_boundary_layer_output_dtypes(test_case, convert_to_jax):
    """
    Test that leaf_boundary_layer returns outputs with correct data types.
    
    Verifies that:
    - All outputs (gbh, gbv, gbc) are JAX arrays
    - All outputs have float dtype
    
    Args:
        test_case: Test case dictionary with inputs and metadata
        convert_to_jax: Fixture to convert inputs to JAX arrays
    """
    inputs = convert_to_jax(test_case["inputs"])
    result = leaf_boundary_layer(**inputs)
    
    # Check that outputs are JAX arrays
    assert isinstance(result.gbh, jnp.ndarray), \
        f"gbh should be JAX array, got {type(result.gbh)}"
    assert isinstance(result.gbv, jnp.ndarray), \
        f"gbv should be JAX array, got {type(result.gbv)}"
    assert isinstance(result.gbc, jnp.ndarray), \
        f"gbc should be JAX array, got {type(result.gbc)}"
    
    # Check dtypes are float
    assert jnp.issubdtype(result.gbh.dtype, jnp.floating), \
        f"gbh dtype should be float, got {result.gbh.dtype}"
    assert jnp.issubdtype(result.gbv.dtype, jnp.floating), \
        f"gbv dtype should be float, got {result.gbv.dtype}"
    assert jnp.issubdtype(result.gbc.dtype, jnp.floating), \
        f"gbc dtype should be float, got {result.gbc.dtype}"


# Physical constraint tests
@pytest.mark.parametrize(
    "test_case",
    TEST_DATA["test_cases"],
    ids=generate_test_ids(TEST_DATA["test_cases"])
)
def test_leaf_boundary_layer_positive_conductances(test_case, convert_to_jax):
    """
    Test that all boundary layer conductances are non-negative.
    
    Physical constraint: Conductances represent mass/heat transfer rates
    and must be >= 0. Zero conductance is physically possible (no transfer),
    but negative values are non-physical.
    
    Args:
        test_case: Test case dictionary with inputs and metadata
        convert_to_jax: Fixture to convert inputs to JAX arrays
    """
    inputs = convert_to_jax(test_case["inputs"])
    result = leaf_boundary_layer(**inputs)
    
    # Check all conductances are non-negative
    assert jnp.all(result.gbh >= 0), \
        f"Heat conductance gbh has negative values: min={jnp.min(result.gbh)}"
    assert jnp.all(result.gbv >= 0), \
        f"Vapor conductance gbv has negative values: min={jnp.min(result.gbv)}"
    assert jnp.all(result.gbc >= 0), \
        f"CO2 conductance gbc has negative values: min={jnp.min(result.gbc)}"


@pytest.mark.parametrize(
    "test_case",
    TEST_DATA["test_cases"],
    ids=generate_test_ids(TEST_DATA["test_cases"])
)
def test_leaf_boundary_layer_finite_values(test_case, convert_to_jax):
    """
    Test that all outputs are finite (no NaN or Inf values).
    
    Verifies numerical stability across all test conditions including
    edge cases like zero wind and isothermal conditions.
    
    Args:
        test_case: Test case dictionary with inputs and metadata
        convert_to_jax: Fixture to convert inputs to JAX arrays
    """
    inputs = convert_to_jax(test_case["inputs"])
    result = leaf_boundary_layer(**inputs)
    
    # Check for finite values
    assert jnp.all(jnp.isfinite(result.gbh)), \
        f"gbh contains non-finite values (NaN/Inf) in test {test_case['name']}"
    assert jnp.all(jnp.isfinite(result.gbv)), \
        f"gbv contains non-finite values (NaN/Inf) in test {test_case['name']}"
    assert jnp.all(jnp.isfinite(result.gbc)), \
        f"gbc contains non-finite values (NaN/Inf) in test {test_case['name']}"


# Relative magnitude tests
@pytest.mark.parametrize(
    "test_case",
    get_test_cases_by_type(TEST_DATA, "nominal"),
    ids=generate_test_ids(get_test_cases_by_type(TEST_DATA, "nominal"))
)
def test_leaf_boundary_layer_conductance_ordering(test_case, convert_to_jax):
    """
    Test expected ordering of conductances based on molecular diffusivities.
    
    Physical expectation: For similar conditions, conductances should follow
    the order of molecular diffusivities:
    - Water vapor diffusivity > CO2 diffusivity > Heat diffusivity
    - Therefore: gbv >= gbc >= gbh (approximately)
    
    This test applies to nominal conditions where forced/free convection
    effects are similar across species. Edge cases may violate this due to
    different Prandtl/Schmidt numbers affecting laminar vs turbulent regimes.
    
    Args:
        test_case: Test case dictionary with inputs and metadata
        convert_to_jax: Fixture to convert inputs to JAX arrays
    """
    inputs = convert_to_jax(test_case["inputs"])
    result = leaf_boundary_layer(**inputs)
    
    # For most conditions, expect gbv > gbc due to higher diffusivity
    # Allow some tolerance for numerical effects
    ratio_v_to_c = result.gbv / (result.gbc + 1e-10)
    assert jnp.mean(ratio_v_to_c) > 0.95, \
        f"Expected gbv >= gbc on average, got mean ratio {jnp.mean(ratio_v_to_c):.3f}"


# Wind speed dependency tests
def test_leaf_boundary_layer_wind_speed_dependency(convert_to_jax):
    """
    Test that conductances increase with wind speed.
    
    Physical expectation: Higher wind speeds increase forced convection,
    leading to higher boundary layer conductances. This test uses identical
    conditions except for wind speed.
    """
    # Base case with moderate wind
    base_inputs = {
        "dleaf": [0.05],
        "tref": [298.15],
        "pref": [101325.0],
        "wind": [[2.0]],
        "tair": [[296.15]],
        "tleaf": [[298.15]],
        "rhomol": [[41.5]],
        "dpai": [[1.5]],
        "params": None
    }
    
    # High wind case
    high_wind_inputs = base_inputs.copy()
    high_wind_inputs["wind"] = [[8.0]]
    
    base_result = leaf_boundary_layer(**convert_to_jax(base_inputs))
    high_wind_result = leaf_boundary_layer(**convert_to_jax(high_wind_inputs))
    
    # Higher wind should give higher conductances
    assert jnp.all(high_wind_result.gbh > base_result.gbh), \
        "Heat conductance should increase with wind speed"
    assert jnp.all(high_wind_result.gbv > base_result.gbv), \
        "Vapor conductance should increase with wind speed"
    assert jnp.all(high_wind_result.gbc > base_result.gbc), \
        "CO2 conductance should increase with wind speed"


def test_leaf_boundary_layer_leaf_size_dependency(convert_to_jax):
    """
    Test that conductances decrease with increasing leaf size.
    
    Physical expectation: Larger leaves have thicker boundary layers,
    resulting in lower conductances (higher resistance to transfer).
    This follows from Reynolds number scaling.
    """
    # Small leaf case
    small_leaf_inputs = {
        "dleaf": [0.02],
        "tref": [298.15],
        "pref": [101325.0],
        "wind": [[2.0]],
        "tair": [[296.15]],
        "tleaf": [[298.15]],
        "rhomol": [[41.5]],
        "dpai": [[1.5]],
        "params": None
    }
    
    # Large leaf case
    large_leaf_inputs = small_leaf_inputs.copy()
    large_leaf_inputs["dleaf"] = [0.20]
    
    small_result = leaf_boundary_layer(**convert_to_jax(small_leaf_inputs))
    large_result = leaf_boundary_layer(**convert_to_jax(large_leaf_inputs))
    
    # Smaller leaves should have higher conductances
    assert jnp.all(small_result.gbh > large_result.gbh), \
        "Small leaves should have higher heat conductance than large leaves"
    assert jnp.all(small_result.gbv > large_result.gbv), \
        "Small leaves should have higher vapor conductance than large leaves"
    assert jnp.all(small_result.gbc > large_result.gbc), \
        "Small leaves should have higher CO2 conductance than large leaves"


# Edge case specific tests
def test_leaf_boundary_layer_zero_wind_nonzero_conductance(convert_to_jax):
    """
    Test that conductances are non-zero even with zero wind speed.
    
    Physical expectation: With zero forced convection, free convection
    (driven by temperature difference) should still provide some conductance.
    This test verifies the free convection mechanism is working.
    """
    # Zero wind but with temperature difference
    inputs = {
        "dleaf": [0.05],
        "tref": [298.15],
        "pref": [101325.0],
        "wind": [[0.0]],
        "tair": [[295.15]],
        "tleaf": [[300.15]],  # 5K warmer than air
        "rhomol": [[41.5]],
        "dpai": [[1.5]],
        "params": None
    }
    
    result = leaf_boundary_layer(**convert_to_jax(inputs))
    
    # Should have non-zero conductances due to free convection
    assert jnp.all(result.gbh > 0), \
        "Heat conductance should be > 0 with zero wind (free convection)"
    assert jnp.all(result.gbv > 0), \
        "Vapor conductance should be > 0 with zero wind (free convection)"
    assert jnp.all(result.gbc > 0), \
        "CO2 conductance should be > 0 with zero wind (free convection)"


def test_leaf_boundary_layer_isothermal_minimal_conductance(convert_to_jax):
    """
    Test behavior under isothermal conditions with zero wind.
    
    Physical expectation: With no forced convection (zero wind) and no
    free convection (isothermal), conductances should be at minimum values
    (molecular diffusion only). This is the most challenging numerical case.
    """
    inputs = {
        "dleaf": [0.05],
        "tref": [298.15],
        "pref": [101325.0],
        "wind": [[0.0]],
        "tair": [[298.15]],
        "tleaf": [[298.15]],  # Same as air
        "rhomol": [[41.5]],
        "dpai": [[1.5]],
        "params": None
    }
    
    result = leaf_boundary_layer(**convert_to_jax(inputs))
    
    # Should still have small positive conductances
    assert jnp.all(result.gbh > 0), \
        "Heat conductance should be > 0 even in isothermal zero-wind case"
    assert jnp.all(result.gbv > 0), \
        "Vapor conductance should be > 0 even in isothermal zero-wind case"
    assert jnp.all(result.gbc > 0), \
        "CO2 conductance should be > 0 even in isothermal zero-wind case"
    
    # Values should be relatively small (molecular diffusion regime)
    assert jnp.all(result.gbh < 1.0), \
        "Heat conductance should be small in isothermal zero-wind case"


def test_leaf_boundary_layer_extreme_temperature_gradient(convert_to_jax):
    """
    Test that extreme temperature gradients produce reasonable conductances.
    
    Physical expectation: Large temperature differences drive strong free
    convection (high Grashof number), which should significantly increase
    conductances even with low wind speeds.
    """
    # Low wind but extreme temperature difference
    inputs = {
        "dleaf": [0.05],
        "tref": [298.15],
        "pref": [101325.0],
        "wind": [[0.5]],
        "tair": [[288.15]],
        "tleaf": [[318.15]],  # 30K difference
        "rhomol": [[41.5]],
        "dpai": [[1.5]],
        "params": None
    }
    
    result = leaf_boundary_layer(**convert_to_jax(inputs))
    
    # Should have substantial conductances due to strong free convection
    assert jnp.all(result.gbh > 0.5), \
        "Heat conductance should be substantial with extreme temperature gradient"
    assert jnp.all(result.gbv > 0.5), \
        "Vapor conductance should be substantial with extreme temperature gradient"
    assert jnp.all(result.gbc > 0.5), \
        "CO2 conductance should be substantial with extreme temperature gradient"


# Parameter variation tests
@pytest.mark.parametrize("gb_type", [0, 1, 2, 3])
def test_leaf_boundary_layer_gb_type_parameter(gb_type, convert_to_jax, default_params):
    """
    Test that different gb_type values produce valid outputs.
    
    Tests all four boundary layer calculation types:
    - 0: Simplified CLM5
    - 1: Laminar only
    - 2: Max of laminar/turbulent
    - 3: Max of laminar/turbulent plus free convection
    
    Args:
        gb_type: Boundary layer calculation type (0-3)
        convert_to_jax: Fixture to convert inputs to JAX arrays
        default_params: Default boundary layer parameters
    """
    # Create custom parameters with specified gb_type
    custom_params = BoundaryLayerParams(
        gb_type=gb_type,
        gb_factor=default_params.gb_factor,
        visc0=default_params.visc0,
        dh0=default_params.dh0,
        dv0=default_params.dv0,
        dc0=default_params.dc0,
        tfrz=default_params.tfrz,
        grav=default_params.grav
    )
    
    inputs = {
        "dleaf": [0.05],
        "tref": [298.15],
        "pref": [101325.0],
        "wind": [[2.0]],
        "tair": [[296.15]],
        "tleaf": [[298.15]],
        "rhomol": [[41.5]],
        "dpai": [[1.5]],
        "params": custom_params
    }
    
    result = leaf_boundary_layer(**convert_to_jax(inputs))
    
    # Verify valid outputs for this gb_type
    assert jnp.all(jnp.isfinite(result.gbh)), \
        f"gbh should be finite for gb_type={gb_type}"
    assert jnp.all(jnp.isfinite(result.gbv)), \
        f"gbv should be finite for gb_type={gb_type}"
    assert jnp.all(jnp.isfinite(result.gbc)), \
        f"gbc should be finite for gb_type={gb_type}"
    assert jnp.all(result.gbh > 0), \
        f"gbh should be positive for gb_type={gb_type}"


def test_leaf_boundary_layer_default_params_vs_explicit(convert_to_jax, default_params):
    """
    Test that using default params (None) gives same results as explicit defaults.
    
    Verifies that the default parameter handling is consistent.
    """
    inputs_default = {
        "dleaf": [0.05],
        "tref": [298.15],
        "pref": [101325.0],
        "wind": [[2.0]],
        "tair": [[296.15]],
        "tleaf": [[298.15]],
        "rhomol": [[41.5]],
        "dpai": [[1.5]],
        "params": None
    }
    
    inputs_explicit = inputs_default.copy()
    inputs_explicit["params"] = default_params
    
    result_default = leaf_boundary_layer(**convert_to_jax(inputs_default))
    result_explicit = leaf_boundary_layer(**convert_to_jax(inputs_explicit))
    
    # Results should be identical
    np.testing.assert_allclose(
        result_default.gbh, result_explicit.gbh,
        rtol=1e-10, atol=1e-10,
        err_msg="Default params should match explicit default params for gbh"
    )
    np.testing.assert_allclose(
        result_default.gbv, result_explicit.gbv,
        rtol=1e-10, atol=1e-10,
        err_msg="Default params should match explicit default params for gbv"
    )
    np.testing.assert_allclose(
        result_default.gbc, result_explicit.gbc,
        rtol=1e-10, atol=1e-10,
        err_msg="Default params should match explicit default params for gbc"
    )


# Broadcast and dimension tests
def test_leaf_boundary_layer_single_patch_multiple_layers(convert_to_jax):
    """
    Test proper broadcasting with single patch and multiple canopy layers.
    
    Verifies that scalar patch-level inputs (dleaf, tref, pref) are correctly
    broadcast across multiple canopy layers.
    """
    inputs = {
        "dleaf": [0.05],
        "tref": [298.15],
        "pref": [101325.0],
        "wind": [[3.0, 2.0, 1.0]],
        "tair": [[296.15, 295.15, 294.15]],
        "tleaf": [[298.15, 297.15, 296.15]],
        "rhomol": [[41.5, 41.7, 41.9]],
        "dpai": [[2.0, 1.5, 0.8]],
        "params": None
    }
    
    result = leaf_boundary_layer(**convert_to_jax(inputs))
    
    # Output should have shape (1, 3)
    assert result.gbh.shape == (1, 3), \
        f"Expected shape (1, 3), got {result.gbh.shape}"
    
    # Conductances should vary across layers due to different wind/temp
    assert not jnp.allclose(result.gbh[0, 0], result.gbh[0, 1]), \
        "Conductances should differ across canopy layers"


def test_leaf_boundary_layer_multiple_patches_single_layer(convert_to_jax):
    """
    Test proper handling of multiple patches with single canopy layer.
    
    Verifies that different patches with different conditions produce
    different conductances.
    """
    inputs = {
        "dleaf": [0.03, 0.08, 0.05],
        "tref": [298.15, 300.15, 295.15],
        "pref": [101325.0, 95000.0, 101325.0],
        "wind": [[2.0], [3.0], [1.5]],
        "tair": [[296.15], [298.15], [293.15]],
        "tleaf": [[298.15], [300.15], [295.15]],
        "rhomol": [[41.5], [40.5], [42.5]],
        "dpai": [[1.5], [2.0], [1.2]],
        "params": None
    }
    
    result = leaf_boundary_layer(**convert_to_jax(inputs))
    
    # Output should have shape (3, 1)
    assert result.gbh.shape == (3, 1), \
        f"Expected shape (3, 1), got {result.gbh.shape}"
    
    # Conductances should vary across patches due to different conditions
    assert not jnp.allclose(result.gbh[0, 0], result.gbh[1, 0]), \
        "Conductances should differ across patches"


# Consistency tests
def test_leaf_boundary_layer_symmetry_identical_inputs(convert_to_jax):
    """
    Test that identical patches produce identical outputs.
    
    Verifies computational consistency when inputs are duplicated.
    """
    inputs = {
        "dleaf": [0.05, 0.05],
        "tref": [298.15, 298.15],
        "pref": [101325.0, 101325.0],
        "wind": [[2.0, 1.5], [2.0, 1.5]],
        "tair": [[296.15, 295.15], [296.15, 295.15]],
        "tleaf": [[298.15, 297.15], [298.15, 297.15]],
        "rhomol": [[41.5, 41.7], [41.5, 41.7]],
        "dpai": [[1.5, 0.8], [1.5, 0.8]],
        "params": None
    }
    
    result = leaf_boundary_layer(**convert_to_jax(inputs))
    
    # Identical patches should produce identical results
    np.testing.assert_allclose(
        result.gbh[0, :], result.gbh[1, :],
        rtol=1e-10, atol=1e-10,
        err_msg="Identical patches should produce identical gbh"
    )
    np.testing.assert_allclose(
        result.gbv[0, :], result.gbv[1, :],
        rtol=1e-10, atol=1e-10,
        err_msg="Identical patches should produce identical gbv"
    )
    np.testing.assert_allclose(
        result.gbc[0, :], result.gbc[1, :],
        rtol=1e-10, atol=1e-10,
        err_msg="Identical patches should produce identical gbc"
    )


# Reasonable magnitude tests
@pytest.mark.parametrize(
    "test_case",
    get_test_cases_by_type(TEST_DATA, "nominal"),
    ids=generate_test_ids(get_test_cases_by_type(TEST_DATA, "nominal"))
)
def test_leaf_boundary_layer_reasonable_magnitudes(test_case, convert_to_jax):
    """
    Test that conductances fall within reasonable physical ranges.
    
    Typical boundary layer conductances for leaves range from about
    0.1 to 10 mol/m2/s depending on conditions. This test ensures
    outputs are in a physically plausible range for nominal conditions.
    
    Args:
        test_case: Test case dictionary with inputs and metadata
        convert_to_jax: Fixture to convert inputs to JAX arrays
    """
    inputs = convert_to_jax(test_case["inputs"])
    result = leaf_boundary_layer(**inputs)
    
    # Check reasonable ranges (0.01 to 20 mol/m2/s covers most conditions)
    assert jnp.all(result.gbh >= 0.01) and jnp.all(result.gbh <= 20.0), \
        f"gbh outside reasonable range [0.01, 20]: min={jnp.min(result.gbh):.3f}, max={jnp.max(result.gbh):.3f}"
    assert jnp.all(result.gbv >= 0.01) and jnp.all(result.gbv <= 20.0), \
        f"gbv outside reasonable range [0.01, 20]: min={jnp.min(result.gbv):.3f}, max={jnp.max(result.gbv):.3f}"
    assert jnp.all(result.gbc >= 0.01) and jnp.all(result.gbc <= 20.0), \
        f"gbc outside reasonable range [0.01, 20]: min={jnp.min(result.gbc):.3f}, max={jnp.max(result.gbc):.3f}"


# Documentation and metadata tests
def test_test_data_structure():
    """
    Test that the test data structure is valid and complete.
    
    Verifies:
    - All required fields are present
    - Test cases have proper structure
    - Metadata is complete
    """
    assert "function_name" in TEST_DATA
    assert "test_cases" in TEST_DATA
    assert TEST_DATA["function_name"] == "leaf_boundary_layer"
    
    for case in TEST_DATA["test_cases"]:
        assert "name" in case
        assert "inputs" in case
        assert "metadata" in case
        
        # Check inputs have all required fields
        inputs = case["inputs"]
        required_inputs = ["dleaf", "tref", "pref", "wind", "tair", 
                          "tleaf", "rhomol", "dpai", "params"]
        for req in required_inputs:
            assert req in inputs, f"Missing required input '{req}' in {case['name']}"
        
        # Check metadata structure
        metadata = case["metadata"]
        assert "type" in metadata
        assert "description" in metadata
        assert metadata["type"] in ["nominal", "edge"]


def test_test_coverage_completeness(test_data):
    """
    Test that test suite covers all important scenarios.
    
    Verifies that we have:
    - Multiple nominal test cases
    - Multiple edge case tests
    - Tests for different biomes/conditions
    - Tests for dimension variations
    """
    nominal_cases = get_test_cases_by_type(test_data, "nominal")
    edge_cases = get_test_cases_by_type(test_data, "edge")
    
    assert len(nominal_cases) >= 4, \
        f"Should have at least 4 nominal test cases, got {len(nominal_cases)}"
    assert len(edge_cases) >= 4, \
        f"Should have at least 4 edge test cases, got {len(edge_cases)}"
    
    # Check for dimension variety
    all_cases = test_data["test_cases"]
    n_patches_list = [len(case["inputs"]["dleaf"]) for case in all_cases]
    n_layers_list = [len(case["inputs"]["wind"][0]) for case in all_cases]
    
    assert len(set(n_patches_list)) >= 2, \
        "Should test multiple different numbers of patches"
    assert len(set(n_layers_list)) >= 2, \
        "Should test multiple different numbers of canopy layers"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])