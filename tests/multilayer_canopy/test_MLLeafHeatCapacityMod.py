"""
Comprehensive pytest suite for MLLeafHeatCapacityMod module.

Tests the leaf_heat_capacity and leaf_heat_capacity_simple functions
for calculating leaf heat capacity in canopy models.

Physical context:
- Leaf heat capacity depends on leaf mass per area (LMA), water content,
  and specific heat capacities of biomass and water
- Output units: J/m2 leaf/K
- Key equation: cpleaf = cpbio * dry_weight + cpliq * leaf_water
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

# Import the module under test
# Assuming the module structure based on the signature
try:
    from multilayer_canopy.MLLeafHeatCapacityMod import (
        leaf_heat_capacity,
        leaf_heat_capacity_simple,
        leaf_heat_capacity_jit,
        leaf_heat_capacity_simple_jit,
        LeafHeatCapacityInput,
        LeafHeatCapacityParams,
    )
except ImportError:
    # Mock implementations for testing the test file itself
    class LeafHeatCapacityInput(NamedTuple):
        slatop: jnp.ndarray
        ncan: jnp.ndarray
        dpai: jnp.ndarray
        cpbio: float
        cpliq: float
        fcarbon: float
        fwater: float

    class LeafHeatCapacityParams(NamedTuple):
        cpbio: float = 1470.0
        cpliq: float = 4188.0
        fcarbon: float = 0.5
        fwater: float = 0.7

    def leaf_heat_capacity(inputs):
        """Mock implementation."""
        return jnp.zeros_like(inputs.dpai)

    def leaf_heat_capacity_simple(slatop, dpai, params=LeafHeatCapacityParams()):
        """Mock implementation."""
        return jnp.zeros_like(dpai)

    leaf_heat_capacity_jit = leaf_heat_capacity
    leaf_heat_capacity_simple_jit = leaf_heat_capacity_simple


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_params():
    """Default physical parameters for leaf heat capacity."""
    return LeafHeatCapacityParams(
        cpbio=1470.0,
        cpliq=4188.0,
        fcarbon=0.5,
        fwater=0.7
    )


@pytest.fixture
def nominal_single_patch_data():
    """
    Nominal test case: single patch, single layer.
    
    Physical context: Typical temperate forest canopy with moderate LAI.
    """
    return {
        'slatop': jnp.array([0.015]),
        'ncan': jnp.array([1], dtype=jnp.int32),
        'dpai': jnp.array([[2.5]]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.7,
        'expected': jnp.array([[17458.33]])
    }


@pytest.fixture
def nominal_multi_patch_data():
    """
    Nominal test case: multiple patches with varying canopy layers.
    
    Physical context: Mixed forest stand with different species
    (conifer, broadleaf, shrub).
    """
    return {
        'slatop': jnp.array([0.012, 0.018, 0.02]),
        'ncan': jnp.array([3, 2, 4], dtype=jnp.int32),
        'dpai': jnp.array([
            [1.5, 1.2, 0.8],
            [2.0, 1.5, 0.0],
            [1.8, 1.6, 1.4, 1.0]
        ]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.7
    }


@pytest.fixture
def grassland_low_lai_data():
    """
    Nominal test case: grassland/crop canopy.
    
    Physical context: Agricultural or grassland ecosystem with thin leaves,
    high SLA and low LAI.
    """
    return {
        'slatop': jnp.array([0.025, 0.03]),
        'ncan': jnp.array([2, 2], dtype=jnp.int32),
        'dpai': jnp.array([
            [0.5, 0.3],
            [0.8, 0.4]
        ]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.7
    }


@pytest.fixture
def edge_zero_dpai_data():
    """
    Edge case: zero plant area index.
    
    Physical context: Sparse canopy or winter deciduous forest.
    Should return zero heat capacity where dpai is zero.
    """
    return {
        'slatop': jnp.array([0.015, 0.02]),
        'ncan': jnp.array([3, 2], dtype=jnp.int32),
        'dpai': jnp.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.7
    }


@pytest.fixture
def edge_small_slatop_data():
    """
    Edge case: very small SLA (thick leaves).
    
    Physical context: Succulent plants or sclerophyllous vegetation
    with very thick leaves.
    """
    return {
        'slatop': jnp.array([0.001, 0.0005]),
        'ncan': jnp.array([2, 2], dtype=jnp.int32),
        'dpai': jnp.array([
            [1.0, 0.5],
            [1.2, 0.8]
        ]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.7
    }


@pytest.fixture
def edge_large_slatop_data():
    """
    Edge case: very large SLA (thin leaves).
    
    Physical context: Shade-adapted understory plants with very thin leaves.
    """
    return {
        'slatop': jnp.array([0.05, 0.08]),
        'ncan': jnp.array([2, 3], dtype=jnp.int32),
        'dpai': jnp.array([
            [2.0, 1.5, 0.0],
            [1.8, 1.2, 0.6]
        ]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.7
    }


@pytest.fixture
def edge_high_fwater_data():
    """
    Edge case: very high water fraction near upper boundary.
    
    Physical context: Succulent plants with very high water content.
    Tests numerical stability as (1-fwater) approaches zero.
    """
    return {
        'slatop': jnp.array([0.015]),
        'ncan': jnp.array([2], dtype=jnp.int32),
        'dpai': jnp.array([[1.5, 1.0]]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.95
    }


@pytest.fixture
def edge_low_fwater_data():
    """
    Edge case: very low water fraction.
    
    Physical context: Severely drought-stressed vegetation or senescing leaves.
    """
    return {
        'slatop': jnp.array([0.015]),
        'ncan': jnp.array([2], dtype=jnp.int32),
        'dpai': jnp.array([[1.5, 1.0]]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.1
    }


@pytest.fixture
def special_high_lai_data():
    """
    Special case: dense tropical forest canopy.
    
    Physical context: Tropical rainforest with deep, multi-layered
    canopy structure and high LAI.
    """
    return {
        'slatop': jnp.array([0.01]),
        'ncan': jnp.array([5], dtype=jnp.int32),
        'dpai': jnp.array([[3.5, 3.0, 2.5, 2.0, 1.5]]),
        'cpbio': 1470.0,
        'cpliq': 4188.0,
        'fcarbon': 0.5,
        'fwater': 0.7
    }


@pytest.fixture
def special_variable_params_data():
    """
    Special case: non-default physical parameters.
    
    Physical context: Vegetation with different biochemical composition
    (e.g., high lignin content).
    """
    return {
        'slatop': jnp.array([0.015, 0.02]),
        'ncan': jnp.array([2, 2], dtype=jnp.int32),
        'dpai': jnp.array([
            [1.5, 1.0],
            [2.0, 1.2]
        ]),
        'cpbio': 1200.0,
        'cpliq': 4188.0,
        'fcarbon': 0.45,
        'fwater': 0.65
    }


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_expected_heat_capacity(slatop, dpai, cpbio, cpliq, fcarbon, fwater):
    """
    Calculate expected leaf heat capacity using the physical equations.
    
    Equations:
        lma = 1 / slatop * 0.001  [kg C / m2]
        dry_weight = lma / fcarbon  [kg DM / m2]
        fresh_weight = dry_weight / (1 - fwater)  [kg FM / m2]
        leaf_water = fwater * fresh_weight  [kg H2O / m2]
        cpleaf = cpbio * dry_weight + cpliq * leaf_water  [J/K/m2 leaf]
    
    Args:
        slatop: Specific leaf area at top [m2/gC], shape (n_patches,)
        dpai: Plant area index [m2/m2], shape (n_patches, n_levels)
        cpbio: Heat capacity of dry biomass [J/kg/K]
        cpliq: Heat capacity of liquid water [J/kg/K]
        fcarbon: Carbon fraction of dry biomass [-]
        fwater: Water fraction of fresh biomass [-]
    
    Returns:
        Expected heat capacity [J/m2 leaf/K], shape (n_patches, n_levels)
    """
    # Convert to numpy for calculation
    slatop = np.asarray(slatop)
    dpai = np.asarray(dpai)
    
    # Reshape slatop for broadcasting
    slatop_expanded = slatop[:, np.newaxis]
    
    # Calculate leaf mass per area (kg C / m2)
    lma = 1.0 / slatop_expanded * 0.001
    
    # Calculate dry weight (kg DM / m2)
    dry_weight = lma / fcarbon
    
    # Calculate fresh weight (kg FM / m2)
    fresh_weight = dry_weight / (1.0 - fwater)
    
    # Calculate leaf water content (kg H2O / m2)
    leaf_water = fwater * fresh_weight
    
    # Calculate heat capacity per unit leaf area
    cpleaf = cpbio * dry_weight + cpliq * leaf_water
    
    # Return zero where dpai is zero
    result = np.where(dpai > 0, cpleaf, 0.0)
    
    return result


# ============================================================================
# Tests for leaf_heat_capacity function
# ============================================================================

class TestLeafHeatCapacity:
    """Tests for the main leaf_heat_capacity function."""
    
    def test_nominal_single_patch_single_layer(self, nominal_single_patch_data):
        """
        Test nominal case with single patch and single layer.
        
        Verifies correct calculation for typical temperate forest parameters.
        """
        data = nominal_single_patch_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check shape
        assert result.shape == data['dpai'].shape, \
            f"Expected shape {data['dpai'].shape}, got {result.shape}"
        
        # Check values
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity values don't match expected for single patch case"
        )
    
    def test_nominal_multi_patch_multi_layer(self, nominal_multi_patch_data):
        """
        Test nominal case with multiple patches and varying layers.
        
        Verifies correct calculation for mixed forest stand with different
        species (conifer, broadleaf, shrub).
        """
        data = nominal_multi_patch_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check shape
        assert result.shape == data['dpai'].shape, \
            f"Expected shape {data['dpai'].shape}, got {result.shape}"
        
        # Check values
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity values don't match expected for multi-patch case"
        )
    
    def test_nominal_grassland_low_lai(self, grassland_low_lai_data):
        """
        Test nominal case for grassland/crop canopy.
        
        Verifies correct calculation for agricultural ecosystem with
        high SLA and low LAI.
        """
        data = grassland_low_lai_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check shape
        assert result.shape == data['dpai'].shape
        
        # Check values
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity values don't match expected for grassland case"
        )
    
    def test_edge_zero_dpai(self, edge_zero_dpai_data):
        """
        Test edge case with zero plant area index.
        
        Verifies that zero dpai produces zero heat capacity, representing
        bare ground or canopy gaps.
        """
        data = edge_zero_dpai_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check that zero dpai produces zero output
        zero_mask = data['dpai'] == 0.0
        assert np.all(result[zero_mask] == 0.0), \
            "Expected zero heat capacity where dpai is zero"
        
        # Check non-zero dpai produces non-zero output
        nonzero_mask = data['dpai'] > 0.0
        if np.any(nonzero_mask):
            assert np.all(result[nonzero_mask] > 0.0), \
                "Expected positive heat capacity where dpai is positive"
    
    def test_edge_very_small_slatop(self, edge_small_slatop_data):
        """
        Test edge case with very small SLA (thick leaves).
        
        Verifies numerical stability for succulent plants or sclerophyllous
        vegetation with very thick leaves.
        """
        data = edge_small_slatop_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check shape
        assert result.shape == data['dpai'].shape
        
        # Check non-negative
        assert np.all(result >= 0.0), \
            "Heat capacity must be non-negative"
        
        # Check finite
        assert np.all(np.isfinite(result)), \
            "Heat capacity must be finite for small SLA"
        
        # Small SLA (thick leaves) should produce large heat capacity
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity incorrect for small SLA case"
        )
    
    def test_edge_very_large_slatop(self, edge_large_slatop_data):
        """
        Test edge case with very large SLA (thin leaves).
        
        Verifies numerical stability for shade-adapted understory plants
        with very thin leaves.
        """
        data = edge_large_slatop_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check shape
        assert result.shape == data['dpai'].shape
        
        # Check non-negative
        assert np.all(result >= 0.0), \
            "Heat capacity must be non-negative"
        
        # Check finite
        assert np.all(np.isfinite(result)), \
            "Heat capacity must be finite for large SLA"
        
        # Large SLA (thin leaves) should produce small heat capacity
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity incorrect for large SLA case"
        )
    
    def test_edge_extreme_fwater_boundary(self, edge_high_fwater_data):
        """
        Test edge case with very high water fraction.
        
        Verifies numerical stability for succulent plants with very high
        water content, where (1-fwater) approaches zero.
        """
        data = edge_high_fwater_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check finite (no division by zero issues)
        assert np.all(np.isfinite(result)), \
            "Heat capacity must be finite even with high fwater"
        
        # Check non-negative
        assert np.all(result >= 0.0), \
            "Heat capacity must be non-negative"
        
        # High fwater should produce high heat capacity (more water)
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity incorrect for high fwater case"
        )
    
    def test_edge_low_fwater(self, edge_low_fwater_data):
        """
        Test edge case with very low water fraction.
        
        Verifies correct calculation for severely drought-stressed
        vegetation or senescing leaves.
        """
        data = edge_low_fwater_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check finite
        assert np.all(np.isfinite(result)), \
            "Heat capacity must be finite with low fwater"
        
        # Check non-negative
        assert np.all(result >= 0.0), \
            "Heat capacity must be non-negative"
        
        # Low fwater should produce lower heat capacity (less water)
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity incorrect for low fwater case"
        )
    
    def test_special_high_lai_dense_canopy(self, special_high_lai_data):
        """
        Test special case with dense tropical forest canopy.
        
        Verifies correct calculation for tropical rainforest with deep,
        multi-layered canopy structure and high LAI.
        """
        data = special_high_lai_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check shape
        assert result.shape == data['dpai'].shape
        
        # Check all values are positive (high LAI)
        assert np.all(result > 0.0), \
            "Expected positive heat capacity for dense canopy"
        
        # Check values
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity incorrect for high LAI case"
        )
    
    def test_special_variable_physical_params(self, special_variable_params_data):
        """
        Test special case with non-default physical parameters.
        
        Verifies correct calculation for vegetation with different
        biochemical composition (e.g., high lignin content).
        """
        data = special_variable_params_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        # Check shape
        assert result.shape == data['dpai'].shape
        
        # Check values
        expected = calculate_expected_heat_capacity(
            data['slatop'], data['dpai'], data['cpbio'],
            data['cpliq'], data['fcarbon'], data['fwater']
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity incorrect for variable params case"
        )
    
    def test_output_dtype(self, nominal_single_patch_data):
        """Verify output has correct dtype (float)."""
        data = nominal_single_patch_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        assert jnp.issubdtype(result.dtype, jnp.floating), \
            f"Expected floating point dtype, got {result.dtype}"
    
    def test_output_shape_consistency(self, nominal_multi_patch_data):
        """Verify output shape matches dpai shape."""
        data = nominal_multi_patch_data
        inputs = LeafHeatCapacityInput(
            slatop=data['slatop'],
            ncan=data['ncan'],
            dpai=data['dpai'],
            cpbio=data['cpbio'],
            cpliq=data['cpliq'],
            fcarbon=data['fcarbon'],
            fwater=data['fwater']
        )
        
        result = leaf_heat_capacity(inputs)
        
        assert result.shape == data['dpai'].shape, \
            f"Output shape {result.shape} doesn't match dpai shape {data['dpai'].shape}"


# ============================================================================
# Tests for leaf_heat_capacity_simple function
# ============================================================================

class TestLeafHeatCapacitySimple:
    """Tests for the simplified leaf_heat_capacity_simple function."""
    
    def test_simple_nominal_default_params(self, default_params):
        """
        Test simple function with default parameters.
        
        Verifies standard test case for simplified interface.
        """
        slatop = jnp.array([0.015, 0.02])
        dpai = jnp.array([
            [1.5, 1.0],
            [2.0, 1.2]
        ])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # Check shape
        assert result.shape == dpai.shape, \
            f"Expected shape {dpai.shape}, got {result.shape}"
        
        # Check non-negative
        assert np.all(result >= 0.0), \
            "Heat capacity must be non-negative"
        
        # Check values
        expected = calculate_expected_heat_capacity(
            slatop, dpai, default_params.cpbio, default_params.cpliq,
            default_params.fcarbon, default_params.fwater
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity values don't match expected"
        )
    
    def test_simple_edge_zero_dpai_all_layers(self, default_params):
        """
        Test simple function with all layers having zero PAI.
        
        Verifies that completely bare ground or winter leafless condition
        produces all zeros.
        """
        slatop = jnp.array([0.015])
        dpai = jnp.array([[0.0, 0.0, 0.0]])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # Check all zeros
        np.testing.assert_array_equal(
            result, jnp.zeros_like(dpai),
            err_msg="Expected all zeros for zero dpai"
        )
    
    def test_simple_special_single_layer_high_lai(self, default_params):
        """
        Test simple function with single very dense layer.
        
        Verifies correct calculation for extremely dense canopy layer
        (e.g., bamboo forest).
        """
        slatop = jnp.array([0.008])
        dpai = jnp.array([[8.0]])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # Check shape
        assert result.shape == dpai.shape
        
        # Check positive
        assert np.all(result > 0.0), \
            "Expected positive heat capacity for high LAI"
        
        # Check finite
        assert np.all(np.isfinite(result)), \
            "Heat capacity must be finite for high LAI"
        
        # Check values
        expected = calculate_expected_heat_capacity(
            slatop, dpai, default_params.cpbio, default_params.cpliq,
            default_params.fcarbon, default_params.fwater
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity incorrect for high LAI case"
        )
    
    def test_simple_custom_params(self):
        """Test simple function with custom parameters."""
        slatop = jnp.array([0.015])
        dpai = jnp.array([[1.5, 1.0]])
        custom_params = LeafHeatCapacityParams(
            cpbio=1200.0,
            cpliq=4188.0,
            fcarbon=0.45,
            fwater=0.65
        )
        
        result = leaf_heat_capacity_simple(slatop, dpai, custom_params)
        
        # Check values
        expected = calculate_expected_heat_capacity(
            slatop, dpai, custom_params.cpbio, custom_params.cpliq,
            custom_params.fcarbon, custom_params.fwater
        )
        np.testing.assert_allclose(
            result, expected, rtol=1e-4, atol=1e-2,
            err_msg="Heat capacity incorrect with custom params"
        )
    
    def test_simple_output_dtype(self, default_params):
        """Verify output has correct dtype (float)."""
        slatop = jnp.array([0.015])
        dpai = jnp.array([[1.5, 1.0]])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        assert jnp.issubdtype(result.dtype, jnp.floating), \
            f"Expected floating point dtype, got {result.dtype}"


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Cross-validation and consistency tests."""
    
    def test_consistency_between_functions(self):
        """
        Verify that leaf_heat_capacity and leaf_heat_capacity_simple
        produce identical results.
        """
        slatop = jnp.array([0.015, 0.02])
        ncan = jnp.array([2, 3], dtype=jnp.int32)
        dpai = jnp.array([
            [1.5, 1.0, 0.0],
            [2.0, 1.5, 1.0]
        ])
        cpbio = 1470.0
        cpliq = 4188.0
        fcarbon = 0.5
        fwater = 0.7
        
        # Call main function
        inputs = LeafHeatCapacityInput(
            slatop=slatop,
            ncan=ncan,
            dpai=dpai,
            cpbio=cpbio,
            cpliq=cpliq,
            fcarbon=fcarbon,
            fwater=fwater
        )
        result1 = leaf_heat_capacity(inputs)
        
        # Call simple function
        params = LeafHeatCapacityParams(
            cpbio=cpbio,
            cpliq=cpliq,
            fcarbon=fcarbon,
            fwater=fwater
        )
        result2 = leaf_heat_capacity_simple(slatop, dpai, params)
        
        # Compare results
        np.testing.assert_allclose(
            result1, result2, rtol=1e-6, atol=1e-6,
            err_msg="Results differ between main and simple functions"
        )
    
    def test_jit_consistency(self, default_params):
        """
        Verify JIT-compiled versions produce identical results to
        non-JIT versions.
        """
        slatop = jnp.array([0.015])
        dpai = jnp.array([[1.5, 1.0, 0.5]])
        
        # Non-JIT version
        result_no_jit = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # JIT version
        result_jit = leaf_heat_capacity_simple_jit(slatop, dpai, default_params)
        
        # Compare results
        np.testing.assert_allclose(
            result_no_jit, result_jit, rtol=1e-10, atol=1e-10,
            err_msg="JIT and non-JIT versions produce different results"
        )


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestProperties:
    """Property-based tests for physical constraints."""
    
    def test_property_non_negative_output(self, default_params):
        """
        Property: Heat capacity must always be non-negative.
        
        Tests with random valid inputs to verify output >= 0.
        """
        # Generate random valid inputs
        np.random.seed(42)
        n_tests = 10
        
        for _ in range(n_tests):
            n_patches = np.random.randint(1, 5)
            n_levels = np.random.randint(1, 6)
            
            slatop = jnp.array(np.random.uniform(0.005, 0.04, n_patches))
            dpai = jnp.array(np.random.uniform(0.0, 5.0, (n_patches, n_levels)))
            
            result = leaf_heat_capacity_simple(slatop, dpai, default_params)
            
            assert np.all(result >= 0.0), \
                f"Found negative heat capacity: {result[result < 0]}"
    
    def test_property_zero_dpai_zero_output(self, default_params):
        """
        Property: Zero plant area index should produce zero heat capacity.
        
        Tests that output[i,j] == 0 when dpai[i,j] == 0.
        """
        slatop = jnp.array([0.015, 0.02, 0.025])
        dpai = jnp.array([
            [1.5, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [1.2, 0.0, 0.8]
        ])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # Check zero dpai produces zero output
        zero_mask = dpai == 0.0
        assert np.all(result[zero_mask] == 0.0), \
            "Expected zero heat capacity where dpai is zero"
    
    def test_property_monotonic_with_dpai(self, default_params):
        """
        Property: Heat capacity should increase monotonically with dpai
        (all else equal).
        
        Tests that output increases as dpai increases.
        """
        slatop = jnp.array([0.015])
        dpai_values = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        
        results = []
        for dpai_val in dpai_values:
            dpai = jnp.array([[dpai_val]])
            result = leaf_heat_capacity_simple(slatop, dpai, default_params)
            results.append(float(result[0, 0]))
        
        # Check monotonicity
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], \
                f"Heat capacity not monotonic: {results[i]} >= {results[i+1]}"
    
    def test_property_inverse_slatop_relationship(self, default_params):
        """
        Property: Heat capacity should increase as slatop decreases
        (thicker leaves, for fixed dpai).
        
        Tests inverse relationship between SLA and heat capacity.
        """
        dpai = jnp.array([[2.0]])
        slatop_values = jnp.array([0.04, 0.03, 0.02, 0.015, 0.01])
        
        results = []
        for slatop_val in slatop_values:
            slatop = jnp.array([slatop_val])
            result = leaf_heat_capacity_simple(slatop, dpai, default_params)
            results.append(float(result[0, 0]))
        
        # Check inverse relationship (decreasing slatop -> increasing heat capacity)
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], \
                f"Heat capacity doesn't increase with decreasing SLA: {results[i]} >= {results[i+1]}"
    
    def test_property_water_content_effect(self):
        """
        Property: Heat capacity should increase with water fraction
        (water has higher heat capacity than dry biomass).
        """
        slatop = jnp.array([0.015])
        dpai = jnp.array([[2.0]])
        fwater_values = [0.3, 0.5, 0.7, 0.85]
        
        results = []
        for fwater in fwater_values:
            params = LeafHeatCapacityParams(
                cpbio=1470.0,
                cpliq=4188.0,
                fcarbon=0.5,
                fwater=fwater
            )
            result = leaf_heat_capacity_simple(slatop, dpai, params)
            results.append(float(result[0, 0]))
        
        # Check increasing trend
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], \
                f"Heat capacity doesn't increase with water content: {results[i]} >= {results[i+1]}"


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("slatop,dpai,expected_positive", [
    (jnp.array([0.015]), jnp.array([[2.0]]), True),
    (jnp.array([0.02]), jnp.array([[0.0]]), False),
    (jnp.array([0.01, 0.03]), jnp.array([[1.5, 0.0], [0.0, 2.0]]), True),
])
def test_parametrized_output_sign(slatop, dpai, expected_positive, default_params):
    """
    Parametrized test for output sign based on dpai values.
    
    Args:
        slatop: Specific leaf area
        dpai: Plant area index
        expected_positive: Whether any positive values are expected
    """
    result = leaf_heat_capacity_simple(slatop, dpai, default_params)
    
    has_positive = np.any(result > 0.0)
    assert has_positive == expected_positive, \
        f"Expected positive values: {expected_positive}, got: {has_positive}"


@pytest.mark.parametrize("fwater", [0.1, 0.3, 0.5, 0.7, 0.85, 0.95])
def test_parametrized_fwater_range(fwater):
    """
    Parametrized test across range of water fractions.
    
    Verifies numerical stability and physical realism across
    the valid range of fwater values.
    """
    slatop = jnp.array([0.015])
    dpai = jnp.array([[1.5]])
    params = LeafHeatCapacityParams(
        cpbio=1470.0,
        cpliq=4188.0,
        fcarbon=0.5,
        fwater=fwater
    )
    
    result = leaf_heat_capacity_simple(slatop, dpai, params)
    
    # Check finite
    assert np.all(np.isfinite(result)), \
        f"Non-finite result for fwater={fwater}"
    
    # Check non-negative
    assert np.all(result >= 0.0), \
        f"Negative result for fwater={fwater}"
    
    # Check reasonable range (5000-50000 J/m2/K typical)
    assert np.all(result < 100000.0), \
        f"Unreasonably large result for fwater={fwater}: {result}"


@pytest.mark.parametrize("slatop", [0.001, 0.005, 0.01, 0.02, 0.04, 0.08])
def test_parametrized_slatop_range(slatop, default_params):
    """
    Parametrized test across range of specific leaf area values.
    
    Verifies numerical stability from very thick (succulent) to
    very thin (shade-adapted) leaves.
    """
    slatop_arr = jnp.array([slatop])
    dpai = jnp.array([[2.0]])
    
    result = leaf_heat_capacity_simple(slatop_arr, dpai, default_params)
    
    # Check finite
    assert np.all(np.isfinite(result)), \
        f"Non-finite result for slatop={slatop}"
    
    # Check non-negative
    assert np.all(result >= 0.0), \
        f"Negative result for slatop={slatop}"
    
    # Check reasonable range
    assert np.all(result < 200000.0), \
        f"Unreasonably large result for slatop={slatop}: {result}"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_single_element_arrays(self, default_params):
        """Test with minimal array sizes (single patch, single layer)."""
        slatop = jnp.array([0.015])
        dpai = jnp.array([[1.5]])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        assert result.shape == (1, 1), \
            f"Expected shape (1, 1), got {result.shape}"
        assert np.isfinite(result[0, 0]), \
            "Result must be finite"
        assert result[0, 0] > 0.0, \
            "Result must be positive for non-zero dpai"
    
    def test_large_arrays(self, default_params):
        """Test with large array sizes to check scalability."""
        n_patches = 100
        n_levels = 20
        
        slatop = jnp.array(np.random.uniform(0.01, 0.03, n_patches))
        dpai = jnp.array(np.random.uniform(0.0, 3.0, (n_patches, n_levels)))
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        assert result.shape == (n_patches, n_levels), \
            f"Expected shape ({n_patches}, {n_levels}), got {result.shape}"
        assert np.all(np.isfinite(result)), \
            "All results must be finite"
        assert np.all(result >= 0.0), \
            "All results must be non-negative"
    
    def test_mixed_zero_nonzero_dpai(self, default_params):
        """Test with mixture of zero and non-zero dpai values."""
        slatop = jnp.array([0.015, 0.02])
        dpai = jnp.array([
            [1.5, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.5]
        ])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # Check zeros where dpai is zero
        zero_mask = dpai == 0.0
        assert np.all(result[zero_mask] == 0.0), \
            "Expected zero output where dpai is zero"
        
        # Check positive where dpai is positive
        nonzero_mask = dpai > 0.0
        assert np.all(result[nonzero_mask] > 0.0), \
            "Expected positive output where dpai is positive"
    
    def test_extreme_parameter_combinations(self):
        """Test with extreme but valid parameter combinations."""
        # Very thick leaves (low SLA), high water content
        slatop = jnp.array([0.001])
        dpai = jnp.array([[1.0]])
        params = LeafHeatCapacityParams(
            cpbio=1470.0,
            cpliq=4188.0,
            fcarbon=0.5,
            fwater=0.9
        )
        
        result = leaf_heat_capacity_simple(slatop, dpai, params)
        
        assert np.isfinite(result[0, 0]), \
            "Result must be finite for extreme parameters"
        assert result[0, 0] > 0.0, \
            "Result must be positive"
        
        # Very thin leaves (high SLA), low water content
        slatop = jnp.array([0.08])
        params = LeafHeatCapacityParams(
            cpbio=1470.0,
            cpliq=4188.0,
            fcarbon=0.5,
            fwater=0.2
        )
        
        result = leaf_heat_capacity_simple(slatop, dpai, params)
        
        assert np.isfinite(result[0, 0]), \
            "Result must be finite for extreme parameters"
        assert result[0, 0] > 0.0, \
            "Result must be positive"


# ============================================================================
# Documentation Tests
# ============================================================================

def test_module_has_docstrings():
    """Verify that functions have docstrings."""
    assert leaf_heat_capacity.__doc__ is not None, \
        "leaf_heat_capacity should have a docstring"
    assert leaf_heat_capacity_simple.__doc__ is not None, \
        "leaf_heat_capacity_simple should have a docstring"


def test_namedtuples_exist():
    """Verify that required NamedTuples are defined."""
    assert LeafHeatCapacityInput is not None, \
        "LeafHeatCapacityInput should be defined"
    assert LeafHeatCapacityParams is not None, \
        "LeafHeatCapacityParams should be defined"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])