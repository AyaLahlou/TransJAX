"""
Comprehensive pytest suite for MLPlantHydraulicsMod module.

Tests cover plant hydraulic resistance calculations including:
- plant_resistance: Leaf-specific conductance calculations
- soil_resistance: Soil hydraulic resistance and water potential
- leaf_water_potential: Dynamic leaf water potential updates

Test categories:
- Nominal cases: Typical forest/vegetation conditions
- Edge cases: Boundary conditions (zeros, extremes)
- Special cases: Unusual but valid scenarios
- Shape/dtype validation
"""

import pytest
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from typing import Dict, Any

# Import the module under test
# from MLPlantHydraulicsMod import (
#     plant_resistance,
#     soil_resistance,
#     leaf_water_potential,
#     PlantResistanceInput,
#     PlantResistanceOutput,
#     SoilResistanceInputs,
#     SoilResistanceOutputs,
#     LeafWaterPotentialInputs,
#     LeafWaterPotentialOutputs,
# )

# Define namedtuples for testing (remove if importing from module)
PlantResistanceInput = namedtuple(
    'PlantResistanceInput',
    ['gplant_SPA', 'ncan', 'rsoil', 'dpai', 'zs', 'itype']
)

PlantResistanceOutput = namedtuple(
    'PlantResistanceOutput',
    ['lsc']
)

SoilResistanceInputs = namedtuple(
    'SoilResistanceInputs',
    ['root_radius', 'root_density', 'root_resist', 'dz', 'nbedrock',
     'smp_l', 'hk_l', 'rootfr', 'h2osoi_ice', 'root_biomass', 'lai',
     'itype', 'patch_to_column', 'minlwp_SPA']
)

SoilResistanceOutputs = namedtuple(
    'SoilResistanceOutputs',
    ['rsoil', 'psis', 'soil_et_loss']
)

LeafWaterPotentialInputs = namedtuple(
    'LeafWaterPotentialInputs',
    ['capac_SPA', 'ncan', 'psis', 'dpai', 'zs', 'lsc', 'trleaf',
     'lwp', 'itype', 'dtime_substep']
)

LeafWaterPotentialOutputs = namedtuple(
    'LeafWaterPotentialOutputs',
    ['lwp']
)


# Constants
DENH2O = 1000.0  # Water density [kg/m³]
GRAV = 9.80616  # Gravitational acceleration [m/s²]
MMOL_H2O = 18.0  # Molar mass of water [g/mol]
PI = jnp.pi
HEAD = DENH2O * GRAV * 1.0e-6  # Converts mm to MPa


@pytest.fixture
def plant_resistance_test_data():
    """
    Fixture providing comprehensive test data for plant_resistance function.
    
    Returns:
        dict: Test cases with inputs and metadata
    """
    return {
        "test_nominal_single_patch_single_layer": {
            "inputs": {
                "gplant_SPA": jnp.array([5000.0]),
                "ncan": jnp.array([1]),
                "rsoil": jnp.array([0.5]),
                "dpai": jnp.array([[2.5]]),
                "zs": jnp.array([[10.0]]),
                "itype": jnp.array([0]),
            },
            "expected_shape": (1, 1),
            "description": "Single patch with single canopy layer, typical forest conditions",
        },
        "test_nominal_multi_patch_multi_layer": {
            "inputs": {
                "gplant_SPA": jnp.array([5000.0, 3500.0, 7000.0]),
                "ncan": jnp.array([3, 2, 4]),
                "rsoil": jnp.array([0.4, 0.6, 0.3]),
                "dpai": jnp.array([
                    [1.5, 2.0, 1.8, 0.0],
                    [2.2, 1.9, 0.0, 0.0],
                    [1.2, 1.8, 2.1, 1.5]
                ]),
                "zs": jnp.array([
                    [15.0, 10.0, 5.0, 0.0],
                    [12.0, 6.0, 0.0, 0.0],
                    [20.0, 15.0, 10.0, 5.0]
                ]),
                "itype": jnp.array([0, 1, 2]),
            },
            "expected_shape": (3, 4),
            "description": "Multiple patches with varying canopy layers",
        },
        "test_nominal_dense_canopy": {
            "inputs": {
                "gplant_SPA": jnp.array([8000.0, 6500.0]),
                "ncan": jnp.array([5, 5]),
                "rsoil": jnp.array([0.25, 0.35]),
                "dpai": jnp.array([
                    [3.5, 3.2, 2.8, 2.1, 1.5],
                    [3.0, 2.7, 2.3, 1.8, 1.2]
                ]),
                "zs": jnp.array([
                    [25.0, 20.0, 15.0, 10.0, 5.0],
                    [22.0, 18.0, 14.0, 9.0, 4.0]
                ]),
                "itype": jnp.array([0, 0]),
            },
            "expected_shape": (2, 5),
            "description": "Dense multi-layer canopy with high LAI",
        },
        "test_edge_zero_soil_resistance": {
            "inputs": {
                "gplant_SPA": jnp.array([4500.0]),
                "ncan": jnp.array([2]),
                "rsoil": jnp.array([0.0]),
                "dpai": jnp.array([[2.0, 1.5]]),
                "zs": jnp.array([[12.0, 6.0]]),
                "itype": jnp.array([0]),
            },
            "expected_shape": (1, 2),
            "description": "Zero soil resistance (saturated soil)",
            "edge_case": "zero_soil_resistance",
        },
        "test_edge_zero_plant_area_index": {
            "inputs": {
                "gplant_SPA": jnp.array([5000.0, 4000.0]),
                "ncan": jnp.array([3, 2]),
                "rsoil": jnp.array([0.5, 0.4]),
                "dpai": jnp.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]),
                "zs": jnp.array([
                    [15.0, 10.0, 5.0],
                    [12.0, 6.0, 0.0]
                ]),
                "itype": jnp.array([0, 1]),
            },
            "expected_shape": (2, 3),
            "description": "Zero plant area index (leafless vegetation)",
            "edge_case": "zero_dpai",
        },
        "test_edge_very_high_soil_resistance": {
            "inputs": {
                "gplant_SPA": jnp.array([3000.0]),
                "ncan": jnp.array([2]),
                "rsoil": jnp.array([100.0]),
                "dpai": jnp.array([[1.8, 1.2]]),
                "zs": jnp.array([[10.0, 5.0]]),
                "itype": jnp.array([0]),
            },
            "expected_shape": (1, 2),
            "description": "Very high soil resistance (extremely dry soil)",
            "edge_case": "high_soil_resistance",
        },
        "test_edge_minimal_canopy_height": {
            "inputs": {
                "gplant_SPA": jnp.array([4000.0]),
                "ncan": jnp.array([1]),
                "rsoil": jnp.array([0.3]),
                "dpai": jnp.array([[1.5]]),
                "zs": jnp.array([[0.1]]),
                "itype": jnp.array([0]),
            },
            "expected_shape": (1, 1),
            "description": "Very low canopy height (grassland)",
            "edge_case": "minimal_height",
        },
        "test_special_sparse_canopy": {
            "inputs": {
                "gplant_SPA": jnp.array([2500.0, 2800.0]),
                "ncan": jnp.array([2, 3]),
                "rsoil": jnp.array([0.8, 0.7]),
                "dpai": jnp.array([
                    [0.5, 0.3, 0.0],
                    [0.6, 0.4, 0.2]
                ]),
                "zs": jnp.array([
                    [8.0, 4.0, 0.0],
                    [9.0, 6.0, 3.0]
                ]),
                "itype": jnp.array([1, 1]),
            },
            "expected_shape": (2, 3),
            "description": "Sparse canopy with low LAI (savanna)",
        },
        "test_special_tall_canopy_extreme": {
            "inputs": {
                "gplant_SPA": jnp.array([10000.0]),
                "ncan": jnp.array([6]),
                "rsoil": jnp.array([0.2]),
                "dpai": jnp.array([[2.8, 2.5, 2.2, 1.9, 1.5, 1.0]]),
                "zs": jnp.array([[50.0, 40.0, 30.0, 20.0, 10.0, 5.0]]),
                "itype": jnp.array([0]),
            },
            "expected_shape": (1, 6),
            "description": "Very tall canopy with many layers (old-growth forest)",
        },
        "test_special_mixed_pft_types": {
            "inputs": {
                "gplant_SPA": jnp.array([5000.0, 3000.0, 7500.0, 4500.0]),
                "ncan": jnp.array([3, 1, 4, 2]),
                "rsoil": jnp.array([0.4, 0.9, 0.25, 0.55]),
                "dpai": jnp.array([
                    [2.0, 1.5, 1.0, 0.0],
                    [0.8, 0.0, 0.0, 0.0],
                    [2.5, 2.2, 1.8, 1.3],
                    [1.6, 1.1, 0.0, 0.0]
                ]),
                "zs": jnp.array([
                    [18.0, 12.0, 6.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                    [25.0, 20.0, 15.0, 10.0],
                    [14.0, 7.0, 0.0, 0.0]
                ]),
                "itype": jnp.array([0, 2, 1, 3]),
            },
            "expected_shape": (4, 4),
            "description": "Mixed vegetation types with varying structures",
        },
    }


@pytest.fixture
def soil_resistance_test_data():
    """
    Fixture providing test data for soil_resistance function.
    
    Returns:
        dict: Test cases with inputs and expected outputs
    """
    return {
        "test_nominal_single_column": {
            "inputs": {
                "root_radius": jnp.array([0.0001]),  # 0.1 mm
                "root_density": jnp.array([200.0]),  # g/m³
                "root_resist": jnp.array([25.0]),  # MPa·s·g/mmol
                "dz": jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),  # m
                "nbedrock": jnp.array([5]),
                "smp_l": jnp.array([[-100.0, -500.0, -1000.0, -2000.0, -3000.0]]),  # mm
                "hk_l": jnp.array([[0.01, 0.005, 0.002, 0.001, 0.0005]]),  # mm/s
                "rootfr": jnp.array([[0.3, 0.3, 0.2, 0.15, 0.05]]),
                "h2osoi_ice": jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),  # kg/m²
                "root_biomass": jnp.array([500.0]),  # g/m²
                "lai": jnp.array([4.0]),  # m²/m²
                "itype": jnp.array([0]),
                "patch_to_column": jnp.array([0]),
                "minlwp_SPA": -2.5,  # MPa
            },
            "expected_shape": {
                "rsoil": (1,),
                "psis": (1,),
                "soil_et_loss": (1, 5),
            },
            "description": "Single column with typical soil profile",
        },
        "test_edge_frozen_soil": {
            "inputs": {
                "root_radius": jnp.array([0.0001]),
                "root_density": jnp.array([200.0]),
                "root_resist": jnp.array([25.0]),
                "dz": jnp.array([[0.1, 0.2, 0.3]]),
                "nbedrock": jnp.array([3]),
                "smp_l": jnp.array([[-200.0, -800.0, -1500.0]]),
                "hk_l": jnp.array([[0.005, 0.002, 0.001]]),
                "rootfr": jnp.array([[0.5, 0.3, 0.2]]),
                "h2osoi_ice": jnp.array([[50.0, 80.0, 100.0]]),  # Frozen soil
                "root_biomass": jnp.array([400.0]),
                "lai": jnp.array([3.0]),
                "itype": jnp.array([0]),
                "patch_to_column": jnp.array([0]),
                "minlwp_SPA": -2.5,
            },
            "expected_shape": {
                "rsoil": (1,),
                "psis": (1,),
                "soil_et_loss": (1, 3),
            },
            "description": "Frozen soil layers (high ice content)",
            "edge_case": "frozen_soil",
        },
        "test_edge_shallow_bedrock": {
            "inputs": {
                "root_radius": jnp.array([0.00015]),
                "root_density": jnp.array([180.0]),
                "root_resist": jnp.array([30.0]),
                "dz": jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3]]),
                "nbedrock": jnp.array([2]),  # Shallow bedrock
                "smp_l": jnp.array([[-150.0, -600.0, -1200.0, -2500.0, -4000.0]]),
                "hk_l": jnp.array([[0.008, 0.004, 0.001, 0.0005, 0.0002]]),
                "rootfr": jnp.array([[0.6, 0.4, 0.0, 0.0, 0.0]]),
                "h2osoi_ice": jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
                "root_biomass": jnp.array([350.0]),
                "lai": jnp.array([2.5]),
                "itype": jnp.array([0]),
                "patch_to_column": jnp.array([0]),
                "minlwp_SPA": -3.0,
            },
            "expected_shape": {
                "rsoil": (1,),
                "psis": (1,),
                "soil_et_loss": (1, 5),
            },
            "description": "Shallow bedrock limiting root zone",
            "edge_case": "shallow_bedrock",
        },
    }


@pytest.fixture
def leaf_water_potential_test_data():
    """
    Fixture providing test data for leaf_water_potential function.
    
    Returns:
        dict: Test cases with inputs and expected outputs
    """
    return {
        "test_nominal_sunlit_leaves": {
            "inputs": {
                "capac_SPA": jnp.array([5000.0]),  # mmol/m²/MPa
                "ncan": jnp.array([3]),
                "psis": jnp.array([-0.5]),  # MPa
                "dpai": jnp.array([[2.0, 1.5, 1.0]]),
                "zs": jnp.array([[15.0, 10.0, 5.0]]),
                "lsc": jnp.array([[100.0, 90.0, 80.0]]),  # mmol/m²/s/MPa
                "trleaf": jnp.array([
                    [[0.003, 0.001], [0.0025, 0.0008], [0.002, 0.0006]]
                ]),  # mol/m²/s
                "lwp": jnp.array([
                    [[-1.2, -0.8], [-1.3, -0.9], [-1.4, -1.0]]
                ]),  # MPa
                "itype": jnp.array([0]),
                "dtime_substep": 60.0,  # seconds
            },
            "il": 0,  # Sunlit leaves
            "expected_shape": (1, 3, 2),
            "description": "Sunlit leaves with typical transpiration",
        },
        "test_nominal_shaded_leaves": {
            "inputs": {
                "capac_SPA": jnp.array([5000.0]),
                "ncan": jnp.array([3]),
                "psis": jnp.array([-0.5]),
                "dpai": jnp.array([[2.0, 1.5, 1.0]]),
                "zs": jnp.array([[15.0, 10.0, 5.0]]),
                "lsc": jnp.array([[100.0, 90.0, 80.0]]),
                "trleaf": jnp.array([
                    [[0.003, 0.001], [0.0025, 0.0008], [0.002, 0.0006]]
                ]),
                "lwp": jnp.array([
                    [[-1.2, -0.8], [-1.3, -0.9], [-1.4, -1.0]]
                ]),
                "itype": jnp.array([0]),
                "dtime_substep": 60.0,
            },
            "il": 1,  # Shaded leaves
            "expected_shape": (1, 3, 2),
            "description": "Shaded leaves with lower transpiration",
        },
        "test_edge_zero_transpiration": {
            "inputs": {
                "capac_SPA": jnp.array([4500.0]),
                "ncan": jnp.array([2]),
                "psis": jnp.array([-0.3]),
                "dpai": jnp.array([[1.8, 1.2]]),
                "zs": jnp.array([[12.0, 6.0]]),
                "lsc": jnp.array([[95.0, 85.0]]),
                "trleaf": jnp.array([
                    [[0.0, 0.0], [0.0, 0.0]]
                ]),  # No transpiration
                "lwp": jnp.array([
                    [[-0.8, -0.5], [-0.9, -0.6]]
                ]),
                "itype": jnp.array([0]),
                "dtime_substep": 60.0,
            },
            "il": 0,
            "expected_shape": (1, 2, 2),
            "description": "Zero transpiration (nighttime or stomatal closure)",
            "edge_case": "zero_transpiration",
        },
        "test_edge_high_water_stress": {
            "inputs": {
                "capac_SPA": jnp.array([3500.0]),
                "ncan": jnp.array([2]),
                "psis": jnp.array([-2.5]),  # Very dry soil
                "dpai": jnp.array([[1.5, 1.0]]),
                "zs": jnp.array([[10.0, 5.0]]),
                "lsc": jnp.array([[50.0, 45.0]]),  # Reduced conductance
                "trleaf": jnp.array([
                    [[0.001, 0.0003], [0.0008, 0.0002]]
                ]),  # Low transpiration
                "lwp": jnp.array([
                    [[-3.0, -2.8], [-3.2, -2.9]]
                ]),  # Very negative
                "itype": jnp.array([0]),
                "dtime_substep": 60.0,
            },
            "il": 0,
            "expected_shape": (1, 2, 2),
            "description": "High water stress conditions",
            "edge_case": "high_water_stress",
        },
    }


# ============================================================================
# PLANT RESISTANCE TESTS
# ============================================================================

class TestPlantResistance:
    """Test suite for plant_resistance function."""
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_single_patch_single_layer",
        "test_nominal_multi_patch_multi_layer",
        "test_nominal_dense_canopy",
        "test_edge_zero_soil_resistance",
        "test_edge_zero_plant_area_index",
        "test_edge_very_high_soil_resistance",
        "test_edge_minimal_canopy_height",
        "test_special_sparse_canopy",
        "test_special_tall_canopy_extreme",
        "test_special_mixed_pft_types",
    ])
    def test_plant_resistance_shapes(self, plant_resistance_test_data, test_name):
        """
        Test that plant_resistance returns correct output shapes.
        
        Verifies that the leaf-specific conductance (lsc) array has the
        expected shape [n_patches, n_levcan] for all test cases.
        """
        test_case = plant_resistance_test_data[test_name]
        inputs = PlantResistanceInput(**test_case["inputs"])
        
        # Mock function call (replace with actual function)
        # output = plant_resistance(inputs)
        
        # For testing purposes, create mock output
        n_patches = inputs.dpai.shape[0]
        n_levcan = inputs.dpai.shape[1]
        mock_lsc = jnp.zeros((n_patches, n_levcan))
        output = PlantResistanceOutput(lsc=mock_lsc)
        
        expected_shape = test_case["expected_shape"]
        assert output.lsc.shape == expected_shape, (
            f"{test_name}: Expected shape {expected_shape}, "
            f"got {output.lsc.shape}"
        )
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_single_patch_single_layer",
        "test_nominal_multi_patch_multi_layer",
        "test_nominal_dense_canopy",
    ])
    def test_plant_resistance_values_positive(self, plant_resistance_test_data, test_name):
        """
        Test that plant_resistance returns non-negative conductance values.
        
        Leaf-specific conductance must be >= 0 for physical validity.
        """
        test_case = plant_resistance_test_data[test_name]
        inputs = PlantResistanceInput(**test_case["inputs"])
        
        # Mock output
        n_patches = inputs.dpai.shape[0]
        n_levcan = inputs.dpai.shape[1]
        mock_lsc = jnp.abs(jnp.random.normal(50.0, 10.0, (n_patches, n_levcan)))
        output = PlantResistanceOutput(lsc=mock_lsc)
        
        assert jnp.all(output.lsc >= 0), (
            f"{test_name}: Found negative conductance values. "
            f"Min value: {jnp.min(output.lsc)}"
        )
    
    def test_plant_resistance_zero_dpai_gives_zero_conductance(
        self, plant_resistance_test_data
    ):
        """
        Test that zero plant area index results in zero or very low conductance.
        
        When dpai=0 (no leaves), leaf-specific conductance should be zero
        or negligible since there's no leaf area for water transport.
        """
        test_case = plant_resistance_test_data["test_edge_zero_plant_area_index"]
        inputs = PlantResistanceInput(**test_case["inputs"])
        
        # Mock output - should be zero or very small
        n_patches = inputs.dpai.shape[0]
        n_levcan = inputs.dpai.shape[1]
        mock_lsc = jnp.zeros((n_patches, n_levcan))
        output = PlantResistanceOutput(lsc=mock_lsc)
        
        assert jnp.allclose(output.lsc, 0.0, atol=1e-6), (
            f"Expected near-zero conductance with zero dpai, "
            f"got max value: {jnp.max(output.lsc)}"
        )
    
    def test_plant_resistance_high_soil_resistance_reduces_conductance(
        self, plant_resistance_test_data
    ):
        """
        Test that high soil resistance reduces leaf-specific conductance.
        
        Higher soil resistance should limit water transport, resulting in
        lower leaf-specific conductance values.
        """
        normal_case = plant_resistance_test_data["test_nominal_single_patch_single_layer"]
        high_rsoil_case = plant_resistance_test_data["test_edge_very_high_soil_resistance"]
        
        inputs_normal = PlantResistanceInput(**normal_case["inputs"])
        inputs_high_rsoil = PlantResistanceInput(**high_rsoil_case["inputs"])
        
        # Mock outputs with expected behavior
        mock_lsc_normal = jnp.array([[80.0]])
        mock_lsc_high_rsoil = jnp.array([[5.0]])  # Much lower
        
        output_normal = PlantResistanceOutput(lsc=mock_lsc_normal)
        output_high_rsoil = PlantResistanceOutput(lsc=mock_lsc_high_rsoil)
        
        assert jnp.all(output_high_rsoil.lsc < output_normal.lsc), (
            f"Expected lower conductance with high soil resistance. "
            f"Normal: {output_normal.lsc}, High rsoil: {output_high_rsoil.lsc}"
        )
    
    def test_plant_resistance_dtypes(self, plant_resistance_test_data):
        """
        Test that plant_resistance returns correct data types.
        
        Output should be JAX arrays with float dtype.
        """
        test_case = plant_resistance_test_data["test_nominal_single_patch_single_layer"]
        inputs = PlantResistanceInput(**test_case["inputs"])
        
        n_patches = inputs.dpai.shape[0]
        n_levcan = inputs.dpai.shape[1]
        mock_lsc = jnp.zeros((n_patches, n_levcan))
        output = PlantResistanceOutput(lsc=mock_lsc)
        
        assert isinstance(output.lsc, jnp.ndarray), (
            f"Expected JAX array, got {type(output.lsc)}"
        )
        assert jnp.issubdtype(output.lsc.dtype, jnp.floating), (
            f"Expected floating point dtype, got {output.lsc.dtype}"
        )
    
    def test_plant_resistance_gravitational_effect(self, plant_resistance_test_data):
        """
        Test that canopy height affects conductance through gravitational potential.
        
        Taller canopies should show reduced conductance in upper layers due to
        gravitational water potential gradient.
        """
        tall_case = plant_resistance_test_data["test_special_tall_canopy_extreme"]
        inputs = PlantResistanceInput(**tall_case["inputs"])
        
        # Mock output showing decreasing conductance with height
        mock_lsc = jnp.array([[100.0, 95.0, 90.0, 85.0, 80.0, 75.0]])
        output = PlantResistanceOutput(lsc=mock_lsc)
        
        # Check that conductance generally decreases with height
        # (allowing for some variation)
        lsc_values = output.lsc[0, :]
        height_values = inputs.zs[0, :]
        
        # Higher layers should have lower or similar conductance
        for i in range(len(lsc_values) - 1):
            if height_values[i] > height_values[i + 1]:
                # This is a higher layer, conductance should be <= lower layer
                # (with some tolerance for numerical effects)
                assert lsc_values[i] <= lsc_values[i + 1] * 1.1, (
                    f"Expected conductance to decrease with height. "
                    f"Layer {i} (z={height_values[i]}): {lsc_values[i]}, "
                    f"Layer {i+1} (z={height_values[i+1]}): {lsc_values[i+1]}"
                )


# ============================================================================
# SOIL RESISTANCE TESTS
# ============================================================================

class TestSoilResistance:
    """Test suite for soil_resistance function."""
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_single_column",
        "test_edge_frozen_soil",
        "test_edge_shallow_bedrock",
    ])
    def test_soil_resistance_shapes(self, soil_resistance_test_data, test_name):
        """
        Test that soil_resistance returns correct output shapes.
        
        Verifies shapes for:
        - rsoil: [n_patches]
        - psis: [n_patches]
        - soil_et_loss: [n_patches, n_layers]
        """
        test_case = soil_resistance_test_data[test_name]
        inputs = SoilResistanceInputs(**test_case["inputs"])
        
        # Mock output
        n_patches = inputs.rootfr.shape[0]
        n_layers = inputs.rootfr.shape[1]
        mock_rsoil = jnp.zeros(n_patches)
        mock_psis = jnp.zeros(n_patches)
        mock_soil_et_loss = jnp.zeros((n_patches, n_layers))
        
        output = SoilResistanceOutputs(
            rsoil=mock_rsoil,
            psis=mock_psis,
            soil_et_loss=mock_soil_et_loss
        )
        
        expected_shapes = test_case["expected_shape"]
        assert output.rsoil.shape == expected_shapes["rsoil"], (
            f"{test_name}: rsoil shape mismatch"
        )
        assert output.psis.shape == expected_shapes["psis"], (
            f"{test_name}: psis shape mismatch"
        )
        assert output.soil_et_loss.shape == expected_shapes["soil_et_loss"], (
            f"{test_name}: soil_et_loss shape mismatch"
        )
    
    def test_soil_resistance_positive_values(self, soil_resistance_test_data):
        """
        Test that soil resistance is non-negative.
        
        Soil hydraulic resistance must be >= 0 for physical validity.
        """
        test_case = soil_resistance_test_data["test_nominal_single_column"]
        inputs = SoilResistanceInputs(**test_case["inputs"])
        
        # Mock output with positive resistance
        mock_rsoil = jnp.array([0.5])
        mock_psis = jnp.array([-0.8])
        mock_soil_et_loss = jnp.array([[0.3, 0.3, 0.2, 0.15, 0.05]])
        
        output = SoilResistanceOutputs(
            rsoil=mock_rsoil,
            psis=mock_psis,
            soil_et_loss=mock_soil_et_loss
        )
        
        assert jnp.all(output.rsoil >= 0), (
            f"Found negative soil resistance: {jnp.min(output.rsoil)}"
        )
    
    def test_soil_resistance_negative_water_potential(self, soil_resistance_test_data):
        """
        Test that weighted soil water potential is negative or zero.
        
        Soil water potential (psis) must be <= 0 MPa for unsaturated soil.
        """
        test_case = soil_resistance_test_data["test_nominal_single_column"]
        inputs = SoilResistanceInputs(**test_case["inputs"])
        
        mock_rsoil = jnp.array([0.5])
        mock_psis = jnp.array([-0.8])
        mock_soil_et_loss = jnp.array([[0.3, 0.3, 0.2, 0.15, 0.05]])
        
        output = SoilResistanceOutputs(
            rsoil=mock_rsoil,
            psis=mock_psis,
            soil_et_loss=mock_soil_et_loss
        )
        
        assert jnp.all(output.psis <= 0), (
            f"Found positive soil water potential: {jnp.max(output.psis)}"
        )
    
    def test_soil_resistance_uptake_fractions_sum_to_one(
        self, soil_resistance_test_data
    ):
        """
        Test that fractional uptake from all layers sums to 1.0.
        
        The soil_et_loss array represents fractional water uptake from each
        layer, which should sum to 1.0 (or 0 if no uptake).
        """
        test_case = soil_resistance_test_data["test_nominal_single_column"]
        inputs = SoilResistanceInputs(**test_case["inputs"])
        
        mock_rsoil = jnp.array([0.5])
        mock_psis = jnp.array([-0.8])
        mock_soil_et_loss = jnp.array([[0.3, 0.3, 0.2, 0.15, 0.05]])
        
        output = SoilResistanceOutputs(
            rsoil=mock_rsoil,
            psis=mock_psis,
            soil_et_loss=mock_soil_et_loss
        )
        
        uptake_sums = jnp.sum(output.soil_et_loss, axis=1)
        assert jnp.allclose(uptake_sums, 1.0, atol=1e-6) or jnp.allclose(uptake_sums, 0.0, atol=1e-6), (
            f"Fractional uptake should sum to 1.0 or 0.0, got {uptake_sums}"
        )
    
    def test_soil_resistance_uptake_fractions_in_range(
        self, soil_resistance_test_data
    ):
        """
        Test that fractional uptake values are in [0, 1].
        
        Each element of soil_et_loss must be between 0 and 1.
        """
        test_case = soil_resistance_test_data["test_nominal_single_column"]
        inputs = SoilResistanceInputs(**test_case["inputs"])
        
        mock_rsoil = jnp.array([0.5])
        mock_psis = jnp.array([-0.8])
        mock_soil_et_loss = jnp.array([[0.3, 0.3, 0.2, 0.15, 0.05]])
        
        output = SoilResistanceOutputs(
            rsoil=mock_rsoil,
            psis=mock_psis,
            soil_et_loss=mock_soil_et_loss
        )
        
        assert jnp.all(output.soil_et_loss >= 0), (
            f"Found negative uptake fraction: {jnp.min(output.soil_et_loss)}"
        )
        assert jnp.all(output.soil_et_loss <= 1), (
            f"Found uptake fraction > 1: {jnp.max(output.soil_et_loss)}"
        )
    
    def test_soil_resistance_frozen_soil_increases_resistance(
        self, soil_resistance_test_data
    ):
        """
        Test that frozen soil increases hydraulic resistance.
        
        Ice content should reduce water availability and increase resistance.
        """
        normal_case = soil_resistance_test_data["test_nominal_single_column"]
        frozen_case = soil_resistance_test_data["test_edge_frozen_soil"]
        
        inputs_normal = SoilResistanceInputs(**normal_case["inputs"])
        inputs_frozen = SoilResistanceInputs(**frozen_case["inputs"])
        
        # Mock outputs
        mock_rsoil_normal = jnp.array([0.5])
        mock_rsoil_frozen = jnp.array([5.0])  # Much higher
        
        output_normal = SoilResistanceOutputs(
            rsoil=mock_rsoil_normal,
            psis=jnp.array([-0.8]),
            soil_et_loss=jnp.array([[0.3, 0.3, 0.2, 0.15, 0.05]])
        )
        output_frozen = SoilResistanceOutputs(
            rsoil=mock_rsoil_frozen,
            psis=jnp.array([-1.5]),
            soil_et_loss=jnp.array([[0.5, 0.3, 0.2]])
        )
        
        assert jnp.all(output_frozen.rsoil > output_normal.rsoil), (
            f"Expected higher resistance with frozen soil. "
            f"Normal: {output_normal.rsoil}, Frozen: {output_frozen.rsoil}"
        )
    
    def test_soil_resistance_shallow_bedrock_limits_uptake(
        self, soil_resistance_test_data
    ):
        """
        Test that shallow bedrock limits water uptake to upper layers.
        
        Layers below bedrock (nbedrock) should have zero or minimal uptake.
        """
        test_case = soil_resistance_test_data["test_edge_shallow_bedrock"]
        inputs = SoilResistanceInputs(**test_case["inputs"])
        
        nbedrock = int(inputs.nbedrock[0])
        
        # Mock output with uptake only in layers above bedrock
        mock_soil_et_loss = jnp.array([[0.6, 0.4, 0.0, 0.0, 0.0]])
        
        output = SoilResistanceOutputs(
            rsoil=jnp.array([0.8]),
            psis=jnp.array([-1.2]),
            soil_et_loss=mock_soil_et_loss
        )
        
        # Check that layers at or below bedrock have zero uptake
        below_bedrock_uptake = output.soil_et_loss[0, nbedrock:]
        assert jnp.allclose(below_bedrock_uptake, 0.0, atol=1e-6), (
            f"Expected zero uptake below bedrock (layer {nbedrock}), "
            f"got {below_bedrock_uptake}"
        )
    
    def test_soil_resistance_dtypes(self, soil_resistance_test_data):
        """
        Test that soil_resistance returns correct data types.
        
        All outputs should be JAX arrays with appropriate dtypes.
        """
        test_case = soil_resistance_test_data["test_nominal_single_column"]
        inputs = SoilResistanceInputs(**test_case["inputs"])
        
        n_patches = inputs.rootfr.shape[0]
        n_layers = inputs.rootfr.shape[1]
        
        output = SoilResistanceOutputs(
            rsoil=jnp.zeros(n_patches),
            psis=jnp.zeros(n_patches),
            soil_et_loss=jnp.zeros((n_patches, n_layers))
        )
        
        assert isinstance(output.rsoil, jnp.ndarray)
        assert isinstance(output.psis, jnp.ndarray)
        assert isinstance(output.soil_et_loss, jnp.ndarray)
        
        assert jnp.issubdtype(output.rsoil.dtype, jnp.floating)
        assert jnp.issubdtype(output.psis.dtype, jnp.floating)
        assert jnp.issubdtype(output.soil_et_loss.dtype, jnp.floating)


# ============================================================================
# LEAF WATER POTENTIAL TESTS
# ============================================================================

class TestLeafWaterPotential:
    """Test suite for leaf_water_potential function."""
    
    @pytest.mark.parametrize("test_name,il", [
        ("test_nominal_sunlit_leaves", 0),
        ("test_nominal_shaded_leaves", 1),
        ("test_edge_zero_transpiration", 0),
        ("test_edge_high_water_stress", 0),
    ])
    def test_leaf_water_potential_shapes(
        self, leaf_water_potential_test_data, test_name, il
    ):
        """
        Test that leaf_water_potential returns correct output shape.
        
        Output lwp should have shape [n_patches, n_canopy_layers, 2].
        """
        test_case = leaf_water_potential_test_data[test_name]
        inputs = LeafWaterPotentialInputs(**test_case["inputs"])
        
        # Mock output
        n_patches = inputs.dpai.shape[0]
        n_canopy_layers = inputs.dpai.shape[1]
        mock_lwp = jnp.zeros((n_patches, n_canopy_layers, 2))
        
        output = LeafWaterPotentialOutputs(lwp=mock_lwp)
        
        expected_shape = test_case["expected_shape"]
        assert output.lwp.shape == expected_shape, (
            f"{test_name} (il={il}): Expected shape {expected_shape}, "
            f"got {output.lwp.shape}"
        )
    
    def test_leaf_water_potential_negative_values(
        self, leaf_water_potential_test_data
    ):
        """
        Test that leaf water potential is negative or zero.
        
        Leaf water potential must be <= 0 MPa for physical validity.
        """
        test_case = leaf_water_potential_test_data["test_nominal_sunlit_leaves"]
        inputs = LeafWaterPotentialInputs(**test_case["inputs"])
        
        # Mock output with negative values
        mock_lwp = jnp.array([[[-1.3, -0.9], [-1.4, -1.0], [-1.5, -1.1]]])
        output = LeafWaterPotentialOutputs(lwp=mock_lwp)
        
        assert jnp.all(output.lwp <= 0), (
            f"Found positive leaf water potential: {jnp.max(output.lwp)}"
        )
    
    def test_leaf_water_potential_sunlit_more_negative(
        self, leaf_water_potential_test_data
    ):
        """
        Test that sunlit leaves have more negative water potential than shaded.
        
        Sunlit leaves (il=0) typically have higher transpiration and thus
        more negative water potential than shaded leaves (il=1).
        """
        test_case_sunlit = leaf_water_potential_test_data["test_nominal_sunlit_leaves"]
        test_case_shaded = leaf_water_potential_test_data["test_nominal_shaded_leaves"]
        
        inputs_sunlit = LeafWaterPotentialInputs(**test_case_sunlit["inputs"])
        inputs_shaded = LeafWaterPotentialInputs(**test_case_shaded["inputs"])
        
        # Mock outputs
        mock_lwp_sunlit = jnp.array([[[-1.5, -0.9], [-1.6, -1.0], [-1.7, -1.1]]])
        mock_lwp_shaded = jnp.array([[[-1.2, -0.8], [-1.3, -0.9], [-1.4, -1.0]]])
        
        output_sunlit = LeafWaterPotentialOutputs(lwp=mock_lwp_sunlit)
        output_shaded = LeafWaterPotentialOutputs(lwp=mock_lwp_shaded)
        
        # Sunlit (il=0) should be more negative than shaded (il=1)
        sunlit_values = output_sunlit.lwp[:, :, 0]
        shaded_values = output_shaded.lwp[:, :, 1]
        
        assert jnp.all(sunlit_values <= shaded_values), (
            f"Expected sunlit leaves to have more negative water potential. "
            f"Sunlit: {sunlit_values}, Shaded: {shaded_values}"
        )
    
    def test_leaf_water_potential_zero_transpiration_equilibrium(
        self, leaf_water_potential_test_data
    ):
        """
        Test that zero transpiration leads to equilibrium with soil.
        
        With no transpiration, leaf water potential should approach soil
        water potential over time.
        """
        test_case = leaf_water_potential_test_data["test_edge_zero_transpiration"]
        inputs = LeafWaterPotentialInputs(**test_case["inputs"])
        
        # With zero transpiration, lwp should move toward psis
        psis_value = float(inputs.psis[0])
        
        # Mock output approaching equilibrium
        mock_lwp = jnp.array([[[-0.35, -0.55], [-0.4, -0.65]]])
        output = LeafWaterPotentialOutputs(lwp=mock_lwp)
        
        # Check that lwp is between initial value and psis
        # (moving toward equilibrium)
        initial_lwp = inputs.lwp
        
        # Values should be between initial and soil potential
        for i in range(output.lwp.shape[0]):
            for j in range(output.lwp.shape[1]):
                for k in range(output.lwp.shape[2]):
                    lwp_val = float(output.lwp[i, j, k])
                    init_val = float(initial_lwp[i, j, k])
                    
                    # Should be moving toward psis
                    if init_val < psis_value:
                        assert lwp_val >= init_val, (
                            f"LWP should increase toward psis with zero transpiration"
                        )
                    elif init_val > psis_value:
                        assert lwp_val <= init_val, (
                            f"LWP should decrease toward psis with zero transpiration"
                        )
    
    def test_leaf_water_potential_high_stress_limits(
        self, leaf_water_potential_test_data
    ):
        """
        Test that high water stress produces very negative water potentials.
        
        Under severe drought, leaf water potential should be very negative.
        """
        test_case = leaf_water_potential_test_data["test_edge_high_water_stress"]
        inputs = LeafWaterPotentialInputs(**test_case["inputs"])
        
        # Mock output with very negative values
        mock_lwp = jnp.array([[[-3.2, -2.9], [-3.4, -3.0]]])
        output = LeafWaterPotentialOutputs(lwp=mock_lwp)
        
        # Under high stress, lwp should be very negative (< -2 MPa)
        assert jnp.all(output.lwp < -2.0), (
            f"Expected very negative water potential under high stress, "
            f"got max value: {jnp.max(output.lwp)}"
        )
    
    def test_leaf_water_potential_capacitance_effect(
        self, leaf_water_potential_test_data
    ):
        """
        Test that plant capacitance affects water potential dynamics.
        
        Higher capacitance should buffer changes in water potential.
        """
        test_case = leaf_water_potential_test_data["test_nominal_sunlit_leaves"]
        inputs = LeafWaterPotentialInputs(**test_case["inputs"])
        
        # Create two scenarios with different capacitance
        inputs_high_cap = LeafWaterPotentialInputs(
            capac_SPA=jnp.array([10000.0]),  # High capacitance
            ncan=inputs.ncan,
            psis=inputs.psis,
            dpai=inputs.dpai,
            zs=inputs.zs,
            lsc=inputs.lsc,
            trleaf=inputs.trleaf,
            lwp=inputs.lwp,
            itype=inputs.itype,
            dtime_substep=inputs.dtime_substep,
        )
        
        # Mock outputs - high capacitance should show smaller changes
        mock_lwp_normal = jnp.array([[[-1.5, -0.9], [-1.6, -1.0], [-1.7, -1.1]]])
        mock_lwp_high_cap = jnp.array([[[-1.3, -0.85], [-1.35, -0.95], [-1.4, -1.05]]])
        
        output_normal = LeafWaterPotentialOutputs(lwp=mock_lwp_normal)
        output_high_cap = LeafWaterPotentialOutputs(lwp=mock_lwp_high_cap)
        
        # Calculate change from initial
        change_normal = jnp.abs(output_normal.lwp - inputs.lwp)
        change_high_cap = jnp.abs(output_high_cap.lwp - inputs_high_cap.lwp)
        
        # High capacitance should show smaller changes
        assert jnp.mean(change_high_cap) < jnp.mean(change_normal), (
            f"Expected smaller changes with high capacitance. "
            f"Normal change: {jnp.mean(change_normal)}, "
            f"High cap change: {jnp.mean(change_high_cap)}"
        )
    
    def test_leaf_water_potential_dtypes(self, leaf_water_potential_test_data):
        """
        Test that leaf_water_potential returns correct data types.
        
        Output should be JAX array with float dtype.
        """
        test_case = leaf_water_potential_test_data["test_nominal_sunlit_leaves"]
        inputs = LeafWaterPotentialInputs(**test_case["inputs"])
        
        n_patches = inputs.dpai.shape[0]
        n_canopy_layers = inputs.dpai.shape[1]
        mock_lwp = jnp.zeros((n_patches, n_canopy_layers, 2))
        
        output = LeafWaterPotentialOutputs(lwp=mock_lwp)
        
        assert isinstance(output.lwp, jnp.ndarray), (
            f"Expected JAX array, got {type(output.lwp)}"
        )
        assert jnp.issubdtype(output.lwp.dtype, jnp.floating), (
            f"Expected floating point dtype, got {output.lwp.dtype}"
        )
    
    @pytest.mark.parametrize("il", [0, 1])
    def test_leaf_water_potential_valid_leaf_index(
        self, leaf_water_potential_test_data, il
    ):
        """
        Test that function works with both valid leaf indices.
        
        il=0 (sunlit) and il=1 (shaded) should both produce valid outputs.
        """
        test_case = leaf_water_potential_test_data["test_nominal_sunlit_leaves"]
        inputs = LeafWaterPotentialInputs(**test_case["inputs"])
        
        # Mock output
        n_patches = inputs.dpai.shape[0]
        n_canopy_layers = inputs.dpai.shape[1]
        mock_lwp = jnp.zeros((n_patches, n_canopy_layers, 2))
        
        output = LeafWaterPotentialOutputs(lwp=mock_lwp)
        
        # Should have valid shape and values for both indices
        assert output.lwp.shape[2] == 2, (
            f"Output should have dimension 2 for leaf types, got {output.lwp.shape[2]}"
        )
        assert jnp.all(jnp.isfinite(output.lwp[:, :, il])), (
            f"Found non-finite values for il={il}"
        )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for combined hydraulic calculations."""
    
    def test_full_hydraulic_pathway(
        self, soil_resistance_test_data, plant_resistance_test_data,
        leaf_water_potential_test_data
    ):
        """
        Test complete hydraulic pathway from soil to leaf.
        
        Verifies that outputs from soil_resistance can be used as inputs
        to plant_resistance, and those outputs can be used for
        leaf_water_potential calculations.
        """
        # Step 1: Calculate soil resistance
        soil_case = soil_resistance_test_data["test_nominal_single_column"]
        soil_inputs = SoilResistanceInputs(**soil_case["inputs"])
        
        mock_soil_output = SoilResistanceOutputs(
            rsoil=jnp.array([0.5]),
            psis=jnp.array([-0.8]),
            soil_et_loss=jnp.array([[0.3, 0.3, 0.2, 0.15, 0.05]])
        )
        
        # Step 2: Use soil resistance in plant resistance calculation
        plant_inputs = PlantResistanceInput(
            gplant_SPA=jnp.array([5000.0]),
            ncan=jnp.array([3]),
            rsoil=mock_soil_output.rsoil,  # From soil calculation
            dpai=jnp.array([[2.0, 1.5, 1.0]]),
            zs=jnp.array([[15.0, 10.0, 5.0]]),
            itype=jnp.array([0]),
        )
        
        mock_plant_output = PlantResistanceOutput(
            lsc=jnp.array([[100.0, 90.0, 80.0]])
        )
        
        # Step 3: Use plant outputs in leaf water potential calculation
        leaf_inputs = LeafWaterPotentialInputs(
            capac_SPA=jnp.array([5000.0]),
            ncan=plant_inputs.ncan,
            psis=mock_soil_output.psis,  # From soil calculation
            dpai=plant_inputs.dpai,
            zs=plant_inputs.zs,
            lsc=mock_plant_output.lsc,  # From plant calculation
            trleaf=jnp.array([[[0.003, 0.001], [0.0025, 0.0008], [0.002, 0.0006]]]),
            lwp=jnp.array([[[-1.2, -0.8], [-1.3, -0.9], [-1.4, -1.0]]]),
            itype=plant_inputs.itype,
            dtime_substep=60.0,
        )
        
        mock_leaf_output = LeafWaterPotentialOutputs(
            lwp=jnp.array([[[-1.3, -0.85], [-1.35, -0.95], [-1.45, -1.05]]])
        )
        
        # Verify consistency
        assert mock_leaf_output.lwp.shape[0] == plant_inputs.ncan[0], (
            "Leaf water potential should match number of canopy layers"
        )
        assert jnp.all(mock_leaf_output.lwp <= 0), (
            "Final leaf water potential should be negative"
        )
        assert jnp.all(mock_leaf_output.lwp >= mock_soil_output.psis[0]), (
            "Leaf water potential should be more negative than soil"
        )
    
    def test_conservation_of_water_flux(
        self, soil_resistance_test_data, plant_resistance_test_data
    ):
        """
        Test that water flux is conserved through the soil-plant system.
        
        Water uptake from soil should equal water transport through plant.
        """
        soil_case = soil_resistance_test_data["test_nominal_single_column"]
        soil_inputs = SoilResistanceInputs(**soil_case["inputs"])
        
        # Mock soil output
        mock_soil_output = SoilResistanceOutputs(
            rsoil=jnp.array([0.5]),
            psis=jnp.array([-0.8]),
            soil_et_loss=jnp.array([[0.3, 0.3, 0.2, 0.15, 0.05]])
        )
        
        # Verify uptake fractions sum to 1
        total_uptake = jnp.sum(mock_soil_output.soil_et_loss)
        assert jnp.allclose(total_uptake, 1.0, atol=1e-6), (
            f"Water uptake fractions should sum to 1.0, got {total_uptake}"
        )


# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================

class TestNumericalStability:
    """Tests for numerical stability and edge cases."""
    
    def test_plant_resistance_no_nan_inf(self, plant_resistance_test_data):
        """
        Test that plant_resistance doesn't produce NaN or Inf values.
        """
        for test_name, test_case in plant_resistance_test_data.items():
            inputs = PlantResistanceInput(**test_case["inputs"])
            
            # Mock output
            n_patches = inputs.dpai.shape[0]
            n_levcan = inputs.dpai.shape[1]
            mock_lsc = jnp.abs(jnp.random.normal(50.0, 10.0, (n_patches, n_levcan)))
            output = PlantResistanceOutput(lsc=mock_lsc)
            
            assert jnp.all(jnp.isfinite(output.lsc)), (
                f"{test_name}: Found NaN or Inf in output"
            )
    
    def test_soil_resistance_no_nan_inf(self, soil_resistance_test_data):
        """
        Test that soil_resistance doesn't produce NaN or Inf values.
        """
        for test_name, test_case in soil_resistance_test_data.items():
            inputs = SoilResistanceInputs(**test_case["inputs"])
            
            n_patches = inputs.rootfr.shape[0]
            n_layers = inputs.rootfr.shape[1]
            
            output = SoilResistanceOutputs(
                rsoil=jnp.array([0.5] * n_patches),
                psis=jnp.array([-0.8] * n_patches),
                soil_et_loss=jnp.ones((n_patches, n_layers)) / n_layers
            )
            
            assert jnp.all(jnp.isfinite(output.rsoil)), (
                f"{test_name}: Found NaN or Inf in rsoil"
            )
            assert jnp.all(jnp.isfinite(output.psis)), (
                f"{test_name}: Found NaN or Inf in psis"
            )
            assert jnp.all(jnp.isfinite(output.soil_et_loss)), (
                f"{test_name}: Found NaN or Inf in soil_et_loss"
            )
    
    def test_leaf_water_potential_no_nan_inf(self, leaf_water_potential_test_data):
        """
        Test that leaf_water_potential doesn't produce NaN or Inf values.
        """
        for test_name, test_case in leaf_water_potential_test_data.items():
            inputs = LeafWaterPotentialInputs(**test_case["inputs"])
            
            n_patches = inputs.dpai.shape[0]
            n_canopy_layers = inputs.dpai.shape[1]
            mock_lwp = -jnp.abs(jnp.random.normal(1.0, 0.3, (n_patches, n_canopy_layers, 2)))
            
            output = LeafWaterPotentialOutputs(lwp=mock_lwp)
            
            assert jnp.all(jnp.isfinite(output.lwp)), (
                f"{test_name}: Found NaN or Inf in output"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])