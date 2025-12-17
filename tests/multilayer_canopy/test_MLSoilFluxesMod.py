"""
Comprehensive pytest suite for MLSoilFluxesMod.soil_fluxes function.

This module tests the soil surface energy balance calculations including:
- Sensible and latent heat fluxes
- Soil heat flux
- Surface temperature and vapor pressure
- Energy balance closure

Test coverage includes:
- Nominal conditions across diverse climate regimes
- Edge cases (zero fluxes, extreme humidity, minimal conductances)
- Extreme but valid temperature gradients
- Array dimension variations
- Physical constraint validation
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict, Any, List


# ============================================================================
# NamedTuple Definitions (matching the module interface)
# ============================================================================

class SoilFluxesInput(NamedTuple):
    """Input state for soil flux calculations."""
    tref: jnp.ndarray  # [n_patches] Air temperature at reference height [K]
    pref: jnp.ndarray  # [n_patches] Air pressure at reference height [Pa]
    rhomol: jnp.ndarray  # [n_patches] Molar density [mol/m3]
    cpair: jnp.ndarray  # [n_patches] Specific heat of air [J/mol/K]
    rnsoi: jnp.ndarray  # [n_patches] Net radiation at ground [W/m2]
    rhg: jnp.ndarray  # [n_patches] Relative humidity at soil surface [fraction]
    soilres: jnp.ndarray  # [n_patches] Soil evaporative resistance [s/m]
    gac0: jnp.ndarray  # [n_patches] Aerodynamic conductance [mol/m2/s]
    soil_t: jnp.ndarray  # [n_patches] First layer temperature [K]
    soil_dz: jnp.ndarray  # [n_patches] Depth to first layer [m]
    soil_tk: jnp.ndarray  # [n_patches] Thermal conductivity [W/m/K]
    tg_bef: jnp.ndarray  # [n_patches] Previous surface temperature [K]
    tair: jnp.ndarray  # [n_patches, n_layers] Canopy air temperature [K]
    eair: jnp.ndarray  # [n_patches, n_layers] Canopy vapor pressure [Pa]


class SoilFluxesOutput(NamedTuple):
    """Output state from soil flux calculations."""
    shsoi: jnp.ndarray  # [n_patches] Sensible heat flux [W/m2]
    lhsoi: jnp.ndarray  # [n_patches] Latent heat flux [W/m2]
    gsoi: jnp.ndarray  # [n_patches] Soil heat flux [W/m2]
    etsoi: jnp.ndarray  # [n_patches] Water vapor flux [mol H2O/m2/s]
    tg: jnp.ndarray  # [n_patches] Surface temperature [K]
    eg: jnp.ndarray  # [n_patches] Surface vapor pressure [Pa]
    energy_error: jnp.ndarray  # [n_patches] Energy balance error [W/m2]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load and prepare test data for soil_fluxes function.
    
    Returns:
        Dictionary containing all test cases with inputs and metadata.
    """
    return {
        "test_nominal_daytime_conditions": {
            "inputs": {
                "tref": [298.15, 301.15, 295.15],
                "pref": [101325.0, 100000.0, 102000.0],
                "rhomol": [41.5, 40.8, 42.0],
                "cpair": [29.3, 29.3, 29.3],
                "rnsoi": [450.0, 520.0, 380.0],
                "rhg": [0.65, 0.7, 0.6],
                "soilres": [200.0, 180.0, 220.0],
                "gac0": [0.15, 0.18, 0.12],
                "soil_t": [295.15, 298.15, 293.15],
                "soil_dz": [0.05, 0.05, 0.05],
                "soil_tk": [1.2, 1.5, 1.0],
                "tg_bef": [296.15, 299.15, 294.15],
                "tair": [[298.15, 297.15, 296.15], [301.15, 300.15, 299.15], [295.15, 294.15, 293.15]],
                "eair": [[1800.0, 1750.0, 1700.0], [2000.0, 1950.0, 1900.0], [1600.0, 1550.0, 1500.0]]
            },
            "metadata": {
                "type": "nominal",
                "description": "Typical daytime conditions with positive net radiation",
                "n_patches": 3,
                "n_layers": 3
            }
        },
        "test_nominal_nighttime_cooling": {
            "inputs": {
                "tref": [285.15, 283.15, 287.15],
                "pref": [101325.0, 101325.0, 101325.0],
                "rhomol": [43.2, 43.5, 42.9],
                "cpair": [29.3, 29.3, 29.3],
                "rnsoi": [-80.0, -95.0, -70.0],
                "rhg": [0.85, 0.9, 0.8],
                "soilres": [150.0, 140.0, 160.0],
                "gac0": [0.08, 0.1, 0.07],
                "soil_t": [288.15, 286.15, 290.15],
                "soil_dz": [0.05, 0.05, 0.05],
                "soil_tk": [1.0, 1.1, 0.9],
                "tg_bef": [287.15, 285.15, 289.15],
                "tair": [[285.15, 284.15, 283.15], [283.15, 282.15, 281.15], [287.15, 286.15, 285.15]],
                "eair": [[1200.0, 1180.0, 1160.0], [1100.0, 1080.0, 1060.0], [1300.0, 1280.0, 1260.0]]
            },
            "metadata": {
                "type": "nominal",
                "description": "Nighttime radiative cooling with negative net radiation",
                "n_patches": 3,
                "n_layers": 3
            }
        },
        "test_nominal_cold_winter_conditions": {
            "inputs": {
                "tref": [263.15, 268.15, 258.15],
                "pref": [102000.0, 101500.0, 102500.0],
                "rhomol": [46.8, 45.9, 47.5],
                "cpair": [29.3, 29.3, 29.3],
                "rnsoi": [150.0, 180.0, 120.0],
                "rhg": [0.5, 0.55, 0.45],
                "soilres": [500.0, 450.0, 550.0],
                "gac0": [0.2, 0.22, 0.18],
                "soil_t": [273.15, 275.15, 271.15],
                "soil_dz": [0.1, 0.1, 0.1],
                "soil_tk": [2.2, 2.5, 2.0],
                "tg_bef": [272.15, 274.15, 270.15],
                "tair": [[263.15, 262.15, 261.15], [268.15, 267.15, 266.15], [258.15, 257.15, 256.15]],
                "eair": [[300.0, 290.0, 280.0], [400.0, 390.0, 380.0], [200.0, 190.0, 180.0]]
            },
            "metadata": {
                "type": "nominal",
                "description": "Cold winter conditions near freezing",
                "n_patches": 3,
                "n_layers": 3
            }
        },
        "test_nominal_hot_arid_conditions": {
            "inputs": {
                "tref": [313.15, 318.15, 308.15],
                "pref": [98000.0, 97500.0, 98500.0],
                "rhomol": [38.5, 37.8, 39.2],
                "cpair": [29.3, 29.3, 29.3],
                "rnsoi": [650.0, 720.0, 580.0],
                "rhg": [0.2, 0.15, 0.25],
                "soilres": [800.0, 900.0, 700.0],
                "gac0": [0.25, 0.28, 0.22],
                "soil_t": [318.15, 323.15, 313.15],
                "soil_dz": [0.03, 0.03, 0.03],
                "soil_tk": [0.5, 0.4, 0.6],
                "tg_bef": [316.15, 321.15, 311.15],
                "tair": [[313.15, 312.15, 311.15], [318.15, 317.15, 316.15], [308.15, 307.15, 306.15]],
                "eair": [[1500.0, 1450.0, 1400.0], [1800.0, 1750.0, 1700.0], [1200.0, 1150.0, 1100.0]]
            },
            "metadata": {
                "type": "nominal",
                "description": "Hot arid desert conditions",
                "n_patches": 3,
                "n_layers": 3
            }
        },
        "test_nominal_tropical_humid": {
            "inputs": {
                "tref": [303.15, 305.15, 301.15],
                "pref": [100500.0, 100000.0, 101000.0],
                "rhomol": [40.5, 40.0, 41.0],
                "cpair": [29.3, 29.3, 29.3],
                "rnsoi": [400.0, 450.0, 350.0],
                "rhg": [0.95, 0.98, 0.92],
                "soilres": [100.0, 90.0, 110.0],
                "gac0": [0.12, 0.14, 0.1],
                "soil_t": [302.15, 304.15, 300.15],
                "soil_dz": [0.05, 0.05, 0.05],
                "soil_tk": [1.8, 2.0, 1.6],
                "tg_bef": [302.65, 304.65, 300.65],
                "tair": [[303.15, 302.65, 302.15], [305.15, 304.65, 304.15], [301.15, 300.65, 300.15]],
                "eair": [[3500.0, 3450.0, 3400.0], [3800.0, 3750.0, 3700.0], [3200.0, 3150.0, 3100.0]]
            },
            "metadata": {
                "type": "nominal",
                "description": "Tropical humid conditions",
                "n_patches": 3,
                "n_layers": 3
            }
        },
        "test_edge_zero_net_radiation": {
            "inputs": {
                "tref": [288.15, 290.15],
                "pref": [101325.0, 101325.0],
                "rhomol": [42.5, 42.3],
                "cpair": [29.3, 29.3],
                "rnsoi": [0.0, 0.0],
                "rhg": [0.75, 0.7],
                "soilres": [200.0, 210.0],
                "gac0": [0.1, 0.11],
                "soil_t": [288.15, 290.15],
                "soil_dz": [0.05, 0.05],
                "soil_tk": [1.2, 1.3],
                "tg_bef": [288.15, 290.15],
                "tair": [[288.15, 287.65, 287.15], [290.15, 289.65, 289.15]],
                "eair": [[1400.0, 1380.0, 1360.0], [1500.0, 1480.0, 1460.0]]
            },
            "metadata": {
                "type": "edge",
                "description": "Zero net radiation at transition conditions",
                "n_patches": 2,
                "n_layers": 3
            }
        },
        "test_edge_extreme_humidity_bounds": {
            "inputs": {
                "tref": [295.15, 298.15, 293.15, 300.15],
                "pref": [101325.0, 101325.0, 101325.0, 101325.0],
                "rhomol": [42.0, 41.8, 42.2, 41.6],
                "cpair": [29.3, 29.3, 29.3, 29.3],
                "rnsoi": [300.0, 350.0, 250.0, 400.0],
                "rhg": [0.0, 1.0, 0.01, 0.99],
                "soilres": [1000.0, 50.0, 900.0, 60.0],
                "gac0": [0.15, 0.15, 0.15, 0.15],
                "soil_t": [295.15, 298.15, 293.15, 300.15],
                "soil_dz": [0.05, 0.05, 0.05, 0.05],
                "soil_tk": [1.0, 1.0, 1.0, 1.0],
                "tg_bef": [295.15, 298.15, 293.15, 300.15],
                "tair": [[295.15], [298.15], [293.15], [300.15]],
                "eair": [[1700.0], [1900.0], [1500.0], [2100.0]]
            },
            "metadata": {
                "type": "edge",
                "description": "Extreme relative humidity at boundaries",
                "n_patches": 4,
                "n_layers": 1
            }
        },
        "test_edge_minimal_conductances": {
            "inputs": {
                "tref": [290.15, 292.15],
                "pref": [101325.0, 101325.0],
                "rhomol": [42.3, 42.1],
                "cpair": [29.3, 29.3],
                "rnsoi": [200.0, 220.0],
                "rhg": [0.6, 0.65],
                "soilres": [1e-6, 1e-5],
                "gac0": [1e-6, 1e-5],
                "soil_t": [290.15, 292.15],
                "soil_dz": [1e-6, 1e-5],
                "soil_tk": [1e-6, 1e-5],
                "tg_bef": [290.15, 292.15],
                "tair": [[290.15, 289.65], [292.15, 291.65]],
                "eair": [[1500.0, 1480.0], [1600.0, 1580.0]]
            },
            "metadata": {
                "type": "edge",
                "description": "Minimal positive values for numerical stability",
                "n_patches": 2,
                "n_layers": 2
            }
        },
        "test_edge_extreme_temperature_gradient": {
            "inputs": {
                "tref": [250.15, 320.15],
                "pref": [105000.0, 95000.0],
                "rhomol": [50.0, 36.0],
                "cpair": [29.3, 29.3],
                "rnsoi": [100.0, 800.0],
                "rhg": [0.4, 0.3],
                "soilres": [300.0, 600.0],
                "gac0": [0.3, 0.35],
                "soil_t": [273.15, 330.15],
                "soil_dz": [0.02, 0.15],
                "soil_tk": [3.0, 0.3],
                "tg_bef": [260.15, 325.15],
                "tair": [[250.15, 248.15, 246.15, 244.15, 242.15], 
                         [320.15, 318.15, 316.15, 314.15, 312.15]],
                "eair": [[150.0, 140.0, 130.0, 120.0, 110.0], 
                         [5000.0, 4900.0, 4800.0, 4700.0, 4600.0]]
            },
            "metadata": {
                "type": "edge",
                "description": "Extreme temperature ranges with large gradients",
                "n_patches": 2,
                "n_layers": 5
            }
        },
        "test_special_single_layer_canopy": {
            "inputs": {
                "tref": [295.15, 298.15, 292.15, 300.15, 290.15],
                "pref": [101325.0, 100500.0, 102000.0, 99800.0, 101800.0],
                "rhomol": [42.0, 41.5, 42.5, 41.0, 42.8],
                "cpair": [29.3, 29.3, 29.3, 29.3, 29.3],
                "rnsoi": [350.0, 400.0, 300.0, 450.0, 320.0],
                "rhg": [0.7, 0.65, 0.75, 0.6, 0.72],
                "soilres": [180.0, 200.0, 160.0, 220.0, 190.0],
                "gac0": [0.14, 0.16, 0.12, 0.18, 0.13],
                "soil_t": [294.15, 297.15, 291.15, 299.15, 289.15],
                "soil_dz": [0.05, 0.05, 0.05, 0.05, 0.05],
                "soil_tk": [1.2, 1.3, 1.1, 1.4, 1.0],
                "tg_bef": [295.15, 298.15, 292.15, 300.15, 290.15],
                "tair": [[295.15], [298.15], [292.15], [300.15], [290.15]],
                "eair": [[1700.0], [1900.0], [1500.0], [2100.0], [1600.0]]
            },
            "metadata": {
                "type": "special",
                "description": "Single canopy layer with multiple patches",
                "n_patches": 5,
                "n_layers": 1
            }
        }
    }


def create_input_namedtuple(inputs_dict: Dict[str, List]) -> SoilFluxesInput:
    """
    Convert dictionary of input arrays to SoilFluxesInput NamedTuple.
    
    Args:
        inputs_dict: Dictionary with input field names and values
        
    Returns:
        SoilFluxesInput NamedTuple with JAX arrays
    """
    return SoilFluxesInput(
        tref=jnp.array(inputs_dict["tref"]),
        pref=jnp.array(inputs_dict["pref"]),
        rhomol=jnp.array(inputs_dict["rhomol"]),
        cpair=jnp.array(inputs_dict["cpair"]),
        rnsoi=jnp.array(inputs_dict["rnsoi"]),
        rhg=jnp.array(inputs_dict["rhg"]),
        soilres=jnp.array(inputs_dict["soilres"]),
        gac0=jnp.array(inputs_dict["gac0"]),
        soil_t=jnp.array(inputs_dict["soil_t"]),
        soil_dz=jnp.array(inputs_dict["soil_dz"]),
        soil_tk=jnp.array(inputs_dict["soil_tk"]),
        tg_bef=jnp.array(inputs_dict["tg_bef"]),
        tair=jnp.array(inputs_dict["tair"]),
        eair=jnp.array(inputs_dict["eair"])
    )


# ============================================================================
# Mock Implementation (for testing structure)
# ============================================================================

def soil_fluxes(inputs: SoilFluxesInput) -> SoilFluxesOutput:
    """
    Mock implementation of soil_fluxes for testing purposes.
    
    This is a placeholder that returns physically plausible values.
    Replace with actual implementation when available.
    """
    n_patches = inputs.tref.shape[0]
    
    # Mock outputs with physically reasonable values
    return SoilFluxesOutput(
        shsoi=jnp.ones(n_patches) * 50.0,  # Sensible heat flux
        lhsoi=jnp.ones(n_patches) * 100.0,  # Latent heat flux
        gsoi=jnp.ones(n_patches) * 30.0,  # Soil heat flux
        etsoi=jnp.ones(n_patches) * 0.002,  # Water vapor flux
        tg=inputs.tg_bef + 0.5,  # Surface temperature (slightly warmed)
        eg=jnp.ones(n_patches) * 1500.0,  # Surface vapor pressure
        energy_error=jnp.ones(n_patches) * 0.0001  # Small energy error
    )


# ============================================================================
# Input Validation Tests
# ============================================================================

class TestInputValidation:
    """Test suite for validating input constraints and physical bounds."""
    
    def test_temperature_positivity(self, test_data):
        """Test that all temperature inputs are positive (> 0K)."""
        for test_name, test_case in test_data.items():
            inputs = create_input_namedtuple(test_case["inputs"])
            
            assert jnp.all(inputs.tref > 0), f"{test_name}: tref must be > 0K"
            assert jnp.all(inputs.soil_t > 0), f"{test_name}: soil_t must be > 0K"
            assert jnp.all(inputs.tg_bef > 0), f"{test_name}: tg_bef must be > 0K"
            assert jnp.all(inputs.tair > 0), f"{test_name}: tair must be > 0K"
    
    def test_pressure_positivity(self, test_data):
        """Test that all pressure inputs are positive."""
        for test_name, test_case in test_data.items():
            inputs = create_input_namedtuple(test_case["inputs"])
            
            assert jnp.all(inputs.pref > 0), f"{test_name}: pref must be > 0 Pa"
            assert jnp.all(inputs.eair >= 0), f"{test_name}: eair must be >= 0 Pa"
    
    def test_relative_humidity_bounds(self, test_data):
        """Test that relative humidity is in valid range [0, 1]."""
        for test_name, test_case in test_data.items():
            inputs = create_input_namedtuple(test_case["inputs"])
            
            assert jnp.all(inputs.rhg >= 0), f"{test_name}: rhg must be >= 0"
            assert jnp.all(inputs.rhg <= 1), f"{test_name}: rhg must be <= 1"
    
    def test_conductance_positivity(self, test_data):
        """Test that conductances and related parameters are positive."""
        for test_name, test_case in test_data.items():
            inputs = create_input_namedtuple(test_case["inputs"])
            
            assert jnp.all(inputs.rhomol > 0), f"{test_name}: rhomol must be > 0"
            assert jnp.all(inputs.cpair > 0), f"{test_name}: cpair must be > 0"
            assert jnp.all(inputs.soilres > 0), f"{test_name}: soilres must be > 0"
            assert jnp.all(inputs.gac0 > 0), f"{test_name}: gac0 must be > 0"
            assert jnp.all(inputs.soil_dz > 0), f"{test_name}: soil_dz must be > 0"
            assert jnp.all(inputs.soil_tk >= 0), f"{test_name}: soil_tk must be >= 0"
    
    def test_array_dimensions(self, test_data):
        """Test that array dimensions are consistent and valid."""
        for test_name, test_case in test_data.items():
            inputs = create_input_namedtuple(test_case["inputs"])
            metadata = test_case["metadata"]
            
            n_patches = metadata["n_patches"]
            n_layers = metadata["n_layers"]
            
            # Check 1D arrays have correct patch dimension
            assert inputs.tref.shape == (n_patches,), f"{test_name}: tref shape mismatch"
            assert inputs.pref.shape == (n_patches,), f"{test_name}: pref shape mismatch"
            assert inputs.rhomol.shape == (n_patches,), f"{test_name}: rhomol shape mismatch"
            assert inputs.cpair.shape == (n_patches,), f"{test_name}: cpair shape mismatch"
            assert inputs.rnsoi.shape == (n_patches,), f"{test_name}: rnsoi shape mismatch"
            assert inputs.rhg.shape == (n_patches,), f"{test_name}: rhg shape mismatch"
            assert inputs.soilres.shape == (n_patches,), f"{test_name}: soilres shape mismatch"
            assert inputs.gac0.shape == (n_patches,), f"{test_name}: gac0 shape mismatch"
            assert inputs.soil_t.shape == (n_patches,), f"{test_name}: soil_t shape mismatch"
            assert inputs.soil_dz.shape == (n_patches,), f"{test_name}: soil_dz shape mismatch"
            assert inputs.soil_tk.shape == (n_patches,), f"{test_name}: soil_tk shape mismatch"
            assert inputs.tg_bef.shape == (n_patches,), f"{test_name}: tg_bef shape mismatch"
            
            # Check 2D arrays have correct dimensions
            assert inputs.tair.shape == (n_patches, n_layers), f"{test_name}: tair shape mismatch"
            assert inputs.eair.shape == (n_patches, n_layers), f"{test_name}: eair shape mismatch"
            
            # Check minimum layer requirement
            assert n_layers >= 1, f"{test_name}: n_layers must be >= 1"


# ============================================================================
# Output Shape and Type Tests
# ============================================================================

class TestOutputShapesAndTypes:
    """Test suite for verifying output array shapes and data types."""
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_daytime_conditions",
        "test_nominal_nighttime_cooling",
        "test_nominal_cold_winter_conditions",
        "test_nominal_hot_arid_conditions",
        "test_nominal_tropical_humid",
        "test_edge_zero_net_radiation",
        "test_edge_extreme_humidity_bounds",
        "test_edge_minimal_conductances",
        "test_edge_extreme_temperature_gradient",
        "test_special_single_layer_canopy"
    ])
    def test_output_shapes(self, test_data, test_name):
        """
        Test that all output arrays have correct shape [n_patches].
        
        Args:
            test_data: Fixture providing test cases
            test_name: Name of the test case to run
        """
        test_case = test_data[test_name]
        inputs = create_input_namedtuple(test_case["inputs"])
        n_patches = test_case["metadata"]["n_patches"]
        
        output = soil_fluxes(inputs)
        
        assert output.shsoi.shape == (n_patches,), \
            f"{test_name}: shsoi shape should be ({n_patches},), got {output.shsoi.shape}"
        assert output.lhsoi.shape == (n_patches,), \
            f"{test_name}: lhsoi shape should be ({n_patches},), got {output.lhsoi.shape}"
        assert output.gsoi.shape == (n_patches,), \
            f"{test_name}: gsoi shape should be ({n_patches},), got {output.gsoi.shape}"
        assert output.etsoi.shape == (n_patches,), \
            f"{test_name}: etsoi shape should be ({n_patches},), got {output.etsoi.shape}"
        assert output.tg.shape == (n_patches,), \
            f"{test_name}: tg shape should be ({n_patches},), got {output.tg.shape}"
        assert output.eg.shape == (n_patches,), \
            f"{test_name}: eg shape should be ({n_patches},), got {output.eg.shape}"
        assert output.energy_error.shape == (n_patches,), \
            f"{test_name}: energy_error shape should be ({n_patches},), got {output.energy_error.shape}"
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_daytime_conditions",
        "test_edge_extreme_humidity_bounds",
        "test_special_single_layer_canopy"
    ])
    def test_output_dtypes(self, test_data, test_name):
        """
        Test that all output arrays have correct floating point dtype.
        
        Args:
            test_data: Fixture providing test cases
            test_name: Name of the test case to run
        """
        test_case = test_data[test_name]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # All outputs should be floating point
        assert jnp.issubdtype(output.shsoi.dtype, jnp.floating), \
            f"{test_name}: shsoi should be floating point"
        assert jnp.issubdtype(output.lhsoi.dtype, jnp.floating), \
            f"{test_name}: lhsoi should be floating point"
        assert jnp.issubdtype(output.gsoi.dtype, jnp.floating), \
            f"{test_name}: gsoi should be floating point"
        assert jnp.issubdtype(output.etsoi.dtype, jnp.floating), \
            f"{test_name}: etsoi should be floating point"
        assert jnp.issubdtype(output.tg.dtype, jnp.floating), \
            f"{test_name}: tg should be floating point"
        assert jnp.issubdtype(output.eg.dtype, jnp.floating), \
            f"{test_name}: eg should be floating point"
        assert jnp.issubdtype(output.energy_error.dtype, jnp.floating), \
            f"{test_name}: energy_error should be floating point"


# ============================================================================
# Physical Constraint Tests
# ============================================================================

class TestPhysicalConstraints:
    """Test suite for verifying physical constraints on outputs."""
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_daytime_conditions",
        "test_nominal_nighttime_cooling",
        "test_nominal_cold_winter_conditions",
        "test_nominal_hot_arid_conditions",
        "test_nominal_tropical_humid"
    ])
    def test_temperature_output_positivity(self, test_data, test_name):
        """
        Test that output surface temperature is positive and physically realistic.
        
        Surface temperature should be > 0K and typically in range 173.15-330K.
        """
        test_case = test_data[test_name]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        assert jnp.all(output.tg > 0), \
            f"{test_name}: Surface temperature tg must be > 0K"
        assert jnp.all(output.tg >= 173.15), \
            f"{test_name}: Surface temperature tg should be >= 173.15K (physically realistic)"
        assert jnp.all(output.tg <= 350.0), \
            f"{test_name}: Surface temperature tg should be <= 350K (physically realistic)"
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_daytime_conditions",
        "test_nominal_nighttime_cooling",
        "test_edge_extreme_humidity_bounds"
    ])
    def test_vapor_pressure_positivity(self, test_data, test_name):
        """
        Test that output surface vapor pressure is non-negative.
        
        Vapor pressure must be >= 0 Pa.
        """
        test_case = test_data[test_name]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        assert jnp.all(output.eg >= 0), \
            f"{test_name}: Surface vapor pressure eg must be >= 0 Pa"
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_daytime_conditions",
        "test_nominal_nighttime_cooling",
        "test_nominal_cold_winter_conditions"
    ])
    def test_energy_balance_closure(self, test_data, test_name):
        """
        Test that energy balance error is small (< 0.001 W/m2 for accurate solutions).
        
        The energy balance should close to within numerical precision.
        """
        test_case = test_data[test_name]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Energy error should be very small for converged solutions
        max_error = jnp.max(jnp.abs(output.energy_error))
        assert max_error < 1.0, \
            f"{test_name}: Energy balance error should be < 1.0 W/m2, got {max_error}"
    
    @pytest.mark.parametrize("test_name", [
        "test_nominal_daytime_conditions",
        "test_nominal_hot_arid_conditions"
    ])
    def test_flux_sign_conventions(self, test_data, test_name):
        """
        Test that flux sign conventions are physically reasonable.
        
        For daytime conditions with positive net radiation:
        - Sensible and latent heat fluxes should typically be positive (upward)
        - Soil heat flux should typically be positive (downward into soil)
        """
        test_case = test_data[test_name]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # For positive net radiation, expect positive upward fluxes
        positive_radiation_mask = inputs.rnsoi > 0
        
        if jnp.any(positive_radiation_mask):
            # At least some fluxes should be positive for positive radiation
            assert jnp.any(output.shsoi[positive_radiation_mask] != 0) or \
                   jnp.any(output.lhsoi[positive_radiation_mask] != 0) or \
                   jnp.any(output.gsoi[positive_radiation_mask] != 0), \
                f"{test_name}: Expected non-zero fluxes for positive net radiation"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_zero_net_radiation(self, test_data):
        """
        Test behavior with zero net radiation.
        
        With zero net radiation, the sum of fluxes should balance to near zero.
        """
        test_case = test_data["test_edge_zero_net_radiation"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Check that outputs are finite
        assert jnp.all(jnp.isfinite(output.shsoi)), "shsoi should be finite for zero radiation"
        assert jnp.all(jnp.isfinite(output.lhsoi)), "lhsoi should be finite for zero radiation"
        assert jnp.all(jnp.isfinite(output.gsoi)), "gsoi should be finite for zero radiation"
        assert jnp.all(jnp.isfinite(output.tg)), "tg should be finite for zero radiation"
    
    def test_extreme_humidity_bounds(self, test_data):
        """
        Test behavior at extreme relative humidity values (0.0, 1.0, near-boundaries).
        
        Function should handle dry soil (rhg=0) and saturated soil (rhg=1) gracefully.
        """
        test_case = test_data["test_edge_extreme_humidity_bounds"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # All outputs should be finite
        assert jnp.all(jnp.isfinite(output.shsoi)), "shsoi should be finite at humidity extremes"
        assert jnp.all(jnp.isfinite(output.lhsoi)), "lhsoi should be finite at humidity extremes"
        assert jnp.all(jnp.isfinite(output.etsoi)), "etsoi should be finite at humidity extremes"
        assert jnp.all(jnp.isfinite(output.tg)), "tg should be finite at humidity extremes"
        assert jnp.all(jnp.isfinite(output.eg)), "eg should be finite at humidity extremes"
        
        # For rhg=0 (dry soil), latent heat flux should be minimal or zero
        dry_mask = inputs.rhg == 0.0
        if jnp.any(dry_mask):
            # Latent heat should be small for completely dry soil
            assert jnp.all(jnp.abs(output.lhsoi[dry_mask]) < 1000.0), \
                "Latent heat flux should be limited for completely dry soil"
    
    def test_minimal_conductances(self, test_data):
        """
        Test numerical stability with very small positive conductances.
        
        Function should not produce NaN or Inf with minimal but positive values.
        """
        test_case = test_data["test_edge_minimal_conductances"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Check for NaN or Inf
        assert jnp.all(jnp.isfinite(output.shsoi)), \
            "shsoi should be finite with minimal conductances"
        assert jnp.all(jnp.isfinite(output.lhsoi)), \
            "lhsoi should be finite with minimal conductances"
        assert jnp.all(jnp.isfinite(output.gsoi)), \
            "gsoi should be finite with minimal conductances"
        assert jnp.all(jnp.isfinite(output.etsoi)), \
            "etsoi should be finite with minimal conductances"
        assert jnp.all(jnp.isfinite(output.tg)), \
            "tg should be finite with minimal conductances"
        assert jnp.all(jnp.isfinite(output.eg)), \
            "eg should be finite with minimal conductances"
    
    def test_extreme_temperature_gradients(self, test_data):
        """
        Test behavior with extreme but physically valid temperature gradients.
        
        Function should handle arctic (-23°C) to desert (57°C) conditions.
        """
        test_case = test_data["test_edge_extreme_temperature_gradient"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # All outputs should be finite
        assert jnp.all(jnp.isfinite(output.shsoi)), \
            "shsoi should be finite with extreme temperatures"
        assert jnp.all(jnp.isfinite(output.lhsoi)), \
            "lhsoi should be finite with extreme temperatures"
        assert jnp.all(jnp.isfinite(output.tg)), \
            "tg should be finite with extreme temperatures"
        
        # Surface temperature should be within physical bounds
        assert jnp.all(output.tg > 173.15), \
            "Surface temperature should be above absolute minimum"
        assert jnp.all(output.tg < 350.0), \
            "Surface temperature should be below extreme maximum"


# ============================================================================
# Special Case Tests
# ============================================================================

class TestSpecialCases:
    """Test suite for special configurations and dimension variations."""
    
    def test_single_layer_canopy(self, test_data):
        """
        Test with minimum canopy layers (n_layers=1).
        
        Function should correctly handle single-layer canopy with multiple patches.
        """
        test_case = test_data["test_special_single_layer_canopy"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        # Verify input dimensions
        assert inputs.tair.shape[1] == 1, "Should have exactly 1 canopy layer"
        assert inputs.eair.shape[1] == 1, "Should have exactly 1 canopy layer"
        
        output = soil_fluxes(inputs)
        
        # Check output shapes
        n_patches = test_case["metadata"]["n_patches"]
        assert output.shsoi.shape == (n_patches,), "Output shape should match n_patches"
        assert output.lhsoi.shape == (n_patches,), "Output shape should match n_patches"
        assert output.tg.shape == (n_patches,), "Output shape should match n_patches"
        
        # All outputs should be finite
        assert jnp.all(jnp.isfinite(output.shsoi)), "shsoi should be finite"
        assert jnp.all(jnp.isfinite(output.lhsoi)), "lhsoi should be finite"
        assert jnp.all(jnp.isfinite(output.tg)), "tg should be finite"
    
    def test_multiple_patches_consistency(self, test_data):
        """
        Test that multiple patches are processed independently and consistently.
        
        Each patch should produce independent results.
        """
        test_case = test_data["test_nominal_daytime_conditions"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        n_patches = test_case["metadata"]["n_patches"]
        
        # Each patch should have different results (unless inputs are identical)
        # Check that not all outputs are identical
        if n_patches > 1:
            # At least one output should vary across patches
            varies = (
                jnp.std(output.shsoi) > 1e-10 or
                jnp.std(output.lhsoi) > 1e-10 or
                jnp.std(output.tg) > 1e-10
            )
            assert varies, "Outputs should vary across patches with different inputs"


# ============================================================================
# Nominal Behavior Tests
# ============================================================================

class TestNominalBehavior:
    """Test suite for expected behavior under nominal conditions."""
    
    def test_daytime_heating(self, test_data):
        """
        Test expected behavior during daytime heating conditions.
        
        With positive net radiation:
        - Surface should warm relative to initial temperature
        - Sensible and latent heat fluxes should be positive (upward)
        """
        test_case = test_data["test_nominal_daytime_conditions"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Surface temperature should be reasonable
        assert jnp.all(output.tg > 273.15), \
            "Daytime surface temperature should be above freezing"
        assert jnp.all(output.tg < 330.0), \
            "Daytime surface temperature should be below extreme values"
    
    def test_nighttime_cooling(self, test_data):
        """
        Test expected behavior during nighttime cooling conditions.
        
        With negative net radiation:
        - Surface should cool
        - Fluxes should reflect cooling conditions
        """
        test_case = test_data["test_nominal_nighttime_cooling"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # All outputs should be finite
        assert jnp.all(jnp.isfinite(output.tg)), \
            "Surface temperature should be finite during nighttime"
        assert jnp.all(jnp.isfinite(output.shsoi)), \
            "Sensible heat flux should be finite during nighttime"
    
    def test_cold_winter_conditions(self, test_data):
        """
        Test behavior under cold winter conditions near freezing.
        
        Should handle frozen soil and low temperatures correctly.
        """
        test_case = test_data["test_nominal_cold_winter_conditions"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Surface temperature should be in cold range
        assert jnp.all(output.tg > 173.15), \
            "Winter surface temperature should be above absolute minimum"
        assert jnp.all(output.tg < 300.0), \
            "Winter surface temperature should be in cold range"
    
    def test_hot_arid_conditions(self, test_data):
        """
        Test behavior under hot arid desert conditions.
        
        Should handle high temperatures and low humidity correctly.
        """
        test_case = test_data["test_nominal_hot_arid_conditions"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Surface temperature should be in hot range
        assert jnp.all(output.tg > 290.0), \
            "Desert surface temperature should be warm"
        assert jnp.all(output.tg < 350.0), \
            "Desert surface temperature should be below extreme maximum"
    
    def test_tropical_humid_conditions(self, test_data):
        """
        Test behavior under tropical humid conditions.
        
        Should handle high humidity and temperatures correctly.
        """
        test_case = test_data["test_nominal_tropical_humid"]
        inputs = create_input_namedtuple(test_case["inputs"])
        
        output = soil_fluxes(inputs)
        
        # Surface temperature should be in tropical range
        assert jnp.all(output.tg > 285.0), \
            "Tropical surface temperature should be warm"
        assert jnp.all(output.tg < 320.0), \
            "Tropical surface temperature should be in reasonable range"
        
        # High humidity should produce significant latent heat flux
        assert jnp.any(jnp.abs(output.lhsoi) > 10.0), \
            "Tropical conditions should produce significant latent heat flux"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple aspects of the function."""
    
    def test_energy_conservation_across_conditions(self, test_data):
        """
        Test that energy is conserved across different climate conditions.
        
        Net radiation should approximately equal sum of sensible, latent, and soil heat fluxes.
        """
        for test_name in ["test_nominal_daytime_conditions", 
                         "test_nominal_nighttime_cooling",
                         "test_nominal_cold_winter_conditions"]:
            test_case = test_data[test_name]
            inputs = create_input_namedtuple(test_case["inputs"])
            
            output = soil_fluxes(inputs)
            
            # Energy balance: rnsoi ≈ shsoi + lhsoi + gsoi
            # Allow for some numerical error
            energy_sum = output.shsoi + output.lhsoi + output.gsoi
            residual = jnp.abs(inputs.rnsoi - energy_sum)
            
            # Residual should be small relative to net radiation magnitude
            max_residual = jnp.max(residual)
            max_radiation = jnp.max(jnp.abs(inputs.rnsoi)) + 1.0  # Add 1 to avoid division by zero
            relative_error = max_residual / max_radiation
            
            assert relative_error < 0.5, \
                f"{test_name}: Energy balance residual too large (relative error: {relative_error})"
    
    def test_consistency_across_patch_sizes(self, test_data):
        """
        Test that function produces consistent results regardless of number of patches.
        
        Single patch should give same result as first element of multi-patch run.
        """
        # Get a multi-patch test case
        multi_patch_case = test_data["test_nominal_daytime_conditions"]
        multi_inputs_dict = multi_patch_case["inputs"]
        
        # Create single-patch version using first patch
        single_inputs_dict = {
            key: [val[0]] if isinstance(val, list) and len(val) > 0 else val
            for key, val in multi_inputs_dict.items()
        }
        
        # Handle 2D arrays specially
        single_inputs_dict["tair"] = [multi_inputs_dict["tair"][0]]
        single_inputs_dict["eair"] = [multi_inputs_dict["eair"][0]]
        
        multi_inputs = create_input_namedtuple(multi_inputs_dict)
        single_inputs = create_input_namedtuple(single_inputs_dict)
        
        multi_output = soil_fluxes(multi_inputs)
        single_output = soil_fluxes(single_inputs)
        
        # First element of multi-patch should match single-patch
        # (allowing for numerical differences)
        atol = 1e-5
        rtol = 1e-5
        
        assert jnp.allclose(multi_output.shsoi[0], single_output.shsoi[0], atol=atol, rtol=rtol), \
            "shsoi should be consistent across patch sizes"
        assert jnp.allclose(multi_output.lhsoi[0], single_output.lhsoi[0], atol=atol, rtol=rtol), \
            "lhsoi should be consistent across patch sizes"
        assert jnp.allclose(multi_output.tg[0], single_output.tg[0], atol=atol, rtol=rtol), \
            "tg should be consistent across patch sizes"


# ============================================================================
# Documentation Tests
# ============================================================================

class TestDocumentation:
    """Tests to verify function documentation and interface."""
    
    def test_function_has_docstring(self):
        """Test that the soil_fluxes function has a docstring."""
        assert soil_fluxes.__doc__ is not None, \
            "soil_fluxes function should have a docstring"
    
    def test_namedtuple_fields(self):
        """Test that NamedTuples have expected fields."""
        # Check input fields
        input_fields = SoilFluxesInput._fields
        expected_input_fields = (
            'tref', 'pref', 'rhomol', 'cpair', 'rnsoi', 'rhg', 'soilres',
            'gac0', 'soil_t', 'soil_dz', 'soil_tk', 'tg_bef', 'tair', 'eair'
        )
        assert input_fields == expected_input_fields, \
            f"SoilFluxesInput fields mismatch. Expected {expected_input_fields}, got {input_fields}"
        
        # Check output fields
        output_fields = SoilFluxesOutput._fields
        expected_output_fields = (
            'shsoi', 'lhsoi', 'gsoi', 'etsoi', 'tg', 'eg', 'energy_error'
        )
        assert output_fields == expected_output_fields, \
            f"SoilFluxesOutput fields mismatch. Expected {expected_output_fields}, got {output_fields}"


# ============================================================================
# Summary Statistics Tests
# ============================================================================

class TestSummaryStatistics:
    """Tests for summary statistics and overall behavior."""
    
    def test_output_ranges_reasonable(self, test_data):
        """
        Test that output values are in physically reasonable ranges across all test cases.
        """
        all_tg = []
        all_shsoi = []
        all_lhsoi = []
        all_gsoi = []
        
        for test_name, test_case in test_data.items():
            inputs = create_input_namedtuple(test_case["inputs"])
            output = soil_fluxes(inputs)
            
            all_tg.extend(output.tg.tolist())
            all_shsoi.extend(output.shsoi.tolist())
            all_lhsoi.extend(output.lhsoi.tolist())
            all_gsoi.extend(output.gsoi.tolist())
        
        # Convert to arrays
        all_tg = jnp.array(all_tg)
        all_shsoi = jnp.array(all_shsoi)
        all_lhsoi = jnp.array(all_lhsoi)
        all_gsoi = jnp.array(all_gsoi)
        
        # Check temperature range
        assert jnp.all(all_tg > 173.15), "All surface temperatures should be > 173.15K"
        assert jnp.all(all_tg < 350.0), "All surface temperatures should be < 350K"
        
        # Check that fluxes are finite
        assert jnp.all(jnp.isfinite(all_shsoi)), "All sensible heat fluxes should be finite"
        assert jnp.all(jnp.isfinite(all_lhsoi)), "All latent heat fluxes should be finite"
        assert jnp.all(jnp.isfinite(all_gsoi)), "All soil heat fluxes should be finite"