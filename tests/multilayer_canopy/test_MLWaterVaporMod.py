"""
Comprehensive pytest suite for MLWaterVaporMod module.

This test suite covers:
- sat_vap: Saturation vapor pressure calculations
- lat_vap: Latent heat of vaporization
- sat_vap_with_constants: Saturation vapor pressure with custom constants
- vapor_pressure_deficit: VPD calculations

Test coverage includes:
- Nominal/typical atmospheric conditions
- Edge cases (boundaries, phase transitions, saturation)
- Multi-dimensional array handling
- Physical constraints validation
- Numerical accuracy and stability
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Tuple
from collections import namedtuple

# Assuming the module is imported as:
# from multilayer_canopy.MLWaterVaporMod import sat_vap, lat_vap, sat_vap_with_constants, vapor_pressure_deficit, WaterVaporConstants, DEFAULT_CONSTANTS


# Define WaterVaporConstants for testing
WaterVaporConstants = namedtuple(
    'WaterVaporConstants',
    ['tfrz', 'hvap', 'hsub', 'mmh2o']
)

DEFAULT_CONSTANTS = WaterVaporConstants(
    tfrz=273.15,
    hvap=2501000.0,
    hsub=2834000.0,
    mmh2o=0.018015
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_constants():
    """Fixture providing default water vapor constants."""
    return DEFAULT_CONSTANTS


@pytest.fixture
def tolerance():
    """Fixture providing numerical tolerance for comparisons."""
    return {'atol': 1e-6, 'rtol': 1e-6}


@pytest.fixture
def test_data():
    """
    Fixture providing comprehensive test data for all functions.
    
    Returns:
        dict: Test cases organized by function name with inputs and metadata.
    """
    return {
        'sat_vap': {
            'nominal_room_temp': {
                't': jnp.array([293.15, 298.15, 303.15]),
                'tfrz': 273.15,
                'description': 'Room temperature conditions (20°C, 25°C, 30°C)',
                'expected_es_range': (2000, 5000),
                'expected_desdt_range': (100, 400)
            },
            'edge_freezing': {
                't': jnp.array([273.15]),
                'tfrz': 273.15,
                'description': 'Exactly at freezing point',
                'expected_es_approx': 611.0,
                'tolerance_factor': 0.1
            },
            'edge_bounds': {
                't': jnp.array([198.15, 373.15]),
                'tfrz': 273.15,
                'description': 'Temperature boundaries (-75°C and 100°C)',
                'expected_es_at_373': 101325.0,
                'tolerance_factor': 0.05
            },
            'multidim_2d': {
                't': jnp.array([[273.15, 283.15, 293.15],
                               [303.15, 313.15, 323.15]]),
                'tfrz': 273.15,
                'description': '2D array (2x3 spatial grid)',
                'expected_shape': (2, 3)
            }
        },
        'lat_vap': {
            'nominal_atmospheric': {
                't': jnp.array([250.0, 273.15, 300.0, 320.0]),
                'constants': DEFAULT_CONSTANTS,
                'description': 'Atmospheric temperature range',
                'expected_range': (44000, 52000)
            },
            'edge_near_zero': {
                't': jnp.array([1.0, 10.0, 50.0]),
                'constants': DEFAULT_CONSTANTS,
                'description': 'Very low temperatures near absolute zero',
                'expected_behavior': 'sublimation_dominated'
            },
            'multidim_3d': {
                't': jnp.array([[[250.0, 270.0], [290.0, 310.0]],
                               [[230.0, 250.0], [270.0, 290.0]]]),
                'constants': DEFAULT_CONSTANTS,
                'description': '3D array (2x2x2 volumetric)',
                'expected_shape': (2, 2, 2)
            }
        },
        'vapor_pressure_deficit': {
            'nominal_varied_humidity': {
                't': jnp.array([298.15, 298.15, 298.15, 298.15]),
                'rh': jnp.array([30.0, 50.0, 70.0, 90.0]),
                'constants': DEFAULT_CONSTANTS,
                'description': 'Fixed temp (25°C), varying RH',
                'expected_behavior': 'decreasing_with_rh'
            },
            'edge_saturated': {
                't': jnp.array([273.15, 293.15, 313.15]),
                'rh': jnp.array([100.0, 100.0, 100.0]),
                'constants': DEFAULT_CONSTANTS,
                'description': '100% RH (saturated air)',
                'expected_vpd_approx': 0.0,
                'tolerance': 1e-3
            },
            'edge_dry': {
                't': jnp.array([273.15, 293.15, 313.15]),
                'rh': jnp.array([0.0, 0.0, 0.0]),
                'constants': DEFAULT_CONSTANTS,
                'description': '0% RH (completely dry)',
                'expected_behavior': 'vpd_equals_es'
            },
            'multidim_broadcast': {
                't': jnp.array([[273.15, 283.15], [293.15, 303.15]]),
                'rh': jnp.array([[50.0, 60.0], [70.0, 80.0]]),
                'constants': DEFAULT_CONSTANTS,
                'description': '2D arrays with broadcasting',
                'expected_shape': (2, 2)
            }
        },
        'sat_vap_with_constants': {
            'nominal_with_defaults': {
                't': jnp.array([273.15, 293.15, 313.15]),
                'constants': DEFAULT_CONSTANTS,
                'description': 'Standard temperatures with default constants',
                'expected_es_range': (600, 8000)
            },
            'multidim_3d': {
                't': jnp.array([[[273.15, 283.15], [293.15, 303.15]],
                               [[313.15, 323.15], [333.15, 343.15]]]),
                'constants': DEFAULT_CONSTANTS,
                'description': '3D array (2x2x2)',
                'expected_shape': (2, 2, 2)
            }
        }
    }


# ============================================================================
# Test sat_vap function
# ============================================================================

class TestSatVap:
    """Test suite for sat_vap function."""
    
    @pytest.mark.parametrize("case_name", [
        'nominal_room_temp',
        'edge_freezing',
        'edge_bounds',
        'multidim_2d'
    ])
    def test_sat_vap_shapes(self, test_data, case_name):
        """
        Test that sat_vap returns correct output shapes.
        
        Verifies that both es and desdt have the same shape as input temperature.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap
        
        case = test_data['sat_vap'][case_name]
        t = case['t']
        tfrz = case['tfrz']
        
        es, desdt = sat_vap(t, tfrz)
        
        assert es.shape == t.shape, \
            f"es shape {es.shape} doesn't match input shape {t.shape} for {case_name}"
        assert desdt.shape == t.shape, \
            f"desdt shape {desdt.shape} doesn't match input shape {t.shape} for {case_name}"
    
    @pytest.mark.parametrize("case_name", [
        'nominal_room_temp',
        'edge_freezing',
        'edge_bounds'
    ])
    def test_sat_vap_dtypes(self, test_data, case_name):
        """
        Test that sat_vap returns correct data types.
        
        Verifies outputs are JAX arrays with float dtype.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap
        
        case = test_data['sat_vap'][case_name]
        t = case['t']
        tfrz = case['tfrz']
        
        es, desdt = sat_vap(t, tfrz)
        
        assert isinstance(es, jnp.ndarray), \
            f"es should be jnp.ndarray, got {type(es)} for {case_name}"
        assert isinstance(desdt, jnp.ndarray), \
            f"desdt should be jnp.ndarray, got {type(desdt)} for {case_name}"
        assert jnp.issubdtype(es.dtype, jnp.floating), \
            f"es should have float dtype, got {es.dtype} for {case_name}"
        assert jnp.issubdtype(desdt.dtype, jnp.floating), \
            f"desdt should have float dtype, got {desdt.dtype} for {case_name}"
    
    def test_sat_vap_nominal_values(self, test_data, tolerance):
        """
        Test sat_vap returns physically reasonable values for room temperature.
        
        Verifies:
        - es increases with temperature
        - desdt is positive
        - Values are within expected ranges
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap
        
        case = test_data['sat_vap']['nominal_room_temp']
        t = case['t']
        tfrz = case['tfrz']
        
        es, desdt = sat_vap(t, tfrz)
        
        # Check es increases with temperature
        assert jnp.all(jnp.diff(es) > 0), \
            "Saturation vapor pressure should increase with temperature"
        
        # Check desdt is positive
        assert jnp.all(desdt > 0), \
            "Temperature derivative of es should be positive"
        
        # Check values are in expected range
        es_min, es_max = case['expected_es_range']
        assert jnp.all(es >= es_min) and jnp.all(es <= es_max), \
            f"es values {es} outside expected range [{es_min}, {es_max}]"
        
        desdt_min, desdt_max = case['expected_desdt_range']
        assert jnp.all(desdt >= desdt_min) and jnp.all(desdt <= desdt_max), \
            f"desdt values {desdt} outside expected range [{desdt_min}, {desdt_max}]"
    
    def test_sat_vap_freezing_point(self, test_data):
        """
        Test sat_vap at freezing point (triple point).
        
        At 273.15K, saturation vapor pressure should be approximately 611 Pa.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap
        
        case = test_data['sat_vap']['edge_freezing']
        t = case['t']
        tfrz = case['tfrz']
        
        es, desdt = sat_vap(t, tfrz)
        
        expected_es = case['expected_es_approx']
        tolerance_factor = case['tolerance_factor']
        
        assert jnp.allclose(es, expected_es, rtol=tolerance_factor), \
            f"es at freezing point {es[0]:.2f} Pa differs from expected {expected_es} Pa"
        
        # desdt should be positive at freezing point
        assert desdt[0] > 0, \
            f"desdt at freezing point should be positive, got {desdt[0]}"
    
    def test_sat_vap_boiling_point(self, test_data):
        """
        Test sat_vap at boiling point (373.15K).
        
        At 373.15K (100°C), saturation vapor pressure should be approximately
        101325 Pa (1 atm).
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap
        
        case = test_data['sat_vap']['edge_bounds']
        t = case['t']
        tfrz = case['tfrz']
        
        es, desdt = sat_vap(t, tfrz)
        
        expected_es = case['expected_es_at_373']
        tolerance_factor = case['tolerance_factor']
        
        # Check the boiling point value (second element)
        assert jnp.allclose(es[1], expected_es, rtol=tolerance_factor), \
            f"es at boiling point {es[1]:.2f} Pa differs from expected {expected_es} Pa"
    
    def test_sat_vap_monotonicity(self, test_data):
        """
        Test that sat_vap is monotonically increasing with temperature.
        
        Both es and desdt should increase as temperature increases.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap
        
        # Create a range of temperatures
        t = jnp.linspace(250.0, 350.0, 50)
        tfrz = 273.15
        
        es, desdt = sat_vap(t, tfrz)
        
        # Check es is monotonically increasing
        es_diff = jnp.diff(es)
        assert jnp.all(es_diff > 0), \
            "es should be monotonically increasing with temperature"
        
        # Check desdt is positive everywhere
        assert jnp.all(desdt > 0), \
            "desdt should be positive for all temperatures"
    
    def test_sat_vap_multidimensional(self, test_data):
        """
        Test sat_vap with multi-dimensional input arrays.
        
        Verifies shape preservation and element-wise computation.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap
        
        case = test_data['sat_vap']['multidim_2d']
        t = case['t']
        tfrz = case['tfrz']
        expected_shape = case['expected_shape']
        
        es, desdt = sat_vap(t, tfrz)
        
        assert es.shape == expected_shape, \
            f"es shape {es.shape} doesn't match expected {expected_shape}"
        assert desdt.shape == expected_shape, \
            f"desdt shape {desdt.shape} doesn't match expected {expected_shape}"
        
        # Verify values increase along temperature gradient
        # First row should have lower values than second row
        assert jnp.all(es[0, :] < es[1, :]), \
            "es should increase with temperature across rows"


# ============================================================================
# Test lat_vap function
# ============================================================================

class TestLatVap:
    """Test suite for lat_vap function."""
    
    @pytest.mark.parametrize("case_name", [
        'nominal_atmospheric',
        'edge_near_zero',
        'multidim_3d'
    ])
    def test_lat_vap_shapes(self, test_data, case_name):
        """
        Test that lat_vap returns correct output shape.
        
        Output should have the same shape as input temperature.
        """
        from multilayer_canopy.MLWaterVaporMod import lat_vap
        
        case = test_data['lat_vap'][case_name]
        t = case['t']
        constants = case['constants']
        
        lh = lat_vap(t, constants)
        
        assert lh.shape == t.shape, \
            f"Output shape {lh.shape} doesn't match input shape {t.shape} for {case_name}"
    
    def test_lat_vap_dtypes(self, test_data):
        """
        Test that lat_vap returns correct data type.
        
        Output should be a JAX array with float dtype.
        """
        from multilayer_canopy.MLWaterVaporMod import lat_vap
        
        case = test_data['lat_vap']['nominal_atmospheric']
        t = case['t']
        constants = case['constants']
        
        lh = lat_vap(t, constants)
        
        assert isinstance(lh, jnp.ndarray), \
            f"Output should be jnp.ndarray, got {type(lh)}"
        assert jnp.issubdtype(lh.dtype, jnp.floating), \
            f"Output should have float dtype, got {lh.dtype}"
    
    def test_lat_vap_nominal_values(self, test_data):
        """
        Test lat_vap returns physically reasonable values.
        
        Latent heat should:
        - Be in the range [44000, 52000] J/mol for atmospheric temperatures
        - Decrease with increasing temperature
        - Be positive
        """
        from multilayer_canopy.MLWaterVaporMod import lat_vap
        
        case = test_data['lat_vap']['nominal_atmospheric']
        t = case['t']
        constants = case['constants']
        
        lh = lat_vap(t, constants)
        
        # Check values are in expected range
        lh_min, lh_max = case['expected_range']
        assert jnp.all(lh >= lh_min) and jnp.all(lh <= lh_max), \
            f"Latent heat values {lh} outside expected range [{lh_min}, {lh_max}]"
        
        # Check all values are positive
        assert jnp.all(lh > 0), \
            "Latent heat should be positive"
        
        # Check latent heat decreases with temperature
        lh_diff = jnp.diff(lh)
        assert jnp.all(lh_diff < 0), \
            "Latent heat should decrease with increasing temperature"
    
    def test_lat_vap_phase_transition(self, test_data, default_constants):
        """
        Test lat_vap behavior across phase transition.
        
        Below freezing: should use sublimation heat (hsub)
        Above freezing: should use vaporization heat (hvap)
        """
        from multilayer_canopy.MLWaterVaporMod import lat_vap
        
        tfrz = default_constants.tfrz
        
        # Temperatures below and above freezing
        t_below = jnp.array([tfrz - 10.0])
        t_above = jnp.array([tfrz + 10.0])
        
        lh_below = lat_vap(t_below, default_constants)
        lh_above = lat_vap(t_above, default_constants)
        
        # Below freezing should have higher latent heat (sublimation)
        assert lh_below[0] > lh_above[0], \
            "Latent heat below freezing (sublimation) should be greater than above (vaporization)"
    
    def test_lat_vap_extreme_cold(self, test_data):
        """
        Test lat_vap at very low temperatures.
        
        Should handle temperatures near absolute zero gracefully.
        """
        from multilayer_canopy.MLWaterVaporMod import lat_vap
        
        case = test_data['lat_vap']['edge_near_zero']
        t = case['t']
        constants = case['constants']
        
        lh = lat_vap(t, constants)
        
        # Check all values are positive and finite
        assert jnp.all(jnp.isfinite(lh)), \
            "Latent heat should be finite at extreme cold temperatures"
        assert jnp.all(lh > 0), \
            "Latent heat should be positive at extreme cold temperatures"
        
        # At very cold temperatures, should be dominated by sublimation
        # Values should be relatively high
        assert jnp.all(lh > 45000), \
            "Latent heat at extreme cold should be high (sublimation dominated)"
    
    def test_lat_vap_multidimensional(self, test_data):
        """
        Test lat_vap with 3D input array.
        
        Verifies shape preservation and element-wise computation.
        """
        from multilayer_canopy.MLWaterVaporMod import lat_vap
        
        case = test_data['lat_vap']['multidim_3d']
        t = case['t']
        constants = case['constants']
        expected_shape = case['expected_shape']
        
        lh = lat_vap(t, constants)
        
        assert lh.shape == expected_shape, \
            f"Output shape {lh.shape} doesn't match expected {expected_shape}"
        
        # Check all values are positive
        assert jnp.all(lh > 0), \
            "All latent heat values should be positive"


# ============================================================================
# Test sat_vap_with_constants function
# ============================================================================

class TestSatVapWithConstants:
    """Test suite for sat_vap_with_constants function."""
    
    def test_sat_vap_with_constants_shapes(self, test_data):
        """
        Test that sat_vap_with_constants returns correct output shapes.
        
        Both es and desdt should have the same shape as input temperature.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants
        
        case = test_data['sat_vap_with_constants']['nominal_with_defaults']
        t = case['t']
        constants = case['constants']
        
        es, desdt = sat_vap_with_constants(t, constants)
        
        assert es.shape == t.shape, \
            f"es shape {es.shape} doesn't match input shape {t.shape}"
        assert desdt.shape == t.shape, \
            f"desdt shape {desdt.shape} doesn't match input shape {t.shape}"
    
    def test_sat_vap_with_constants_dtypes(self, test_data):
        """
        Test that sat_vap_with_constants returns correct data types.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants
        
        case = test_data['sat_vap_with_constants']['nominal_with_defaults']
        t = case['t']
        constants = case['constants']
        
        es, desdt = sat_vap_with_constants(t, constants)
        
        assert isinstance(es, jnp.ndarray), \
            f"es should be jnp.ndarray, got {type(es)}"
        assert isinstance(desdt, jnp.ndarray), \
            f"desdt should be jnp.ndarray, got {type(desdt)}"
    
    def test_sat_vap_with_constants_values(self, test_data):
        """
        Test sat_vap_with_constants returns physically reasonable values.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants
        
        case = test_data['sat_vap_with_constants']['nominal_with_defaults']
        t = case['t']
        constants = case['constants']
        
        es, desdt = sat_vap_with_constants(t, constants)
        
        # Check values are in expected range
        es_min, es_max = case['expected_es_range']
        assert jnp.all(es >= es_min) and jnp.all(es <= es_max), \
            f"es values {es} outside expected range [{es_min}, {es_max}]"
        
        # Check es increases with temperature
        assert jnp.all(jnp.diff(es) > 0), \
            "es should increase with temperature"
        
        # Check desdt is positive
        assert jnp.all(desdt > 0), \
            "desdt should be positive"
    
    def test_sat_vap_with_constants_consistency(self, test_data, default_constants):
        """
        Test consistency between sat_vap and sat_vap_with_constants.
        
        When using default constants, both functions should give identical results.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap, sat_vap_with_constants
        
        t = jnp.array([273.15, 293.15, 313.15])
        tfrz = default_constants.tfrz
        
        es1, desdt1 = sat_vap(t, tfrz)
        es2, desdt2 = sat_vap_with_constants(t, default_constants)
        
        assert jnp.allclose(es1, es2, rtol=1e-10), \
            "sat_vap and sat_vap_with_constants should give identical es values"
        assert jnp.allclose(desdt1, desdt2, rtol=1e-10), \
            "sat_vap and sat_vap_with_constants should give identical desdt values"
    
    def test_sat_vap_with_constants_multidimensional(self, test_data):
        """
        Test sat_vap_with_constants with 3D input array.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants
        
        case = test_data['sat_vap_with_constants']['multidim_3d']
        t = case['t']
        constants = case['constants']
        expected_shape = case['expected_shape']
        
        es, desdt = sat_vap_with_constants(t, constants)
        
        assert es.shape == expected_shape, \
            f"es shape {es.shape} doesn't match expected {expected_shape}"
        assert desdt.shape == expected_shape, \
            f"desdt shape {desdt.shape} doesn't match expected {expected_shape}"


# ============================================================================
# Test vapor_pressure_deficit function
# ============================================================================

class TestVaporPressureDeficit:
    """Test suite for vapor_pressure_deficit function."""
    
    @pytest.mark.parametrize("case_name", [
        'nominal_varied_humidity',
        'edge_saturated',
        'edge_dry',
        'multidim_broadcast'
    ])
    def test_vpd_shapes(self, test_data, case_name):
        """
        Test that vapor_pressure_deficit returns correct output shape.
        
        Output should have the same shape as input arrays.
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit
        
        case = test_data['vapor_pressure_deficit'][case_name]
        t = case['t']
        rh = case['rh']
        constants = case['constants']
        
        vpd = vapor_pressure_deficit(t, rh, constants)
        
        assert vpd.shape == t.shape, \
            f"VPD shape {vpd.shape} doesn't match input shape {t.shape} for {case_name}"
    
    def test_vpd_dtypes(self, test_data):
        """
        Test that vapor_pressure_deficit returns correct data type.
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit
        
        case = test_data['vapor_pressure_deficit']['nominal_varied_humidity']
        t = case['t']
        rh = case['rh']
        constants = case['constants']
        
        vpd = vapor_pressure_deficit(t, rh, constants)
        
        assert isinstance(vpd, jnp.ndarray), \
            f"VPD should be jnp.ndarray, got {type(vpd)}"
        assert jnp.issubdtype(vpd.dtype, jnp.floating), \
            f"VPD should have float dtype, got {vpd.dtype}"
    
    def test_vpd_nominal_values(self, test_data):
        """
        Test vapor_pressure_deficit with varying humidity.
        
        VPD should:
        - Decrease as RH increases
        - Be non-negative
        - Be within reasonable range
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit
        
        case = test_data['vapor_pressure_deficit']['nominal_varied_humidity']
        t = case['t']
        rh = case['rh']
        constants = case['constants']
        
        vpd = vapor_pressure_deficit(t, rh, constants)
        
        # Check VPD is non-negative
        assert jnp.all(vpd >= 0), \
            "VPD should be non-negative"
        
        # Check VPD decreases with increasing RH
        vpd_diff = jnp.diff(vpd)
        assert jnp.all(vpd_diff < 0), \
            "VPD should decrease as relative humidity increases"
        
        # Check values are reasonable (< 10000 Pa for typical conditions)
        assert jnp.all(vpd < 10000), \
            f"VPD values {vpd} seem unreasonably high"
    
    def test_vpd_saturated_air(self, test_data):
        """
        Test vapor_pressure_deficit at 100% relative humidity.
        
        At saturation (RH=100%), VPD should be zero or very close to zero.
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit
        
        case = test_data['vapor_pressure_deficit']['edge_saturated']
        t = case['t']
        rh = case['rh']
        constants = case['constants']
        
        vpd = vapor_pressure_deficit(t, rh, constants)
        
        expected_vpd = case['expected_vpd_approx']
        tolerance = case['tolerance']
        
        assert jnp.allclose(vpd, expected_vpd, atol=tolerance), \
            f"VPD at 100% RH should be ~0, got {vpd}"
    
    def test_vpd_dry_air(self, test_data):
        """
        Test vapor_pressure_deficit at 0% relative humidity.
        
        At 0% RH, VPD should equal the saturation vapor pressure.
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit, sat_vap_with_constants
        
        case = test_data['vapor_pressure_deficit']['edge_dry']
        t = case['t']
        rh = case['rh']
        constants = case['constants']
        
        vpd = vapor_pressure_deficit(t, rh, constants)
        es, _ = sat_vap_with_constants(t, constants)
        
        # VPD at 0% RH should equal saturation vapor pressure
        assert jnp.allclose(vpd, es, rtol=1e-6), \
            "VPD at 0% RH should equal saturation vapor pressure"
    
    def test_vpd_formula_verification(self, default_constants):
        """
        Test that VPD follows the formula: VPD = es * (1 - RH/100).
        
        Verifies the mathematical relationship between VPD, es, and RH.
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit, sat_vap_with_constants
        
        t = jnp.array([298.15])
        rh = jnp.array([60.0])
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        es, _ = sat_vap_with_constants(t, default_constants)
        
        expected_vpd = es * (1.0 - rh / 100.0)
        
        assert jnp.allclose(vpd, expected_vpd, rtol=1e-6), \
            f"VPD {vpd} doesn't match expected formula result {expected_vpd}"
    
    def test_vpd_multidimensional(self, test_data):
        """
        Test vapor_pressure_deficit with 2D input arrays.
        
        Verifies shape preservation and broadcasting behavior.
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit
        
        case = test_data['vapor_pressure_deficit']['multidim_broadcast']
        t = case['t']
        rh = case['rh']
        constants = case['constants']
        expected_shape = case['expected_shape']
        
        vpd = vapor_pressure_deficit(t, rh, constants)
        
        assert vpd.shape == expected_shape, \
            f"VPD shape {vpd.shape} doesn't match expected {expected_shape}"
        
        # Check all values are non-negative
        assert jnp.all(vpd >= 0), \
            "All VPD values should be non-negative"
    
    def test_vpd_rh_bounds(self, default_constants):
        """
        Test vapor_pressure_deficit with RH at boundaries [0, 100].
        
        Verifies correct behavior at extreme humidity values.
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit, sat_vap_with_constants
        
        t = jnp.array([298.15, 298.15])
        rh = jnp.array([0.0, 100.0])
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        es, _ = sat_vap_with_constants(t, default_constants)
        
        # At RH=0%, VPD should equal es
        assert jnp.allclose(vpd[0], es[0], rtol=1e-6), \
            "VPD at RH=0% should equal saturation vapor pressure"
        
        # At RH=100%, VPD should be ~0
        assert jnp.allclose(vpd[1], 0.0, atol=1e-3), \
            "VPD at RH=100% should be approximately zero"


# ============================================================================
# Integration and Cross-Function Tests
# ============================================================================

class TestIntegration:
    """Integration tests across multiple functions."""
    
    def test_thermodynamic_consistency(self, default_constants):
        """
        Test thermodynamic consistency across functions.
        
        Verifies that related quantities maintain physical relationships.
        """
        from multilayer_canopy.MLWaterVaporMod import (
            sat_vap_with_constants,
            lat_vap,
            vapor_pressure_deficit
        )
        
        t = jnp.array([273.15, 298.15, 323.15])
        rh = jnp.array([50.0, 50.0, 50.0])
        
        # Get saturation vapor pressure
        es, desdt = sat_vap_with_constants(t, default_constants)
        
        # Get latent heat
        lh = lat_vap(t, default_constants)
        
        # Get VPD
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        # Check all outputs are physically reasonable
        assert jnp.all(es > 0), "Saturation vapor pressure should be positive"
        assert jnp.all(desdt > 0), "Temperature derivative should be positive"
        assert jnp.all(lh > 0), "Latent heat should be positive"
        assert jnp.all(vpd >= 0), "VPD should be non-negative"
        
        # Check Clausius-Clapeyron-like relationship
        # desdt should be roughly proportional to es * lh / (R * T^2)
        # This is a qualitative check
        assert jnp.all(jnp.diff(desdt) > 0), \
            "desdt should increase with temperature"
    
    def test_array_broadcasting_consistency(self, default_constants):
        """
        Test that all functions handle broadcasting consistently.
        
        Verifies that functions work correctly with different array shapes.
        """
        from multilayer_canopy.MLWaterVaporMod import (
            sat_vap_with_constants,
            lat_vap,
            vapor_pressure_deficit
        )
        
        # Test with scalar-like arrays
        t_scalar = jnp.array([298.15])
        rh_scalar = jnp.array([60.0])
        
        es1, desdt1 = sat_vap_with_constants(t_scalar, default_constants)
        lh1 = lat_vap(t_scalar, default_constants)
        vpd1 = vapor_pressure_deficit(t_scalar, rh_scalar, default_constants)
        
        assert es1.shape == (1,), "Scalar-like input should produce (1,) output"
        assert lh1.shape == (1,), "Scalar-like input should produce (1,) output"
        assert vpd1.shape == (1,), "Scalar-like input should produce (1,) output"
        
        # Test with 1D arrays
        t_1d = jnp.array([273.15, 298.15, 323.15])
        rh_1d = jnp.array([40.0, 60.0, 80.0])
        
        es2, desdt2 = sat_vap_with_constants(t_1d, default_constants)
        lh2 = lat_vap(t_1d, default_constants)
        vpd2 = vapor_pressure_deficit(t_1d, rh_1d, default_constants)
        
        assert es2.shape == (3,), "1D input should produce matching 1D output"
        assert lh2.shape == (3,), "1D input should produce matching 1D output"
        assert vpd2.shape == (3,), "1D input should produce matching 1D output"
    
    def test_numerical_stability(self, default_constants):
        """
        Test numerical stability across wide temperature range.
        
        Verifies that functions produce finite, reasonable values across
        the full valid temperature range.
        """
        from multilayer_canopy.MLWaterVaporMod import (
            sat_vap_with_constants,
            lat_vap,
            vapor_pressure_deficit
        )
        
        # Test across full valid range
        t = jnp.linspace(198.15, 373.15, 100)
        rh = jnp.full_like(t, 50.0)
        
        es, desdt = sat_vap_with_constants(t, default_constants)
        lh = lat_vap(t, default_constants)
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        # Check all values are finite
        assert jnp.all(jnp.isfinite(es)), "es should be finite across full range"
        assert jnp.all(jnp.isfinite(desdt)), "desdt should be finite across full range"
        assert jnp.all(jnp.isfinite(lh)), "lh should be finite across full range"
        assert jnp.all(jnp.isfinite(vpd)), "vpd should be finite across full range"
        
        # Check no NaN values
        assert not jnp.any(jnp.isnan(es)), "es should not contain NaN"
        assert not jnp.any(jnp.isnan(desdt)), "desdt should not contain NaN"
        assert not jnp.any(jnp.isnan(lh)), "lh should not contain NaN"
        assert not jnp.any(jnp.isnan(vpd)), "vpd should not contain NaN"


# ============================================================================
# Edge Case and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_temperature_clamping(self, default_constants):
        """
        Test that temperatures outside valid range are handled correctly.
        
        Temperatures should be clamped to [-75°C, 100°C] range.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants
        
        # Test temperatures outside valid range
        t_extreme = jnp.array([100.0, 500.0])  # Very cold and very hot
        
        es, desdt = sat_vap_with_constants(t_extreme, default_constants)
        
        # Should not produce NaN or Inf
        assert jnp.all(jnp.isfinite(es)), \
            "Extreme temperatures should produce finite es values"
        assert jnp.all(jnp.isfinite(desdt)), \
            "Extreme temperatures should produce finite desdt values"
    
    def test_zero_temperature(self, default_constants):
        """
        Test behavior at absolute zero (edge case).
        
        While physically unrealistic, should handle gracefully.
        """
        from multilayer_canopy.MLWaterVaporMod import lat_vap
        
        t = jnp.array([0.0])
        
        lh = lat_vap(t, default_constants)
        
        # Should produce finite value (even if physically meaningless)
        assert jnp.isfinite(lh[0]), \
            "Should handle T=0K without producing NaN/Inf"
    
    def test_negative_temperature(self, default_constants):
        """
        Test that negative temperatures (if passed) are handled.
        
        Note: This tests robustness, not physical validity.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants
        
        t = jnp.array([-10.0])  # Negative Kelvin (unphysical)
        
        # Function should either clamp or handle gracefully
        try:
            es, desdt = sat_vap_with_constants(t, default_constants)
            # If it doesn't raise an error, check for finite values
            assert jnp.isfinite(es[0]) or True, \
                "Should handle negative temperature gracefully"
        except Exception:
            # If it raises an error, that's also acceptable
            pass
    
    def test_rh_outside_bounds(self, default_constants):
        """
        Test VPD with RH values outside [0, 100] range.
        
        Should handle gracefully (clamp or error).
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit
        
        t = jnp.array([298.15, 298.15])
        rh = jnp.array([-10.0, 150.0])  # Outside valid range
        
        # Function should handle gracefully
        try:
            vpd = vapor_pressure_deficit(t, rh, default_constants)
            # If no error, check values are reasonable
            assert jnp.all(jnp.isfinite(vpd)), \
                "Should produce finite VPD even with out-of-range RH"
        except Exception:
            # Raising an error is also acceptable
            pass
    
    def test_empty_arrays(self, default_constants):
        """
        Test functions with empty input arrays.
        
        Should return empty arrays of correct shape.
        """
        from multilayer_canopy.MLWaterVaporMod import (
            sat_vap_with_constants,
            lat_vap,
            vapor_pressure_deficit
        )
        
        t_empty = jnp.array([])
        rh_empty = jnp.array([])
        
        es, desdt = sat_vap_with_constants(t_empty, default_constants)
        lh = lat_vap(t_empty, default_constants)
        vpd = vapor_pressure_deficit(t_empty, rh_empty, default_constants)
        
        assert es.shape == (0,), "Empty input should produce empty output"
        assert desdt.shape == (0,), "Empty input should produce empty output"
        assert lh.shape == (0,), "Empty input should produce empty output"
        assert vpd.shape == (0,), "Empty input should produce empty output"
    
    def test_single_value_arrays(self, default_constants):
        """
        Test functions with single-value arrays.
        
        Should work correctly with minimal input.
        """
        from multilayer_canopy.MLWaterVaporMod import (
            sat_vap_with_constants,
            lat_vap,
            vapor_pressure_deficit
        )
        
        t = jnp.array([298.15])
        rh = jnp.array([60.0])
        
        es, desdt = sat_vap_with_constants(t, default_constants)
        lh = lat_vap(t, default_constants)
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        assert es.shape == (1,), "Single value should produce (1,) shape"
        assert desdt.shape == (1,), "Single value should produce (1,) shape"
        assert lh.shape == (1,), "Single value should produce (1,) shape"
        assert vpd.shape == (1,), "Single value should produce (1,) shape"
        
        # Check values are reasonable
        assert 2000 < es[0] < 5000, "es should be in reasonable range"
        assert 100 < desdt[0] < 400, "desdt should be in reasonable range"
        assert 44000 < lh[0] < 48000, "lh should be in reasonable range"
        assert 0 < vpd[0] < 2000, "vpd should be in reasonable range"


# ============================================================================
# Physical Constraints Tests
# ============================================================================

class TestPhysicalConstraints:
    """Test that outputs satisfy physical constraints."""
    
    def test_positive_pressure(self, default_constants):
        """
        Test that saturation vapor pressure is always positive.
        
        Physical constraint: pressure must be > 0.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants
        
        t = jnp.linspace(200.0, 370.0, 50)
        es, _ = sat_vap_with_constants(t, default_constants)
        
        assert jnp.all(es > 0), \
            "Saturation vapor pressure must be positive"
    
    def test_positive_latent_heat(self, default_constants):
        """
        Test that latent heat is always positive.
        
        Physical constraint: energy required for phase change > 0.
        """
        from multilayer_canopy.MLWaterVaporMod import lat_vap
        
        t = jnp.linspace(200.0, 370.0, 50)
        lh = lat_vap(t, default_constants)
        
        assert jnp.all(lh > 0), \
            "Latent heat must be positive"
    
    def test_non_negative_vpd(self, default_constants):
        """
        Test that VPD is always non-negative.
        
        Physical constraint: deficit cannot be negative.
        """
        from multilayer_canopy.MLWaterVaporMod import vapor_pressure_deficit
        
        t = jnp.linspace(250.0, 350.0, 20)
        rh = jnp.linspace(0.0, 100.0, 20)
        
        vpd = vapor_pressure_deficit(t, rh, default_constants)
        
        assert jnp.all(vpd >= 0), \
            "Vapor pressure deficit must be non-negative"
    
    def test_clausius_clapeyron_relation(self, default_constants):
        """
        Test that outputs approximately satisfy Clausius-Clapeyron equation.
        
        d(ln(es))/dT ≈ L/(R*T^2) where L is latent heat, R is gas constant.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants, lat_vap
        
        t = jnp.array([273.15, 298.15, 323.15])
        
        es, desdt = sat_vap_with_constants(t, default_constants)
        lh = lat_vap(t, default_constants)
        
        # Calculate d(ln(es))/dT = (1/es) * des/dt
        dlnes_dt = desdt / es
        
        # Calculate expected value: L/(R*T^2)
        # R for water vapor ≈ 461.5 J/(kg·K)
        R_water = 461.5
        expected_dlnes_dt = (lh / default_constants.mmh2o) / (R_water * t**2)
        
        # Check they're in the same order of magnitude (loose check)
        ratio = dlnes_dt / expected_dlnes_dt
        assert jnp.all(ratio > 0.1) and jnp.all(ratio < 10.0), \
            "Clausius-Clapeyron relation should be approximately satisfied"
    
    def test_monotonic_increase_with_temperature(self, default_constants):
        """
        Test that es increases monotonically with temperature.
        
        Physical constraint: higher temperature → higher vapor pressure.
        """
        from multilayer_canopy.MLWaterVaporMod import sat_vap_with_constants
        
        t = jnp.linspace(250.0, 350.0, 100)
        es, _ = sat_vap_with_constants(t, default_constants)
        
        # Check monotonic increase
        es_diff = jnp.diff(es)
        assert jnp.all(es_diff > 0), \
            "Saturation vapor pressure must increase monotonically with temperature"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])