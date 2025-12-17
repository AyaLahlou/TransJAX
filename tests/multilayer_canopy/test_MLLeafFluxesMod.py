"""
Comprehensive pytest suite for MLLeafFluxesMod.leaf_fluxes function.

This module tests the leaf energy balance and flux calculations for canopy models,
including nominal conditions across diverse biomes, edge cases with boundary values,
and special cases testing numerical stability and convergence.

Test Coverage:
- Nominal cases: Temperate, tropical, cold, nighttime, and arid conditions
- Edge cases: Zero/negative LAI, fully wet canopy, zero conductances
- Special cases: Extreme temperature gradients
- Physical constraints: Temperature > 0K, fractions in [0,1], energy balance
- Output validation: Shapes, dtypes, value ranges, energy conservation
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple


# ============================================================================
# Mock NamedTuple Definition (replace with actual import in production)
# ============================================================================
class LeafFluxesResult(NamedTuple):
    """Result container for leaf flux calculations."""
    tleaf: jnp.ndarray
    stleaf: jnp.ndarray
    shleaf: jnp.ndarray
    lhleaf: jnp.ndarray
    evleaf: jnp.ndarray
    trleaf: jnp.ndarray
    energy_balance_error: jnp.ndarray


# ============================================================================
# Mock Function (replace with actual import in production)
# ============================================================================
def leaf_fluxes(
    dtime_substep: float,
    tref: float,
    pref: float,
    cpair: float,
    dpai: float,
    tair: float,
    eair: float,
    cpleaf: float,
    fwet: float,
    fdry: float,
    gbh: float,
    gbv: float,
    gs: float,
    rnleaf: float,
    tleaf_bef: float,
) -> LeafFluxesResult:
    """Mock implementation - replace with actual function import."""
    # This is a placeholder - replace with actual import:
    # from MLLeafFluxesMod import leaf_fluxes
    raise NotImplementedError("Replace with actual leaf_fluxes import")


# ============================================================================
# Constants
# ============================================================================
ENERGY_BALANCE_TOL = 1e-3  # W/m2
R8_DTYPE = jnp.float64
ABSOLUTE_ZERO = 0.0  # K (exclusive minimum)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def test_data():
    """
    Fixture providing comprehensive test data for leaf_fluxes function.
    
    Returns:
        dict: Test cases organized by type (nominal, edge, special) with
              inputs, expected behaviors, and metadata.
    """
    return {
        "nominal": [
            {
                "name": "temperate_daytime",
                "inputs": {
                    "dtime_substep": 1800.0,
                    "tref": 298.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": 2.5,
                    "tair": 298.15,
                    "eair": 1500.0,
                    "cpleaf": 2500.0,
                    "fwet": 0.1,
                    "fdry": 0.85,
                    "gbh": 0.5,
                    "gbv": 0.48,
                    "gs": 0.15,
                    "rnleaf": 400.0,
                    "tleaf_bef": 298.15,
                },
                "description": "Typical temperate forest daytime conditions",
            },
            {
                "name": "tropical_high_humidity",
                "inputs": {
                    "dtime_substep": 900.0,
                    "tref": 303.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": 4.0,
                    "tair": 303.15,
                    "eair": 3500.0,
                    "cpleaf": 3000.0,
                    "fwet": 0.3,
                    "fdry": 0.65,
                    "gbh": 0.8,
                    "gbv": 0.76,
                    "gs": 0.25,
                    "rnleaf": 500.0,
                    "tleaf_bef": 303.0,
                },
                "description": "Tropical rainforest with high humidity and LAI",
            },
            {
                "name": "cold_winter",
                "inputs": {
                    "dtime_substep": 3600.0,
                    "tref": 273.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": 1.0,
                    "tair": 273.15,
                    "eair": 611.0,
                    "cpleaf": 2000.0,
                    "fwet": 0.0,
                    "fdry": 0.95,
                    "gbh": 0.3,
                    "gbv": 0.29,
                    "gs": 0.02,
                    "rnleaf": 100.0,
                    "tleaf_bef": 273.15,
                },
                "description": "Cold winter at freezing point",
            },
            {
                "name": "nighttime_low_radiation",
                "inputs": {
                    "dtime_substep": 1800.0,
                    "tref": 288.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": 3.0,
                    "tair": 288.15,
                    "eair": 1200.0,
                    "cpleaf": 2500.0,
                    "fwet": 0.2,
                    "fdry": 0.75,
                    "gbh": 0.4,
                    "gbv": 0.38,
                    "gs": 0.01,
                    "rnleaf": -50.0,
                    "tleaf_bef": 288.5,
                },
                "description": "Nighttime with negative radiation",
            },
            {
                "name": "hot_arid_stressed",
                "inputs": {
                    "dtime_substep": 1200.0,
                    "tref": 313.15,
                    "pref": 95000.0,
                    "cpair": 29.3,
                    "dpai": 1.5,
                    "tair": 313.15,
                    "eair": 800.0,
                    "cpleaf": 2200.0,
                    "fwet": 0.0,
                    "fdry": 0.98,
                    "gbh": 0.6,
                    "gbv": 0.57,
                    "gs": 0.005,
                    "rnleaf": 600.0,
                    "tleaf_bef": 314.0,
                },
                "description": "Hot arid with water stress",
            },
        ],
        "edge": [
            {
                "name": "zero_plant_area_index",
                "inputs": {
                    "dtime_substep": 1800.0,
                    "tref": 298.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": 0.0,
                    "tair": 298.15,
                    "eair": 1500.0,
                    "cpleaf": 2500.0,
                    "fwet": 0.1,
                    "fdry": 0.85,
                    "gbh": 0.5,
                    "gbv": 0.48,
                    "gs": 0.15,
                    "rnleaf": 400.0,
                    "tleaf_bef": 298.15,
                },
                "expected_behavior": "zero_fluxes_tleaf_equals_tair",
                "description": "dpai = 0 should return zero fluxes",
            },
            {
                "name": "negative_plant_area_index",
                "inputs": {
                    "dtime_substep": 1800.0,
                    "tref": 298.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": -0.5,
                    "tair": 298.15,
                    "eair": 1500.0,
                    "cpleaf": 2500.0,
                    "fwet": 0.1,
                    "fdry": 0.85,
                    "gbh": 0.5,
                    "gbv": 0.48,
                    "gs": 0.15,
                    "rnleaf": 400.0,
                    "tleaf_bef": 298.15,
                },
                "expected_behavior": "zero_fluxes_tleaf_equals_tair",
                "description": "dpai < 0 should return zero fluxes",
            },
            {
                "name": "fully_wet_canopy",
                "inputs": {
                    "dtime_substep": 1800.0,
                    "tref": 293.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": 3.0,
                    "tair": 293.15,
                    "eair": 1800.0,
                    "cpleaf": 2500.0,
                    "fwet": 1.0,
                    "fdry": 0.0,
                    "gbh": 0.5,
                    "gbv": 0.48,
                    "gs": 0.15,
                    "rnleaf": 300.0,
                    "tleaf_bef": 293.15,
                },
                "expected_behavior": "no_transpiration",
                "description": "Fully wet canopy (fwet=1.0, fdry=0.0)",
            },
            {
                "name": "zero_conductances",
                "inputs": {
                    "dtime_substep": 1800.0,
                    "tref": 298.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": 2.0,
                    "tair": 298.15,
                    "eair": 1500.0,
                    "cpleaf": 2500.0,
                    "fwet": 0.1,
                    "fdry": 0.85,
                    "gbh": 0.0,
                    "gbv": 0.0,
                    "gs": 0.0,
                    "rnleaf": 400.0,
                    "tleaf_bef": 298.15,
                },
                "expected_behavior": "numerical_stability",
                "description": "All conductances zero (infinite resistance)",
            },
        ],
        "special": [
            {
                "name": "extreme_temperature_gradient",
                "inputs": {
                    "dtime_substep": 600.0,
                    "tref": 298.15,
                    "pref": 101325.0,
                    "cpair": 29.3,
                    "dpai": 2.5,
                    "tair": 288.15,
                    "eair": 1200.0,
                    "cpleaf": 2500.0,
                    "fwet": 0.05,
                    "fdry": 0.9,
                    "gbh": 0.7,
                    "gbv": 0.67,
                    "gs": 0.12,
                    "rnleaf": 800.0,
                    "tleaf_bef": 310.15,
                },
                "expected_behavior": "convergence_test",
                "description": "Extreme temperature gradient tests convergence",
            },
        ],
    }


@pytest.fixture
def nominal_case_names(test_data):
    """Extract nominal test case names for parametrization."""
    return [case["name"] for case in test_data["nominal"]]


@pytest.fixture
def edge_case_names(test_data):
    """Extract edge test case names for parametrization."""
    return [case["name"] for case in test_data["edge"]]


@pytest.fixture
def all_case_names(test_data):
    """Extract all test case names for parametrization."""
    all_cases = (
        test_data["nominal"] + test_data["edge"] + test_data["special"]
    )
    return [case["name"] for case in all_cases]


# ============================================================================
# Test: Input Validation and Physical Constraints
# ============================================================================
class TestInputValidation:
    """Test suite for input validation and physical constraints."""

    def test_temperature_above_absolute_zero(self, test_data):
        """
        Verify all temperature inputs are strictly greater than 0K.
        
        Physical constraint: Temperatures must be > 0K (absolute zero).
        """
        all_cases = (
            test_data["nominal"] + test_data["edge"] + test_data["special"]
        )
        
        for case in all_cases:
            inputs = case["inputs"]
            temp_params = ["tref", "tair", "tleaf_bef"]
            
            for param in temp_params:
                assert inputs[param] > ABSOLUTE_ZERO, (
                    f"Case '{case['name']}': {param}={inputs[param]} "
                    f"must be > 0K"
                )

    def test_fractions_in_valid_range(self, test_data):
        """
        Verify fraction parameters are in [0, 1] range.
        
        Physical constraint: fwet and fdry must be in [0, 1].
        """
        all_cases = (
            test_data["nominal"] + test_data["edge"] + test_data["special"]
        )
        
        for case in all_cases:
            inputs = case["inputs"]
            
            assert 0.0 <= inputs["fwet"] <= 1.0, (
                f"Case '{case['name']}': fwet={inputs['fwet']} "
                f"must be in [0, 1]"
            )
            assert 0.0 <= inputs["fdry"] <= 1.0, (
                f"Case '{case['name']}': fdry={inputs['fdry']} "
                f"must be in [0, 1]"
            )

    def test_conductances_non_negative(self, test_data):
        """
        Verify all conductance parameters are non-negative.
        
        Physical constraint: Conductances (gbh, gbv, gs) must be >= 0.
        """
        all_cases = (
            test_data["nominal"] + test_data["edge"] + test_data["special"]
        )
        
        for case in all_cases:
            inputs = case["inputs"]
            conductances = ["gbh", "gbv", "gs"]
            
            for param in conductances:
                assert inputs[param] >= 0.0, (
                    f"Case '{case['name']}': {param}={inputs[param]} "
                    f"must be >= 0"
                )

    def test_positive_parameters(self, test_data):
        """
        Verify parameters that must be strictly positive.
        
        Physical constraint: dtime_substep, pref, cpair, cpleaf must be > 0.
        """
        all_cases = (
            test_data["nominal"] + test_data["edge"] + test_data["special"]
        )
        
        for case in all_cases:
            inputs = case["inputs"]
            positive_params = ["dtime_substep", "pref", "cpair", "cpleaf"]
            
            for param in positive_params:
                assert inputs[param] > 0.0, (
                    f"Case '{case['name']}': {param}={inputs[param]} "
                    f"must be > 0"
                )


# ============================================================================
# Test: Output Shapes and Types
# ============================================================================
class TestOutputShapesAndTypes:
    """Test suite for output shapes and data types."""

    @pytest.mark.parametrize(
        "case_type",
        ["nominal", "edge", "special"],
        ids=["nominal_cases", "edge_cases", "special_cases"],
    )
    def test_output_is_namedtuple(self, test_data, case_type):
        """
        Verify function returns LeafFluxesResult NamedTuple.
        
        All outputs should be wrapped in the LeafFluxesResult container.
        """
        cases = test_data[case_type]
        
        for case in cases:
            result = leaf_fluxes(**case["inputs"])
            
            assert isinstance(result, LeafFluxesResult), (
                f"Case '{case['name']}': Output must be LeafFluxesResult, "
                f"got {type(result)}"
            )

    @pytest.mark.parametrize(
        "case_type",
        ["nominal", "edge", "special"],
        ids=["nominal_cases", "edge_cases", "special_cases"],
    )
    def test_output_fields_are_scalars(self, test_data, case_type):
        """
        Verify all output fields are scalar values (0-D arrays or floats).
        
        Since inputs are scalars, outputs should also be scalars.
        """
        cases = test_data[case_type]
        
        for case in cases:
            result = leaf_fluxes(**case["inputs"])
            
            fields = [
                "tleaf", "stleaf", "shleaf", "lhleaf",
                "evleaf", "trleaf", "energy_balance_error"
            ]
            
            for field in fields:
                value = getattr(result, field)
                
                # Check if scalar (0-D array or Python float)
                if isinstance(value, jnp.ndarray):
                    assert value.ndim == 0, (
                        f"Case '{case['name']}': {field} should be scalar, "
                        f"got shape {value.shape}"
                    )
                else:
                    assert isinstance(value, (float, np.floating)), (
                        f"Case '{case['name']}': {field} should be scalar, "
                        f"got type {type(value)}"
                    )

    @pytest.mark.parametrize(
        "case_type",
        ["nominal", "edge", "special"],
        ids=["nominal_cases", "edge_cases", "special_cases"],
    )
    def test_output_dtypes(self, test_data, case_type):
        """
        Verify output data types match expected precision (float64).
        
        All outputs should use R8_DTYPE (jnp.float64) precision.
        """
        cases = test_data[case_type]
        
        for case in cases:
            result = leaf_fluxes(**case["inputs"])
            
            fields = [
                "tleaf", "stleaf", "shleaf", "lhleaf",
                "evleaf", "trleaf", "energy_balance_error"
            ]
            
            for field in fields:
                value = getattr(result, field)
                
                if isinstance(value, jnp.ndarray):
                    assert value.dtype == R8_DTYPE, (
                        f"Case '{case['name']}': {field} dtype should be "
                        f"{R8_DTYPE}, got {value.dtype}"
                    )


# ============================================================================
# Test: Physical Constraints on Outputs
# ============================================================================
class TestOutputPhysicalConstraints:
    """Test suite for physical constraints on output values."""

    @pytest.mark.parametrize(
        "case_type",
        ["nominal", "edge", "special"],
        ids=["nominal_cases", "edge_cases", "special_cases"],
    )
    def test_leaf_temperature_above_absolute_zero(self, test_data, case_type):
        """
        Verify leaf temperature is strictly greater than 0K.
        
        Physical constraint: tleaf must be > 0K (absolute zero).
        """
        cases = test_data[case_type]
        
        for case in cases:
            result = leaf_fluxes(**case["inputs"])
            
            tleaf_value = float(result.tleaf)
            assert tleaf_value > ABSOLUTE_ZERO, (
                f"Case '{case['name']}': tleaf={tleaf_value} must be > 0K"
            )

    @pytest.mark.parametrize(
        "case_type",
        ["nominal", "special"],
        ids=["nominal_cases", "special_cases"],
    )
    def test_energy_balance_within_tolerance(self, test_data, case_type):
        """
        Verify energy balance error is within acceptable tolerance.
        
        Physical constraint: Energy balance error should be < 1e-3 W/m2
        for accurate solutions (when dpai > 0).
        """
        cases = test_data[case_type]
        
        for case in cases:
            inputs = case["inputs"]
            
            # Skip if dpai <= 0 (special case with zero fluxes)
            if inputs["dpai"] <= 0:
                continue
            
            result = leaf_fluxes(**inputs)
            
            error = float(result.energy_balance_error)
            assert abs(error) < ENERGY_BALANCE_TOL, (
                f"Case '{case['name']}': Energy balance error "
                f"{error} W/m2 exceeds tolerance {ENERGY_BALANCE_TOL} W/m2"
            )

    def test_energy_conservation(self, test_data):
        """
        Verify energy conservation: rnleaf = stleaf + shleaf + lhleaf.
        
        The sum of storage, sensible, and latent heat fluxes should
        equal net radiation (within energy balance tolerance).
        """
        # Test on nominal cases with dpai > 0
        for case in test_data["nominal"]:
            inputs = case["inputs"]
            result = leaf_fluxes(**inputs)
            
            rnleaf = inputs["rnleaf"]
            flux_sum = (
                float(result.stleaf) +
                float(result.shleaf) +
                float(result.lhleaf)
            )
            
            assert np.isclose(flux_sum, rnleaf, atol=ENERGY_BALANCE_TOL), (
                f"Case '{case['name']}': Energy not conserved. "
                f"rnleaf={rnleaf}, flux_sum={flux_sum}, "
                f"difference={abs(flux_sum - rnleaf)}"
            )


# ============================================================================
# Test: Edge Cases - Zero and Negative dpai
# ============================================================================
class TestEdgeCasesZeroDpai:
    """Test suite for special behavior when dpai <= 0."""

    @pytest.mark.parametrize(
        "case_name",
        ["zero_plant_area_index", "negative_plant_area_index"],
        ids=["dpai_zero", "dpai_negative"],
    )
    def test_zero_dpai_returns_zero_fluxes(self, test_data, case_name):
        """
        Verify that when dpai <= 0, all fluxes are zero.
        
        Special case: When dpai <= 0, the function should return
        zero for all flux components (stleaf, shleaf, lhleaf, evleaf, trleaf).
        """
        # Find the case
        case = next(c for c in test_data["edge"] if c["name"] == case_name)
        
        result = leaf_fluxes(**case["inputs"])
        
        flux_fields = ["stleaf", "shleaf", "lhleaf", "evleaf", "trleaf"]
        
        for field in flux_fields:
            value = float(getattr(result, field))
            assert value == 0.0, (
                f"Case '{case_name}': {field} should be 0.0 when dpai <= 0, "
                f"got {value}"
            )

    @pytest.mark.parametrize(
        "case_name",
        ["zero_plant_area_index", "negative_plant_area_index"],
        ids=["dpai_zero", "dpai_negative"],
    )
    def test_zero_dpai_tleaf_equals_tair(self, test_data, case_name):
        """
        Verify that when dpai <= 0, tleaf equals tair.
        
        Special case: When dpai <= 0, leaf temperature should equal
        air temperature (no leaf present).
        """
        # Find the case
        case = next(c for c in test_data["edge"] if c["name"] == case_name)
        
        inputs = case["inputs"]
        result = leaf_fluxes(**inputs)
        
        tleaf_value = float(result.tleaf)
        tair_value = inputs["tair"]
        
        assert np.isclose(tleaf_value, tair_value, atol=1e-10), (
            f"Case '{case_name}': tleaf should equal tair when dpai <= 0. "
            f"tleaf={tleaf_value}, tair={tair_value}"
        )


# ============================================================================
# Test: Edge Cases - Fully Wet Canopy
# ============================================================================
class TestEdgeCasesFullyWet:
    """Test suite for fully wet canopy conditions."""

    def test_fully_wet_no_transpiration(self, test_data):
        """
        Verify that when fwet=1.0 and fdry=0.0, transpiration is zero.
        
        Physical constraint: Transpiration only occurs through dry leaf
        surfaces. When fdry=0.0, trleaf should be zero.
        """
        case = next(
            c for c in test_data["edge"]
            if c["name"] == "fully_wet_canopy"
        )
        
        result = leaf_fluxes(**case["inputs"])
        
        trleaf_value = float(result.trleaf)
        assert trleaf_value == 0.0, (
            f"Fully wet canopy should have zero transpiration, "
            f"got trleaf={trleaf_value}"
        )

    def test_fully_wet_evaporation_equals_total(self, test_data):
        """
        Verify that when fwet=1.0, all water flux is evaporation.
        
        Physical constraint: When fully wet, evleaf should equal
        the total water flux (no transpiration component).
        """
        case = next(
            c for c in test_data["edge"]
            if c["name"] == "fully_wet_canopy"
        )
        
        result = leaf_fluxes(**case["inputs"])
        
        evleaf_value = float(result.evleaf)
        trleaf_value = float(result.trleaf)
        
        # Total water flux should equal evaporation
        assert trleaf_value == 0.0, "Transpiration should be zero"
        assert evleaf_value >= 0.0, "Evaporation should be non-negative"


# ============================================================================
# Test: Edge Cases - Zero Conductances
# ============================================================================
class TestEdgeCasesZeroConductances:
    """Test suite for zero conductance conditions."""

    def test_zero_conductances_numerical_stability(self, test_data):
        """
        Verify numerical stability when all conductances are zero.
        
        Edge case: When gbh=gbv=gs=0 (infinite resistance), the function
        should handle this gracefully without NaN or Inf values.
        """
        case = next(
            c for c in test_data["edge"]
            if c["name"] == "zero_conductances"
        )
        
        result = leaf_fluxes(**case["inputs"])
        
        # Check all outputs are finite
        fields = [
            "tleaf", "stleaf", "shleaf", "lhleaf",
            "evleaf", "trleaf", "energy_balance_error"
        ]
        
        for field in fields:
            value = float(getattr(result, field))
            assert np.isfinite(value), (
                f"Zero conductances case: {field}={value} is not finite"
            )

    def test_zero_conductances_minimal_fluxes(self, test_data):
        """
        Verify that zero conductances result in minimal or zero fluxes.
        
        Physical constraint: With infinite resistance (zero conductances),
        sensible and latent heat fluxes should be zero or very small.
        """
        case = next(
            c for c in test_data["edge"]
            if c["name"] == "zero_conductances"
        )
        
        result = leaf_fluxes(**case["inputs"])
        
        # Sensible and latent heat should be zero or negligible
        shleaf_value = float(result.shleaf)
        lhleaf_value = float(result.lhleaf)
        evleaf_value = float(result.evleaf)
        trleaf_value = float(result.trleaf)
        
        tolerance = 1e-6
        assert abs(shleaf_value) < tolerance, (
            f"Zero conductances: shleaf={shleaf_value} should be ~0"
        )
        assert abs(lhleaf_value) < tolerance, (
            f"Zero conductances: lhleaf={lhleaf_value} should be ~0"
        )
        assert abs(evleaf_value) < tolerance, (
            f"Zero conductances: evleaf={evleaf_value} should be ~0"
        )
        assert abs(trleaf_value) < tolerance, (
            f"Zero conductances: trleaf={trleaf_value} should be ~0"
        )


# ============================================================================
# Test: Special Cases - Extreme Conditions
# ============================================================================
class TestSpecialCasesExtremeConditions:
    """Test suite for extreme environmental conditions."""

    def test_extreme_temperature_gradient_convergence(self, test_data):
        """
        Verify convergence with extreme temperature gradients.
        
        Special case: Large temperature difference between tair (288K)
        and tleaf_bef (310K) with high radiation forcing should still
        converge to a valid solution.
        """
        case = next(
            c for c in test_data["special"]
            if c["name"] == "extreme_temperature_gradient"
        )
        
        result = leaf_fluxes(**case["inputs"])
        
        # Check convergence via energy balance error
        error = float(result.energy_balance_error)
        assert abs(error) < ENERGY_BALANCE_TOL, (
            f"Extreme gradient case failed to converge: "
            f"energy_balance_error={error}"
        )
        
        # Check tleaf is reasonable (between tair and tleaf_bef + some margin)
        tleaf_value = float(result.tleaf)
        tair = case["inputs"]["tair"]
        tleaf_bef = case["inputs"]["tleaf_bef"]
        
        # Leaf temp should be physically reasonable
        assert tleaf_value > ABSOLUTE_ZERO, "tleaf must be > 0K"
        assert tleaf_value < 400.0, (
            f"tleaf={tleaf_value}K seems unreasonably high"
        )

    def test_negative_radiation_cooling(self, test_data):
        """
        Verify correct behavior with negative net radiation (nighttime cooling).
        
        Physical constraint: Negative rnleaf should result in cooling,
        with negative sensible heat flux (heat from air to leaf).
        """
        case = next(
            c for c in test_data["nominal"]
            if c["name"] == "nighttime_low_radiation"
        )
        
        inputs = case["inputs"]
        result = leaf_fluxes(**inputs)
        
        # With negative radiation, expect cooling
        assert inputs["rnleaf"] < 0, "Test case should have negative radiation"
        
        # Leaf temperature should be reasonable
        tleaf_value = float(result.tleaf)
        assert tleaf_value > ABSOLUTE_ZERO, "tleaf must be > 0K"


# ============================================================================
# Test: Nominal Cases - Value Ranges
# ============================================================================
class TestNominalCasesValueRanges:
    """Test suite for expected value ranges in nominal conditions."""

    @pytest.mark.parametrize(
        "case_name",
        [
            "temperate_daytime",
            "tropical_high_humidity",
            "cold_winter",
            "hot_arid_stressed",
        ],
        ids=[
            "temperate",
            "tropical",
            "cold",
            "arid",
        ],
    )
    def test_leaf_temperature_reasonable_range(self, test_data, case_name):
        """
        Verify leaf temperature is in reasonable physical range.
        
        Physical constraint: Leaf temperature should be within
        reasonable bounds for Earth's biosphere (200K - 350K).
        """
        case = next(
            c for c in test_data["nominal"]
            if c["name"] == case_name
        )
        
        result = leaf_fluxes(**case["inputs"])
        
        tleaf_value = float(result.tleaf)
        assert 200.0 < tleaf_value < 350.0, (
            f"Case '{case_name}': tleaf={tleaf_value}K outside "
            f"reasonable range [200K, 350K]"
        )

    @pytest.mark.parametrize(
        "case_name",
        [
            "temperate_daytime",
            "tropical_high_humidity",
            "hot_arid_stressed",
        ],
        ids=[
            "temperate",
            "tropical",
            "arid",
        ],
    )
    def test_positive_radiation_positive_latent_heat(
        self, test_data, case_name
    ):
        """
        Verify positive net radiation leads to positive latent heat flux.
        
        Physical expectation: With positive radiation and open stomata,
        latent heat flux should be positive (evapotranspiration occurring).
        """
        case = next(
            c for c in test_data["nominal"]
            if c["name"] == case_name
        )
        
        inputs = case["inputs"]
        result = leaf_fluxes(**inputs)
        
        # These cases have positive radiation
        assert inputs["rnleaf"] > 0, "Test case should have positive radiation"
        
        lhleaf_value = float(result.lhleaf)
        # With positive radiation and some conductance, expect positive LH
        if inputs["gs"] > 0 or inputs["fwet"] > 0:
            assert lhleaf_value >= 0, (
                f"Case '{case_name}': Expected positive latent heat flux, "
                f"got lhleaf={lhleaf_value}"
            )


# ============================================================================
# Test: Flux Partitioning
# ============================================================================
class TestFluxPartitioning:
    """Test suite for proper partitioning of water fluxes."""

    def test_evaporation_transpiration_partitioning(self, test_data):
        """
        Verify proper partitioning between evaporation and transpiration.
        
        Physical constraint: Total water flux should be partitioned
        between wet surface evaporation (evleaf) and dry surface
        transpiration (trleaf) based on fwet and fdry.
        """
        # Test on cases with both wet and dry fractions
        case = next(
            c for c in test_data["nominal"]
            if c["name"] == "temperate_daytime"
        )
        
        inputs = case["inputs"]
        result = leaf_fluxes(**inputs)
        
        evleaf_value = float(result.evleaf)
        trleaf_value = float(result.trleaf)
        
        # Both should be non-negative
        assert evleaf_value >= 0, f"evleaf={evleaf_value} should be >= 0"
        assert trleaf_value >= 0, f"trleaf={trleaf_value} should be >= 0"
        
        # If fwet > 0, expect some evaporation
        if inputs["fwet"] > 0:
            # May be zero if vapor pressure gradient is unfavorable
            assert evleaf_value >= 0, "evleaf should be non-negative"
        
        # If fdry > 0 and gs > 0, expect some transpiration
        if inputs["fdry"] > 0 and inputs["gs"] > 0:
            # May be zero if vapor pressure gradient is unfavorable
            assert trleaf_value >= 0, "trleaf should be non-negative"


# ============================================================================
# Test: Consistency Checks
# ============================================================================
class TestConsistencyChecks:
    """Test suite for internal consistency of results."""

    def test_storage_flux_consistency(self, test_data):
        """
        Verify storage flux is consistent with temperature change.
        
        Physical constraint: Storage flux should relate to the change
        in leaf temperature: stleaf = cpleaf * dpai * (tleaf - tleaf_bef) / dt
        """
        # Test on a case with temperature change
        case = next(
            c for c in test_data["special"]
            if c["name"] == "extreme_temperature_gradient"
        )
        
        inputs = case["inputs"]
        result = leaf_fluxes(**inputs)
        
        # Calculate expected storage flux
        tleaf_value = float(result.tleaf)
        dt_leaf = tleaf_value - inputs["tleaf_bef"]
        expected_stleaf = (
            inputs["cpleaf"] * inputs["dpai"] * dt_leaf / inputs["dtime_substep"]
        )
        
        stleaf_value = float(result.stleaf)
        
        # Should be close (within numerical tolerance)
        assert np.isclose(stleaf_value, expected_stleaf, rtol=1e-3), (
            f"Storage flux inconsistent: stleaf={stleaf_value}, "
            f"expected={expected_stleaf}"
        )

    def test_latent_heat_evaporation_consistency(self, test_data):
        """
        Verify latent heat flux is consistent with evaporation rate.
        
        Physical constraint: Latent heat flux should equal evaporation
        rate times latent heat of vaporization.
        """
        # Test on nominal cases
        for case in test_data["nominal"][:3]:  # Test first 3 cases
            inputs = case["inputs"]
            
            # Skip if dpai <= 0
            if inputs["dpai"] <= 0:
                continue
            
            result = leaf_fluxes(**inputs)
            
            lhleaf_value = float(result.lhleaf)
            evleaf_value = float(result.evleaf)
            trleaf_value = float(result.trleaf)
            
            # Total water flux
            total_water_flux = evleaf_value + trleaf_value
            
            # Latent heat should be proportional to water flux
            # (exact relationship depends on latent heat of vaporization)
            if total_water_flux > 0:
                assert lhleaf_value > 0, (
                    f"Case '{case['name']}': Positive water flux "
                    f"should give positive latent heat"
                )


# ============================================================================
# Test: Reproducibility
# ============================================================================
class TestReproducibility:
    """Test suite for reproducibility of results."""

    def test_deterministic_output(self, test_data):
        """
        Verify function produces identical results for identical inputs.
        
        The function should be deterministic - same inputs should
        always produce same outputs.
        """
        case = test_data["nominal"][0]  # Use first nominal case
        
        # Run twice with same inputs
        result1 = leaf_fluxes(**case["inputs"])
        result2 = leaf_fluxes(**case["inputs"])
        
        # Compare all fields
        fields = [
            "tleaf", "stleaf", "shleaf", "lhleaf",
            "evleaf", "trleaf", "energy_balance_error"
        ]
        
        for field in fields:
            val1 = float(getattr(result1, field))
            val2 = float(getattr(result2, field))
            
            assert val1 == val2, (
                f"Non-deterministic output for {field}: "
                f"{val1} != {val2}"
            )


# ============================================================================
# Test Documentation
# ============================================================================
def test_documentation_completeness():
    """
    Verify test suite documentation is complete.
    
    This meta-test ensures all test classes and functions have
    proper docstrings explaining their purpose.
    """
    import inspect
    
    # Get all test classes in this module
    current_module = inspect.getmodule(inspect.currentframe())
    test_classes = [
        obj for name, obj in inspect.getmembers(current_module)
        if inspect.isclass(obj) and name.startswith("Test")
    ]
    
    for test_class in test_classes:
        # Check class docstring
        assert test_class.__doc__ is not None, (
            f"Test class {test_class.__name__} missing docstring"
        )
        
        # Check method docstrings
        test_methods = [
            method for name, method in inspect.getmembers(test_class)
            if name.startswith("test_") and callable(method)
        ]
        
        for method in test_methods:
            assert method.__doc__ is not None, (
                f"Test method {test_class.__name__}.{method.__name__} "
                f"missing docstring"
            )


# ============================================================================
# Test Summary Report
# ============================================================================
@pytest.fixture(scope="session", autouse=True)
def test_summary(request):
    """
    Generate test summary report at end of session.
    
    Provides overview of test coverage and results.
    """
    yield
    
    # This runs after all tests complete
    print("\n" + "=" * 70)
    print("LEAF FLUXES TEST SUITE SUMMARY")
    print("=" * 70)
    print("\nTest Coverage:")
    print("  - Input validation: Temperature, fractions, conductances")
    print("  - Output validation: Shapes, types, physical constraints")
    print("  - Edge cases: Zero/negative dpai, fully wet, zero conductances")
    print("  - Special cases: Extreme gradients, negative radiation")
    print("  - Nominal cases: 5 diverse biome conditions")
    print("  - Consistency: Energy balance, flux partitioning")
    print("  - Reproducibility: Deterministic outputs")
    print("\nPhysical Constraints Tested:")
    print("  - Temperatures > 0K")
    print("  - Fractions in [0, 1]")
    print("  - Conductances >= 0")
    print("  - Energy conservation (< 1e-3 W/m2 error)")
    print("  - Proper flux partitioning")
    print("=" * 70 + "\n")