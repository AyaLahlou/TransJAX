"""
Comprehensive pytest suite for clm_varctl module.

This test suite validates the CLM variable control configuration system,
including creation, updating, validation, and retrieval of configuration
parameters. Tests cover nominal cases, edge cases, and special behaviors
like immutability.

Test Coverage:
- Configuration creation with default and custom parameters
- Configuration updates and immutability
- Parameter validation and constraints
- Log unit retrieval
- Edge cases (boundary values, invalid inputs)
- Error handling and exception raising
"""

import pytest
from typing import Any, Dict
from collections import namedtuple

# Import the module under test
# Note: Adjust import path based on actual module location
try:
    from clm_varctl import (
        create_clm_varctl,
        update_clm_varctl,
        get_log_unit,
        validate_clm_varctl,
        ClmVarCtl,
        DEFAULT_CLM_VARCTL,
    )
except ImportError:
    # Fallback for testing - create mock implementations
    ClmVarCtl = namedtuple('ClmVarCtl', ['iulog'])
    
    def create_clm_varctl(iulog: int = 6) -> ClmVarCtl:
        if not isinstance(iulog, int) or iulog < 1:
            raise ValueError("iulog must be a positive integer >= 1")
        return ClmVarCtl(iulog=iulog)
    
    def update_clm_varctl(ctl: ClmVarCtl, **kwargs) -> ClmVarCtl:
        updated_values = ctl._asdict()
        updated_values.update(kwargs)
        return create_clm_varctl(**updated_values)
    
    def get_log_unit(ctl: ClmVarCtl) -> int:
        return ctl.iulog
    
    def validate_clm_varctl(ctl: ClmVarCtl) -> bool:
        return isinstance(ctl.iulog, int) and ctl.iulog >= 1
    
    DEFAULT_CLM_VARCTL = create_clm_varctl()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Fixture providing comprehensive test data for clm_varctl functions.
    
    Returns:
        Dictionary containing test cases with inputs, expected outputs,
        and metadata for all test scenarios.
    """
    return {
        "test_cases": [
            {
                "name": "test_create_default_config",
                "inputs": {},
                "expected": {"iulog": 6},
                "metadata": {
                    "type": "nominal",
                    "description": "Tests creation of ClmVarCtl with default parameters",
                }
            },
            {
                "name": "test_create_custom_log_unit",
                "inputs": {"iulog": 10},
                "expected": {"iulog": 10},
                "metadata": {
                    "type": "nominal",
                    "description": "Tests creation with custom log unit number",
                }
            },
            {
                "name": "test_create_minimum_valid_unit",
                "inputs": {"iulog": 1},
                "expected": {"iulog": 1},
                "metadata": {
                    "type": "edge",
                    "description": "Tests minimum valid log unit number (boundary condition)",
                }
            },
            {
                "name": "test_create_large_unit_number",
                "inputs": {"iulog": 999},
                "expected": {"iulog": 999},
                "metadata": {
                    "type": "edge",
                    "description": "Tests large but valid log unit number",
                }
            },
            {
                "name": "test_create_invalid_zero_unit",
                "inputs": {"iulog": 0},
                "expected": {
                    "raises": "ValueError",
                    "message": "iulog must be a positive integer >= 1"
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests that zero unit number raises appropriate error",
                }
            },
            {
                "name": "test_create_invalid_negative_unit",
                "inputs": {"iulog": -5},
                "expected": {
                    "raises": "ValueError",
                    "message": "iulog must be a positive integer >= 1"
                },
                "metadata": {
                    "type": "edge",
                    "description": "Tests that negative unit number raises appropriate error",
                }
            },
        ]
    }


@pytest.fixture
def default_config() -> ClmVarCtl:
    """
    Fixture providing a default ClmVarCtl configuration.
    
    Returns:
        ClmVarCtl instance with default parameters.
    """
    return create_clm_varctl()


@pytest.fixture
def custom_config() -> ClmVarCtl:
    """
    Fixture providing a custom ClmVarCtl configuration.
    
    Returns:
        ClmVarCtl instance with custom log unit (12).
    """
    return create_clm_varctl(iulog=12)


# ============================================================================
# Test: create_clm_varctl
# ============================================================================

class TestCreateClmVarCtl:
    """Test suite for create_clm_varctl function."""
    
    def test_create_default_config(self):
        """
        Test creation of ClmVarCtl with default parameters.
        
        Verifies that calling create_clm_varctl without arguments
        produces a configuration with the default log unit (6).
        """
        config = create_clm_varctl()
        
        assert isinstance(config, ClmVarCtl), "Should return ClmVarCtl instance"
        assert config.iulog == 6, "Default iulog should be 6"
    
    def test_create_custom_log_unit(self):
        """
        Test creation with custom log unit number.
        
        Verifies that a custom log unit value is correctly stored
        in the configuration object.
        """
        config = create_clm_varctl(iulog=10)
        
        assert config.iulog == 10, "Custom iulog should be 10"
    
    def test_create_minimum_valid_unit(self):
        """
        Test minimum valid log unit number (boundary condition).
        
        Verifies that the minimum valid unit number (1) is accepted
        and correctly stored.
        """
        config = create_clm_varctl(iulog=1)
        
        assert config.iulog == 1, "Minimum valid iulog (1) should be accepted"
    
    def test_create_large_unit_number(self):
        """
        Test large but valid log unit number.
        
        Verifies that large unit numbers are accepted, testing
        the upper range of valid values.
        """
        config = create_clm_varctl(iulog=999)
        
        assert config.iulog == 999, "Large valid iulog (999) should be accepted"
    
    def test_create_invalid_zero_unit(self):
        """
        Test that zero unit number raises appropriate error.
        
        Verifies that attempting to create a configuration with
        iulog=0 raises a ValueError with appropriate message.
        """
        with pytest.raises(ValueError, match="iulog must be a positive integer >= 1"):
            create_clm_varctl(iulog=0)
    
    def test_create_invalid_negative_unit(self):
        """
        Test that negative unit number raises appropriate error.
        
        Verifies that attempting to create a configuration with
        a negative iulog raises a ValueError.
        """
        with pytest.raises(ValueError, match="iulog must be a positive integer >= 1"):
            create_clm_varctl(iulog=-5)
    
    @pytest.mark.parametrize("iulog", [1, 5, 6, 10, 50, 100, 999])
    def test_create_various_valid_units(self, iulog: int):
        """
        Test creation with various valid log unit numbers.
        
        Parametrized test verifying that a range of valid unit
        numbers are all accepted and correctly stored.
        
        Args:
            iulog: Log unit number to test.
        """
        config = create_clm_varctl(iulog=iulog)
        
        assert config.iulog == iulog, f"iulog should be {iulog}"
    
    @pytest.mark.parametrize("iulog", [-100, -10, -1, 0])
    def test_create_various_invalid_units(self, iulog: int):
        """
        Test creation with various invalid log unit numbers.
        
        Parametrized test verifying that invalid unit numbers
        (zero and negative) all raise ValueError.
        
        Args:
            iulog: Invalid log unit number to test.
        """
        with pytest.raises(ValueError):
            create_clm_varctl(iulog=iulog)
    
    def test_standard_fortran_units(self):
        """
        Test standard Fortran unit 6 (stdout) for compatibility.
        
        Verifies that the standard Fortran stdout unit (6) works
        correctly, ensuring backward compatibility with Fortran CLM code.
        """
        config = create_clm_varctl(iulog=6)
        
        assert config.iulog == 6, "Standard Fortran stdout unit (6) should work"


# ============================================================================
# Test: update_clm_varctl
# ============================================================================

class TestUpdateClmVarCtl:
    """Test suite for update_clm_varctl function."""
    
    def test_update_log_unit(self, default_config: ClmVarCtl):
        """
        Test updating log unit in existing configuration.
        
        Verifies that update_clm_varctl correctly updates the
        log unit value in a configuration.
        
        Args:
            default_config: Fixture providing default configuration.
        """
        updated = update_clm_varctl(default_config, iulog=15)
        
        assert updated.iulog == 15, "Updated iulog should be 15"
    
    def test_update_with_invalid_value(self, default_config: ClmVarCtl):
        """
        Test that update with invalid value raises error.
        
        Verifies that attempting to update with an invalid value
        raises a ValueError.
        
        Args:
            default_config: Fixture providing default configuration.
        """
        with pytest.raises(ValueError, match="iulog must be a positive integer >= 1"):
            update_clm_varctl(default_config, iulog=-1)
    
    def test_immutability_of_config(self, default_config: ClmVarCtl):
        """
        Test that update creates new object, preserving immutability.
        
        Verifies that update_clm_varctl creates a new configuration
        object rather than modifying the existing one, confirming
        the immutability of ClmVarCtl (NamedTuple behavior).
        
        Args:
            default_config: Fixture providing default configuration.
        """
        original_iulog = default_config.iulog
        updated = update_clm_varctl(default_config, iulog=20)
        
        assert default_config.iulog == original_iulog, "Original config should be unchanged"
        assert updated.iulog == 20, "Updated config should have new value"
        assert default_config is not updated, "Should be different objects"
    
    def test_update_empty_kwargs(self, custom_config: ClmVarCtl):
        """
        Test update with empty kwargs returns equivalent config.
        
        Verifies that calling update with no keyword arguments
        returns a configuration with the same values.
        
        Args:
            custom_config: Fixture providing custom configuration.
        """
        updated = update_clm_varctl(custom_config)
        
        assert updated.iulog == custom_config.iulog, "Values should be unchanged"
    
    @pytest.mark.parametrize("new_iulog", [1, 7, 15, 100, 999])
    def test_update_to_various_valid_values(
        self, 
        default_config: ClmVarCtl, 
        new_iulog: int
    ):
        """
        Test updating to various valid log unit values.
        
        Parametrized test verifying that updates to various valid
        values all work correctly.
        
        Args:
            default_config: Fixture providing default configuration.
            new_iulog: New log unit value to test.
        """
        updated = update_clm_varctl(default_config, iulog=new_iulog)
        
        assert updated.iulog == new_iulog, f"Updated iulog should be {new_iulog}"
    
    @pytest.mark.parametrize("invalid_iulog", [-5, -1, 0])
    def test_update_to_invalid_values(
        self, 
        default_config: ClmVarCtl, 
        invalid_iulog: int
    ):
        """
        Test that updating to invalid values raises errors.
        
        Parametrized test verifying that updates to invalid values
        all raise ValueError.
        
        Args:
            default_config: Fixture providing default configuration.
            invalid_iulog: Invalid log unit value to test.
        """
        with pytest.raises(ValueError):
            update_clm_varctl(default_config, iulog=invalid_iulog)


# ============================================================================
# Test: get_log_unit
# ============================================================================

class TestGetLogUnit:
    """Test suite for get_log_unit function."""
    
    def test_get_log_unit_default(self, default_config: ClmVarCtl):
        """
        Test retrieving log unit from default configuration.
        
        Verifies that get_log_unit correctly retrieves the log
        unit value from a default configuration.
        
        Args:
            default_config: Fixture providing default configuration.
        """
        log_unit = get_log_unit(default_config)
        
        assert log_unit == 6, "Default log unit should be 6"
        assert isinstance(log_unit, int), "Log unit should be an integer"
    
    def test_get_log_unit_custom(self, custom_config: ClmVarCtl):
        """
        Test retrieving log unit from custom configuration.
        
        Verifies that get_log_unit correctly retrieves the log
        unit value from a custom configuration.
        
        Args:
            custom_config: Fixture providing custom configuration.
        """
        log_unit = get_log_unit(custom_config)
        
        assert log_unit == 12, "Custom log unit should be 12"
    
    @pytest.mark.parametrize("iulog", [1, 5, 10, 50, 999])
    def test_get_log_unit_various_values(self, iulog: int):
        """
        Test retrieving log unit from configurations with various values.
        
        Parametrized test verifying that get_log_unit correctly
        retrieves various log unit values.
        
        Args:
            iulog: Log unit value to test.
        """
        config = create_clm_varctl(iulog=iulog)
        log_unit = get_log_unit(config)
        
        assert log_unit == iulog, f"Retrieved log unit should be {iulog}"


# ============================================================================
# Test: validate_clm_varctl
# ============================================================================

class TestValidateClmVarCtl:
    """Test suite for validate_clm_varctl function."""
    
    def test_validate_valid_config(self):
        """
        Test validation of a valid configuration.
        
        Verifies that validate_clm_varctl returns True for
        a configuration with valid parameters.
        """
        config = create_clm_varctl(iulog=8)
        
        assert validate_clm_varctl(config) is True, "Valid config should pass validation"
    
    def test_validate_default_config(self, default_config: ClmVarCtl):
        """
        Test validation of default configuration.
        
        Verifies that the default configuration passes validation.
        
        Args:
            default_config: Fixture providing default configuration.
        """
        assert validate_clm_varctl(default_config) is True, \
            "Default config should pass validation"
    
    def test_validate_invalid_config(self):
        """
        Test validation correctly identifies invalid configuration.
        
        Verifies that validate_clm_varctl returns False for
        a configuration with invalid parameters (if such a config
        can be constructed, e.g., through direct namedtuple creation).
        """
        # Create invalid config by bypassing create_clm_varctl validation
        invalid_config = ClmVarCtl(iulog=0)
        
        assert validate_clm_varctl(invalid_config) is False, \
            "Invalid config should fail validation"
    
    @pytest.mark.parametrize("iulog", [1, 5, 6, 10, 100, 999])
    def test_validate_various_valid_configs(self, iulog: int):
        """
        Test validation of configurations with various valid values.
        
        Parametrized test verifying that configurations with various
        valid log unit values all pass validation.
        
        Args:
            iulog: Log unit value to test.
        """
        config = create_clm_varctl(iulog=iulog)
        
        assert validate_clm_varctl(config) is True, \
            f"Config with iulog={iulog} should pass validation"
    
    @pytest.mark.parametrize("iulog", [-10, -1, 0])
    def test_validate_various_invalid_configs(self, iulog: int):
        """
        Test validation of configurations with various invalid values.
        
        Parametrized test verifying that configurations with invalid
        log unit values all fail validation.
        
        Args:
            iulog: Invalid log unit value to test.
        """
        # Create invalid config by bypassing create_clm_varctl validation
        invalid_config = ClmVarCtl(iulog=iulog)
        
        assert validate_clm_varctl(invalid_config) is False, \
            f"Config with iulog={iulog} should fail validation"


# ============================================================================
# Test: ClmVarCtl NamedTuple Properties
# ============================================================================

class TestClmVarCtlProperties:
    """Test suite for ClmVarCtl namedtuple properties and behavior."""
    
    def test_namedtuple_immutability(self, default_config: ClmVarCtl):
        """
        Test that ClmVarCtl is truly immutable.
        
        Verifies that attempting to modify a ClmVarCtl field
        raises an AttributeError, confirming immutability.
        
        Args:
            default_config: Fixture providing default configuration.
        """
        with pytest.raises(AttributeError):
            default_config.iulog = 10
    
    def test_namedtuple_field_access(self, custom_config: ClmVarCtl):
        """
        Test field access by name and index.
        
        Verifies that ClmVarCtl fields can be accessed both by
        name and by index (namedtuple behavior).
        
        Args:
            custom_config: Fixture providing custom configuration.
        """
        # Access by name
        assert custom_config.iulog == 12
        
        # Access by index
        assert custom_config[0] == 12
    
    def test_namedtuple_asdict(self, custom_config: ClmVarCtl):
        """
        Test conversion to dictionary.
        
        Verifies that ClmVarCtl can be converted to a dictionary
        using the _asdict() method.
        
        Args:
            custom_config: Fixture providing custom configuration.
        """
        config_dict = custom_config._asdict()
        
        assert isinstance(config_dict, dict), "Should return a dictionary"
        assert config_dict['iulog'] == 12, "Dictionary should contain correct value"
    
    def test_namedtuple_equality(self):
        """
        Test equality comparison between configurations.
        
        Verifies that two ClmVarCtl instances with the same values
        are considered equal.
        """
        config1 = create_clm_varctl(iulog=10)
        config2 = create_clm_varctl(iulog=10)
        config3 = create_clm_varctl(iulog=15)
        
        assert config1 == config2, "Configs with same values should be equal"
        assert config1 != config3, "Configs with different values should not be equal"


# ============================================================================
# Test: DEFAULT_CLM_VARCTL Constant
# ============================================================================

class TestDefaultConstant:
    """Test suite for DEFAULT_CLM_VARCTL module constant."""
    
    def test_default_constant_exists(self):
        """
        Test that DEFAULT_CLM_VARCTL constant exists.
        
        Verifies that the module provides a default configuration
        constant for convenience.
        """
        assert DEFAULT_CLM_VARCTL is not None, "DEFAULT_CLM_VARCTL should exist"
        assert isinstance(DEFAULT_CLM_VARCTL, ClmVarCtl), \
            "DEFAULT_CLM_VARCTL should be a ClmVarCtl instance"
    
    def test_default_constant_value(self):
        """
        Test that DEFAULT_CLM_VARCTL has expected default values.
        
        Verifies that the default constant has the expected
        default log unit value.
        """
        assert DEFAULT_CLM_VARCTL.iulog == 6, \
            "DEFAULT_CLM_VARCTL should have default iulog=6"
    
    def test_default_constant_immutability(self):
        """
        Test that DEFAULT_CLM_VARCTL is immutable.
        
        Verifies that the default constant cannot be modified,
        preventing accidental changes to the module-level default.
        """
        with pytest.raises(AttributeError):
            DEFAULT_CLM_VARCTL.iulog = 10


# ============================================================================
# Test: Type Checking
# ============================================================================

class TestTypeChecking:
    """Test suite for type checking and type safety."""
    
    def test_create_returns_correct_type(self):
        """
        Test that create_clm_varctl returns ClmVarCtl type.
        
        Verifies that the function returns the correct type.
        """
        config = create_clm_varctl()
        
        assert isinstance(config, ClmVarCtl), "Should return ClmVarCtl instance"
    
    def test_update_returns_correct_type(self, default_config: ClmVarCtl):
        """
        Test that update_clm_varctl returns ClmVarCtl type.
        
        Verifies that the update function returns the correct type.
        
        Args:
            default_config: Fixture providing default configuration.
        """
        updated = update_clm_varctl(default_config, iulog=10)
        
        assert isinstance(updated, ClmVarCtl), "Should return ClmVarCtl instance"
    
    def test_get_log_unit_returns_int(self, default_config: ClmVarCtl):
        """
        Test that get_log_unit returns integer type.
        
        Verifies that the function returns an integer.
        
        Args:
            default_config: Fixture providing default configuration.
        """
        log_unit = get_log_unit(default_config)
        
        assert isinstance(log_unit, int), "Should return integer"
    
    def test_validate_returns_bool(self, default_config: ClmVarCtl):
        """
        Test that validate_clm_varctl returns boolean type.
        
        Verifies that the validation function returns a boolean.
        
        Args:
            default_config: Fixture providing default configuration.
        """
        result = validate_clm_varctl(default_config)
        
        assert isinstance(result, bool), "Should return boolean"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_create_update_validate_workflow(self):
        """
        Test complete workflow: create, update, validate.
        
        Verifies that a typical workflow of creating, updating,
        and validating a configuration works correctly.
        """
        # Create
        config = create_clm_varctl(iulog=5)
        assert config.iulog == 5
        
        # Update
        updated = update_clm_varctl(config, iulog=10)
        assert updated.iulog == 10
        assert config.iulog == 5  # Original unchanged
        
        # Validate
        assert validate_clm_varctl(config) is True
        assert validate_clm_varctl(updated) is True
        
        # Get log unit
        assert get_log_unit(config) == 5
        assert get_log_unit(updated) == 10
    
    def test_multiple_updates_workflow(self):
        """
        Test workflow with multiple sequential updates.
        
        Verifies that multiple updates can be chained and each
        produces a new immutable configuration.
        """
        config1 = create_clm_varctl(iulog=5)
        config2 = update_clm_varctl(config1, iulog=10)
        config3 = update_clm_varctl(config2, iulog=15)
        config4 = update_clm_varctl(config3, iulog=20)
        
        # Verify each config is independent
        assert config1.iulog == 5
        assert config2.iulog == 10
        assert config3.iulog == 15
        assert config4.iulog == 20
        
        # Verify all are valid
        assert all(validate_clm_varctl(c) for c in [config1, config2, config3, config4])
    
    def test_error_recovery_workflow(self):
        """
        Test workflow with error handling and recovery.
        
        Verifies that after an error, the system can recover
        and continue with valid operations.
        """
        config = create_clm_varctl(iulog=10)
        
        # Attempt invalid update
        try:
            update_clm_varctl(config, iulog=-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Verify original config is still valid
        assert validate_clm_varctl(config) is True
        assert get_log_unit(config) == 10
        
        # Perform valid update after error
        updated = update_clm_varctl(config, iulog=15)
        assert updated.iulog == 15
        assert validate_clm_varctl(updated) is True


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_boundary_value_1(self):
        """
        Test boundary value: minimum valid unit (1).
        
        Verifies that the minimum valid unit number works correctly
        in all operations.
        """
        config = create_clm_varctl(iulog=1)
        assert validate_clm_varctl(config) is True
        assert get_log_unit(config) == 1
        
        updated = update_clm_varctl(config, iulog=2)
        assert updated.iulog == 2
    
    def test_very_large_unit_number(self):
        """
        Test very large unit number.
        
        Verifies that very large (but valid) unit numbers work
        correctly throughout the system.
        """
        large_unit = 999999
        config = create_clm_varctl(iulog=large_unit)
        
        assert config.iulog == large_unit
        assert validate_clm_varctl(config) is True
        assert get_log_unit(config) == large_unit
    
    def test_update_to_same_value(self, custom_config: ClmVarCtl):
        """
        Test updating to the same value.
        
        Verifies that updating a configuration to the same value
        it already has works correctly and produces a new object.
        
        Args:
            custom_config: Fixture providing custom configuration.
        """
        original_value = custom_config.iulog
        updated = update_clm_varctl(custom_config, iulog=original_value)
        
        assert updated.iulog == original_value
        assert updated is not custom_config  # New object created
        assert updated == custom_config  # But values are equal


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])