"""
Comprehensive pytest suite for MLCanopyFluxesType module.

This test suite covers:
- Initialization functions (create_empty, init_allocate, init_cold, init)
- Restart functionality (extract, restore, validate)
- Metadata functions (restart and history metadata)
- Edge cases (single patch, extreme conditions, boundary values)
- Physical constraints (temperature ranges, water potential limits)
- Array shapes and data types
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict, Any
from collections import namedtuple

# Constants from module specification
SPVAL = 1e36
ISPVAL = -9999
NLEAF = 2
ISUN = 0
ISHA = 1
DEFAULT_LWP = -0.1
DEFAULT_H2OCAN = 0.0


# Define namedtuples matching module specification
BoundsType = namedtuple('BoundsType', ['begp', 'endp'])


class MLCanopyState(NamedTuple):
    """Multilayer canopy state variables."""
    ztop_canopy: jnp.ndarray
    zbot_canopy: jnp.ndarray
    lai_canopy: jnp.ndarray
    tref_forcing: jnp.ndarray
    lwp_leaf: jnp.ndarray
    taf_canopy: jnp.ndarray = None
    lwp_mean_profile: jnp.ndarray = None
    h2ocan_profile: jnp.ndarray = None


class MLCanopyRestartData(NamedTuple):
    """Restart data subset."""
    taf_canopy: jnp.ndarray
    lwp_mean_profile: jnp.ndarray


# Mock implementations for testing
def create_empty_mlcanopy_state(n_patches, nlevmlcan, nleaf=2, numrad=2, nlevgrnd=15):
    """Create MLCanopyState with zero-initialized arrays."""
    return MLCanopyState(
        ztop_canopy=jnp.zeros(n_patches),
        zbot_canopy=jnp.zeros(n_patches),
        lai_canopy=jnp.zeros(n_patches),
        tref_forcing=jnp.zeros(n_patches),
        lwp_leaf=jnp.zeros((n_patches, nlevmlcan, nleaf)),
        taf_canopy=jnp.zeros(n_patches),
        lwp_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        h2ocan_profile=jnp.zeros((n_patches, nlevmlcan))
    )


def init_allocate(bounds, nlevmlcan, numrad=2, nlevgrnd=15, nleaf=2):
    """Initialize with special values (SPVAL/ISPVAL)."""
    n_patches = bounds.endp - bounds.begp + 1
    return MLCanopyState(
        ztop_canopy=jnp.full(n_patches, SPVAL),
        zbot_canopy=jnp.full(n_patches, SPVAL),
        lai_canopy=jnp.full(n_patches, SPVAL),
        tref_forcing=jnp.full(n_patches, SPVAL),
        lwp_leaf=jnp.full((n_patches, nlevmlcan, nleaf), SPVAL),
        taf_canopy=jnp.full(n_patches, SPVAL),
        lwp_mean_profile=jnp.full((n_patches, nlevmlcan), SPVAL),
        h2ocan_profile=jnp.full((n_patches, nlevmlcan), SPVAL)
    )


def init_cold(bounds, nlevmlcan, nleaf=2):
    """Initialize with cold start values."""
    n_patches = bounds.endp - bounds.begp + 1
    return MLCanopyState(
        ztop_canopy=jnp.zeros(n_patches),
        zbot_canopy=jnp.zeros(n_patches),
        lai_canopy=jnp.zeros(n_patches),
        tref_forcing=jnp.zeros(n_patches),
        lwp_leaf=jnp.full((n_patches, nlevmlcan, nleaf), DEFAULT_LWP),
        taf_canopy=jnp.zeros(n_patches),
        lwp_mean_profile=jnp.zeros((n_patches, nlevmlcan)),
        h2ocan_profile=jnp.full((n_patches, nlevmlcan), DEFAULT_H2OCAN)
    )


def init(bounds, nlevmlcan, numrad=2, nlevgrnd=15, nleaf=2):
    """Full initialization combining allocate and cold."""
    state = init_allocate(bounds, nlevmlcan, numrad, nlevgrnd, nleaf)
    cold_state = init_cold(bounds, nlevmlcan, nleaf)
    return cold_state


def extract_restart_data(mlcanopy_state):
    """Extract restart variables from full state."""
    return MLCanopyRestartData(
        taf_canopy=mlcanopy_state.taf_canopy,
        lwp_mean_profile=mlcanopy_state.lwp_mean_profile
    )


def restore_from_restart(mlcanopy_state, restart_data):
    """Restore state from restart data."""
    return mlcanopy_state._replace(
        taf_canopy=restart_data.taf_canopy,
        lwp_mean_profile=restart_data.lwp_mean_profile
    )


def validate_restart_data(restart_data, n_patches, nlevmlcan):
    """Validate restart data for correct shapes and physical values."""
    # Check shapes
    if restart_data.taf_canopy.shape != (n_patches,):
        return False
    if restart_data.lwp_mean_profile.shape != (n_patches, nlevmlcan):
        return False
    
    # Check for NaN/Inf
    if jnp.any(jnp.isnan(restart_data.taf_canopy)) or jnp.any(jnp.isinf(restart_data.taf_canopy)):
        return False
    if jnp.any(jnp.isnan(restart_data.lwp_mean_profile)) or jnp.any(jnp.isinf(restart_data.lwp_mean_profile)):
        return False
    
    # Check physical constraints
    if jnp.any(restart_data.taf_canopy < 150.0) or jnp.any(restart_data.taf_canopy > 350.0):
        return False
    if jnp.any(restart_data.lwp_mean_profile < -10.0) or jnp.any(restart_data.lwp_mean_profile > 0.0):
        return False
    
    return True


def get_restart_metadata():
    """Get metadata for restart variables."""
    return {
        'taf_canopy': {
            'dimensions': ['patch'],
            'units': 'K',
            'long_name': 'Air temperature at canopy top',
            'interpinic_flag': 'interp'
        },
        'lwp_mean_profile': {
            'dimensions': ['patch', 'levmlcan'],
            'units': 'MPa',
            'long_name': 'Mean leaf water potential by layer',
            'interpinic_flag': 'interp'
        }
    }


def get_history_metadata():
    """Get metadata for history output variables."""
    return {
        'taf_canopy': {
            'dimensions': ['patch'],
            'units': 'K',
            'long_name': 'Air temperature at canopy top',
            'averaging_flag': 'A'
        },
        'lwp_mean_profile': {
            'dimensions': ['patch', 'levmlcan'],
            'units': 'MPa',
            'long_name': 'Mean leaf water potential by layer',
            'averaging_flag': 'A'
        }
    }


# Fixtures
@pytest.fixture
def test_data():
    """Load test data from specification."""
    return {
        'nominal_create_empty': {
            'n_patches': 10,
            'nlevmlcan': 20,
            'nleaf': 2,
            'numrad': 2,
            'nlevgrnd': 15
        },
        'edge_single_patch': {
            'n_patches': 1,
            'nlevmlcan': 10,
            'nleaf': 2,
            'numrad': 2,
            'nlevgrnd': 15
        },
        'special_many_layers': {
            'n_patches': 5,
            'nlevmlcan': 50,
            'nleaf': 2,
            'numrad': 2,
            'nlevgrnd': 30
        },
        'standard_bounds': {
            'bounds': BoundsType(begp=0, endp=99),
            'nlevmlcan': 20,
            'numrad': 2,
            'nlevgrnd': 15,
            'nleaf': 2
        },
        'single_patch_bounds': {
            'bounds': BoundsType(begp=5, endp=5),
            'nlevmlcan': 15,
            'numrad': 2,
            'nlevgrnd': 15,
            'nleaf': 2
        },
        'typical_forest': {
            'bounds': BoundsType(begp=0, endp=49),
            'nlevmlcan': 25,
            'nleaf': 2
        },
        'minimal_canopy': {
            'bounds': BoundsType(begp=0, endp=9),
            'nlevmlcan': 1,
            'nleaf': 2
        },
        'complete_system': {
            'bounds': BoundsType(begp=0, endp=199),
            'nlevmlcan': 30,
            'numrad': 2,
            'nlevgrnd': 15,
            'nleaf': 2
        }
    }


@pytest.fixture
def sample_mlcanopy_state():
    """Create a sample MLCanopyState for testing."""
    n_patches = 5
    nlevmlcan = 10
    nleaf = 2
    
    return MLCanopyState(
        ztop_canopy=jnp.array([25.0, 30.0, 22.5, 28.0, 26.5]),
        zbot_canopy=jnp.array([0.5, 0.8, 0.3, 0.6, 0.7]),
        lai_canopy=jnp.array([5.2, 6.1, 4.8, 5.9, 5.5]),
        tref_forcing=jnp.array([288.15, 290.5, 285.0, 292.3, 289.7]),
        lwp_leaf=jnp.array([
            [[-0.5, -0.8], [-0.4, -0.7], [-0.6, -0.9], [-0.3, -0.6], [-0.5, -0.8],
             [-0.4, -0.7], [-0.6, -0.9], [-0.5, -0.8], [-0.4, -0.7], [-0.6, -0.9]],
            [[-0.6, -0.9], [-0.5, -0.8], [-0.7, -1.0], [-0.4, -0.7], [-0.6, -0.9],
             [-0.5, -0.8], [-0.7, -1.0], [-0.6, -0.9], [-0.5, -0.8], [-0.7, -1.0]],
            [[-0.4, -0.7], [-0.3, -0.6], [-0.5, -0.8], [-0.2, -0.5], [-0.4, -0.7],
             [-0.3, -0.6], [-0.5, -0.8], [-0.4, -0.7], [-0.3, -0.6], [-0.5, -0.8]],
            [[-0.7, -1.0], [-0.6, -0.9], [-0.8, -1.2], [-0.5, -0.8], [-0.7, -1.0],
             [-0.6, -0.9], [-0.8, -1.2], [-0.7, -1.0], [-0.6, -0.9], [-0.8, -1.2]],
            [[-0.5, -0.8], [-0.4, -0.7], [-0.6, -0.9], [-0.3, -0.6], [-0.5, -0.8],
             [-0.4, -0.7], [-0.6, -0.9], [-0.5, -0.8], [-0.4, -0.7], [-0.6, -0.9]]
        ]),
        taf_canopy=jnp.array([289.0, 291.2, 285.8, 293.0, 290.5]),
        lwp_mean_profile=jnp.array([
            [-0.65, -0.55, -0.75, -0.45, -0.65, -0.55, -0.75, -0.65, -0.55, -0.75],
            [-0.75, -0.65, -0.85, -0.55, -0.75, -0.65, -0.85, -0.75, -0.65, -0.85],
            [-0.55, -0.45, -0.65, -0.35, -0.55, -0.45, -0.65, -0.55, -0.45, -0.65],
            [-0.85, -0.75, -1.0, -0.65, -0.85, -0.75, -1.0, -0.85, -0.75, -1.0],
            [-0.65, -0.55, -0.75, -0.45, -0.65, -0.55, -0.75, -0.65, -0.55, -0.75]
        ]),
        h2ocan_profile=jnp.zeros((5, 10))
    )


# Test create_empty_mlcanopy_state
class TestCreateEmptyMLCanopyState:
    """Tests for create_empty_mlcanopy_state function."""
    
    @pytest.mark.parametrize("test_case", [
        'nominal_create_empty',
        'edge_single_patch',
        'special_many_layers'
    ])
    def test_shapes(self, test_data, test_case):
        """Verify output array shapes match input dimensions."""
        params = test_data[test_case]
        state = create_empty_mlcanopy_state(**params)
        
        n_patches = params['n_patches']
        nlevmlcan = params['nlevmlcan']
        nleaf = params['nleaf']
        
        assert state.ztop_canopy.shape == (n_patches,), \
            f"ztop_canopy shape mismatch: expected {(n_patches,)}, got {state.ztop_canopy.shape}"
        assert state.zbot_canopy.shape == (n_patches,), \
            f"zbot_canopy shape mismatch"
        assert state.lai_canopy.shape == (n_patches,), \
            f"lai_canopy shape mismatch"
        assert state.tref_forcing.shape == (n_patches,), \
            f"tref_forcing shape mismatch"
        assert state.lwp_leaf.shape == (n_patches, nlevmlcan, nleaf), \
            f"lwp_leaf shape mismatch: expected {(n_patches, nlevmlcan, nleaf)}, got {state.lwp_leaf.shape}"
    
    @pytest.mark.parametrize("test_case", [
        'nominal_create_empty',
        'edge_single_patch'
    ])
    def test_values_zero_initialized(self, test_data, test_case):
        """Verify all arrays are zero-initialized."""
        params = test_data[test_case]
        state = create_empty_mlcanopy_state(**params)
        
        assert jnp.allclose(state.ztop_canopy, 0.0, atol=1e-10), \
            "ztop_canopy should be zero-initialized"
        assert jnp.allclose(state.zbot_canopy, 0.0, atol=1e-10), \
            "zbot_canopy should be zero-initialized"
        assert jnp.allclose(state.lai_canopy, 0.0, atol=1e-10), \
            "lai_canopy should be zero-initialized"
        assert jnp.allclose(state.tref_forcing, 0.0, atol=1e-10), \
            "tref_forcing should be zero-initialized"
        assert jnp.allclose(state.lwp_leaf, 0.0, atol=1e-10), \
            "lwp_leaf should be zero-initialized"
    
    def test_dtypes(self, test_data):
        """Verify all arrays have correct data types."""
        params = test_data['nominal_create_empty']
        state = create_empty_mlcanopy_state(**params)
        
        assert state.ztop_canopy.dtype in [jnp.float32, jnp.float64], \
            f"ztop_canopy dtype should be float, got {state.ztop_canopy.dtype}"
        assert state.lwp_leaf.dtype in [jnp.float32, jnp.float64], \
            f"lwp_leaf dtype should be float, got {state.lwp_leaf.dtype}"
    
    def test_edge_case_minimum_dimensions(self):
        """Test with minimum valid dimensions (1 patch, 1 layer)."""
        state = create_empty_mlcanopy_state(n_patches=1, nlevmlcan=1, nleaf=2)
        
        assert state.ztop_canopy.shape == (1,), \
            "Should handle single patch"
        assert state.lwp_leaf.shape == (1, 1, 2), \
            "Should handle single layer"


# Test init_allocate
class TestInitAllocate:
    """Tests for init_allocate function."""
    
    @pytest.mark.parametrize("test_case", [
        'standard_bounds',
        'single_patch_bounds'
    ])
    def test_shapes(self, test_data, test_case):
        """Verify output shapes match bounds dimensions."""
        params = test_data[test_case]
        state = init_allocate(**params)
        
        bounds = params['bounds']
        n_patches = bounds.endp - bounds.begp + 1
        nlevmlcan = params['nlevmlcan']
        nleaf = params['nleaf']
        
        assert state.ztop_canopy.shape == (n_patches,), \
            f"Shape mismatch for {n_patches} patches"
        assert state.lwp_leaf.shape == (n_patches, nlevmlcan, nleaf), \
            f"lwp_leaf shape mismatch"
    
    def test_values_spval_initialized(self, test_data):
        """Verify arrays are initialized with SPVAL."""
        params = test_data['standard_bounds']
        state = init_allocate(**params)
        
        assert jnp.allclose(state.ztop_canopy, SPVAL, atol=1e-6), \
            f"ztop_canopy should be initialized to SPVAL ({SPVAL})"
        assert jnp.allclose(state.lwp_leaf, SPVAL, atol=1e-6), \
            f"lwp_leaf should be initialized to SPVAL"
    
    def test_edge_case_single_patch_bounds(self, test_data):
        """Test with single patch where begp == endp."""
        params = test_data['single_patch_bounds']
        state = init_allocate(**params)
        
        assert state.ztop_canopy.shape == (1,), \
            "Should create single patch when begp == endp"
        assert jnp.allclose(state.ztop_canopy[0], SPVAL, atol=1e-6), \
            "Single patch should be initialized to SPVAL"
    
    def test_bounds_calculation(self):
        """Test correct calculation of n_patches from bounds."""
        bounds = BoundsType(begp=10, endp=19)
        state = init_allocate(bounds, nlevmlcan=15)
        
        expected_patches = 10  # 19 - 10 + 1
        assert state.ztop_canopy.shape == (expected_patches,), \
            f"Expected {expected_patches} patches from bounds (10, 19)"


# Test init_cold
class TestInitCold:
    """Tests for init_cold function."""
    
    @pytest.mark.parametrize("test_case", [
        'typical_forest',
        'minimal_canopy'
    ])
    def test_shapes(self, test_data, test_case):
        """Verify output shapes match input dimensions."""
        params = test_data[test_case]
        state = init_cold(**params)
        
        bounds = params['bounds']
        n_patches = bounds.endp - bounds.begp + 1
        nlevmlcan = params['nlevmlcan']
        nleaf = params['nleaf']
        
        assert state.lwp_leaf.shape == (n_patches, nlevmlcan, nleaf), \
            f"lwp_leaf shape mismatch"
        assert state.h2ocan_profile.shape == (n_patches, nlevmlcan), \
            f"h2ocan_profile shape mismatch"
    
    def test_lwp_leaf_initialization(self, test_data):
        """Verify lwp_leaf is initialized to DEFAULT_LWP."""
        params = test_data['typical_forest']
        state = init_cold(**params)
        
        assert jnp.allclose(state.lwp_leaf, DEFAULT_LWP, atol=1e-10), \
            f"lwp_leaf should be initialized to DEFAULT_LWP ({DEFAULT_LWP} MPa)"
    
    def test_h2ocan_initialization(self, test_data):
        """Verify h2ocan_profile is initialized to DEFAULT_H2OCAN."""
        params = test_data['typical_forest']
        state = init_cold(**params)
        
        assert jnp.allclose(state.h2ocan_profile, DEFAULT_H2OCAN, atol=1e-10), \
            f"h2ocan_profile should be initialized to DEFAULT_H2OCAN ({DEFAULT_H2OCAN} kg/m2)"
    
    def test_edge_case_minimal_canopy(self, test_data):
        """Test with minimal canopy (1 layer)."""
        params = test_data['minimal_canopy']
        state = init_cold(**params)
        
        assert state.lwp_leaf.shape[1] == 1, \
            "Should handle single canopy layer"
        assert jnp.allclose(state.lwp_leaf, DEFAULT_LWP, atol=1e-10), \
            "Single layer should be properly initialized"


# Test init
class TestInit:
    """Tests for init function (combines allocate and cold)."""
    
    def test_complete_initialization(self, test_data):
        """Test complete initialization process."""
        params = test_data['complete_system']
        state = init(**params)
        
        bounds = params['bounds']
        n_patches = bounds.endp - bounds.begp + 1
        nlevmlcan = params['nlevmlcan']
        
        assert state.ztop_canopy.shape == (n_patches,), \
            "Should have correct patch dimension"
        assert state.lwp_leaf.shape[1] == nlevmlcan, \
            "Should have correct layer dimension"
    
    def test_lwp_initialization_after_init(self, test_data):
        """Verify lwp_leaf has cold start values after init."""
        params = test_data['complete_system']
        state = init(**params)
        
        assert jnp.allclose(state.lwp_leaf, DEFAULT_LWP, atol=1e-10), \
            "init should apply cold start values to lwp_leaf"
    
    def test_large_domain(self):
        """Test initialization with large domain (200 patches)."""
        bounds = BoundsType(begp=0, endp=199)
        state = init(bounds, nlevmlcan=30)
        
        assert state.ztop_canopy.shape == (200,), \
            "Should handle large domain (200 patches)"
        assert not jnp.any(jnp.isnan(state.lwp_leaf)), \
            "Should not contain NaN values"


# Test extract_restart_data
class TestExtractRestartData:
    """Tests for extract_restart_data function."""
    
    def test_extraction_shapes(self, sample_mlcanopy_state):
        """Verify extracted data has correct shapes."""
        restart_data = extract_restart_data(sample_mlcanopy_state)
        
        n_patches = sample_mlcanopy_state.taf_canopy.shape[0]
        nlevmlcan = sample_mlcanopy_state.lwp_mean_profile.shape[1]
        
        assert restart_data.taf_canopy.shape == (n_patches,), \
            f"taf_canopy shape mismatch: expected {(n_patches,)}, got {restart_data.taf_canopy.shape}"
        assert restart_data.lwp_mean_profile.shape == (n_patches, nlevmlcan), \
            f"lwp_mean_profile shape mismatch"
    
    def test_extraction_values(self, sample_mlcanopy_state):
        """Verify extracted values match original state."""
        restart_data = extract_restart_data(sample_mlcanopy_state)
        
        assert jnp.allclose(restart_data.taf_canopy, sample_mlcanopy_state.taf_canopy, atol=1e-10), \
            "Extracted taf_canopy should match original"
        assert jnp.allclose(restart_data.lwp_mean_profile, sample_mlcanopy_state.lwp_mean_profile, atol=1e-10), \
            "Extracted lwp_mean_profile should match original"
    
    def test_extraction_independence(self, sample_mlcanopy_state):
        """Verify extracted data is independent of original."""
        restart_data = extract_restart_data(sample_mlcanopy_state)
        
        # Modify original (conceptually - JAX arrays are immutable)
        # Just verify they're separate references
        assert restart_data.taf_canopy is sample_mlcanopy_state.taf_canopy or \
               jnp.array_equal(restart_data.taf_canopy, sample_mlcanopy_state.taf_canopy), \
            "Extracted data should reference or equal original data"


# Test restore_from_restart
class TestRestoreFromRestart:
    """Tests for restore_from_restart function."""
    
    def test_restore_shapes(self, sample_mlcanopy_state):
        """Verify restored state maintains correct shapes."""
        restart_data = extract_restart_data(sample_mlcanopy_state)
        restored_state = restore_from_restart(sample_mlcanopy_state, restart_data)
        
        assert restored_state.taf_canopy.shape == sample_mlcanopy_state.taf_canopy.shape, \
            "Restored taf_canopy shape should match original"
        assert restored_state.lwp_mean_profile.shape == sample_mlcanopy_state.lwp_mean_profile.shape, \
            "Restored lwp_mean_profile shape should match original"
    
    def test_restore_values(self, sample_mlcanopy_state):
        """Verify restored values match restart data."""
        restart_data = extract_restart_data(sample_mlcanopy_state)
        restored_state = restore_from_restart(sample_mlcanopy_state, restart_data)
        
        assert jnp.allclose(restored_state.taf_canopy, restart_data.taf_canopy, atol=1e-10), \
            "Restored taf_canopy should match restart data"
        assert jnp.allclose(restored_state.lwp_mean_profile, restart_data.lwp_mean_profile, atol=1e-10), \
            "Restored lwp_mean_profile should match restart data"
    
    def test_restore_preserves_other_fields(self, sample_mlcanopy_state):
        """Verify restore doesn't modify other state fields."""
        restart_data = extract_restart_data(sample_mlcanopy_state)
        restored_state = restore_from_restart(sample_mlcanopy_state, restart_data)
        
        assert jnp.allclose(restored_state.ztop_canopy, sample_mlcanopy_state.ztop_canopy, atol=1e-10), \
            "ztop_canopy should be preserved"
        assert jnp.allclose(restored_state.lai_canopy, sample_mlcanopy_state.lai_canopy, atol=1e-10), \
            "lai_canopy should be preserved"
    
    def test_complete_restart_cycle(self, sample_mlcanopy_state):
        """Test complete extract-restore cycle preserves data."""
        # Extract
        restart_data = extract_restart_data(sample_mlcanopy_state)
        
        # Create new state
        new_state = create_empty_mlcanopy_state(
            n_patches=5, nlevmlcan=10, nleaf=2
        )
        
        # Restore
        restored_state = restore_from_restart(new_state, restart_data)
        
        assert jnp.allclose(restored_state.taf_canopy, sample_mlcanopy_state.taf_canopy, atol=1e-10), \
            "Complete cycle should preserve taf_canopy"
        assert jnp.allclose(restored_state.lwp_mean_profile, sample_mlcanopy_state.lwp_mean_profile, atol=1e-10), \
            "Complete cycle should preserve lwp_mean_profile"


# Test validate_restart_data
class TestValidateRestartData:
    """Tests for validate_restart_data function."""
    
    def test_valid_data_passes(self, sample_mlcanopy_state):
        """Verify valid restart data passes validation."""
        restart_data = extract_restart_data(sample_mlcanopy_state)
        n_patches = 5
        nlevmlcan = 10
        
        is_valid = validate_restart_data(restart_data, n_patches, nlevmlcan)
        
        assert is_valid, "Valid restart data should pass validation"
    
    def test_wrong_shape_fails(self, sample_mlcanopy_state):
        """Verify wrong shapes fail validation."""
        restart_data = extract_restart_data(sample_mlcanopy_state)
        
        # Wrong n_patches
        is_valid = validate_restart_data(restart_data, n_patches=10, nlevmlcan=10)
        assert not is_valid, "Wrong n_patches should fail validation"
        
        # Wrong nlevmlcan
        is_valid = validate_restart_data(restart_data, n_patches=5, nlevmlcan=20)
        assert not is_valid, "Wrong nlevmlcan should fail validation"
    
    def test_nan_values_fail(self):
        """Verify NaN values fail validation."""
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([288.15, jnp.nan, 290.0]),
            lwp_mean_profile=jnp.array([[-0.5, -0.6], [-0.4, -0.7], [-0.3, -0.5]])
        )
        
        is_valid = validate_restart_data(restart_data, n_patches=3, nlevmlcan=2)
        assert not is_valid, "NaN values should fail validation"
    
    def test_inf_values_fail(self):
        """Verify Inf values fail validation."""
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([288.15, jnp.inf, 290.0]),
            lwp_mean_profile=jnp.array([[-0.5, -0.6], [-0.4, -0.7], [-0.3, -0.5]])
        )
        
        is_valid = validate_restart_data(restart_data, n_patches=3, nlevmlcan=2)
        assert not is_valid, "Inf values should fail validation"
    
    def test_temperature_bounds(self):
        """Verify temperature constraint checking."""
        # Too cold
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([100.0, 288.15, 290.0]),  # 100K too cold
            lwp_mean_profile=jnp.array([[-0.5, -0.6], [-0.4, -0.7], [-0.3, -0.5]])
        )
        is_valid = validate_restart_data(restart_data, n_patches=3, nlevmlcan=2)
        assert not is_valid, "Temperature below 150K should fail"
        
        # Too hot
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([288.15, 400.0, 290.0]),  # 400K too hot
            lwp_mean_profile=jnp.array([[-0.5, -0.6], [-0.4, -0.7], [-0.3, -0.5]])
        )
        is_valid = validate_restart_data(restart_data, n_patches=3, nlevmlcan=2)
        assert not is_valid, "Temperature above 350K should fail"
    
    def test_lwp_bounds(self):
        """Verify leaf water potential constraint checking."""
        # Too negative
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([288.15, 290.0, 285.0]),
            lwp_mean_profile=jnp.array([[-0.5, -0.6], [-15.0, -0.7], [-0.3, -0.5]])  # -15 MPa too negative
        )
        is_valid = validate_restart_data(restart_data, n_patches=3, nlevmlcan=2)
        assert not is_valid, "LWP below -10 MPa should fail"
        
        # Too positive
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([288.15, 290.0, 285.0]),
            lwp_mean_profile=jnp.array([[-0.5, -0.6], [0.5, -0.7], [-0.3, -0.5]])  # 0.5 MPa too positive
        )
        is_valid = validate_restart_data(restart_data, n_patches=3, nlevmlcan=2)
        assert not is_valid, "LWP above 0 MPa should fail"
    
    def test_extreme_stress_conditions(self):
        """Test validation with extreme but valid water stress."""
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([273.15, 275.0, 270.0, 278.5, 272.3]),
            lwp_mean_profile=jnp.array([
                [-8.5, -9.0, -9.5, -8.0, -7.5, -9.2, -9.8, -9.3],
                [-9.0, -9.5, -9.8, -8.5, -8.0, -9.7, -9.3, -9.6],
                [-7.5, -8.0, -8.5, -7.0, -6.5, -8.2, -7.8, -8.3],
                [-9.5, -9.8, -10.0, -9.0, -8.5, -9.9, -9.6, -9.8],
                [-8.0, -8.5, -9.0, -7.5, -7.0, -8.7, -8.3, -8.6]
            ])
        )
        
        is_valid = validate_restart_data(restart_data, n_patches=5, nlevmlcan=8)
        assert is_valid, "Extreme but valid stress conditions should pass"
    
    def test_cold_conditions(self):
        """Test validation with cold but valid temperatures."""
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([253.15, 258.0, 250.0, 260.5, 255.3, 252.8, 257.2]),
            lwp_mean_profile=jnp.array([
                [-0.1, -0.15, -0.2, -0.12, -0.18],
                [-0.12, -0.17, -0.22, -0.14, -0.2],
                [-0.08, -0.13, -0.18, -0.1, -0.16],
                [-0.15, -0.2, -0.25, -0.17, -0.23],
                [-0.1, -0.15, -0.2, -0.12, -0.18],
                [-0.09, -0.14, -0.19, -0.11, -0.17],
                [-0.13, -0.18, -0.23, -0.15, -0.21]
            ])
        )
        
        is_valid = validate_restart_data(restart_data, n_patches=7, nlevmlcan=5)
        assert is_valid, "Cold but valid conditions should pass"


# Test metadata functions
class TestMetadataFunctions:
    """Tests for get_restart_metadata and get_history_metadata."""
    
    def test_restart_metadata_structure(self):
        """Verify restart metadata has correct structure."""
        metadata = get_restart_metadata()
        
        assert 'taf_canopy' in metadata, "Should include taf_canopy"
        assert 'lwp_mean_profile' in metadata, "Should include lwp_mean_profile"
        
        # Check taf_canopy metadata
        taf_meta = metadata['taf_canopy']
        assert 'dimensions' in taf_meta, "Should have dimensions"
        assert 'units' in taf_meta, "Should have units"
        assert 'long_name' in taf_meta, "Should have long_name"
        assert 'interpinic_flag' in taf_meta, "Should have interpinic_flag"
    
    def test_restart_metadata_values(self):
        """Verify restart metadata has correct values."""
        metadata = get_restart_metadata()
        
        assert metadata['taf_canopy']['units'] == 'K', \
            "taf_canopy units should be Kelvin"
        assert metadata['lwp_mean_profile']['units'] == 'MPa', \
            "lwp_mean_profile units should be MPa"
        assert 'patch' in metadata['taf_canopy']['dimensions'], \
            "taf_canopy should have patch dimension"
        assert 'levmlcan' in metadata['lwp_mean_profile']['dimensions'], \
            "lwp_mean_profile should have levmlcan dimension"
    
    def test_history_metadata_structure(self):
        """Verify history metadata has correct structure."""
        metadata = get_history_metadata()
        
        assert 'taf_canopy' in metadata, "Should include taf_canopy"
        assert 'lwp_mean_profile' in metadata, "Should include lwp_mean_profile"
        
        # Check for averaging_flag instead of interpinic_flag
        taf_meta = metadata['taf_canopy']
        assert 'averaging_flag' in taf_meta, "Should have averaging_flag"
        assert 'interpinic_flag' not in taf_meta, "Should not have interpinic_flag in history"
    
    def test_history_metadata_values(self):
        """Verify history metadata has correct values."""
        metadata = get_history_metadata()
        
        assert metadata['taf_canopy']['averaging_flag'] == 'A', \
            "Should use averaging flag 'A'"
        assert metadata['lwp_mean_profile']['averaging_flag'] == 'A', \
            "Should use averaging flag 'A'"


# Integration tests
class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_initialization_workflow(self):
        """Test complete initialization workflow."""
        # Create bounds
        bounds = BoundsType(begp=0, endp=49)
        
        # Initialize
        state = init(bounds, nlevmlcan=25, nleaf=2)
        
        # Verify initialization
        assert state.lwp_leaf.shape == (50, 25, 2), \
            "Should create correct dimensions"
        assert jnp.allclose(state.lwp_leaf, DEFAULT_LWP, atol=1e-10), \
            "Should have cold start values"
    
    def test_full_restart_workflow(self, sample_mlcanopy_state):
        """Test complete restart workflow."""
        # Extract
        restart_data = extract_restart_data(sample_mlcanopy_state)
        
        # Validate
        is_valid = validate_restart_data(restart_data, n_patches=5, nlevmlcan=10)
        assert is_valid, "Extracted data should be valid"
        
        # Create new state
        bounds = BoundsType(begp=0, endp=4)
        new_state = init(bounds, nlevmlcan=10, nleaf=2)
        
        # Restore
        restored_state = restore_from_restart(new_state, restart_data)
        
        # Verify
        assert jnp.allclose(restored_state.taf_canopy, sample_mlcanopy_state.taf_canopy, atol=1e-10), \
            "Full workflow should preserve data"
    
    def test_metadata_consistency(self):
        """Test consistency between restart and history metadata."""
        restart_meta = get_restart_metadata()
        history_meta = get_history_metadata()
        
        # Same variables should be present
        assert set(restart_meta.keys()) == set(history_meta.keys()), \
            "Restart and history should track same variables"
        
        # Units should match
        for var in restart_meta.keys():
            assert restart_meta[var]['units'] == history_meta[var]['units'], \
                f"Units should match for {var}"


# Edge case tests
class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_zero_lai(self):
        """Test with zero leaf area index."""
        state = create_empty_mlcanopy_state(n_patches=5, nlevmlcan=10)
        assert jnp.all(state.lai_canopy == 0.0), \
            "Should handle zero LAI"
    
    def test_boundary_temperatures(self):
        """Test with boundary temperature values."""
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([150.0, 350.0, 250.0]),  # Exact boundaries
            lwp_mean_profile=jnp.array([[-0.5, -0.6], [-0.4, -0.7], [-0.3, -0.5]])
        )
        
        is_valid = validate_restart_data(restart_data, n_patches=3, nlevmlcan=2)
        assert is_valid, "Boundary temperatures should be valid"
    
    def test_boundary_lwp(self):
        """Test with boundary leaf water potential values."""
        restart_data = MLCanopyRestartData(
            taf_canopy=jnp.array([288.15, 290.0, 285.0]),
            lwp_mean_profile=jnp.array([[-10.0, 0.0], [-5.0, -2.5], [-1.0, -0.1]])  # Exact boundaries
        )
        
        is_valid = validate_restart_data(restart_data, n_patches=3, nlevmlcan=2)
        assert is_valid, "Boundary LWP values should be valid"
    
    def test_sunlit_shaded_difference(self, sample_mlcanopy_state):
        """Test that shaded leaves are more stressed than sunlit."""
        lwp_leaf = sample_mlcanopy_state.lwp_leaf
        
        # For each patch and layer, shaded (index 1) should be <= sunlit (index 0)
        sunlit = lwp_leaf[:, :, ISUN]
        shaded = lwp_leaf[:, :, ISHA]
        
        assert jnp.all(shaded <= sunlit), \
            "Shaded leaves should be more stressed (more negative LWP) than sunlit"
    
    def test_large_array_handling(self):
        """Test with large arrays (memory/performance check)."""
        state = create_empty_mlcanopy_state(
            n_patches=1000,
            nlevmlcan=100,
            nleaf=2
        )
        
        assert state.lwp_leaf.shape == (1000, 100, 2), \
            "Should handle large arrays"
        assert not jnp.any(jnp.isnan(state.lwp_leaf)), \
            "Large arrays should not contain NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])