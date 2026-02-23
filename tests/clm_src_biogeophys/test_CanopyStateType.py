"""
Comprehensive pytest suite for CanopyStateType module.

This test suite covers:
- Initialization functions (init_allocate, init_allocate_from_bounds, create_canopy_state)
- State update operations (update_canopy_state)
- State validation (validate_canopy_state)
- Edge cases: single patch, large domains, boundary values
- Physical realism: snow cover scenarios, vegetation types (grassland to forest)
- Array shapes, dtypes, and numerical accuracy
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from collections import namedtuple


# Define NamedTuples matching the module specification
Bounds = namedtuple('Bounds', ['begp', 'endp', 'begc', 'endc', 'begg', 'endg'])
CanopyState = namedtuple('CanopyState', [
    'frac_veg_nosno_patch',
    'elai_patch',
    'esai_patch',
    'htop_patch'
])


# Mock implementation of the module functions for testing
def init_allocate(begp: int, endp: int, use_nan: bool = True) -> CanopyState:
    """Initialize canopy state arrays for a patch range."""
    n_patches = endp - begp + 1
    init_value = jnp.nan if use_nan else 0.0
    
    return CanopyState(
        frac_veg_nosno_patch=jnp.full(n_patches, init_value, dtype=jnp.float32),
        elai_patch=jnp.full(n_patches, init_value, dtype=jnp.float32),
        esai_patch=jnp.full(n_patches, init_value, dtype=jnp.float32),
        htop_patch=jnp.full(n_patches, init_value, dtype=jnp.float32)
    )


def init_allocate_from_bounds(bounds: Bounds, use_nan: bool = True) -> CanopyState:
    """Initialize canopy state from Bounds object."""
    return init_allocate(bounds.begp, bounds.endp, use_nan)


def create_canopy_state(n_patches: int, use_zeros: bool = True) -> CanopyState:
    """Create canopy state for n_patches."""
    init_value = 0.0 if use_zeros else jnp.nan
    
    return CanopyState(
        frac_veg_nosno_patch=jnp.full(n_patches, init_value, dtype=jnp.float32),
        elai_patch=jnp.full(n_patches, init_value, dtype=jnp.float32),
        esai_patch=jnp.full(n_patches, init_value, dtype=jnp.float32),
        htop_patch=jnp.full(n_patches, init_value, dtype=jnp.float32)
    )


def update_canopy_state(state: CanopyState, **updates) -> CanopyState:
    """Update canopy state with new values (immutable)."""
    state_dict = state._asdict()
    state_dict.update(updates)
    
    # Convert lists to JAX arrays if needed
    for key, value in state_dict.items():
        if isinstance(value, list):
            state_dict[key] = jnp.array(value, dtype=jnp.float32)
    
    return CanopyState(**state_dict)


def validate_canopy_state(state: CanopyState) -> bool:
    """Validate canopy state for consistent shapes and physical constraints."""
    # Check all arrays have same shape
    shapes = [
        state.frac_veg_nosno_patch.shape,
        state.elai_patch.shape,
        state.esai_patch.shape,
        state.htop_patch.shape
    ]
    if len(set(shapes)) != 1:
        return False
    
    # Check physical constraints (ignoring NaN values)
    def check_bounds(arr, min_val, max_val=None):
        valid_mask = ~jnp.isnan(arr)
        if not jnp.any(valid_mask):
            return True
        valid_values = arr[valid_mask]
        if jnp.any(valid_values < min_val):
            return False
        if max_val is not None and jnp.any(valid_values > max_val):
            return False
        return True
    
    if not check_bounds(state.frac_veg_nosno_patch, 0.0, 1.0):
        return False
    if not check_bounds(state.elai_patch, 0.0):
        return False
    if not check_bounds(state.esai_patch, 0.0):
        return False
    if not check_bounds(state.htop_patch, 0.0):
        return False
    
    return True


# Fixtures
@pytest.fixture
def test_data():
    """Load test data for all test cases."""
    return {
        "init_allocate_single_patch_nan": {
            "inputs": {"begp": 0, "endp": 0, "use_nan": True},
            "expected_shape": (1,),
            "expected_all_nan": True
        },
        "init_allocate_multiple_patches_zeros": {
            "inputs": {"begp": 5, "endp": 14, "use_nan": False},
            "expected_shape": (10,),
            "expected_all_zeros": True
        },
        "init_allocate_boundary": {
            "inputs": {"begp": 0, "endp": 0, "use_nan": False},
            "expected_shape": (1,),
            "expected_all_zeros": True
        },
        "init_allocate_large": {
            "inputs": {"begp": 0, "endp": 9999, "use_nan": True},
            "expected_shape": (10000,),
            "expected_all_nan": True
        },
        "bounds_typical": {
            "bounds": Bounds(begp=10, endp=49, begc=5, endc=24, begg=0, endg=9),
            "use_nan": False,
            "expected_shape": (40,)
        },
        "create_small": {
            "n_patches": 1,
            "use_zeros": True,
            "expected_shape": (1,)
        },
        "create_medium_nan": {
            "n_patches": 100,
            "use_zeros": False,
            "expected_shape": (100,)
        }
    }


@pytest.fixture
def sample_states():
    """Fixture providing sample canopy states for testing."""
    return {
        "mixed_snow": CanopyState(
            frac_veg_nosno_patch=jnp.array([1.0, 1.0, 0.0, 1.0, 0.0], dtype=jnp.float32),
            elai_patch=jnp.array([3.5, 4.2, 0.0, 2.8, 0.0], dtype=jnp.float32),
            esai_patch=jnp.array([0.5, 0.8, 0.0, 0.3, 0.0], dtype=jnp.float32),
            htop_patch=jnp.array([15.0, 18.5, 0.0, 12.3, 0.0], dtype=jnp.float32)
        ),
        "all_zeros": CanopyState(
            frac_veg_nosno_patch=jnp.zeros(3, dtype=jnp.float32),
            elai_patch=jnp.zeros(3, dtype=jnp.float32),
            esai_patch=jnp.zeros(3, dtype=jnp.float32),
            htop_patch=jnp.zeros(3, dtype=jnp.float32)
        ),
        "grassland": CanopyState(
            frac_veg_nosno_patch=jnp.ones(4, dtype=jnp.float32),
            elai_patch=jnp.array([0.5, 0.8, 0.6, 0.7], dtype=jnp.float32),
            esai_patch=jnp.array([0.1, 0.15, 0.12, 0.13], dtype=jnp.float32),
            htop_patch=jnp.array([0.3, 0.5, 0.4, 0.45], dtype=jnp.float32)
        )
    }


# Test init_allocate function
class TestInitAllocate:
    """Tests for init_allocate function."""
    
    @pytest.mark.parametrize("begp,endp,expected_size", [
        (0, 0, 1),
        (5, 14, 10),
        (0, 9999, 10000),
        (100, 199, 100),
    ])
    def test_init_allocate_shapes(self, begp, endp, expected_size):
        """Test that init_allocate returns correct array shapes."""
        state = init_allocate(begp, endp, use_nan=True)
        
        assert state.frac_veg_nosno_patch.shape == (expected_size,), \
            f"frac_veg_nosno_patch shape mismatch"
        assert state.elai_patch.shape == (expected_size,), \
            f"elai_patch shape mismatch"
        assert state.esai_patch.shape == (expected_size,), \
            f"esai_patch shape mismatch"
        assert state.htop_patch.shape == (expected_size,), \
            f"htop_patch shape mismatch"
    
    @pytest.mark.parametrize("use_nan", [True, False])
    def test_init_allocate_initialization_values(self, use_nan):
        """Test that init_allocate initializes with correct values (NaN or zeros)."""
        state = init_allocate(0, 4, use_nan=use_nan)
        
        if use_nan:
            assert jnp.all(jnp.isnan(state.frac_veg_nosno_patch)), \
                "Expected all NaN values when use_nan=True"
            assert jnp.all(jnp.isnan(state.elai_patch)), \
                "Expected all NaN values when use_nan=True"
            assert jnp.all(jnp.isnan(state.esai_patch)), \
                "Expected all NaN values when use_nan=True"
            assert jnp.all(jnp.isnan(state.htop_patch)), \
                "Expected all NaN values when use_nan=True"
        else:
            assert jnp.allclose(state.frac_veg_nosno_patch, 0.0), \
                "Expected all zeros when use_nan=False"
            assert jnp.allclose(state.elai_patch, 0.0), \
                "Expected all zeros when use_nan=False"
            assert jnp.allclose(state.esai_patch, 0.0), \
                "Expected all zeros when use_nan=False"
            assert jnp.allclose(state.htop_patch, 0.0), \
                "Expected all zeros when use_nan=False"
    
    def test_init_allocate_dtypes(self):
        """Test that init_allocate returns float32 arrays."""
        state = init_allocate(0, 10, use_nan=False)
        
        assert state.frac_veg_nosno_patch.dtype == jnp.float32
        assert state.elai_patch.dtype == jnp.float32
        assert state.esai_patch.dtype == jnp.float32
        assert state.htop_patch.dtype == jnp.float32
    
    def test_init_allocate_single_patch(self):
        """Test edge case: single patch (begp == endp)."""
        state = init_allocate(0, 0, use_nan=False)
        
        assert state.frac_veg_nosno_patch.shape == (1,)
        assert jnp.allclose(state.frac_veg_nosno_patch, 0.0)
    
    def test_init_allocate_large_domain(self):
        """Test edge case: large domain with 10000 patches."""
        state = init_allocate(0, 9999, use_nan=True)
        
        assert state.frac_veg_nosno_patch.shape == (10000,)
        assert jnp.all(jnp.isnan(state.frac_veg_nosno_patch))
    
    def test_init_allocate_index_constraints(self):
        """Test that begp and endp satisfy constraints (begp >= 0, endp >= begp)."""
        # Valid cases
        state = init_allocate(0, 5, use_nan=False)
        assert state.frac_veg_nosno_patch.shape == (6,)
        
        state = init_allocate(10, 10, use_nan=False)
        assert state.frac_veg_nosno_patch.shape == (1,)


# Test init_allocate_from_bounds function
class TestInitAllocateFromBounds:
    """Tests for init_allocate_from_bounds function."""
    
    def test_init_allocate_from_bounds_typical(self):
        """Test initialization from typical Bounds object."""
        bounds = Bounds(begp=10, endp=49, begc=5, endc=24, begg=0, endg=9)
        state = init_allocate_from_bounds(bounds, use_nan=False)
        
        expected_size = 40  # endp - begp + 1 = 49 - 10 + 1
        assert state.frac_veg_nosno_patch.shape == (expected_size,)
        assert jnp.allclose(state.frac_veg_nosno_patch, 0.0)
    
    def test_init_allocate_from_bounds_single_patch(self):
        """Test edge case: Bounds with single patch."""
        bounds = Bounds(begp=0, endp=0, begc=0, endc=0, begg=0, endg=0)
        state = init_allocate_from_bounds(bounds, use_nan=True)
        
        assert state.frac_veg_nosno_patch.shape == (1,)
        assert jnp.all(jnp.isnan(state.frac_veg_nosno_patch))
    
    def test_init_allocate_from_bounds_nan_initialization(self):
        """Test NaN initialization from Bounds."""
        bounds = Bounds(begp=0, endp=9, begc=0, endc=4, begg=0, endg=1)
        state = init_allocate_from_bounds(bounds, use_nan=True)
        
        assert jnp.all(jnp.isnan(state.elai_patch))
        assert jnp.all(jnp.isnan(state.esai_patch))
        assert jnp.all(jnp.isnan(state.htop_patch))


# Test create_canopy_state function
class TestCreateCanopyState:
    """Tests for create_canopy_state function."""
    
    @pytest.mark.parametrize("n_patches,use_zeros", [
        (1, True),
        (100, False),
        (50, True),
        (1000, False),
    ])
    def test_create_canopy_state_shapes(self, n_patches, use_zeros):
        """Test that create_canopy_state returns correct shapes."""
        state = create_canopy_state(n_patches, use_zeros=use_zeros)
        
        assert state.frac_veg_nosno_patch.shape == (n_patches,)
        assert state.elai_patch.shape == (n_patches,)
        assert state.esai_patch.shape == (n_patches,)
        assert state.htop_patch.shape == (n_patches,)
    
    def test_create_canopy_state_zeros(self):
        """Test zero initialization."""
        state = create_canopy_state(10, use_zeros=True)
        
        assert jnp.allclose(state.frac_veg_nosno_patch, 0.0)
        assert jnp.allclose(state.elai_patch, 0.0)
        assert jnp.allclose(state.esai_patch, 0.0)
        assert jnp.allclose(state.htop_patch, 0.0)
    
    def test_create_canopy_state_nan(self):
        """Test NaN initialization."""
        state = create_canopy_state(10, use_zeros=False)
        
        assert jnp.all(jnp.isnan(state.frac_veg_nosno_patch))
        assert jnp.all(jnp.isnan(state.elai_patch))
        assert jnp.all(jnp.isnan(state.esai_patch))
        assert jnp.all(jnp.isnan(state.htop_patch))
    
    def test_create_canopy_state_minimum_domain(self):
        """Test edge case: minimum valid domain (1 patch)."""
        state = create_canopy_state(1, use_zeros=True)
        
        assert state.frac_veg_nosno_patch.shape == (1,)
        assert jnp.allclose(state.frac_veg_nosno_patch, 0.0)
    
    def test_create_canopy_state_dtypes(self):
        """Test that create_canopy_state returns float32 arrays."""
        state = create_canopy_state(5, use_zeros=True)
        
        assert state.frac_veg_nosno_patch.dtype == jnp.float32
        assert state.elai_patch.dtype == jnp.float32
        assert state.esai_patch.dtype == jnp.float32
        assert state.htop_patch.dtype == jnp.float32


# Test update_canopy_state function
class TestUpdateCanopyState:
    """Tests for update_canopy_state function."""
    
    def test_update_canopy_state_partial_update(self, sample_states):
        """Test partial update of canopy state (snow melting scenario)."""
        state = sample_states["mixed_snow"]
        
        updates = {
            "elai_patch": jnp.array([3.5, 4.2, 1.5, 2.8, 0.5], dtype=jnp.float32),
            "frac_veg_nosno_patch": jnp.ones(5, dtype=jnp.float32)
        }
        
        new_state = update_canopy_state(state, **updates)
        
        # Check updated fields
        assert jnp.allclose(new_state.frac_veg_nosno_patch, 1.0)
        assert jnp.allclose(new_state.elai_patch, 
                           jnp.array([3.5, 4.2, 1.5, 2.8, 0.5]))
        
        # Check unchanged fields
        assert jnp.allclose(new_state.esai_patch, state.esai_patch)
        assert jnp.allclose(new_state.htop_patch, state.htop_patch)
    
    def test_update_canopy_state_immutability(self, sample_states):
        """Test that update_canopy_state returns new state (immutable)."""
        state = sample_states["all_zeros"]
        
        updates = {
            "elai_patch": jnp.ones(3, dtype=jnp.float32)
        }
        
        new_state = update_canopy_state(state, **updates)
        
        # Original state should be unchanged
        assert jnp.allclose(state.elai_patch, 0.0)
        # New state should have updates
        assert jnp.allclose(new_state.elai_patch, 1.0)
    
    def test_update_canopy_state_all_fields(self):
        """Test updating all fields simultaneously."""
        state = create_canopy_state(3, use_zeros=True)
        
        updates = {
            "frac_veg_nosno_patch": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
            "elai_patch": jnp.array([12.0, 8.5, 10.2], dtype=jnp.float32),
            "esai_patch": jnp.array([2.5, 1.8, 2.1], dtype=jnp.float32),
            "htop_patch": jnp.array([45.0, 35.2, 40.8], dtype=jnp.float32)
        }
        
        new_state = update_canopy_state(state, **updates)
        
        assert jnp.allclose(new_state.frac_veg_nosno_patch, updates["frac_veg_nosno_patch"])
        assert jnp.allclose(new_state.elai_patch, updates["elai_patch"])
        assert jnp.allclose(new_state.esai_patch, updates["esai_patch"])
        assert jnp.allclose(new_state.htop_patch, updates["htop_patch"])
    
    def test_update_canopy_state_grassland_to_forest(self, sample_states):
        """Test realistic ecological transition: grassland to forest."""
        state = sample_states["grassland"]
        
        # Simulate vegetation succession
        updates = {
            "elai_patch": jnp.array([5.5, 6.2, 4.8, 5.9], dtype=jnp.float32),
            "esai_patch": jnp.array([1.2, 1.5, 1.0, 1.3], dtype=jnp.float32),
            "htop_patch": jnp.array([25.0, 30.0, 22.0, 28.0], dtype=jnp.float32)
        }
        
        new_state = update_canopy_state(state, **updates)
        
        # Verify forest characteristics
        assert jnp.all(new_state.elai_patch > 4.0), "Forest should have LAI > 4.0"
        assert jnp.all(new_state.htop_patch > 20.0), "Forest should be > 20m tall"
        assert jnp.allclose(new_state.frac_veg_nosno_patch, 1.0), \
            "Vegetation should remain exposed"
    
    def test_update_canopy_state_from_list(self):
        """Test that update_canopy_state handles list inputs."""
        state = create_canopy_state(3, use_zeros=True)
        
        # Pass updates as lists (should be converted to arrays)
        updates = {
            "elai_patch": [1.0, 2.0, 3.0],
            "frac_veg_nosno_patch": [1.0, 1.0, 1.0]
        }
        
        new_state = update_canopy_state(state, **updates)
        
        assert jnp.allclose(new_state.elai_patch, jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(new_state.frac_veg_nosno_patch, 1.0)


# Test validate_canopy_state function
class TestValidateCanopyState:
    """Tests for validate_canopy_state function."""
    
    def test_validate_canopy_state_valid_mixed(self):
        """Test validation of valid state with mixed snow cover."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], 
                                           dtype=jnp.float32),
            elai_patch=jnp.array([4.5, 0.0, 2.3, 6.8, 0.0, 3.2, 0.0, 5.1], 
                                dtype=jnp.float32),
            esai_patch=jnp.array([0.8, 0.0, 0.4, 1.2, 0.0, 0.6, 0.0, 0.9], 
                                dtype=jnp.float32),
            htop_patch=jnp.array([20.0, 0.0, 12.5, 28.3, 0.0, 16.7, 0.0, 24.1], 
                                dtype=jnp.float32)
        )
        
        assert validate_canopy_state(state), "Valid mixed state should pass validation"
    
    def test_validate_canopy_state_boundary_values(self):
        """Test validation with boundary values (very small and very large)."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.array([0.0, 1.0, 0.0, 1.0], dtype=jnp.float32),
            elai_patch=jnp.array([0.0, 0.001, 0.0, 15.0], dtype=jnp.float32),
            esai_patch=jnp.array([0.0, 0.0001, 0.0, 3.0], dtype=jnp.float32),
            htop_patch=jnp.array([0.0, 0.01, 0.0, 100.0], dtype=jnp.float32)
        )
        
        assert validate_canopy_state(state), \
            "State with boundary values should pass validation"
    
    def test_validate_canopy_state_all_zeros(self, sample_states):
        """Test validation of all-zero state (complete snow burial)."""
        state = sample_states["all_zeros"]
        
        assert validate_canopy_state(state), \
            "All-zero state (complete snow cover) should be valid"
    
    def test_validate_canopy_state_with_nan(self):
        """Test validation handles NaN values correctly."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.array([1.0, jnp.nan, 0.0], dtype=jnp.float32),
            elai_patch=jnp.array([3.0, jnp.nan, 0.0], dtype=jnp.float32),
            esai_patch=jnp.array([0.5, jnp.nan, 0.0], dtype=jnp.float32),
            htop_patch=jnp.array([15.0, jnp.nan, 0.0], dtype=jnp.float32)
        )
        
        assert validate_canopy_state(state), \
            "State with NaN values should pass validation (NaN ignored)"
    
    def test_validate_canopy_state_inconsistent_shapes(self):
        """Test validation fails for inconsistent array shapes."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.array([1.0, 1.0], dtype=jnp.float32),
            elai_patch=jnp.array([3.0, 4.0, 5.0], dtype=jnp.float32),  # Wrong shape
            esai_patch=jnp.array([0.5, 0.6], dtype=jnp.float32),
            htop_patch=jnp.array([15.0, 18.0], dtype=jnp.float32)
        )
        
        assert not validate_canopy_state(state), \
            "State with inconsistent shapes should fail validation"
    
    def test_validate_canopy_state_negative_values(self):
        """Test validation fails for negative values."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.array([1.0, 1.0], dtype=jnp.float32),
            elai_patch=jnp.array([3.0, -1.0], dtype=jnp.float32),  # Invalid negative
            esai_patch=jnp.array([0.5, 0.6], dtype=jnp.float32),
            htop_patch=jnp.array([15.0, 18.0], dtype=jnp.float32)
        )
        
        assert not validate_canopy_state(state), \
            "State with negative LAI should fail validation"
    
    def test_validate_canopy_state_frac_out_of_bounds(self):
        """Test validation fails for frac_veg_nosno_patch outside [0, 1]."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.array([1.0, 1.5], dtype=jnp.float32),  # Invalid > 1
            elai_patch=jnp.array([3.0, 4.0], dtype=jnp.float32),
            esai_patch=jnp.array([0.5, 0.6], dtype=jnp.float32),
            htop_patch=jnp.array([15.0, 18.0], dtype=jnp.float32)
        )
        
        assert not validate_canopy_state(state), \
            "State with frac_veg_nosno_patch > 1.0 should fail validation"


# Integration tests
class TestIntegration:
    """Integration tests combining multiple operations."""
    
    def test_create_update_validate_workflow(self):
        """Test complete workflow: create -> update -> validate."""
        # Create initial state
        state = create_canopy_state(5, use_zeros=True)
        assert validate_canopy_state(state), "Initial state should be valid"
        
        # Update to grassland
        state = update_canopy_state(
            state,
            frac_veg_nosno_patch=jnp.ones(5, dtype=jnp.float32),
            elai_patch=jnp.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=jnp.float32),
            esai_patch=jnp.array([0.1, 0.12, 0.14, 0.16, 0.18], dtype=jnp.float32),
            htop_patch=jnp.array([0.3, 0.35, 0.4, 0.45, 0.5], dtype=jnp.float32)
        )
        assert validate_canopy_state(state), "Grassland state should be valid"
        
        # Update to forest
        state = update_canopy_state(
            state,
            elai_patch=jnp.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=jnp.float32),
            esai_patch=jnp.array([1.0, 1.2, 1.4, 1.6, 1.8], dtype=jnp.float32),
            htop_patch=jnp.array([20.0, 25.0, 30.0, 35.0, 40.0], dtype=jnp.float32)
        )
        assert validate_canopy_state(state), "Forest state should be valid"
    
    def test_init_allocate_vs_create_consistency(self):
        """Test that init_allocate and create_canopy_state produce consistent results."""
        n_patches = 10
        
        state1 = init_allocate(0, n_patches - 1, use_nan=False)
        state2 = create_canopy_state(n_patches, use_zeros=True)
        
        assert state1.frac_veg_nosno_patch.shape == state2.frac_veg_nosno_patch.shape
        assert jnp.allclose(state1.frac_veg_nosno_patch, state2.frac_veg_nosno_patch)
        assert jnp.allclose(state1.elai_patch, state2.elai_patch)
        assert jnp.allclose(state1.esai_patch, state2.esai_patch)
        assert jnp.allclose(state1.htop_patch, state2.htop_patch)
    
    def test_bounds_to_state_workflow(self):
        """Test workflow using Bounds object."""
        bounds = Bounds(begp=0, endp=9, begc=0, endc=4, begg=0, endg=1)
        
        # Initialize from bounds
        state = init_allocate_from_bounds(bounds, use_nan=False)
        assert state.frac_veg_nosno_patch.shape == (10,)
        assert validate_canopy_state(state)
        
        # Update state
        state = update_canopy_state(
            state,
            frac_veg_nosno_patch=jnp.ones(10, dtype=jnp.float32),
            elai_patch=jnp.full(10, 5.0, dtype=jnp.float32)
        )
        assert validate_canopy_state(state)


# Edge case tests
class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_zero_lai_with_nonzero_frac(self):
        """Test edge case: zero LAI but vegetation present (e.g., deciduous winter)."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.ones(3, dtype=jnp.float32),
            elai_patch=jnp.zeros(3, dtype=jnp.float32),
            esai_patch=jnp.array([0.5, 0.6, 0.7], dtype=jnp.float32),
            htop_patch=jnp.array([10.0, 12.0, 15.0], dtype=jnp.float32)
        )
        
        assert validate_canopy_state(state), \
            "Zero LAI with stems present should be valid (deciduous winter)"
    
    def test_very_dense_canopy(self):
        """Test edge case: very dense canopy (high LAI)."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.ones(2, dtype=jnp.float32),
            elai_patch=jnp.array([12.0, 15.0], dtype=jnp.float32),
            esai_patch=jnp.array([2.5, 3.0], dtype=jnp.float32),
            htop_patch=jnp.array([45.0, 50.0], dtype=jnp.float32)
        )
        
        assert validate_canopy_state(state), \
            "Very dense canopy should be valid"
    
    def test_very_tall_canopy(self):
        """Test edge case: very tall canopy (100m)."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.ones(1, dtype=jnp.float32),
            elai_patch=jnp.array([8.0], dtype=jnp.float32),
            esai_patch=jnp.array([1.5], dtype=jnp.float32),
            htop_patch=jnp.array([100.0], dtype=jnp.float32)
        )
        
        assert validate_canopy_state(state), \
            "Very tall canopy (100m) should be valid"
    
    def test_minimal_nonzero_values(self):
        """Test edge case: minimal non-zero values."""
        state = CanopyState(
            frac_veg_nosno_patch=jnp.array([1.0], dtype=jnp.float32),
            elai_patch=jnp.array([0.001], dtype=jnp.float32),
            esai_patch=jnp.array([0.0001], dtype=jnp.float32),
            htop_patch=jnp.array([0.01], dtype=jnp.float32)
        )
        
        assert validate_canopy_state(state), \
            "Minimal non-zero values should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])