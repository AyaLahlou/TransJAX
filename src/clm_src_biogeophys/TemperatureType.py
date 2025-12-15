"""
Temperature State Variables and Initialization.

Translated from CTSM's TemperatureType.F90 (lines 1-68)

This module defines the temperature state variables used throughout CTSM,
including soil/snow temperatures and atmospheric reference temperatures.
It provides initialization routines for setting up temperature arrays
with proper dimensions based on domain decomposition.

Key variables:
    - t_soisno: Soil and snow layer temperatures [K]
    - t_a10: 10-day running mean 2m air temperature [K]
    - t_ref2m: 2m reference air temperature [K]

Key equations:
    None - this is a state variable container module

Physics:
    Temperature state is fundamental to:
    - Soil thermal dynamics
    - Snow thermodynamics
    - Plant phenology (via t_a10)
    - Surface energy balance
    - Biogeochemical processes

Vertical indexing:
    Fortran uses -nlevsno+1:nlevgrnd indexing
    JAX uses 0-based indexing with shape (nlevsno + nlevgrnd)
    
    Layer mapping:
        Fortran index -nlevsno+1 -> JAX index 0 (top snow)
        Fortran index 0 -> JAX index nlevsno-1 (bottom snow/top soil interface)
        Fortran index 1 -> JAX index nlevsno (top soil)
        Fortran index nlevgrnd -> JAX index nlevsno+nlevgrnd-1 (bottom soil)
"""

from typing import NamedTuple
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================


class BoundsType(NamedTuple):
    """Domain decomposition bounds.
    
    Defines the index ranges for patches, columns, and grid cells
    in the current processor's subdomain.
    
    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
        begc: Beginning column index
        endc: Ending column index
        begg: Beginning grid cell index
        endg: Ending grid cell index
    """
    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


class TemperatureState(NamedTuple):
    """Temperature state variables for CTSM.
    
    Fortran source: TemperatureType.F90, lines 20-24
    
    This corresponds to the Fortran type temperature_type, containing
    temperature fields at column and patch levels.
    
    Attributes:
        t_soisno_col: Soil and snow layer temperatures [K]
                      Shape: [n_columns, n_levtot] where n_levtot = nlevsno + nlevgrnd
                      Indexing: layer 0 is top snow layer (if present), 
                               layer nlevsno is top soil layer
        t_a10_patch: 10-day running mean of 2m air temperature [K]
                     Shape: [n_patches]
                     Used for phenology and acclimation calculations
        t_ref2m_patch: 2m height surface air temperature [K]
                       Shape: [n_patches]
                       Diagnostic output and forcing for biogeochemistry
    
    Note:
        In Fortran, t_soisno_col is indexed as (-nlevsno+1:nlevgrnd).
        In JAX, we use 0-based indexing with shape (nlevsno + nlevgrnd).
    """
    
    t_soisno_col: jnp.ndarray  # [n_columns, n_levtot] in K
    t_a10_patch: jnp.ndarray   # [n_patches] in K
    t_ref2m_patch: jnp.ndarray # [n_patches] in K


# =============================================================================
# Constants
# =============================================================================

# Default initialization temperature [K]
# Corresponds to 0°C, a reasonable starting point for most simulations
DEFAULT_INIT_TEMP = 273.15

# Typical vertical layer counts (can be overridden)
DEFAULT_NLEVSNO = 5   # Maximum number of snow layers
DEFAULT_NLEVGRND = 15 # Number of soil layers


# =============================================================================
# Initialization Functions
# =============================================================================


def init_allocate_temperature(
    bounds: BoundsType,
    nlevsno: int = DEFAULT_NLEVSNO,
    nlevgrnd: int = DEFAULT_NLEVGRND,
) -> TemperatureState:
    """Initialize temperature state arrays with memory allocation.
    
    Allocates and initializes all temperature state variables with NaN values.
    The soil/snow temperature array spans from the top snow layer to the bottom
    soil layer. NaN initialization helps detect uninitialized data during
    debugging.
    
    Fortran source: TemperatureType.F90, lines 47-68
    
    Args:
        bounds: Spatial bounds containing patch and column indices
            - begp, endp: Patch index bounds
            - begc, endc: Column index bounds
            - begg, endg: Grid cell index bounds
        nlevsno: Maximum number of snow layers (typically 5)
        nlevgrnd: Number of ground (soil) layers (typically 15)
        
    Returns:
        Initialized temperature state with NaN-filled arrays
        
    Note:
        Arrays are initialized with NaN to help detect uninitialized values
        during debugging. The vertical dimension for t_soisno_col ranges from
        -nlevsno+1 (top snow) to nlevgrnd (bottom soil) in Fortran indexing,
        which becomes 0 to (nlevsno + nlevgrnd - 1) in JAX.
        
    Example:
        >>> bounds = BoundsType(begp=0, endp=99, begc=0, endc=49, begg=0, endg=24)
        >>> temp_state = init_allocate_temperature(bounds, nlevsno=5, nlevgrnd=15)
        >>> temp_state.t_soisno_col.shape
        (50, 20)  # 50 columns, 20 vertical layers (5 snow + 15 soil)
    """
    # Extract bounds (Fortran lines 61-62)
    begp = bounds.begp
    endp = bounds.endp
    begc = bounds.begc
    endc = bounds.endc
    
    # Calculate array sizes
    n_patches = endp - begp + 1
    n_columns = endc - begc + 1
    n_vertical_layers = nlevsno + nlevgrnd  # -nlevsno+1 to nlevgrnd inclusive
    
    # Allocate and initialize arrays with NaN (Fortran lines 64-66)
    # Using float32 for memory efficiency while maintaining sufficient precision
    t_soisno_col = jnp.full(
        (n_columns, n_vertical_layers), 
        jnp.nan, 
        dtype=jnp.float32
    )
    t_a10_patch = jnp.full(
        (n_patches,), 
        jnp.nan, 
        dtype=jnp.float32
    )
    t_ref2m_patch = jnp.full(
        (n_patches,), 
        jnp.nan, 
        dtype=jnp.float32
    )
    
    return TemperatureState(
        t_soisno_col=t_soisno_col,
        t_a10_patch=t_a10_patch,
        t_ref2m_patch=t_ref2m_patch,
    )


def init_temperature(
    bounds: BoundsType,
    nlevsno: int = DEFAULT_NLEVSNO,
    nlevgrnd: int = DEFAULT_NLEVGRND,
) -> TemperatureState:
    """Initialize temperature state variables (high-level interface).
    
    This function provides a high-level interface for initializing the
    temperature state. It delegates to init_allocate_temperature to perform
    the actual memory allocation and initialization.
    
    Fortran source: TemperatureType.F90, lines 37-44
    
    Args:
        bounds: Domain bounds defining array dimensions
            - begp, endp: Patch index bounds
            - begc, endc: Column index bounds
            - begg, endg: Grid cell index bounds
        nlevsno: Maximum number of snow layers (typically 5)
        nlevgrnd: Number of ground (soil) layers (typically 15)
            
    Returns:
        Initialized TemperatureState with allocated arrays
        
    Note:
        This is a wrapper that delegates to init_allocate_temperature.
        In the Fortran code, this calls this%InitAllocate(bounds).
        In JAX, we create immutable NamedTuples rather than allocating
        mutable arrays.
        
    Example:
        >>> bounds = BoundsType(begp=0, endp=99, begc=0, endc=49, begg=0, endg=24)
        >>> temp_state = init_temperature(bounds)
        >>> jnp.all(jnp.isnan(temp_state.t_a10_patch))
        True
    """
    # Delegate to the allocation routine
    # In the Fortran code, this calls this%InitAllocate(bounds)
    return init_allocate_temperature(bounds, nlevsno, nlevgrnd)


def init_temperature_state(
    n_columns: int,
    n_patches: int,
    n_levtot: int,
    initial_temp: float = DEFAULT_INIT_TEMP,
) -> TemperatureState:
    """Initialize temperature state with specified values (alternative interface).
    
    Creates initial temperature state with all temperatures set to a
    reference value (default is 273.15 K = 0°C). This is an alternative
    initialization method that directly specifies array dimensions and
    initial values rather than using bounds.
    
    Fortran source: TemperatureType.F90, lines 26-27 (Init, InitAllocate procedures)
    
    Args:
        n_columns: Number of columns in domain
        n_patches: Number of patches in domain
        n_levtot: Total number of soil + snow layers (nlevsno + nlevgrnd)
        initial_temp: Initial temperature for all fields [K]
                     Default is 273.15 K (0°C)
    
    Returns:
        Initialized TemperatureState with all temperatures set to initial_temp
        
    Note:
        This function is useful for testing or when you want to initialize
        with specific temperature values rather than NaN. For production
        runs, prefer init_temperature() which uses NaN to detect uninitialized
        values.
        
    Example:
        >>> temp_state = init_temperature_state(50, 100, 20, initial_temp=280.0)
        >>> temp_state.t_a10_patch[0]
        280.0
        >>> temp_state.t_soisno_col.shape
        (50, 20)
    """
    return TemperatureState(
        t_soisno_col=jnp.full(
            (n_columns, n_levtot), 
            initial_temp, 
            dtype=jnp.float32
        ),
        t_a10_patch=jnp.full(
            n_patches, 
            initial_temp, 
            dtype=jnp.float32
        ),
        t_ref2m_patch=jnp.full(
            n_patches, 
            initial_temp, 
            dtype=jnp.float32
        ),
    )


# =============================================================================
# Utility Functions
# =============================================================================


def get_soil_temperature(
    temp_state: TemperatureState,
    column_idx: int,
    nlevsno: int = DEFAULT_NLEVSNO,
) -> jnp.ndarray:
    """Extract soil-only temperatures from combined soil/snow array.
    
    Args:
        temp_state: Temperature state containing t_soisno_col
        column_idx: Column index to extract
        nlevsno: Number of snow layers (to skip)
        
    Returns:
        Soil temperatures [K] [nlevgrnd]
        
    Example:
        >>> temp_state = init_temperature_state(10, 20, 20, initial_temp=280.0)
        >>> soil_temp = get_soil_temperature(temp_state, column_idx=0, nlevsno=5)
        >>> soil_temp.shape
        (15,)  # nlevgrnd layers
    """
    return temp_state.t_soisno_col[column_idx, nlevsno:]


def get_snow_temperature(
    temp_state: TemperatureState,
    column_idx: int,
    nlevsno: int = DEFAULT_NLEVSNO,
) -> jnp.ndarray:
    """Extract snow-only temperatures from combined soil/snow array.
    
    Args:
        temp_state: Temperature state containing t_soisno_col
        column_idx: Column index to extract
        nlevsno: Number of snow layers
        
    Returns:
        Snow temperatures [K] [nlevsno]
        
    Example:
        >>> temp_state = init_temperature_state(10, 20, 20, initial_temp=270.0)
        >>> snow_temp = get_snow_temperature(temp_state, column_idx=0, nlevsno=5)
        >>> snow_temp.shape
        (5,)  # nlevsno layers
    """
    return temp_state.t_soisno_col[column_idx, :nlevsno]


def update_temperature(
    temp_state: TemperatureState,
    t_soisno_col: jnp.ndarray = None,
    t_a10_patch: jnp.ndarray = None,
    t_ref2m_patch: jnp.ndarray = None,
) -> TemperatureState:
    """Update temperature state with new values (functional update).
    
    Creates a new TemperatureState with updated fields. Only provided
    fields are updated; others retain their original values.
    
    Args:
        temp_state: Current temperature state
        t_soisno_col: New soil/snow temperatures [K] (optional)
        t_a10_patch: New 10-day mean temperatures [K] (optional)
        t_ref2m_patch: New 2m reference temperatures [K] (optional)
        
    Returns:
        New TemperatureState with updated fields
        
    Example:
        >>> temp_state = init_temperature_state(10, 20, 20)
        >>> new_t_a10 = jnp.full(20, 285.0)
        >>> updated_state = update_temperature(temp_state, t_a10_patch=new_t_a10)
        >>> updated_state.t_a10_patch[0]
        285.0
    """
    return TemperatureState(
        t_soisno_col=t_soisno_col if t_soisno_col is not None else temp_state.t_soisno_col,
        t_a10_patch=t_a10_patch if t_a10_patch is not None else temp_state.t_a10_patch,
        t_ref2m_patch=t_ref2m_patch if t_ref2m_patch is not None else temp_state.t_ref2m_patch,
    )