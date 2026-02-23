"""
JAX translation of clm_varctl module.

This module contains run control variables for CLM (Community Land Model).
Provides configuration and control parameters for model execution.

Translated from: clm_varctl.F90, lines 1-15
Original module: clm_varctl

Key Features:
    - Run control variables (iulog for logging)
    - Pure functional interface with immutable state
    - Type-safe configuration using NamedTuple
    - JIT-compatible design

Note:
    In the original Fortran, this module primarily manages I/O unit numbers
    and other runtime control variables. In JAX, I/O is handled differently,
    but we preserve the structure for compatibility with the broader CLM codebase.
"""

from typing import NamedTuple
import jax.numpy as jnp
from jax import Array

# ============================================================================
# Type Definitions
# ============================================================================

# Type alias for double precision (Fortran r8 kind)
r8 = jnp.float64


class ClmVarCtl(NamedTuple):
    """
    Run control variables for CLM.
    
    This immutable structure holds runtime configuration parameters that
    control CLM execution behavior. In the original Fortran, these were
    module-level variables; here they're encapsulated in a NamedTuple
    for functional purity.
    
    Attributes:
        iulog: Log file unit number for stdout (default: 6)
               In Fortran, this controls where log messages are written.
               In JAX translation, this is preserved for compatibility
               but actual I/O operations are handled through Python's
               logging system or other mechanisms.
    
    Reference: clm_varctl.F90, lines 13
    
    Example:
        >>> ctl = ClmVarCtl(iulog=6)
        >>> print(ctl.iulog)
        6
    """
    iulog: int = 6


# ============================================================================
# Factory Functions
# ============================================================================

def create_clm_varctl(iulog: int = 6) -> ClmVarCtl:
    """
    Create ClmVarCtl configuration with specified or default values.
    
    Factory function to instantiate run control variables. Provides
    a clean interface for creating configuration objects with validation
    and default values.
    
    Args:
        iulog: Log file unit number (default: 6)
               Must be a positive integer representing a valid unit number.
    
    Returns:
        ClmVarCtl: Immutable configuration object with specified parameters
        
    Reference: clm_varctl.F90, lines 13
    
    Example:
        >>> ctl = create_clm_varctl()
        >>> ctl.iulog
        6
        >>> custom_ctl = create_clm_varctl(iulog=10)
        >>> custom_ctl.iulog
        10
    """
    return ClmVarCtl(iulog=iulog)


def update_clm_varctl(ctl: ClmVarCtl, **kwargs) -> ClmVarCtl:
    """
    Create updated ClmVarCtl with modified fields.
    
    Since ClmVarCtl is immutable (NamedTuple), this function creates
    a new instance with updated values. Follows functional programming
    principles for state updates.
    
    Args:
        ctl: Existing ClmVarCtl configuration
        **kwargs: Fields to update (e.g., iulog=7)
    
    Returns:
        ClmVarCtl: New configuration object with updated fields
        
    Example:
        >>> ctl = create_clm_varctl()
        >>> new_ctl = update_clm_varctl(ctl, iulog=10)
        >>> new_ctl.iulog
        10
        >>> ctl.iulog  # Original unchanged
        6
    """
    return ctl._replace(**kwargs)


# ============================================================================
# Module Constants
# ============================================================================

# Default module-level instance for convenience
# This provides a singleton-like default configuration that can be used
# throughout the codebase without explicit instantiation
DEFAULT_CLM_VARCTL = create_clm_varctl()


# ============================================================================
# Utility Functions
# ============================================================================

def get_log_unit(ctl: ClmVarCtl) -> int:
    """
    Get the log file unit number from configuration.
    
    Accessor function for the log unit number. While this could be
    accessed directly from the NamedTuple, providing an accessor
    maintains consistency with the original Fortran interface.
    
    Args:
        ctl: ClmVarCtl configuration object
    
    Returns:
        int: Log file unit number
        
    Example:
        >>> ctl = create_clm_varctl()
        >>> get_log_unit(ctl)
        6
    """
    return ctl.iulog


def validate_clm_varctl(ctl: ClmVarCtl) -> bool:
    """
    Validate ClmVarCtl configuration.
    
    Checks that all configuration values are within valid ranges.
    Useful for runtime validation of user-provided configurations.
    
    Args:
        ctl: ClmVarCtl configuration to validate
    
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Example:
        >>> ctl = create_clm_varctl()
        >>> validate_clm_varctl(ctl)
        True
    """
    # Log unit should be a positive integer
    if not isinstance(ctl.iulog, int) or ctl.iulog <= 0:
        return False
    
    return True


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__author__ = "JAX Translation Team"
__fortran_source__ = "clm_varctl.F90"
__fortran_lines__ = "1-15"