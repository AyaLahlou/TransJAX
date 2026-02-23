"""
JAX translation of decompMod module.

This module provides a decomposition into a clumped data structure which can
be mapped back to atmosphere physics chunks. It defines the bounds for the
CLM grid cell (g), land unit (l), column (c), and patch (p) hierarchy.

Fortran source: decompMod.F90, lines 1-49

Key components:
- BoundsType: NamedTuple defining spatial decomposition indices
- get_clump_bounds: Function to retrieve bounds for a processor clump
- create_bounds: Helper function to construct BoundsType instances

Note:
    This implementation assumes a single grid cell with one land unit,
    one column, and one patch. This is a simplified version suitable for
    single-point simulations or testing.
"""

from typing import NamedTuple
import jax.numpy as jnp


# ============================================================================
# Type Definitions
# ============================================================================


class BoundsType(NamedTuple):
    """Domain decomposition bounds for gridcells, landunits, columns, and patches.
    
    This type defines the beginning and ending indices for different spatial
    decomposition levels in the CTSM model hierarchy:
    - gridcell (g): coarsest level
    - landunit (l): intermediate level
    - column (c): finer level
    - patch (p): finest level
    
    Fortran source: decompMod.F90, lines 12-17
    
    Attributes:
        begg: Beginning gridcell index
        endg: Ending gridcell index
        begl: Beginning landunit index
        endl: Ending landunit index
        begc: Beginning column index
        endc: Ending column index
        begp: Beginning patch index
        endp: Ending patch index
    """
    begg: int  # Beginning gridcell index
    endg: int  # Ending gridcell index
    begl: int  # Beginning landunit index
    endl: int  # Ending landunit index
    begc: int  # Beginning column index
    endc: int  # Ending column index
    begp: int  # Beginning patch index
    endp: int  # Ending patch index


# ============================================================================
# Public Functions
# ============================================================================


def get_clump_bounds(n: int) -> BoundsType:
    """Get clump bounds for processor clump index.
    
    Define grid cell (g), land unit (l), column (c), and patch (p) bounds
    for CLM g/l/c/p hierarchy. CLM processes clumps of gridcells (and
    associated subgrid-scale entities) with length defined by
    begg/endg, begl/endl, begc/endc, and begp/endp. This code assumes
    that a grid cell has one land unit with one column and one patch. It
    processes a single grid cell.
    
    Fortran source: decompMod.F90, lines 22-47
    
    Args:
        n: Processor clump index (unused in current implementation but kept
           for interface compatibility).
    
    Returns:
        BoundsType: Bounds structure with all begin/end indices set to 1,
                    representing a single grid cell with one land unit,
                    one column, and one patch.
    
    Note:
        This is a simplified implementation that always returns bounds for
        a single grid cell. The clump index n is accepted but not used,
        maintaining compatibility with the Fortran interface.
    
    Example:
        >>> bounds = get_clump_bounds(1)
        >>> bounds.begg, bounds.endg
        (1, 1)
        >>> bounds.begc, bounds.endc
        (1, 1)
    """
    # Lines 35-46: Set all bounds to 1 for single grid cell processing
    return BoundsType(
        begg=1,  # Line 35
        endg=1,  # Line 36
        begl=1,  # Line 38
        endl=1,  # Line 39
        begc=1,  # Line 41
        endc=1,  # Line 42
        begp=1,  # Line 44
        endp=1   # Line 45
    )


def create_bounds(
    begg: int,
    endg: int,
    begl: int,
    endl: int,
    begc: int,
    endc: int,
    begp: int,
    endp: int,
) -> BoundsType:
    """Create a BoundsType instance with specified indices.
    
    Helper function to construct custom bounds for testing or specialized
    decomposition scenarios.
    
    Args:
        begg: Beginning gridcell index
        endg: Ending gridcell index
        begl: Beginning landunit index
        endl: Ending landunit index
        begc: Beginning column index
        endc: Ending column index
        begp: Beginning patch index
        endp: Ending patch index
    
    Returns:
        BoundsType: Instance with the specified bounds
    
    Fortran source: decompMod.F90, lines 12-16
    
    Example:
        >>> bounds = create_bounds(1, 10, 1, 10, 1, 10, 1, 10)
        >>> bounds.begg, bounds.endg
        (1, 10)
    """
    return BoundsType(
        begg=begg,
        endg=endg,
        begl=begl,
        endl=endl,
        begc=begc,
        endc=endc,
        begp=begp,
        endp=endp,
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'BoundsType',
    'get_clump_bounds',
    'create_bounds',
]