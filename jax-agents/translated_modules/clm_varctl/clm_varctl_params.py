"""
Parameters for clm_varctl module.

This file contains constants and default values used by the clm_varctl module.
Since the original Fortran module is minimal, this parameters file is also
minimal but follows the standard structure for consistency.

Reference: clm_varctl.F90, lines 1-15
"""

# ============================================================================
# Default Values
# ============================================================================

# Default log file unit number (Fortran standard output)
DEFAULT_IULOG = 6

# ============================================================================
# Valid Ranges
# ============================================================================

# Minimum valid log unit number
MIN_LOG_UNIT = 1

# Maximum valid log unit number (arbitrary but reasonable limit)
MAX_LOG_UNIT = 999

# ============================================================================
# Module Information
# ============================================================================

MODULE_NAME = "clm_varctl"
MODULE_DESCRIPTION = "Run control variables for CLM"