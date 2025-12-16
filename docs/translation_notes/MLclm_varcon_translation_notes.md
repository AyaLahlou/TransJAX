# Assembly Notes

The MLclm_varcon module is already a complete, self-contained constants module. It consists of a single translation unit that defines:

1. **MLCanopyConstants**: A NamedTuple containing all physical constants and parameters for the multilayer canopy model
2. **RSLPsihatLookupTables**: A NamedTuple for roughness sublayer psihat lookup tables
3. **Helper functions**: Factory function to create empty lookup tables

The module has no computational functions - it only provides immutable constants and data structures. No assembly is needed beyond the single translated unit.

Key design decisions:
- All constants stored in immutable NamedTuple for JAX compatibility
- SPVAL (1.0e36) defined as module-level constant for uninitialized values
- Default instance (ML_CANOPY_CONSTANTS) provided for convenient access
- Lookup table dimensions (n_z=276, n_l=41) included in constants
- Comprehensive docstrings with line number references to original Fortran

---

# Complete JAX Module