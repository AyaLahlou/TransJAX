# Assembly Notes

This module combines 6 translated units into a complete JAX implementation of SoilStateInitTimeConstMod:

1. **Module structure (unit_019)**: Base types and module interface
2. **Root function (unit_020)**: Main entry point and configuration
3. **Root profile computation (unit_021)**: Zeng2001 and Jackson1996 methods
4. **Bedrock adjustment (unit_022)**: Root redistribution and soil texture
5. **Hydraulic properties (unit_023)**: Texture lookup and hydraulic parameters
6. **Thermal properties (units_024-025)**: Conductivity and heat capacity

Key integration decisions:
- Consolidated all NamedTuple definitions at module level
- Merged overlapping state structures (SoilPropertiesState, SoilHydraulicState, SoilThermalProperties)
- Created unified SoilStateType with all computed properties
- Organized functions hierarchically: low-level computations → layer-level → column-level → full domain
- Added vectorization wrappers for efficient batch processing
- Included all physical constants inline with references to Fortran line numbers

# Complete JAX Module