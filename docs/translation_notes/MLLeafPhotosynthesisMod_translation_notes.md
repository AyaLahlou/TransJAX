# Assembly Notes

This module combines 25 translation units into a complete JAX implementation of CTSM's leaf photosynthesis and stomatal conductance model. The assembly follows this structure:

1. **Module header** (unit 205): Core documentation and imports
2. **Temperature response functions** (units 225-227): ft, fth, fth25
3. **Main photosynthesis routine** (units 206-212): LeafPhotosynthesis with 6 inner sections
4. **Ci solver** (units 213-217): CiFunc with 4 inner sections
5. **Gs solver** (units 218-221): CiFuncGs with 3 inner sections
6. **Optimization routines** (units 222-223): StomataOptimization and StomataEfficiency
7. **Isotope fractionation** (unit 224): C13Fractionation

Key integration points:
- All inner units are composed into their parent root functions
- Shared parameters consolidated into NamedTuples
- Constants defined at module level
- Pure functional design with no side effects

# Complete JAX Module