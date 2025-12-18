# Assembly Notes

## Module Structure
The SurfaceAlbedoMod module has been assembled from 3 translation units:
1. **Module definition (042)**: Core types and state containers
2. **SurfaceAlbedoInitTimeConst (043)**: Initialization of soil color and albedo lookup tables
3. **SoilAlbedo (044)**: Runtime soil albedo calculation based on moisture

## Key Integration Points
- Consolidated `SurfaceAlbedoState` and `SurfaceAlbedoConstants` types
- Merged module-level and function-specific imports
- Integrated soil color lookup tables (8 and 20 class systems) directly into init function
- Connected water state and albedo calculation through consistent type interfaces

## Physics Preservation
- Exact soil albedo lookup tables from Fortran (lines 62-77)
- Moisture correction formula: `inc = max(0.11 - 0.40 * h2osoi_vol, 0.0)` (line 127)
- Albedo bounded between saturated and dry values (line 128)
- Separate visible (ivis=0) and near-infrared (inir=1) wavebands

## JAX Compatibility
- All functions are JIT-compatible with pure operations
- Used `jnp.where` and `jnp.maximum/minimum` for conditionals
- Vectorized operations over columns and wavebands
- Immutable updates with `.at[].set()` syntax

---

# Complete JAX Module