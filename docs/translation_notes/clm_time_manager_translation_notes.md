# Assembly Notes

## Module Structure
Assembled `clm_time_manager.py` from 6 translated units:
1. **Module definition** (lines 1-151): Core types, constants, and basic functions
2. **get_curr_date** (lines 152-233): Current date calculation with month/year overflow
3. **get_prev_date** (lines 236-318): Previous date calculation
4. **get_curr_time** (lines 321-340): Elapsed time calculation
5. **get_curr_calday** (lines 343-412): Current calendar day with offset support
6. **get_prev_calday** (lines 415-462): Previous calendar day

## Key Integration Points
- **Circular dependencies resolved**: `get_curr_calday` and `get_prev_calday` cross-reference each other; handled via forward declarations
- **Shared constants**: `MDAY`, `MDAYCUM`, `MDAYLEAP`, `MDAYLEAPCUM` defined once at module level
- **State management**: Single `TimeManagerState` NamedTuple used throughout
- **JIT compatibility**: All functions use `jnp.where` and `lax.while_loop` instead of Python control flow

## Modifications from Individual Units
1. Removed duplicate imports and constant definitions
2. Consolidated `isleap` into single JAX-compatible version
3. Fixed import paths (removed relative imports for self-contained module)
4. Added missing array definitions (`MDAYCUM`, `MDAYLEAPCUM`) from Fortran lines 77-82
5. Removed placeholder `NotImplementedError` functions - all are now fully implemented

## Physics Preservation
- Exact leap year logic (divisible by 4, 100, 400 rules)
- Integer arithmetic for date calculations
- Gregorian calendar hack for day 366 compatibility with orbital calculations
- Month/day overflow handling via iterative correction

---

# Complete JAX Module