# Assembly Notes

## Module Structure
The MLMathToolsMod module provides mathematical utilities for the multilayer canopy model. I've organized it into:

1. **Module header**: Comprehensive docstring explaining all mathematical tools
2. **Imports**: JAX, typing, and protocol definitions
3. **Constants**: Machine epsilon and numerical stability constants
4. **Configuration**: MLMathToolsConfig NamedTuple for solver parameters
5. **Function protocol**: MLMathFunction protocol for function interfaces
6. **Root finding**: hybrid() and zbrent() with supporting state types
7. **Linear algebra**: quadratic(), tridiag(), tridiag_2eq()
8. **Special functions**: log_gamma_function(), beta_function(), beta_distribution_pdf/cdf()
9. **Helper functions**: beta_function_incomplete_cf() for continued fractions

## Key Integration Points
- All functions are pure and JIT-compatible
- State types (HybridState, ZbrentState) use NamedTuples for immutability
- Iterative algorithms use `jax.lax.while_loop` and `jax.lax.scan` for JIT
- No external dependencies except JAX (self-contained module)
- All numerical constants defined inline

## Notes
- The `func` interface (lines 29-36) is translated as a Protocol since it's just an interface definition
- The hybrid root finder has two versions: one standalone and one that integrates zbrent
- Tridiagonal solvers use scan for forward/backward passes (JIT-compatible)
- Beta distribution functions depend on each other in a specific order
- All convergence checks use `jnp.where` instead of Python conditionals

---

# Complete JAX Module