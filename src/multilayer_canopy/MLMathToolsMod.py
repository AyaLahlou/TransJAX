"""
Mathematical Tools for Multilayer Canopy Model.

Translated from CTSM's MLMathToolsMod.F90

This module provides mathematical utilities for the multilayer canopy model:

1. **Root Finding**:
   - hybrid(): Hybrid secant/Brent method for fast convergence
   - zbrent(): Brent's method for guaranteed convergence when root is bracketed

2. **Linear Algebra**:
   - quadratic(): Numerically stable quadratic equation solver
   - tridiag(): Thomas algorithm for tridiagonal systems
   - tridiag_2eq(): Coupled tridiagonal system solver (2 equations per layer)

3. **Special Functions**:
   - log_gamma_function(): Natural log of gamma function
   - beta_function(): Beta function B(a,b)
   - beta_distribution_pdf(): Beta distribution probability density
   - beta_distribution_cdf(): Beta distribution cumulative distribution

All functions are implemented as pure JAX operations for JIT compilation.
No side effects, immutable data structures, vectorized operations.

Key Algorithms:
    Root Finding (hybrid):
        1. Start with secant method: dx = -f1*(x1-x0)/(f1-f0)
        2. Switch to Brent if sign change detected
        3. Fall back to minimum |f| if max iterations exceeded
        
    Root Finding (zbrent):
        1. Maintain bracket [a,b] with f(a)*f(b) < 0
        2. Use inverse quadratic interpolation when possible
        3. Fall back to bisection for robustness
        
    Tridiagonal (Thomas algorithm):
        1. Forward elimination: eliminate lower diagonal
        2. Backward substitution: solve for unknowns
        
    Coupled Tridiagonal:
        1. Forward elimination with 2x2 matrix inversions
        2. Backward substitution for both variables simultaneously

Reference:
    MLMathToolsMod.F90 lines 1-609
"""

from typing import NamedTuple, Tuple, Callable, Protocol, Union
import jax
import jax.numpy as jnp
from jax import lax

# ============================================================================
# Module Constants
# ============================================================================

# Machine precision constants
EPSILON = jnp.finfo(jnp.float64).eps
TINY = jnp.finfo(jnp.float64).tiny


# ============================================================================
# Configuration Types
# ============================================================================

class MLMathToolsConfig(NamedTuple):
    """Configuration for math tools module.
    
    Attributes:
        max_iterations: Maximum iterations for iterative solvers
        tolerance: Convergence tolerance for root finding
        epsilon: Machine epsilon for numerical stability
    """
    max_iterations: int = 100
    tolerance: float = 1e-8
    epsilon: float = EPSILON


# ============================================================================
# Function Interface Protocol
# ============================================================================

class MLMathFunction(Protocol):
    """Protocol defining the interface for multilayer canopy math functions.
    
    This protocol matches the Fortran subroutine interface defined in
    MLMathToolsMod.F90 lines 29-36. Concrete implementations should follow
    this signature for use with root finding and optimization algorithms.
    
    Note:
        In JAX translation, we use a Protocol instead of an actual function
        since the Fortran code only defines an interface. Actual implementations
        will be provided by modules that use this interface.
    """
    
    def __call__(
        self,
        p: int,
        ic: int,
        il: int,
        mlcanopy_inst: any,  # Would be MLCanopyState in actual use
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate mathematical function at point x.
        
        Args:
            p: Patch index [scalar]
            ic: Canopy layer index [scalar]
            il: Leaf layer index [scalar]
            mlcanopy_inst: Multilayer canopy state containing all physics variables
            x: Input value at which to evaluate function [scalar or array]
            
        Returns:
            Function value at x [same shape as x]
            
        Reference:
            MLMathToolsMod.F90:29-36
        """
        ...


def create_ml_math_function(
    func_impl: callable,
) -> MLMathFunction:
    """Create a multilayer canopy math function from an implementation.
    
    This helper wraps a concrete implementation to ensure it matches the
    MLMathFunction protocol.
    
    Args:
        func_impl: Concrete function implementation
        
    Returns:
        Function matching MLMathFunction protocol
        
    Example:
        >>> def my_func(p, ic, il, mlcanopy_inst, x):
        ...     # Some physics calculation
        ...     return x**2 + mlcanopy_inst.some_field[p, ic, il]
        >>> ml_func = create_ml_math_function(my_func)
    """
    return func_impl


# ============================================================================
# Root Finding - State Types
# ============================================================================

class HybridState(NamedTuple):
    """State for hybrid root-finding iteration.
    
    Attributes:
        x0: Previous root estimate
        x1: Current root estimate
        f0: Function value at x0
        f1: Function value at x1
        minx: x value giving minimum |f|
        minf: Minimum |f| value found
        iter: Iteration counter
        converged: Whether convergence achieved
        use_brent: Whether to switch to Brent's method
    """
    x0: jnp.ndarray
    x1: jnp.ndarray
    f0: jnp.ndarray
    f1: jnp.ndarray
    minx: jnp.ndarray
    minf: jnp.ndarray
    iter: jnp.ndarray
    converged: jnp.ndarray
    use_brent: jnp.ndarray


class ZbrentState(NamedTuple):
    """State for Brent's method iteration.
    
    Attributes:
        a: Left bracket point
        b: Right bracket point (current best estimate)
        c: Previous value of a
        d: Step size from previous iteration
        e: Step size from iteration before last
        fa: Function value at a
        fb: Function value at b
        fc: Function value at c
        iter: Current iteration number
        converged: Whether algorithm has converged
        error: Whether an error occurred
    """
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray
    e: jnp.ndarray
    fa: jnp.ndarray
    fb: jnp.ndarray
    fc: jnp.ndarray
    iter: jnp.ndarray
    converged: jnp.ndarray
    error: jnp.ndarray


# ============================================================================
# Root Finding Functions
# ============================================================================

def hybrid_root_finder(
    func: Callable[[jnp.ndarray], jnp.ndarray],
    xa: jnp.ndarray,
    xb: jnp.ndarray,
    tol: float,
    itmax: int = 40,
) -> jnp.ndarray:
    """Find root using hybrid secant/Brent method.
    
    Solves for x where func(x) = 0 using initial estimates xa and xb.
    Combines secant method for speed with Brent's method for robustness.
    
    Algorithm (lines 42-136):
    1. Evaluate function at initial points xa and xb
    2. Check for immediate root (f=0)
    3. Track minimum function value as fallback
    4. Iterate using secant method:
       - Compute dx = -f1*(x1-x0)/(f1-f0)
       - Update x = x1 + dx
       - Check convergence: |dx| < tol
       - If sign change detected, switch to Brent's method
       - If max iterations exceeded, return minimum point
    
    Args:
        func: Function to find root of, signature func(x) -> f
        xa: First initial estimate of root [arbitrary units]
        xb: Second initial estimate of root [arbitrary units]
        tol: Convergence tolerance [same units as xa, xb]
        itmax: Maximum iterations (default 40, line 68)
        
    Returns:
        Root estimate where func(root) ≈ 0 [same units as xa, xb]
        
    Note:
        This is a vectorized version that can handle batched inputs.
        The original Fortran uses external function calls and zbrent
        as a subroutine. This version returns the secant method result
        or minimum if convergence fails. For full Brent integration,
        use hybrid_root_finder_with_brent().
        
    Reference:
        Lines 42-136 in MLMathToolsMod.F90
    """
    # Evaluate function at initial points (lines 73-84)
    f0 = func(xa)
    f1 = func(xb)
    
    # Check for immediate roots (lines 74-84)
    is_root_a = jnp.abs(f0) < 1e-15
    is_root_b = jnp.abs(f1) < 1e-15
    
    # Initialize tracking of minimum function value (lines 86-92)
    minx = jnp.where(jnp.abs(f1) < jnp.abs(f0), xb, xa)
    minf = jnp.where(jnp.abs(f1) < jnp.abs(f0), jnp.abs(f1), jnp.abs(f0))
    
    # Initialize state for iteration
    init_state = HybridState(
        x0=xa,
        x1=xb,
        f0=f0,
        f1=f1,
        minx=minx,
        minf=minf,
        iter=jnp.array(0),
        converged=is_root_a | is_root_b,
        use_brent=jnp.array(False),
    )
    
    def iteration_body(state: HybridState) -> HybridState:
        """Single iteration of hybrid method (lines 97-127)."""
        # Secant method update (lines 99-101)
        dx = -state.f1 * (state.x1 - state.x0) / (state.f1 - state.f0 + 1e-30)
        x_new = state.x1 + dx
        
        # Check convergence (lines 102-105)
        converged = jnp.abs(dx) < tol
        
        # Update for next iteration (lines 106-108)
        x0_next = state.x1
        f0_next = state.f1
        x1_next = x_new
        f1_next = func(x_new)
        
        # Update minimum tracking (lines 109-112)
        minx_next = jnp.where(jnp.abs(f1_next) < state.minf, x1_next, state.minx)
        minf_next = jnp.where(jnp.abs(f1_next) < state.minf, jnp.abs(f1_next), state.minf)
        
        # Check for sign change indicating root zone (lines 116-120)
        sign_change = (f1_next * f0_next) < 0.0
        use_brent = sign_change
        
        # Check for max iterations (lines 124-128)
        iter_next = state.iter + 1
        max_iter_reached = iter_next > itmax
        
        # If max iterations, fall back to minimum
        x0_final = jnp.where(max_iter_reached, state.minx, x0_next)
        x1_final = jnp.where(max_iter_reached, state.minx, x1_next)
        converged_final = converged | max_iter_reached | use_brent
        
        return HybridState(
            x0=x0_final,
            x1=x1_final,
            f0=f0_next,
            f1=f1_next,
            minx=minx_next,
            minf=minf_next,
            iter=iter_next,
            converged=converged_final,
            use_brent=use_brent,
        )
    
    def cond_fun(state: HybridState) -> jnp.ndarray:
        """Continue while not converged."""
        return ~state.converged
    
    # Run iteration loop (lines 97-130)
    final_state = lax.while_loop(cond_fun, iteration_body, init_state)
    
    # Return root (line 132)
    root = jnp.where(
        is_root_a,
        xa,
        jnp.where(is_root_b, xb, final_state.x0)
    )
    
    return root


def hybrid_root_finder_with_brent(
    func: Callable[[jnp.ndarray], jnp.ndarray],
    zbrent_func: Callable[[Callable, jnp.ndarray, jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    xa: jnp.ndarray,
    xb: jnp.ndarray,
    tol: float,
    itmax: int = 40,
) -> jnp.ndarray:
    """Hybrid root finder with Brent's method integration.
    
    This version properly integrates zbrent when a root zone is found,
    matching the original Fortran logic exactly (lines 116-120).
    
    Args:
        func: Function to find root of
        zbrent_func: Brent's method implementation
        xa: First initial estimate
        xb: Second initial estimate
        tol: Convergence tolerance
        itmax: Maximum iterations
        
    Returns:
        Root estimate
        
    Reference:
        Lines 42-136 in MLMathToolsMod.F90
    """
    # Initial evaluations (lines 73-84)
    f0 = func(xa)
    f1 = func(xb)
    
    # Check immediate roots
    is_root_a = jnp.abs(f0) < 1e-15
    is_root_b = jnp.abs(f1) < 1e-15
    
    # Track minimum (lines 86-92)
    minx = jnp.where(jnp.abs(f1) < jnp.abs(f0), xb, xa)
    minf = jnp.where(jnp.abs(f1) < jnp.abs(f0), jnp.abs(f1), jnp.abs(f0))
    
    # Initialize state
    init_state = HybridState(
        x0=xa,
        x1=xb,
        f0=f0,
        f1=f1,
        minx=minx,
        minf=minf,
        iter=jnp.array(0),
        converged=is_root_a | is_root_b,
        use_brent=jnp.array(False),
    )
    
    def iteration_body(state: HybridState) -> HybridState:
        """Iteration with Brent's method integration."""
        # Secant update (lines 99-101)
        dx = -state.f1 * (state.x1 - state.x0) / (state.f1 - state.f0 + 1e-30)
        x_new = state.x1 + dx
        
        # Convergence check (lines 102-105)
        converged = jnp.abs(dx) < tol
        
        # Update state (lines 106-108)
        x0_next = state.x1
        f0_next = state.f1
        x1_next = x_new
        f1_next = func(x_new)
        
        # Track minimum (lines 109-112)
        minx_next = jnp.where(jnp.abs(f1_next) < state.minf, x1_next, state.minx)
        minf_next = jnp.where(jnp.abs(f1_next) < state.minf, jnp.abs(f1_next), state.minf)
        
        # Check for root zone (lines 116-120)
        sign_change = (f1_next * f0_next) < 0.0
        
        # If sign change, use Brent's method
        x_brent, _, _ = zbrent_func(func, x0_next, x1_next, tol)
        x_final = jnp.where(sign_change, x_brent, x1_next)
        
        # Check max iterations (lines 124-128)
        iter_next = state.iter + 1
        max_iter_reached = iter_next > itmax
        x_final = jnp.where(max_iter_reached, state.minx, x_final)
        
        converged_final = converged | max_iter_reached | sign_change
        
        return HybridState(
            x0=x_final,
            x1=x_final,
            f0=f0_next,
            f1=f1_next,
            minx=minx_next,
            minf=minf_next,
            iter=iter_next,
            converged=converged_final,
            use_brent=sign_change,
        )
    
    def cond_fun(state: HybridState) -> jnp.ndarray:
        return ~state.converged
    
    # Run iteration (lines 97-130)
    final_state = lax.while_loop(cond_fun, iteration_body, init_state)
    
    # Return root (line 132)
    root = jnp.where(
        is_root_a,
        xa,
        jnp.where(is_root_b, xb, final_state.x0)
    )
    
    return root


def zbrent(
    func: Callable[[jnp.ndarray], jnp.ndarray],
    xa: jnp.ndarray,
    xb: jnp.ndarray,
    tol: float,
    itmax: int = 50,
    eps: float = 1.0e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find root of function using Brent's method.
    
    Uses Brent's method to find the root of a function, which is known to exist
    between xa and xb. The root is updated until its accuracy is tol.
    
    Algorithm (lines 139-244):
    1. Initialize bracket [a,b] and evaluate function at endpoints
    2. Check that root is bracketed (fa and fb have opposite signs)
    3. Iterate until convergence or max iterations:
       a. Ensure |fc| >= |fb| by swapping if needed
       b. Check convergence: |xm| <= tol1 or fb == 0
       c. Attempt inverse quadratic interpolation or secant method
       d. Fall back to bisection if interpolation step is too small
       e. Update bracket and function values
    
    Args:
        func: Function to find root of, signature f(x) -> f
        xa: Minimum of variable domain to search [n_points]
        xb: Maximum of variable domain to search [n_points]
        tol: Error tolerance (absolute)
        itmax: Maximum number of iterations (default 50, line 161)
        eps: Relative error tolerance (default 1e-8, line 162)
        
    Returns:
        Tuple of:
            root: Root of function [n_points]
            converged: Whether algorithm converged [n_points, bool]
            error: Whether bracketing error occurred [n_points, bool]
            
    Note:
        Lines 165-173: Initialize and check bracketing
        Lines 174-236: Main iteration loop
        Lines 238-243: Check for max iterations exceeded
        
    Reference:
        Lines 139-244 in MLMathToolsMod.F90
    """
    # Initialize (lines 165-167)
    a = xa
    b = xb
    fa = func(a)
    fb = func(b)
    
    # Check that root is bracketed (lines 169-173)
    same_sign = (fa > 0.0) & (fb > 0.0) | (fa < 0.0) & (fb < 0.0)
    error = same_sign
    
    # Initialize state (lines 174-176)
    c = b
    fc = fb
    d = jnp.zeros_like(b)
    e = jnp.zeros_like(b)
    
    initial_state = ZbrentState(
        a=a,
        b=b,
        c=c,
        d=d,
        e=e,
        fa=fa,
        fb=fb,
        fc=fc,
        iter=jnp.zeros_like(b, dtype=jnp.int32),
        converged=jnp.zeros_like(b, dtype=bool),
        error=error,
    )
    
    def iteration_body(state: ZbrentState) -> ZbrentState:
        """Single iteration of Brent's method (lines 178-234)."""
        # Unpack state
        a, b, c, d, e = state.a, state.b, state.c, state.d, state.e
        fa, fb, fc = state.fa, state.fb, state.fc
        
        # Lines 180-185: Ensure |fc| >= |fb|
        need_swap = (fb > 0.0) & (fc > 0.0) | (fb < 0.0) & (fc < 0.0)
        c = jnp.where(need_swap, a, c)
        fc = jnp.where(need_swap, fa, fc)
        d = jnp.where(need_swap, b - a, d)
        e = jnp.where(need_swap, d, e)
        
        # Lines 186-193: Swap a and b if |fc| < |fb|
        need_swap2 = jnp.abs(fc) < jnp.abs(fb)
        a_new = jnp.where(need_swap2, b, a)
        b_new = jnp.where(need_swap2, c, b)
        c_new = jnp.where(need_swap2, a_new, c)
        fa_new = jnp.where(need_swap2, fb, fa)
        fb_new = jnp.where(need_swap2, fc, fb)
        fc_new = jnp.where(need_swap2, fa_new, fc)
        
        a, b, c = a_new, b_new, c_new
        fa, fb, fc = fa_new, fb_new, fc_new
        
        # Lines 194-196: Check convergence
        tol1 = 2.0 * eps * jnp.abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        converged = (jnp.abs(xm) <= tol1) | (fb == 0.0)
        
        # Lines 197-221: Attempt interpolation or use bisection
        can_interpolate = (jnp.abs(e) >= tol1) & (jnp.abs(fa) > jnp.abs(fb))
        
        # Compute interpolation step
        s = fb / fa
        
        # Lines 199-207: Linear or inverse quadratic interpolation
        is_linear = a == c
        pp_linear = 2.0 * xm * s
        q_linear = 1.0 - s
        
        q_quad = fa / fc
        r_quad = fb / fc
        pp_quad = s * (2.0 * xm * q_quad * (q_quad - r_quad) - 
                       (b - a) * (r_quad - 1.0))
        q_quad_final = (q_quad - 1.0) * (r_quad - 1.0) * (s - 1.0)
        
        pp = jnp.where(is_linear, pp_linear, pp_quad)
        q = jnp.where(is_linear, q_linear, q_quad_final)
        
        # Lines 208-209: Adjust sign
        q = jnp.where(pp > 0.0, -q, q)
        pp = jnp.abs(pp)
        
        # Lines 210-216: Check if interpolation is acceptable
        min_val = jnp.minimum(3.0 * xm * q - jnp.abs(tol1 * q), 
                              jnp.abs(e * q))
        use_interpolation = (2.0 * pp < min_val) & can_interpolate
        
        d_interp = pp / q
        e_interp = d
        
        d_bisect = xm
        e_bisect = d_bisect
        
        # Lines 217-221: Choose step
        d = jnp.where(use_interpolation, d_interp, d_bisect)
        e = jnp.where(use_interpolation, e_interp, e_bisect)
        
        # Lines 222-230: Update a and b
        a = b
        fa = fb
        
        # Lines 226-230: Update b
        use_large_step = jnp.abs(d) > tol1
        b = jnp.where(use_large_step, b + d, b + jnp.sign(xm) * tol1)
        
        # Line 231: Evaluate function at new b
        fb = func(b)
        
        # Line 232: Check if root found exactly
        converged = converged | (fb == 0.0)
        
        return ZbrentState(
            a=a,
            b=b,
            c=c,
            d=d,
            e=e,
            fa=fa,
            fb=fb,
            fc=fc,
            iter=state.iter + 1,
            converged=converged,
            error=state.error,
        )
    
    def continue_condition(state: ZbrentState) -> jnp.ndarray:
        """Check if iteration should continue (line 177)."""
        return (state.iter < itmax) & ~state.converged & ~state.error
    
    # Run iterations using while_loop for JIT compatibility
    final_state = jax.lax.while_loop(
        continue_condition,
        iteration_body,
        initial_state,
    )
    
    # Line 234: Return root
    root = final_state.b
    
    # Lines 236-243: Check for max iterations exceeded
    max_iter_exceeded = (final_state.iter >= itmax) & ~final_state.converged
    error_final = final_state.error | max_iter_exceeded
    
    return root, final_state.converged, error_final


# ============================================================================
# Linear Algebra Functions
# ============================================================================

def quadratic(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve quadratic equation for its two roots.
    
    Solves ax² + bx + c = 0 using a numerically stable algorithm that
    avoids catastrophic cancellation.
    
    The algorithm uses the standard quadratic formula but chooses the sign
    in the numerator to avoid subtracting nearly equal numbers:
        q = -0.5 * (b + sign(b) * sqrt(b² - 4ac))
        r1 = q / a
        r2 = c / q
    
    Args:
        a: Coefficient of x² term [any units] [...]
        b: Coefficient of x term [any units] [...]
        c: Constant term [any units] [...]
        
    Returns:
        Tuple of (r1, r2) where:
            r1: First root [same units as b/a] [...]
            r2: Second root [same units as b/a] [...]
            
    Note:
        - Input arrays must be broadcastable
        - If a = 0, behavior is undefined (caller must ensure a != 0)
        - If discriminant < 0, returns NaN (complex roots)
        - If q = 0, r2 is set to 1e36 (line 278)
        
    Reference:
        Lines 247-281 of MLMathToolsMod.F90
    """
    # Calculate discriminant (line 269, 271)
    discriminant = b * b - 4.0 * a * c
    sqrt_discriminant = jnp.sqrt(discriminant)
    
    # Choose sign to avoid cancellation (lines 269-272)
    q = jnp.where(
        b >= 0.0,
        -0.5 * (b + sqrt_discriminant),
        -0.5 * (b - sqrt_discriminant),
    )
    
    # Calculate first root (line 274)
    r1 = q / a
    
    # Calculate second root (lines 275-278)
    r2 = jnp.where(
        q != 0.0,
        c / q,
        1.0e36,
    )
    
    return r1, r2


def tridiag(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    r: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a tridiagonal system of equations.
    
    Implements the Thomas algorithm for solving F x U = R where F is a
    tridiagonal matrix. This is a direct method with O(n) complexity.
    
    The tridiagonal matrix F is defined by three vectors:
        - a: Lower diagonal (a[0] is undefined and not used)
        - b: Main diagonal
        - c: Upper diagonal (c[n-1] is undefined and not used)
    
    The system is:
        | b[0]  c[0]   0    ...                      |   | u[0]   |   | r[0]   |
        | a[1]  b[1]  c[1]  ...                      |   | u[1]   |   | r[1]   |
        |                   ...                      | x | ...    | = | ...    |
        |                   ... a[n-2] b[n-2] c[n-2] |   | u[n-2] |   | r[n-2] |
        |                   ...   0    a[n-1] b[n-1] |   | u[n-1] |   | r[n-1] |
    
    Algorithm:
        1. Forward elimination: Eliminate lower diagonal
        2. Backward substitution: Solve for unknowns
    
    Args:
        a: Lower diagonal coefficients [n]. Note: a[0] is not used.
        b: Main diagonal coefficients [n]
        c: Upper diagonal coefficients [n]. Note: c[n-1] is not used.
        r: Right-hand side vector [n]
        
    Returns:
        u: Solution vector [n]
        
    Note:
        - Fortran lines 284-331
        - For numerical stability, the matrix should be diagonally dominant
        - JAX implementation uses scan for the forward/backward passes
        
    Reference:
        Lines 284-331 in MLMathToolsMod.F90
    """
    n = len(b)
    
    # Initialize arrays for intermediate calculations
    gam = jnp.zeros(n)
    u = jnp.zeros(n)
    
    # Forward elimination (Fortran lines 323-327)
    # First row: special case
    bet = b[0]
    u = u.at[0].set(r[0] / bet)
    
    # Forward elimination loop: j = 2 to n (Fortran indexing)
    # In Python: j = 1 to n-1
    def forward_step(carry, j):
        """Single step of forward elimination."""
        gam_arr, u_arr, bet_prev = carry
        
        # Calculate modified upper diagonal coefficient
        gam_j = c[j-1] / bet_prev
        gam_arr = gam_arr.at[j].set(gam_j)
        
        # Update main diagonal coefficient
        bet_j = b[j] - a[j] * gam_j
        
        # Calculate intermediate solution value
        u_j = (r[j] - a[j] * u_arr[j-1]) / bet_j
        u_arr = u_arr.at[j].set(u_j)
        
        return (gam_arr, u_arr, bet_j), None
    
    # Run forward elimination for indices 1 to n-1
    indices = jnp.arange(1, n)
    (gam, u, _), _ = jax.lax.scan(
        forward_step,
        (gam, u, bet),
        indices
    )
    
    # Backward substitution (Fortran lines 328-330)
    def backward_step(u_arr, j):
        """Single step of backward substitution."""
        u_j = u_arr[j] - gam[j+1] * u_arr[j+1]
        u_arr = u_arr.at[j].set(u_j)
        return u_arr, None
    
    # Run backward substitution for indices n-2 down to 0
    indices_backward = jnp.arange(n-2, -1, -1)
    u, _ = jax.lax.scan(backward_step, u, indices_backward)
    
    return u


def tridiag_2eq(
    a1: jnp.ndarray,
    b11: jnp.ndarray,
    b12: jnp.ndarray,
    c1: jnp.ndarray,
    d1: jnp.ndarray,
    a2: jnp.ndarray,
    b21: jnp.ndarray,
    b22: jnp.ndarray,
    c2: jnp.ndarray,
    d2: jnp.ndarray,
    n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve a tridiagonal system with two coupled equations per layer.
    
    Solves for air temperature and water vapor at each canopy layer using
    a modified Thomas algorithm that handles 2x2 coupling at each layer.
    
    The system is:
        a1(i)*T(i-1) + b11(i)*T(i) + b12(i)*q(i) + c1(i)*T(i+1) = d1(i)
        a2(i)*q(i-1) + b21(i)*T(i) + b22(i)*q(i) + c2(i)*q(i+1) = d2(i)
    
    The solution uses forward elimination followed by backward substitution,
    where at each step a 2x2 matrix inversion is performed to handle the
    coupling between the two variables.
    
    Key equations (lines 376-379):
        T(i) = f1(i) - e11(i)*T(i+1) - e12(i)*q(i+1)
        q(i) = f2(i) - e21(i)*T(i+1) - e22(i)*q(i+1)
    
    Args:
        a1: Lower diagonal coefficient for temperature equation [n_layers]
        b11: Main diagonal coefficient for T in temperature equation [n_layers]
        b12: Coupling coefficient for q in temperature equation [n_layers]
        c1: Upper diagonal coefficient for temperature equation [n_layers]
        d1: Right-hand side for temperature equation [n_layers]
        a2: Lower diagonal coefficient for vapor equation [n_layers]
        b21: Coupling coefficient for T in vapor equation [n_layers]
        b22: Main diagonal coefficient for q in vapor equation [n_layers]
        c2: Upper diagonal coefficient for vapor equation [n_layers]
        d2: Right-hand side for vapor equation [n_layers]
        n: Number of layers to solve (typically nlevmlcan)
        
    Returns:
        t: Air temperature solution [K] [n_layers]
        q: Water vapor mole fraction solution [mol/mol] [n_layers]
        
    Note:
        Arrays are sized for nlevmlcan but only first n elements are used.
        This follows the Fortran convention for flexible layer counts.
        
    Reference:
        Lines 334-437 in MLMathToolsMod.F90
    """
    nlevmlcan = a1.shape[0]
    
    # Initialize coefficient arrays (line 371-376)
    e11 = jnp.zeros(nlevmlcan + 1)
    e12 = jnp.zeros(nlevmlcan + 1)
    e21 = jnp.zeros(nlevmlcan + 1)
    e22 = jnp.zeros(nlevmlcan + 1)
    f1 = jnp.zeros(nlevmlcan + 1)
    f2 = jnp.zeros(nlevmlcan + 1)
    
    # Forward elimination (lines 378-410)
    def forward_step(i: int, carry: Tuple) -> Tuple:
        """Perform forward elimination for layer i.
        
        Args:
            i: Current layer index (1-based to match Fortran)
            carry: Tuple of (e11, e12, e21, e22, f1, f2) arrays
            
        Returns:
            Updated carry tuple with coefficients for layer i
        """
        e11_arr, e12_arr, e21_arr, e22_arr, f1_arr, f2_arr = carry
        
        # Calculate 2x2 matrix elements to invert (lines 389-393)
        ainv = b11[i-1] - a1[i-1] * e11_arr[i-1]
        binv = b12[i-1] - a1[i-1] * e12_arr[i-1]
        cinv = b21[i-1] - a2[i-1] * e21_arr[i-1]
        dinv = b22[i-1] - a2[i-1] * e22_arr[i-1]
        det = ainv * dinv - binv * cinv
        
        # E(i) = [B(i) - A(i)*E(i-1)]^(-1) * C(i) (lines 397-400)
        e11_i = dinv * c1[i-1] / det
        e12_i = -binv * c2[i-1] / det
        e21_i = -cinv * c1[i-1] / det
        e22_i = ainv * c2[i-1] / det
        
        # F(i) = [B(i) - A(i)*E(i-1)]^(-1) * [D(i) - A(i)*F(i-1)] (lines 404-405)
        f1_i = (dinv * (d1[i-1] - a1[i-1] * f1_arr[i-1]) - 
                binv * (d2[i-1] - a2[i-1] * f2_arr[i-1])) / det
        f2_i = (-cinv * (d1[i-1] - a1[i-1] * f1_arr[i-1]) + 
                ainv * (d2[i-1] - a2[i-1] * f2_arr[i-1])) / det
        
        # Update arrays at index i
        e11_arr = e11_arr.at[i].set(e11_i)
        e12_arr = e12_arr.at[i].set(e12_i)
        e21_arr = e21_arr.at[i].set(e21_i)
        e22_arr = e22_arr.at[i].set(e22_i)
        f1_arr = f1_arr.at[i].set(f1_i)
        f2_arr = f2_arr.at[i].set(f2_i)
        
        return (e11_arr, e12_arr, e21_arr, e22_arr, f1_arr, f2_arr)
    
    # Execute forward elimination
    carry_init = (e11, e12, e21, e22, f1, f2)
    e11, e12, e21, e22, f1, f2 = jax.lax.fori_loop(
        1, n + 1, forward_step, carry_init
    )
    
    # Initialize solution arrays
    t = jnp.zeros(nlevmlcan)
    q = jnp.zeros(nlevmlcan)
    
    # Solution for top layer (lines 414-416)
    t = t.at[n-1].set(f1[n])
    q = q.at[n-1].set(f2[n])
    
    # Backward substitution (lines 420-423)
    def backward_step(i: int, carry: Tuple) -> Tuple:
        """Perform backward substitution for layer i.
        
        Args:
            i: Current layer index (1-based, counting down from n-1 to 1)
            carry: Tuple of (t, q) arrays
            
        Returns:
            Updated carry tuple with solution at layer i
        """
        t_arr, q_arr = carry
        
        # T(i) = f1(i) - e11(i)*T(i+1) - e12(i)*q(i+1) (line 421)
        # q(i) = f2(i) - e21(i)*T(i+1) - e22(i)*q(i+1) (line 422)
        t_i = f1[i] - e11[i] * t_arr[i] - e12[i] * q_arr[i]
        q_i = f2[i] - e21[i] * t_arr[i] - e22[i] * q_arr[i]
        
        t_arr = t_arr.at[i-1].set(t_i)
        q_arr = q_arr.at[i-1].set(q_i)
        
        return (t_arr, q_arr)
    
    # Execute backward substitution from n-1 down to 1
    carry_init = (t, q)
    t, q = jax.lax.fori_loop(
        1, n, 
        lambda i, carry: backward_step(n - i, carry),
        carry_init
    )
    
    return t, q


# ============================================================================
# Special Functions
# ============================================================================

def log_gamma_function(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Calculate natural logarithm of gamma function.
    
    Computes ln(Γ(x)) using Lanczos approximation. This is numerically stable
    for x > 0 and avoids overflow issues that would occur with Γ(x) directly.
    
    The gamma function Γ(x) is defined as:
        Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
    
    This implementation returns ln(Γ(x)) to avoid overflow for large x.
    
    The implementation follows the algorithm from Numerical Recipes:
    1. Compute temporary value: tmp = (x + 0.5) * ln(x + 5.5) - (x + 5.5)
    2. Sum series expansion with 6 coefficients
    3. Return: tmp + ln(√(2π) * series / x)
    
    Args:
        x: Input argument [dimensionless] [scalar or array]
           Must be positive (x > 0)
        
    Returns:
        Natural logarithm of gamma function: ln(Γ(x)) [dimensionless]
        [same shape as input]
        
    Reference:
        Lines 440-471 from MLMathToolsMod.F90
        Numerical Recipes in Fortran 77, Section 6.1
    """
    # Lanczos coefficients (line 456-457)
    coef = jnp.array([
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-02,
        -0.5395239384953e-05
    ])
    
    # Constant: √(2π) (line 458)
    stp = 2.5066282746310005
    
    # Initialize working variable (line 461)
    y = x
    
    # Compute temporary value (lines 462-463)
    tmp = x + 5.5
    tmp = (x + 0.5) * jnp.log(tmp) - tmp
    
    # Initialize series sum (line 464)
    ser = 1.000000000190015
    
    # Accumulate series expansion (lines 465-468)
    for j in range(6):
        y = y + 1.0
        ser = ser + coef[j] / y
    
    # Final result (line 469)
    gammaln = tmp + jnp.log(stp * ser / x)
    
    return gammaln


def beta_function(
    a: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the beta function B(a,b).
    
    Uses the relationship between beta and gamma functions:
        B(a,b) = Γ(a) * Γ(b) / Γ(a+b)
    
    Computed in log space for numerical stability to avoid overflow/underflow:
        B(a,b) = exp(ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b)))
    
    The beta function appears in various statistical distributions and
    integration formulas used in canopy modeling.
    
    Args:
        a: First parameter of beta function [dimensionless] [...]
        b: Second parameter of beta function [dimensionless] [...]
        
    Returns:
        Value of beta function B(a,b) [dimensionless] [...]
        
    Note:
        This function is vectorized and works with arrays of any shape.
        
    Reference:
        Lines 474-492 in MLMathToolsMod.F90
    """
    # Line 490: beta = exp(log_gamma_function(a) + log_gamma_function(b) - log_gamma_function(a+b))
    beta = jnp.exp(
        log_gamma_function(a) + log_gamma_function(b) - log_gamma_function(a + b)
    )
    
    return beta


def beta_distribution_pdf(
    a: Union[float, jnp.ndarray],
    b: Union[float, jnp.ndarray],
    x: Union[float, jnp.ndarray],
) -> Union[float, jnp.ndarray]:
    """Calculate the beta distribution probability density function.
    
    Computes f(x; a, b) = (1 / B(a,b)) * x^(a-1) * (1-x)^(b-1)
    
    This function evaluates the PDF of the beta distribution with shape
    parameters a and b at the point x. The beta distribution is defined
    on the interval [0, 1] and is commonly used to model random variables
    that are constrained to this interval, such as probabilities or proportions.
    
    Args:
        a: First shape parameter (a > 0) [scalar or array]
        b: Second shape parameter (b > 0) [scalar or array]
        x: Point at which to evaluate PDF, must be in [0, 1] [scalar or array]
        
    Returns:
        Value of beta distribution PDF at x [same shape as inputs]
        
    Example:
        >>> # Symmetric beta distribution (a=b=2)
        >>> pdf_val = beta_distribution_pdf(2.0, 2.0, 0.5)
        >>> # Should give 1.5 for symmetric case at midpoint
        
    Reference:
        Lines 495-514 in MLMathToolsMod.F90
    """
    # Line 513: beta_pdf = (1._r8 / beta_function(a,b)) * x**(a-1._r8) * (1._r8 - x)**(b-1._r8)
    beta_pdf = (1.0 / beta_function(a, b)) * jnp.power(x, a - 1.0) * jnp.power(1.0 - x, b - 1.0)
    
    return beta_pdf


def beta_distribution_cdf(
    a: float,
    b: float,
    x: float,
) -> float:
    """Calculate the beta distribution cumulative distribution function.
    
    Computes F(x;a,b) = I_x(a,b), the regularized incomplete beta function,
    which represents the CDF of the beta distribution with shape parameters
    a and b evaluated at x.
    
    The beta distribution CDF is defined as:
        F(x;a,b) = I_x(a,b) = B_x(a,b) / B(a,b)
    
    Where:
        - B_x(a,b) is the incomplete beta function
        - B(a,b) is the complete beta function
        - a, b are shape parameters (a > 0, b > 0)
        - x is in the interval [0, 1]
    
    The implementation uses a continued fraction representation for numerical
    stability, choosing between I_x(a,b) and 1 - I_(1-x)(b,a) based on which
    converges faster.
    
    Args:
        a: First shape parameter (a > 0)
        b: Second shape parameter (b > 0)
        x: Evaluation point in [0, 1]
        
    Returns:
        Beta distribution CDF value F(x;a,b) in [0, 1]
        
    Note:
        - Uses symmetry relation: I_x(a,b) = 1 - I_(1-x)(b,a)
        - Chooses computation method based on x < (a+1)/(a+b+2) for stability
        - Special cases: F(0;a,b) = 0, F(1;a,b) = 1
        
    Reference:
        Lines 517-548 in MLMathToolsMod.F90
    """
    # Compute the normalization factor bt
    # bt = exp(ln(Γ(a+b)) - ln(Γ(a)) - ln(Γ(b)) + a*ln(x) + b*ln(1-x))
    # Special case: bt = 0 when x = 0 or x = 1 (lines 537-543)
    bt = jnp.where(
        (x == 0.0) | (x == 1.0),
        0.0,
        jnp.exp(
            log_gamma_function(a + b)
            - log_gamma_function(a)
            - log_gamma_function(b)
            + a * jnp.log(x)
            + b * jnp.log(1.0 - x)
        ),
    )
    
    # Choose computation method based on x value for numerical stability
    # If x < (a+1)/(a+b+2), use direct computation
    # Otherwise, use symmetry relation (lines 544-547)
    beta_cdf = jnp.where(
        x < (a + 1.0) / (a + b + 2.0),
        bt * beta_function_incomplete_cf(a, b, x) / a,
        1.0 - bt * beta_function_incomplete_cf(b, a, 1.0 - x) / b,
    )
    
    return beta_cdf


def beta_function_incomplete_cf(
    a: jnp.ndarray,
    b: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate continued fraction for incomplete beta function.
    
    Computes the continued fraction representation of the incomplete beta
    function I_x(a,b) using modified Lentz's method. This is used as part
    of the incomplete beta function calculation.
    
    The continued fraction provides an efficient way to compute the incomplete
    beta function for certain parameter ranges. The algorithm iterates until
    convergence (relative change < eps) or maximum iterations is reached.
    
    Key algorithm:
        - Initialize c=1, d=1-qab*x/qap, h=d
        - For each iteration m:
            - Compute even term: aa = m*(b-m)*x/((qam+2m)*(a+2m))
            - Update d = 1 + aa*d, c = 1 + aa/c, h = h*d*c
            - Compute odd term: aa = -(a+m)*(qab+m)*x/((qap+2m)*(a+2m))
            - Update d = 1 + aa*d, c = 1 + aa/c, h = h*d*c
            - Check convergence: |d*c - 1| < eps
    
    Args:
        a: First shape parameter [dimensionless] [...]
        b: Second shape parameter [dimensionless] [...]
        x: Evaluation point, must be in [0,1] [dimensionless] [...]
        
    Returns:
        Continued fraction value [dimensionless] [...]
        
    Note:
        - Uses fixed iteration limit (maxit=100) and convergence tolerance (eps=3e-7)
        - fpmin=1e-30 prevents division by zero
        - Original code calls endrun on non-convergence; JAX version returns
          the last computed value (convergence should be checked externally if needed)
        
    Reference:
        Lines 551-607 in MLMathToolsMod.F90
        Press et al., Numerical Recipes
    """
    # Constants (lines 574-576)
    maxit = 100
    eps = 3.0e-7
    fpmin = 1.0e-30
    
    # Initialize variables (lines 579-582)
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = jnp.ones_like(a)
    d = 1.0 - qab * x / qap
    
    # Ensure d is not too small (lines 583-584)
    d = jnp.where(jnp.abs(d) < fpmin, fpmin, d)
    d = 1.0 / d
    h = d
    
    # Define the loop body for a single iteration
    def body_fn(carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], 
                m: int) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        """Single iteration of continued fraction evaluation.
        
        Args:
            carry: Tuple of (c, d, h, converged_mask)
            m: Iteration number (1-indexed)
            
        Returns:
            Updated carry and None (no output to collect)
        """
        c_prev, d_prev, h_prev, converged = carry
        
        m_float = float(m)
        m2 = 2 * m
        m2_float = float(m2)
        
        # Even term (lines 587-594)
        aa = m_float * (b - m_float) * x / ((qam + m2_float) * (a + m2_float))
        d = 1.0 + aa * d_prev
        d = jnp.where(jnp.abs(d) < fpmin, fpmin, d)
        c = 1.0 + aa / c_prev
        c = jnp.where(jnp.abs(c) < fpmin, fpmin, c)
        d = 1.0 / d
        h = h_prev * d * c
        
        # Odd term (lines 595-601)
        aa = -(a + m_float) * (qab + m_float) * x / ((qap + m2_float) * (a + m2_float))
        d = 1.0 + aa * d
        d = jnp.where(jnp.abs(d) < fpmin, fpmin, d)
        c = 1.0 + aa / c
        c = jnp.where(jnp.abs(c) < fpmin, fpmin, c)
        d = 1.0 / d
        del_val = d * c
        h = h * del_val
        
        # Check convergence (lines 602-605)
        new_converged = converged | (jnp.abs(del_val - 1.0) < eps)
        h = jnp.where(converged, h_prev, h)
        
        return (c, d, h, new_converged), None
    
    # Run the loop (lines 585-606)
    initial_carry = (c, d, h, jnp.zeros_like(a, dtype=bool))
    (c_final, d_final, h_final, converged_final), _ = jax.lax.scan(
        body_fn,
        initial_carry,
        jnp.arange(1, maxit + 1)
    )
    
    # Return the final value (line 604)
    betacf = h_final
    
    return betacf