"""
Comprehensive pytest suite for MLMathToolsMod module.

This test suite covers:
- Root finding algorithms (hybrid_root_finder, zbrent)
- Quadratic equation solver
- Tridiagonal matrix solvers (single and coupled systems)
- Beta functions and distributions
- Log gamma function
- Edge cases, numerical stability, and physical constraints
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple, Union
from functools import partial

# Import the module under test
# Assuming the module is available as:
# from multilayer_canopy.MLMathToolsMod import (
#     hybrid_root_finder,
#     hybrid_root_finder_with_brent,
#     zbrent,
#     quadratic,
#     tridiag,
#     tridiag_2eq,
#     log_gamma_function,
#     beta_function,
#     beta_distribution_pdf,
#     beta_distribution_cdf,
#     beta_function_incomplete_cf,
# )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tolerance_config():
    """Standard tolerance configuration for numerical comparisons."""
    return {
        'atol': 1e-6,
        'rtol': 1e-6,
        'strict_atol': 1e-10,
        'strict_rtol': 1e-10,
    }


@pytest.fixture
def simple_quadratic_func():
    """Simple quadratic function: f(x) = x^2 - 4."""
    return lambda x: x**2 - 4.0


@pytest.fixture
def transcendental_func():
    """Transcendental function: f(x) = exp(x) - 3x."""
    return lambda x: jnp.exp(x) - 3.0 * x


@pytest.fixture
def cubic_polynomial_func():
    """Cubic polynomial: f(x) = (x-1)(x-2)(x-3)."""
    return lambda x: (x - 1.0) * (x - 2.0) * (x - 3.0)


@pytest.fixture
def linear_func():
    """Simple linear function: f(x) = x."""
    return lambda x: x


@pytest.fixture
def no_root_func():
    """Function with no real roots: f(x) = x^2 + 1."""
    return lambda x: x**2 + 1.0


@pytest.fixture
def sin_func():
    """Sine function shifted: f(x) = sin(x) - 0.5."""
    return lambda x: jnp.sin(x) - 0.5


@pytest.fixture
def cubic_func():
    """Cubic function: f(x) = x^3 - x - 2."""
    return lambda x: x**3 - x - 2.0


# ============================================================================
# Test: hybrid_root_finder
# ============================================================================

class TestHybridRootFinder:
    """Test suite for hybrid_root_finder function."""

    @pytest.mark.parametrize("xa,xb,expected_root,tol", [
        (jnp.array([1.0, 1.5, 1.8]), jnp.array([3.0, 2.5, 2.2]), 
         jnp.array([2.0, 2.0, 2.0]), 1e-6),
    ])
    def test_simple_quadratic(self, simple_quadratic_func, xa, xb, expected_root, 
                             tol, tolerance_config):
        """Test hybrid root finder on simple quadratic with known root at x=2."""
        from multilayer_canopy.MLMathToolsMod import hybrid_root_finder
        
        result = hybrid_root_finder(simple_quadratic_func, xa, xb, tol, itmax=40)
        
        # Check shape
        assert result.shape == expected_root.shape, \
            f"Expected shape {expected_root.shape}, got {result.shape}"
        
        # Check values
        np.testing.assert_allclose(
            result, expected_root, 
            atol=1e-5, rtol=1e-5,
            err_msg="Root values do not match expected for simple quadratic"
        )
        
        # Verify root is actually a root
        func_at_root = simple_quadratic_func(result)
        np.testing.assert_allclose(
            func_at_root, jnp.zeros_like(func_at_root),
            atol=tol * 10, rtol=tol * 10,
            err_msg="Function value at root is not close to zero"
        )

    @pytest.mark.parametrize("xa,xb,expected_root,tol", [
        (jnp.array([0.5, 1.0, 1.2]), jnp.array([2.0, 2.5, 2.0]), 
         jnp.array([1.512134, 1.512134, 1.512134]), 1e-7),
    ])
    def test_transcendental(self, transcendental_func, xa, xb, expected_root, 
                           tol, tolerance_config):
        """Test hybrid root finder on transcendental equation exp(x) = 3x."""
        from multilayer_canopy.MLMathToolsMod import hybrid_root_finder
        
        result = hybrid_root_finder(transcendental_func, xa, xb, tol, itmax=50)
        
        np.testing.assert_allclose(
            result, expected_root,
            atol=1e-5, rtol=1e-5,
            err_msg="Root values do not match for transcendental equation"
        )

    @pytest.mark.parametrize("xa,xb,expected_root,tol", [
        (jnp.array([0.3, 0.4, 0.45]), jnp.array([0.7, 0.6, 0.55]),
         jnp.array([0.5236, 0.5236, 0.5236]), 1e-12),
    ])
    def test_tight_tolerance(self, sin_func, xa, xb, expected_root, tol):
        """Test hybrid root finder with very tight tolerance requirements."""
        from multilayer_canopy.MLMathToolsMod import hybrid_root_finder
        
        result = hybrid_root_finder(sin_func, xa, xb, tol, itmax=100)
        
        np.testing.assert_allclose(
            result, expected_root,
            atol=1e-10, rtol=1e-10,
            err_msg="High precision root finding failed"
        )

    @pytest.mark.parametrize("xa,xb,tol", [
        (jnp.array([1.0, 1.2, 1.3, 1.4, 1.45]), 
         jnp.array([2.0, 1.8, 1.7, 1.6, 1.55]), 1e-8),
    ])
    def test_vectorized_operation(self, cubic_func, xa, xb, tol):
        """Test vectorized root finding with multiple initial brackets."""
        from multilayer_canopy.MLMathToolsMod import hybrid_root_finder
        
        result = hybrid_root_finder(cubic_func, xa, xb, tol, itmax=60)
        expected_root = jnp.array([1.5214, 1.5214, 1.5214, 1.5214, 1.5214])
        
        # All should converge to same root
        np.testing.assert_allclose(
            result, expected_root,
            atol=1e-6, rtol=1e-6,
            err_msg="Vectorized root finding failed to converge consistently"
        )

    def test_output_dtype(self, simple_quadratic_func):
        """Test that output has correct dtype."""
        from multilayer_canopy.MLMathToolsMod import hybrid_root_finder
        
        xa = jnp.array([1.0, 1.5])
        xb = jnp.array([3.0, 2.5])
        result = hybrid_root_finder(simple_quadratic_func, xa, xb, 1e-6)
        
        assert result.dtype == jnp.float32 or result.dtype == jnp.float64, \
            f"Expected float dtype, got {result.dtype}"


# ============================================================================
# Test: zbrent
# ============================================================================

class TestZbrent:
    """Test suite for zbrent (Brent's method) root finder."""

    @pytest.mark.parametrize("xa,xb,expected_roots", [
        (jnp.array([0.5, 1.5, 2.5]), jnp.array([1.5, 2.5, 3.5]),
         jnp.array([1.0, 2.0, 3.0])),
    ])
    def test_polynomial_roots(self, cubic_polynomial_func, xa, xb, expected_roots):
        """Test zbrent on cubic polynomial with three known roots."""
        from multilayer_canopy.MLMathToolsMod import zbrent
        
        root, converged, error = zbrent(cubic_polynomial_func, xa, xb, 
                                       tol=1e-8, itmax=50, eps=1e-8)
        
        # Check shapes
        assert root.shape == expected_roots.shape
        assert converged.shape == expected_roots.shape
        assert error.shape == expected_roots.shape
        
        # Check convergence
        assert jnp.all(converged), "Not all roots converged"
        assert jnp.all(~error), "Bracketing errors occurred"
        
        # Check root values
        np.testing.assert_allclose(
            root, expected_roots,
            atol=1e-7, rtol=1e-7,
            err_msg="Root values do not match expected"
        )

    @pytest.mark.parametrize("xa,xb,expected_root", [
        (jnp.array([-1.0, -0.5, -0.1]), jnp.array([1.0, 0.5, 0.1]),
         jnp.array([0.0, 0.0, 0.0])),
    ])
    def test_zero_crossing(self, linear_func, xa, xb, expected_root):
        """Test zbrent with root exactly at zero."""
        from multilayer_canopy.MLMathToolsMod import zbrent
        
        root, converged, error = zbrent(linear_func, xa, xb, 
                                       tol=1e-10, itmax=50, eps=1e-10)
        
        assert jnp.all(converged), "Failed to converge to zero root"
        assert jnp.all(~error), "Unexpected bracketing errors"
        
        np.testing.assert_allclose(
            root, expected_root,
            atol=1e-9, rtol=1e-9,
            err_msg="Zero root not found accurately"
        )

    @pytest.mark.parametrize("xa,xb", [
        (jnp.array([0.0, -1.0, -2.0]), jnp.array([1.0, 1.0, 2.0])),
    ])
    def test_no_bracket_error(self, no_root_func, xa, xb):
        """Test zbrent error handling when function has no real roots."""
        from multilayer_canopy.MLMathToolsMod import zbrent
        
        root, converged, error = zbrent(no_root_func, xa, xb, 
                                       tol=1e-6, itmax=50, eps=1e-8)
        
        # Should detect bracketing error
        assert jnp.all(error), "Failed to detect bracketing error for no-root function"
        assert jnp.all(~converged), "Incorrectly reported convergence"

    def test_return_tuple_structure(self, simple_quadratic_func):
        """Test that zbrent returns proper tuple structure."""
        from multilayer_canopy.MLMathToolsMod import zbrent
        
        xa = jnp.array([1.0])
        xb = jnp.array([3.0])
        result = zbrent(simple_quadratic_func, xa, xb, tol=1e-6)
        
        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 3, "Result should have 3 elements"
        
        root, converged, error = result
        assert isinstance(root, jnp.ndarray)
        assert isinstance(converged, jnp.ndarray)
        assert isinstance(error, jnp.ndarray)


# ============================================================================
# Test: quadratic
# ============================================================================

class TestQuadratic:
    """Test suite for quadratic equation solver."""

    @pytest.mark.parametrize("a,b,c,expected_r1,expected_r2", [
        (jnp.array([1.0, 2.0, 1.0, -1.0]), 
         jnp.array([-3.0, -8.0, -4.0, 2.0]),
         jnp.array([2.0, 6.0, 4.0, 1.0]),
         jnp.array([2.0, 3.0, 2.0, 1.0]),
         jnp.array([1.0, 1.0, 2.0, -1.0])),
    ])
    def test_standard_cases(self, a, b, c, expected_r1, expected_r2):
        """Test quadratic solver on standard cases with real roots."""
        from multilayer_canopy.MLMathToolsMod import quadratic
        
        r1, r2 = quadratic(a, b, c)
        
        # Check shapes
        assert r1.shape == expected_r1.shape
        assert r2.shape == expected_r2.shape
        
        # Check values (order may vary)
        np.testing.assert_allclose(
            r1, expected_r1,
            atol=1e-6, rtol=1e-6,
            err_msg="First root does not match expected"
        )
        np.testing.assert_allclose(
            r2, expected_r2,
            atol=1e-6, rtol=1e-6,
            err_msg="Second root does not match expected"
        )
        
        # Verify roots satisfy equation: a*r^2 + b*r + c = 0
        residual1 = a * r1**2 + b * r1 + c
        residual2 = a * r2**2 + b * r2 + c
        np.testing.assert_allclose(
            residual1, jnp.zeros_like(residual1),
            atol=1e-5, rtol=1e-5,
            err_msg="First root does not satisfy quadratic equation"
        )
        np.testing.assert_allclose(
            residual2, jnp.zeros_like(residual2),
            atol=1e-5, rtol=1e-5,
            err_msg="Second root does not satisfy quadratic equation"
        )

    @pytest.mark.parametrize("a,b,c,expected_r1,expected_r2", [
        (jnp.array([1.0, 1.0, 0.001]),
         jnp.array([0.0, -2.0, 0.0]),
         jnp.array([-4.0, 1.0, -1e-6]),
         jnp.array([2.0, 1.0, 0.001]),
         jnp.array([-2.0, 1.0, -0.001])),
    ])
    def test_edge_cases(self, a, b, c, expected_r1, expected_r2):
        """Test edge cases: zero b, repeated root, small coefficients."""
        from multilayer_canopy.MLMathToolsMod import quadratic
        
        r1, r2 = quadratic(a, b, c)
        
        np.testing.assert_allclose(
            r1, expected_r1,
            atol=1e-5, rtol=1e-5,
            err_msg="Edge case first root incorrect"
        )
        np.testing.assert_allclose(
            r2, expected_r2,
            atol=1e-5, rtol=1e-5,
            err_msg="Edge case second root incorrect"
        )

    @pytest.mark.parametrize("a,b,c", [
        (jnp.array([1.0, 1.0, 1.0]),
         jnp.array([-2.0, -2.000001, -1.999999]),
         jnp.array([1.0, 1.0, 1.0])),
    ])
    def test_near_zero_discriminant(self, a, b, c):
        """Test numerical stability with near-repeated roots."""
        from multilayer_canopy.MLMathToolsMod import quadratic
        
        r1, r2 = quadratic(a, b, c)
        
        # Roots should be very close to each other
        expected_r1 = jnp.array([1.0, 1.0000005, 0.9999995])
        expected_r2 = jnp.array([1.0, 1.0000005, 0.9999995])
        
        np.testing.assert_allclose(
            r1, expected_r1,
            atol=1e-6, rtol=1e-6,
            err_msg="Near-zero discriminant handling failed"
        )

    def test_return_tuple_structure(self):
        """Test that quadratic returns proper tuple of two arrays."""
        from multilayer_canopy.MLMathToolsMod import quadratic
        
        a = jnp.array([1.0])
        b = jnp.array([-3.0])
        c = jnp.array([2.0])
        
        result = quadratic(a, b, c)
        
        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should have 2 elements"
        
        r1, r2 = result
        assert isinstance(r1, jnp.ndarray)
        assert isinstance(r2, jnp.ndarray)


# ============================================================================
# Test: tridiag
# ============================================================================

class TestTridiag:
    """Test suite for tridiagonal matrix solver."""

    def test_simple_system(self):
        """Test tridiagonal solver on simple 4x4 system."""
        from multilayer_canopy.MLMathToolsMod import tridiag
        
        a = jnp.array([0.0, 1.0, 1.0, 1.0])
        b = jnp.array([2.0, 2.0, 2.0, 2.0])
        c = jnp.array([1.0, 1.0, 1.0, 0.0])
        r = jnp.array([1.0, 2.0, 3.0, 4.0])
        
        solution = tridiag(a, b, c, r)
        expected = jnp.array([-1.0, 2.0, 0.0, 2.0])
        
        # Check shape
        assert solution.shape == expected.shape
        
        # Check values
        np.testing.assert_allclose(
            solution, expected,
            atol=1e-6, rtol=1e-6,
            err_msg="Tridiagonal solution incorrect"
        )
        
        # Verify solution satisfies system
        # b[0]*x[0] + c[0]*x[1] = r[0]
        # a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = r[i]
        # a[n-1]*x[n-2] + b[n-1]*x[n-1] = r[n-1]
        residual = jnp.zeros_like(r)
        n = len(r)
        residual = residual.at[0].set(b[0]*solution[0] + c[0]*solution[1] - r[0])
        for i in range(1, n-1):
            residual = residual.at[i].set(
                a[i]*solution[i-1] + b[i]*solution[i] + c[i]*solution[i+1] - r[i]
            )
        residual = residual.at[n-1].set(
            a[n-1]*solution[n-2] + b[n-1]*solution[n-1] - r[n-1]
        )
        
        np.testing.assert_allclose(
            residual, jnp.zeros_like(residual),
            atol=1e-5, rtol=1e-5,
            err_msg="Solution does not satisfy tridiagonal system"
        )

    def test_diagonal_dominant(self):
        """Test strongly diagonally dominant system for numerical stability."""
        from multilayer_canopy.MLMathToolsMod import tridiag
        
        a = jnp.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.1])
        b = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        c = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
        r = jnp.array([10.1, 10.2, 10.2, 10.2, 10.2, 10.1])
        
        solution = tridiag(a, b, c, r)
        expected = jnp.ones(6)
        
        np.testing.assert_allclose(
            solution, expected,
            atol=1e-8, rtol=1e-8,
            err_msg="Diagonal dominant system solution incorrect"
        )

    def test_output_shape(self):
        """Test that output shape matches input."""
        from multilayer_canopy.MLMathToolsMod import tridiag
        
        n = 5
        a = jnp.ones(n)
        b = jnp.ones(n) * 2
        c = jnp.ones(n)
        r = jnp.ones(n)
        
        solution = tridiag(a, b, c, r)
        
        assert solution.shape == (n,), f"Expected shape ({n},), got {solution.shape}"


# ============================================================================
# Test: tridiag_2eq
# ============================================================================

class TestTridiag2Eq:
    """Test suite for coupled tridiagonal system solver."""

    def test_coupled_system(self):
        """Test coupled temperature-vapor system."""
        from multilayer_canopy.MLMathToolsMod import tridiag_2eq
        
        n = 5
        a1 = jnp.array([0.0, -0.1, -0.1, -0.1, -0.1])
        b11 = jnp.array([1.0, 1.2, 1.2, 1.2, 1.0])
        b12 = jnp.array([0.05, 0.05, 0.05, 0.05, 0.05])
        c1 = jnp.array([-0.1, -0.1, -0.1, -0.1, 0.0])
        d1 = jnp.array([300.0, 295.0, 290.0, 285.0, 280.0])
        
        a2 = jnp.array([0.0, -0.05, -0.05, -0.05, -0.05])
        b21 = jnp.array([0.02, 0.02, 0.02, 0.02, 0.02])
        b22 = jnp.array([1.0, 1.1, 1.1, 1.1, 1.0])
        c2 = jnp.array([-0.05, -0.05, -0.05, -0.05, 0.0])
        d2 = jnp.array([0.01, 0.012, 0.014, 0.016, 0.018])
        
        t, q = tridiag_2eq(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)
        
        # Check shapes
        assert t.shape == (n,), f"Temperature shape incorrect: {t.shape}"
        assert q.shape == (n,), f"Vapor shape incorrect: {q.shape}"
        
        # Check physical constraints
        assert jnp.all(t > 0), "Temperatures should be positive (Kelvin)"
        assert jnp.all(q >= 0), "Vapor fractions should be non-negative"
        assert jnp.all(q <= 1), "Vapor fractions should be <= 1"
        
        # Approximate expected values
        expected_t = jnp.array([298.0, 293.0, 288.0, 283.0, 278.0])
        expected_q = jnp.array([0.0095, 0.0108, 0.0125, 0.0143, 0.0165])
        
        np.testing.assert_allclose(
            t, expected_t,
            atol=0.001, rtol=0.001,
            err_msg="Temperature solution incorrect"
        )
        np.testing.assert_allclose(
            q, expected_q,
            atol=0.001, rtol=0.001,
            err_msg="Vapor solution incorrect"
        )

    def test_atmospheric_profile(self):
        """Test realistic atmospheric temperature-moisture profile."""
        from multilayer_canopy.MLMathToolsMod import tridiag_2eq
        
        n = 4
        a1 = jnp.array([0.0, -0.05, -0.05, -0.05])
        b11 = jnp.array([1.1, 1.15, 1.15, 1.1])
        b12 = jnp.array([0.01, 0.01, 0.01, 0.01])
        c1 = jnp.array([-0.05, -0.05, -0.05, 0.0])
        d1 = jnp.array([288.15, 285.0, 282.0, 279.0])
        
        a2 = jnp.array([0.0, -0.02, -0.02, -0.02])
        b21 = jnp.array([0.005, 0.005, 0.005, 0.005])
        b22 = jnp.array([1.05, 1.08, 1.08, 1.05])
        c2 = jnp.array([-0.02, -0.02, -0.02, 0.0])
        d2 = jnp.array([0.008, 0.01, 0.012, 0.014])
        
        t, q = tridiag_2eq(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)
        
        # Check physical realism
        assert jnp.all(t > 273.15), "Temperatures should be above freezing"
        assert jnp.all(t < 320.0), "Temperatures should be reasonable"
        assert jnp.all(q > 0), "Vapor should be positive"
        assert jnp.all(q < 0.05), "Vapor fractions should be realistic"

    def test_return_tuple_structure(self):
        """Test that tridiag_2eq returns proper tuple structure."""
        from multilayer_canopy.MLMathToolsMod import tridiag_2eq
        
        n = 3
        a1 = jnp.zeros(n)
        b11 = jnp.ones(n)
        b12 = jnp.ones(n) * 0.1
        c1 = jnp.zeros(n)
        d1 = jnp.ones(n) * 300
        
        a2 = jnp.zeros(n)
        b21 = jnp.ones(n) * 0.1
        b22 = jnp.ones(n)
        c2 = jnp.zeros(n)
        d2 = jnp.ones(n) * 0.01
        
        result = tridiag_2eq(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)
        
        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should have 2 elements"
        
        t, q = result
        assert isinstance(t, jnp.ndarray)
        assert isinstance(q, jnp.ndarray)


# ============================================================================
# Test: log_gamma_function
# ============================================================================

class TestLogGammaFunction:
    """Test suite for log gamma function."""

    @pytest.mark.parametrize("x,expected", [
        (1.0, 0.0),  # Gamma(1) = 1, log(1) = 0
        (2.0, 0.0),  # Gamma(2) = 1, log(1) = 0
        (0.5, 0.5723),  # Gamma(0.5) = sqrt(pi)
        (10.0, 12.8018),
        (100.0, 359.1342),
        (1e-6, 13.8155),
    ])
    def test_special_values(self, x, expected):
        """Test log gamma at special points and extreme values."""
        from multilayer_canopy.MLMathToolsMod import log_gamma_function
        
        result = log_gamma_function(x)
        
        np.testing.assert_allclose(
            result, expected,
            atol=0.001, rtol=0.001,
            err_msg=f"Log gamma incorrect at x={x}"
        )

    def test_array_input(self):
        """Test log gamma with array input."""
        from multilayer_canopy.MLMathToolsMod import log_gamma_function
        
        x = jnp.array([1.0, 2.0, 0.5, 10.0])
        expected = jnp.array([0.0, 0.0, 0.5723, 12.8018])
        
        result = log_gamma_function(x)
        
        assert result.shape == x.shape
        np.testing.assert_allclose(
            result, expected,
            atol=0.001, rtol=0.001,
            err_msg="Log gamma array computation incorrect"
        )

    def test_positive_constraint(self):
        """Test that log gamma is only defined for positive x."""
        from multilayer_canopy.MLMathToolsMod import log_gamma_function
        
        # Very small positive should work
        result = log_gamma_function(1e-10)
        assert jnp.isfinite(result), "Log gamma should be finite for small positive x"


# ============================================================================
# Test: beta_function
# ============================================================================

class TestBetaFunction:
    """Test suite for beta function."""

    @pytest.mark.parametrize("a,b,expected", [
        (jnp.array([2.0, 3.0, 0.5, 5.0]),
         jnp.array([3.0, 2.0, 0.5, 5.0]),
         jnp.array([0.08333, 0.08333, 3.14159, 0.00317])),
    ])
    def test_standard_values(self, a, b, expected):
        """Test beta function at standard parameter values."""
        from multilayer_canopy.MLMathToolsMod import beta_function
        
        result = beta_function(a, b)
        
        assert result.shape == expected.shape
        np.testing.assert_allclose(
            result, expected,
            atol=0.0001, rtol=0.0001,
            err_msg="Beta function values incorrect"
        )

    def test_symmetry(self):
        """Test that beta function is symmetric: B(a,b) = B(b,a)."""
        from multilayer_canopy.MLMathToolsMod import beta_function
        
        a = jnp.array([2.0, 3.0, 5.0])
        b = jnp.array([3.0, 5.0, 2.0])
        
        result_ab = beta_function(a, b)
        result_ba = beta_function(b, a)
        
        np.testing.assert_allclose(
            result_ab, result_ba,
            atol=1e-10, rtol=1e-10,
            err_msg="Beta function not symmetric"
        )

    def test_positive_constraint(self):
        """Test that beta function requires positive parameters."""
        from multilayer_canopy.MLMathToolsMod import beta_function
        
        # Small positive values should work
        a = jnp.array([0.1, 0.01])
        b = jnp.array([0.1, 0.01])
        result = beta_function(a, b)
        
        assert jnp.all(jnp.isfinite(result)), "Beta should be finite for positive params"
        assert jnp.all(result > 0), "Beta should be positive"


# ============================================================================
# Test: beta_distribution_pdf
# ============================================================================

class TestBetaDistributionPDF:
    """Test suite for beta distribution PDF."""

    @pytest.mark.parametrize("a,b,x,expected", [
        (2.0, 5.0, 0.3, 2.1609),
        (5.0, 2.0, 0.7, 2.1609),
        (0.5, 0.5, 0.5, 0.6366),
        (2.0, 2.0, 0.0, 0.0),
    ])
    def test_pdf_values(self, a, b, x, expected):
        """Test beta distribution PDF at various points."""
        from multilayer_canopy.MLMathToolsMod import beta_distribution_pdf
        
        result = beta_distribution_pdf(a, b, x)
        
        np.testing.assert_allclose(
            result, expected,
            atol=0.001, rtol=0.001,
            err_msg=f"Beta PDF incorrect at a={a}, b={b}, x={x}"
        )

    def test_pdf_array_input(self):
        """Test beta PDF with array inputs."""
        from multilayer_canopy.MLMathToolsMod import beta_distribution_pdf
        
        a = jnp.array([2.0, 5.0, 0.5, 2.0])
        b = jnp.array([5.0, 2.0, 0.5, 2.0])
        x = jnp.array([0.3, 0.7, 0.5, 0.0])
        
        result = beta_distribution_pdf(a, b, x)
        
        assert result.shape == a.shape
        assert jnp.all(result >= 0), "PDF should be non-negative"

    def test_pdf_boundary_behavior(self):
        """Test PDF behavior at boundaries x=0 and x=1."""
        from multilayer_canopy.MLMathToolsMod import beta_distribution_pdf
        
        # At x=0 with a>1, PDF should be 0
        result = beta_distribution_pdf(2.0, 2.0, 0.0)
        assert result == 0.0, "PDF should be 0 at x=0 when a>1"
        
        # At x=1 with b>1, PDF should be 0
        result = beta_distribution_pdf(2.0, 2.0, 1.0)
        assert result == 0.0, "PDF should be 0 at x=1 when b>1"


# ============================================================================
# Test: beta_distribution_cdf
# ============================================================================

class TestBetaDistributionCDF:
    """Test suite for beta distribution CDF."""

    def test_cdf_lower_boundary(self):
        """Test that CDF at x=0 is exactly 0."""
        from multilayer_canopy.MLMathToolsMod import beta_distribution_cdf
        
        result = beta_distribution_cdf(2.0, 3.0, 0.0)
        
        np.testing.assert_allclose(
            result, 0.0,
            atol=1e-10, rtol=1e-10,
            err_msg="CDF should be 0 at x=0"
        )

    def test_cdf_upper_boundary(self):
        """Test that CDF at x=1 is exactly 1."""
        from multilayer_canopy.MLMathToolsMod import beta_distribution_cdf
        
        result = beta_distribution_cdf(2.0, 3.0, 1.0)
        
        np.testing.assert_allclose(
            result, 1.0,
            atol=1e-10, rtol=1e-10,
            err_msg="CDF should be 1 at x=1"
        )

    def test_cdf_monotonicity(self):
        """Test that CDF is monotonically increasing."""
        from multilayer_canopy.MLMathToolsMod import beta_distribution_cdf
        
        a, b = 2.0, 3.0
        x_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        cdf_values = [beta_distribution_cdf(a, b, x) for x in x_values]
        
        # Check monotonicity
        for i in range(len(cdf_values) - 1):
            assert cdf_values[i] <= cdf_values[i+1], \
                f"CDF not monotonic: {cdf_values[i]} > {cdf_values[i+1]}"

    def test_cdf_range(self):
        """Test that CDF values are in [0, 1]."""
        from multilayer_canopy.MLMathToolsMod import beta_distribution_cdf
        
        a, b = 2.0, 3.0
        x_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for x in x_values:
            result = beta_distribution_cdf(a, b, x)
            assert 0.0 <= result <= 1.0, \
                f"CDF value {result} out of range [0,1] at x={x}"


# ============================================================================
# Test: beta_function_incomplete_cf
# ============================================================================

class TestBetaFunctionIncompleteCF:
    """Test suite for incomplete beta function continued fraction."""

    @pytest.mark.parametrize("a,b,x,expected", [
        (jnp.array([2.0, 5.0, 0.5]),
         jnp.array([3.0, 2.0, 0.5]),
         jnp.array([0.5, 0.3, 0.9]),
         jnp.array([0.6875, 0.1631, 0.9])),
    ])
    def test_cf_convergence(self, a, b, x, expected):
        """Test continued fraction convergence at various parameter combinations."""
        from multilayer_canopy.MLMathToolsMod import beta_function_incomplete_cf
        
        result = beta_function_incomplete_cf(a, b, x)
        
        assert result.shape == expected.shape
        np.testing.assert_allclose(
            result, expected,
            atol=0.0001, rtol=0.0001,
            err_msg="Incomplete beta CF values incorrect"
        )

    def test_cf_range(self):
        """Test that CF values are in valid range."""
        from multilayer_canopy.MLMathToolsMod import beta_function_incomplete_cf
        
        a = jnp.array([2.0, 3.0, 5.0])
        b = jnp.array([3.0, 2.0, 5.0])
        x = jnp.array([0.3, 0.5, 0.7])
        
        result = beta_function_incomplete_cf(a, b, x)
        
        assert jnp.all(result >= 0), "CF values should be non-negative"
        assert jnp.all(result <= 1), "CF values should be <= 1"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_root_finder_with_quadratic(self):
        """Test that root finders work with quadratic solutions."""
        from multilayer_canopy.MLMathToolsMod import hybrid_root_finder, quadratic
        
        # Solve x^2 - 5x + 6 = 0, roots at x=2 and x=3
        a, b, c = 1.0, -5.0, 6.0
        r1, r2 = quadratic(jnp.array([a]), jnp.array([b]), jnp.array([c]))
        
        # Use root finder to verify
        func = lambda x: a * x**2 + b * x + c
        root = hybrid_root_finder(func, jnp.array([1.5]), jnp.array([2.5]), 1e-8)
        
        # Should find one of the roots
        assert jnp.abs(root[0] - r1[0]) < 1e-6 or jnp.abs(root[0] - r2[0]) < 1e-6

    def test_beta_pdf_integrates_to_one(self):
        """Test that beta PDF integrates to approximately 1."""
        from multilayer_canopy.MLMathToolsMod import beta_distribution_pdf
        
        a, b = 2.0, 3.0
        
        # Numerical integration using trapezoidal rule
        x = jnp.linspace(0, 1, 1000)
        pdf_values = jnp.array([beta_distribution_pdf(a, b, xi) for xi in x])
        integral = jnp.trapz(pdf_values, x)
        
        np.testing.assert_allclose(
            integral, 1.0,
            atol=0.01, rtol=0.01,
            err_msg="Beta PDF does not integrate to 1"
        )


# ============================================================================
# Edge Case and Error Handling Tests
# ============================================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_root_finder_max_iterations(self):
        """Test root finder behavior when max iterations reached."""
        from multilayer_canopy.MLMathToolsMod import hybrid_root_finder
        
        # Difficult function with very tight tolerance and few iterations
        func = lambda x: jnp.exp(x) - 10*x
        xa = jnp.array([1.0])
        xb = jnp.array([2.0])
        
        # Should still return a result even if not fully converged
        result = hybrid_root_finder(func, xa, xb, tol=1e-15, itmax=5)
        assert jnp.isfinite(result[0]), "Should return finite result"

    def test_tridiag_single_element(self):
        """Test tridiagonal solver with single element."""
        from multilayer_canopy.MLMathToolsMod import tridiag
        
        a = jnp.array([0.0])
        b = jnp.array([2.0])
        c = jnp.array([0.0])
        r = jnp.array([4.0])
        
        solution = tridiag(a, b, c, r)
        expected = jnp.array([2.0])
        
        np.testing.assert_allclose(solution, expected, atol=1e-10)

    def test_beta_extreme_parameters(self):
        """Test beta function with extreme parameter values."""
        from multilayer_canopy.MLMathToolsMod import beta_function
        
        # Very small parameters
        a = jnp.array([0.01])
        b = jnp.array([0.01])
        result = beta_function(a, b)
        
        assert jnp.isfinite(result[0]), "Beta should be finite for small params"
        assert result[0] > 0, "Beta should be positive"


# ============================================================================
# Documentation Tests
# ============================================================================

def test_module_imports():
    """Test that all required functions can be imported."""
    try:
        from multilayer_canopy.MLMathToolsMod import (
            hybrid_root_finder,
            zbrent,
            quadratic,
            tridiag,
            tridiag_2eq,
            log_gamma_function,
            beta_function,
            beta_distribution_pdf,
            beta_distribution_cdf,
            beta_function_incomplete_cf,
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required functions: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])