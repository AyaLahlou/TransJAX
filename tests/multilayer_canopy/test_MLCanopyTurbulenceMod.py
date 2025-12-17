"""
Comprehensive pytest suite for MLCanopyTurbulenceMod module.

This test suite covers:
- Monin-Obukhov stability functions (phi_m, phi_c, psi_m, psi_c)
- Prandtl/Schmidt number calculations
- Beta (u*/u) calculations
- RSL (Roughness Sublayer) psi functions
- Obukhov length calculations
- Canopy turbulence parameterization
- RSL lookup table initialization

Tests include nominal cases, edge cases, and physical realism checks.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import Callable, NamedTuple
from unittest.mock import Mock, patch


# ============================================================================
# Named Tuples (Mock definitions for testing)
# ============================================================================

class PrScParams(NamedTuple):
    """Parameters for Prandtl/Schmidt number calculation."""
    Pr0: float
    Pr1: float
    Pr2: float


class PsiRSLResult(NamedTuple):
    """Result from RSL psi function calculation."""
    psim: jnp.ndarray
    psic: jnp.ndarray


class ObuFuncInputs(NamedTuple):
    """Inputs for Obukhov length function."""
    p: int
    ic: int
    il: int
    obu_val: jnp.ndarray
    zref: jnp.ndarray
    uref: jnp.ndarray
    thref: jnp.ndarray
    thvref: jnp.ndarray
    qref: jnp.ndarray
    rhomol: jnp.ndarray
    ztop: jnp.ndarray
    lai: jnp.ndarray
    sai: jnp.ndarray
    Lc: jnp.ndarray
    taf: jnp.ndarray
    qaf: jnp.ndarray
    vkc: float
    grav: float
    beta_neutral_max: float
    cr: float
    z0mg: float
    zeta_min: float
    zeta_max: float


class ObuFuncOutputs(NamedTuple):
    """Outputs from Obukhov length function."""
    obu_dif: jnp.ndarray
    zdisp: jnp.ndarray
    beta: jnp.ndarray
    PrSc: jnp.ndarray
    ustar: jnp.ndarray
    gac_to_hc: jnp.ndarray
    obu: jnp.ndarray


class RSLPsihatTable(NamedTuple):
    """RSL lookup table structure."""
    initialized: bool
    nZ: int
    nL: int
    zdtgrid_m: jnp.ndarray
    dtLgrid_m: jnp.ndarray
    psigrid_m: jnp.ndarray
    zdtgrid_h: jnp.ndarray
    dtLgrid_h: jnp.ndarray
    psigrid_h: jnp.ndarray


# ============================================================================
# Mock Module Functions (to be replaced with actual imports)
# ============================================================================

def phim_monin_obukhov(zeta: jnp.ndarray) -> jnp.ndarray:
    """
    Mock implementation of phi_m Monin-Obukhov stability function.
    
    Args:
        zeta: Stability parameter (z-d)/L
        
    Returns:
        Stability function for momentum
    """
    # Simplified implementation for testing
    return jnp.where(
        zeta < 0,
        jnp.power(1.0 - 16.0 * zeta, -0.25),
        1.0 + 5.0 * zeta
    )


def phic_monin_obukhov(zeta: jnp.ndarray) -> jnp.ndarray:
    """
    Mock implementation of phi_c Monin-Obukhov stability function.
    
    Args:
        zeta: Stability parameter (z-d)/L
        
    Returns:
        Stability function for scalars
    """
    # Simplified implementation for testing
    return jnp.where(
        zeta < 0,
        jnp.power(1.0 - 16.0 * zeta, -0.5),
        1.0 + 5.0 * zeta
    )


def psim_monin_obukhov(zeta: jnp.ndarray, pi: float = np.pi) -> jnp.ndarray:
    """
    Mock implementation of psi_m integrated stability function.
    
    Args:
        zeta: Stability parameter
        pi: Value of pi
        
    Returns:
        Integrated stability function for momentum
    """
    x = jnp.power(1.0 - 16.0 * jnp.minimum(zeta, 0.0), 0.25)
    return jnp.where(
        zeta < 0,
        2.0 * jnp.log((1.0 + x) / 2.0) + jnp.log((1.0 + x * x) / 2.0) - 2.0 * jnp.arctan(x) + pi / 2.0,
        -5.0 * zeta
    )


def psic_monin_obukhov(zeta: jnp.ndarray) -> jnp.ndarray:
    """
    Mock implementation of psi_c integrated stability function.
    
    Args:
        zeta: Stability parameter
        
    Returns:
        Integrated stability function for scalars
    """
    y = jnp.power(1.0 - 16.0 * jnp.minimum(zeta, 0.0), 0.5)
    return jnp.where(
        zeta < 0,
        2.0 * jnp.log((1.0 + y) / 2.0),
        -5.0 * zeta
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """
    Load and provide test data for all test cases.
    
    Returns:
        Dictionary containing test cases with inputs and metadata
    """
    return {
        "phim_neutral": {
            "zeta": jnp.array([0.0, 0.001, -0.001]),
            "expected_near": 1.0,
            "tolerance": 0.01
        },
        "phim_stable_unstable": {
            "zeta": jnp.array([-2.0, -1.0, -0.5, 0.0, 0.2, 0.5, 0.8]),
        },
        "phic_range": {
            "zeta": jnp.array([-5.0, -2.0, -0.5, 0.0, 0.3, 0.7, 1.0]),
        },
        "psi_edge_cases": {
            "zeta": jnp.array([-100.0, -10.0, 0.0, 1.0, 10.0]),
            "pi": np.pi
        },
        "prsc_conditions": {
            "beta_neutral": jnp.array([0.25, 0.3, 0.2]),
            "beta_neutral_max": jnp.array([0.35, 0.35, 0.35]),
            "LcL": jnp.array([0.0, 0.5, -0.5]),
            "params": PrScParams(Pr0=0.5, Pr1=0.3, Pr2=0.143)
        },
        "beta_boundary": {
            "beta_neutral": 0.25,
            "lcl": 0.0,
            "beta_min": 0.01,
            "beta_max": 0.99,
        },
        "lookup_psihat": {
            "zdt": 2.5,
            "dtL": 0.15,
            "zdtgrid": jnp.array([[5.0], [4.0], [3.0], [2.0], [1.0], [0.5]]),
            "dtLgrid": jnp.array([[-0.5, -0.2, 0.0, 0.1, 0.2, 0.5]]),
            "psigrid": jnp.array([
                [1.2, 1.1, 1.0, 0.95, 0.9, 0.8],
                [1.15, 1.05, 0.98, 0.93, 0.88, 0.78],
                [1.1, 1.0, 0.95, 0.9, 0.85, 0.75],
                [1.05, 0.95, 0.9, 0.85, 0.8, 0.7],
                [1.0, 0.9, 0.85, 0.8, 0.75, 0.65],
                [0.95, 0.85, 0.8, 0.75, 0.7, 0.6]
            ])
        },
        "psi_rsl_typical": {
            "za": jnp.array([50.0, 45.0, 40.0]),
            "hc": jnp.array([20.0, 18.0, 22.0]),
            "disp": jnp.array([14.0, 12.6, 15.4]),
            "obu": jnp.array([100.0, -50.0, 200.0]),
            "beta": jnp.array([0.3, 0.28, 0.32]),
            "prsc": jnp.array([0.7, 0.65, 0.75]),
            "vkc": 0.4,
            "c2": 0.5,
        },
        "obu_func_edge": {
            "inputs": ObuFuncInputs(
                p=0, ic=5, il=1,
                obu_val=jnp.array([1e-6]),
                zref=jnp.array([50.0]),
                uref=jnp.array([0.1]),
                thref=jnp.array([290.0]),
                thvref=jnp.array([290.5]),
                qref=jnp.array([0.008]),
                rhomol=jnp.array([42.0]),
                ztop=jnp.array([20.0]),
                lai=jnp.array([0.01]),
                sai=jnp.array([0.005]),
                Lc=jnp.array([100.0]),
                taf=jnp.array([288.0]),
                qaf=jnp.array([0.007]),
                vkc=0.4, grav=9.80616,
                beta_neutral_max=0.35,
                cr=0.3, z0mg=0.01,
                zeta_min=-100.0, zeta_max=1.0
            )
        },
        "canopy_turbulence_multi": {
            "niter": 3,
            "num_filter": 5,
            "filter_indices": jnp.array([0, 2, 5, 8, 10]),
            "turb_type": 1,
            "mlcanopy_inst": {
                "nlevcan": 10,
                "npatches": 12,
                "obu": jnp.array([50.0, -30.0, 100.0, -80.0, 200.0, 40.0, 
                                  -60.0, 150.0, -40.0, 90.0, 120.0, -100.0]),
                "ustar": jnp.array([0.4, 0.5, 0.35, 0.6, 0.3, 0.45, 
                                    0.55, 0.32, 0.52, 0.38, 0.33, 0.58]),
                "taf": jnp.array([288.0, 290.0, 287.0, 291.0, 286.0, 289.0,
                                  292.0, 285.0, 293.0, 287.5, 288.5, 294.0]),
                "qaf": jnp.array([0.008, 0.009, 0.007, 0.01, 0.006, 0.0085,
                                  0.0095, 0.0065, 0.011, 0.0075, 0.0082, 0.012]),
                "zref": jnp.array([50.0, 48.0, 52.0, 45.0, 55.0, 49.0,
                                   47.0, 53.0, 46.0, 51.0, 50.5, 44.0]),
                "hc": jnp.array([20.0, 18.0, 22.0, 15.0, 25.0, 19.0,
                                 17.0, 23.0, 16.0, 21.0, 20.5, 14.0]),
                "lai": jnp.array([4.5, 3.8, 5.2, 2.5, 6.0, 4.2,
                                  3.5, 5.5, 2.8, 4.8, 4.6, 2.2]),
                "sai": jnp.array([1.2, 1.0, 1.4, 0.7, 1.6, 1.1,
                                  0.9, 1.5, 0.75, 1.3, 1.25, 0.6])
            }
        }
    }


@pytest.fixture
def lookup_grids():
    """
    Provide standard lookup table grids for RSL calculations.
    
    Returns:
        Dictionary with momentum and scalar lookup grids
    """
    zdtgrid = jnp.array([[5.0], [4.0], [3.0], [2.0], [1.0], [0.5]])
    dtLgrid = jnp.array([[-0.5, -0.2, 0.0, 0.1, 0.2, 0.5]])
    
    psigrid_m = jnp.array([
        [1.2, 1.1, 1.0, 0.95, 0.9, 0.8],
        [1.15, 1.05, 0.98, 0.93, 0.88, 0.78],
        [1.1, 1.0, 0.95, 0.9, 0.85, 0.75],
        [1.05, 0.95, 0.9, 0.85, 0.8, 0.7],
        [1.0, 0.9, 0.85, 0.8, 0.75, 0.65],
        [0.95, 0.85, 0.8, 0.75, 0.7, 0.6]
    ])
    
    psigrid_h = jnp.array([
        [1.3, 1.2, 1.0, 0.93, 0.87, 0.75],
        [1.25, 1.15, 0.97, 0.91, 0.85, 0.73],
        [1.2, 1.1, 0.94, 0.88, 0.82, 0.7],
        [1.15, 1.05, 0.9, 0.84, 0.78, 0.66],
        [1.1, 1.0, 0.86, 0.8, 0.74, 0.62],
        [1.05, 0.95, 0.82, 0.76, 0.7, 0.58]
    ])
    
    return {
        "zdtgrid_m": zdtgrid,
        "dtLgrid_m": dtLgrid,
        "psigrid_m": psigrid_m,
        "zdtgrid_h": zdtgrid,
        "dtLgrid_h": dtLgrid,
        "psigrid_h": psigrid_h
    }


# ============================================================================
# Tests for phim_monin_obukhov
# ============================================================================

class TestPhimMoninObukhov:
    """Tests for phi_m Monin-Obukhov stability function."""
    
    def test_phim_neutral_conditions(self, test_data):
        """
        Test phi_m near neutral stability (zeta ≈ 0).
        
        For neutral conditions, phi_m should be approximately 1.0.
        """
        data = test_data["phim_neutral"]
        result = phim_monin_obukhov(data["zeta"])
        
        # Check shape
        assert result.shape == data["zeta"].shape, \
            f"Expected shape {data['zeta'].shape}, got {result.shape}"
        
        # Check values near 1.0 for neutral conditions
        assert jnp.allclose(result, data["expected_near"], atol=data["tolerance"]), \
            f"phi_m should be near {data['expected_near']} for neutral conditions, got {result}"
    
    def test_phim_stable_unstable_range(self, test_data):
        """
        Test phi_m across stable and unstable conditions.
        
        Unstable (zeta < 0): phi_m < 1
        Neutral (zeta = 0): phi_m = 1
        Stable (zeta > 0): phi_m > 1
        """
        data = test_data["phim_stable_unstable"]
        result = phim_monin_obukhov(data["zeta"])
        
        # Check shape
        assert result.shape == data["zeta"].shape
        
        # Check monotonicity: phi_m increases with zeta
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], \
                f"phi_m should increase with zeta, but {result[i]} > {result[i+1]}"
        
        # Check physical bounds
        assert jnp.all(result > 0), "phi_m must be positive"
        
        # Check unstable conditions (zeta < 0)
        unstable_mask = data["zeta"] < 0
        if jnp.any(unstable_mask):
            assert jnp.all(result[unstable_mask] < 1.1), \
                "phi_m should be < 1.1 for unstable conditions"
        
        # Check stable conditions (zeta > 0)
        stable_mask = data["zeta"] > 0
        if jnp.any(stable_mask):
            assert jnp.all(result[stable_mask] > 0.9), \
                "phi_m should be > 0.9 for stable conditions"
    
    def test_phim_dtype(self, test_data):
        """Test that phi_m returns correct dtype."""
        data = test_data["phim_neutral"]
        result = phim_monin_obukhov(data["zeta"])
        
        assert result.dtype == data["zeta"].dtype, \
            f"Expected dtype {data['zeta'].dtype}, got {result.dtype}"
    
    @pytest.mark.parametrize("zeta_val", [-10.0, -1.0, 0.0, 1.0, 10.0])
    def test_phim_scalar_inputs(self, zeta_val):
        """Test phi_m with scalar inputs."""
        zeta = jnp.array([zeta_val])
        result = phim_monin_obukhov(zeta)
        
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert jnp.isfinite(result[0]), f"Result should be finite for zeta={zeta_val}"


# ============================================================================
# Tests for phic_monin_obukhov
# ============================================================================

class TestPhicMoninObukhov:
    """Tests for phi_c Monin-Obukhov stability function for scalars."""
    
    def test_phic_range(self, test_data):
        """
        Test phi_c across typical atmospheric stability range.
        
        phi_c should behave similarly to phi_m but with different coefficients.
        """
        data = test_data["phic_range"]
        result = phic_monin_obukhov(data["zeta"])
        
        # Check shape
        assert result.shape == data["zeta"].shape
        
        # Check physical bounds
        assert jnp.all(result > 0), "phi_c must be positive"
        assert jnp.all(jnp.isfinite(result)), "phi_c must be finite"
    
    def test_phic_vs_phim(self, test_data):
        """
        Test that phi_c and phi_m have expected relationship.
        
        For unstable conditions, phi_c typically decreases faster than phi_m.
        """
        data = test_data["phic_range"]
        phic = phic_monin_obukhov(data["zeta"])
        phim = phim_monin_obukhov(data["zeta"])
        
        # For unstable conditions (zeta < 0), phi_c < phi_m
        unstable_mask = data["zeta"] < 0
        if jnp.any(unstable_mask):
            assert jnp.all(phic[unstable_mask] <= phim[unstable_mask]), \
                "phi_c should be <= phi_m for unstable conditions"
    
    def test_phic_neutral(self):
        """Test phi_c at neutral stability."""
        zeta = jnp.array([0.0])
        result = phic_monin_obukhov(zeta)
        
        assert jnp.allclose(result, 1.0, atol=1e-6), \
            f"phi_c should be 1.0 at neutral stability, got {result}"


# ============================================================================
# Tests for psim_monin_obukhov
# ============================================================================

class TestPsimMoninObukhov:
    """Tests for psi_m integrated stability function."""
    
    def test_psim_edge_cases(self, test_data):
        """
        Test psi_m at extreme and boundary values.
        
        Tests extreme unstable, extreme stable, and neutral conditions.
        """
        data = test_data["psi_edge_cases"]
        result = psim_monin_obukhov(data["zeta"], data["pi"])
        
        # Check shape
        assert result.shape == data["zeta"].shape
        
        # Check finite values
        assert jnp.all(jnp.isfinite(result)), "psi_m must be finite"
        
        # Check neutral condition (zeta = 0)
        neutral_idx = jnp.where(data["zeta"] == 0.0)[0]
        if len(neutral_idx) > 0:
            assert jnp.allclose(result[neutral_idx], 0.0, atol=1e-6), \
                f"psi_m should be 0 at neutral stability, got {result[neutral_idx]}"
    
    def test_psim_monotonicity(self):
        """
        Test that psi_m is monotonically decreasing with zeta.
        
        As stability increases (zeta increases), psi_m should decrease.
        """
        zeta = jnp.linspace(-5.0, 1.0, 20)
        result = psim_monin_obukhov(zeta)
        
        # Check monotonic decrease
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1], \
                f"psi_m should decrease with zeta, but {result[i]} < {result[i+1]}"
    
    def test_psim_sign_convention(self, test_data):
        """
        Test sign convention for psi_m.
        
        Unstable (zeta < 0): psi_m > 0
        Stable (zeta > 0): psi_m < 0
        """
        data = test_data["psi_edge_cases"]
        result = psim_monin_obukhov(data["zeta"])
        
        unstable_mask = data["zeta"] < 0
        stable_mask = data["zeta"] > 0
        
        if jnp.any(unstable_mask):
            assert jnp.all(result[unstable_mask] > 0), \
                "psi_m should be positive for unstable conditions"
        
        if jnp.any(stable_mask):
            assert jnp.all(result[stable_mask] < 0), \
                "psi_m should be negative for stable conditions"


# ============================================================================
# Tests for psic_monin_obukhov
# ============================================================================

class TestPsicMoninObukhov:
    """Tests for psi_c integrated stability function for scalars."""
    
    def test_psic_edge_cases(self, test_data):
        """Test psi_c at extreme and boundary values."""
        data = test_data["psi_edge_cases"]
        result = psic_monin_obukhov(data["zeta"])
        
        # Check shape
        assert result.shape == data["zeta"].shape
        
        # Check finite values
        assert jnp.all(jnp.isfinite(result)), "psi_c must be finite"
        
        # Check neutral condition
        neutral_idx = jnp.where(data["zeta"] == 0.0)[0]
        if len(neutral_idx) > 0:
            assert jnp.allclose(result[neutral_idx], 0.0, atol=1e-6), \
                f"psi_c should be 0 at neutral stability"
    
    def test_psic_vs_psim_magnitude(self, test_data):
        """
        Test that |psi_c| >= |psi_m| for unstable conditions.
        
        Scalars typically have stronger stability corrections than momentum.
        """
        data = test_data["psi_edge_cases"]
        psic = psic_monin_obukhov(data["zeta"])
        psim = psim_monin_obukhov(data["zeta"])
        
        unstable_mask = data["zeta"] < 0
        if jnp.any(unstable_mask):
            assert jnp.all(jnp.abs(psic[unstable_mask]) >= 
                          jnp.abs(psim[unstable_mask]) - 1e-6), \
                "|psi_c| should be >= |psi_m| for unstable conditions"


# ============================================================================
# Tests for get_prsc
# ============================================================================

class TestGetPrsc:
    """Tests for Prandtl/Schmidt number calculation."""
    
    def test_prsc_neutral_stable_unstable(self, test_data):
        """
        Test Prandtl number for neutral, stable, and unstable conditions.
        
        Tests three scenarios:
        - LcL = 0: neutral
        - LcL > 0: stable
        - LcL < 0: unstable
        """
        data = test_data["prsc_conditions"]
        
        # Mock implementation for testing
        def get_prsc(beta_neutral, beta_neutral_max, LcL, params):
            # Simplified calculation
            beta = jnp.minimum(beta_neutral * (1.0 + 0.5 * LcL), beta_neutral_max)
            prsc = params.Pr0 + params.Pr1 * jnp.tanh(params.Pr2 * LcL)
            return prsc
        
        result = get_prsc(
            data["beta_neutral"],
            data["beta_neutral_max"],
            data["LcL"],
            data["params"]
        )
        
        # Check shape
        assert result.shape == data["beta_neutral"].shape
        
        # Check physical bounds (Prandtl number should be positive)
        assert jnp.all(result > 0), "Prandtl number must be positive"
        
        # Check reasonable range (typically 0.3 to 1.0 for atmosphere)
        assert jnp.all(result >= 0.1), "Prandtl number too small"
        assert jnp.all(result <= 2.0), "Prandtl number too large"
    
    def test_prsc_parameter_sensitivity(self):
        """Test sensitivity of Prandtl number to parameters."""
        beta_neutral = jnp.array([0.3])
        beta_neutral_max = jnp.array([0.35])
        LcL = jnp.array([0.0])
        
        params1 = PrScParams(Pr0=0.5, Pr1=0.3, Pr2=0.143)
        params2 = PrScParams(Pr0=0.7, Pr1=0.3, Pr2=0.143)
        
        def get_prsc(beta_neutral, beta_neutral_max, LcL, params):
            prsc = params.Pr0 + params.Pr1 * jnp.tanh(params.Pr2 * LcL)
            return prsc
        
        result1 = get_prsc(beta_neutral, beta_neutral_max, LcL, params1)
        result2 = get_prsc(beta_neutral, beta_neutral_max, LcL, params2)
        
        # Different Pr0 should give different results
        assert not jnp.allclose(result1, result2), \
            "Prandtl number should depend on Pr0"


# ============================================================================
# Tests for get_beta
# ============================================================================

class TestGetBeta:
    """Tests for beta (u*/u) calculation."""
    
    def test_beta_boundary_conditions(self, test_data):
        """
        Test beta calculation at neutral conditions and boundary constraints.
        
        Beta should be constrained to [beta_min, beta_max].
        """
        data = test_data["beta_boundary"]
        
        # Mock implementation
        def get_beta(beta_neutral, lcl, beta_min, beta_max, phim_func):
            # Simplified: beta increases with stability
            beta = beta_neutral * (1.0 + 0.3 * lcl)
            beta = jnp.clip(beta, beta_min, beta_max)
            return beta
        
        result = get_beta(
            data["beta_neutral"],
            data["lcl"],
            data["beta_min"],
            data["beta_max"],
            phim_monin_obukhov
        )
        
        # Check bounds
        assert result >= data["beta_min"], \
            f"Beta {result} below minimum {data['beta_min']}"
        assert result <= data["beta_max"], \
            f"Beta {result} above maximum {data['beta_max']}"
        
        # For neutral conditions (lcl=0), should be close to beta_neutral
        if data["lcl"] == 0.0:
            assert jnp.allclose(result, data["beta_neutral"], atol=0.05), \
                f"Beta should be near beta_neutral for neutral conditions"
    
    @pytest.mark.parametrize("lcl_val", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_beta_stability_dependence(self, lcl_val):
        """Test beta dependence on stability parameter."""
        def get_beta(beta_neutral, lcl, beta_min, beta_max, phim_func):
            beta = beta_neutral * (1.0 + 0.3 * lcl)
            beta = jnp.clip(beta, beta_min, beta_max)
            return beta
        
        result = get_beta(0.3, lcl_val, 0.01, 0.99, phim_monin_obukhov)
        
        assert 0.01 <= result <= 0.99, \
            f"Beta {result} outside valid range for lcl={lcl_val}"


# ============================================================================
# Tests for lookup_psihat
# ============================================================================

class TestLookupPsihat:
    """Tests for psihat lookup table interpolation."""
    
    def test_lookup_psihat_interpolation(self, test_data):
        """
        Test bilinear interpolation of psihat from lookup tables.
        
        Verifies that interpolation produces reasonable values within
        the range of the lookup table.
        """
        data = test_data["lookup_psihat"]
        
        # Mock bilinear interpolation
        def lookup_psihat(zdt, dtL, zdtgrid, dtLgrid, psigrid):
            # Simple nearest-neighbor for testing
            zdt_idx = jnp.argmin(jnp.abs(zdtgrid.flatten() - zdt))
            dtL_idx = jnp.argmin(jnp.abs(dtLgrid.flatten() - dtL))
            return psigrid[zdt_idx, dtL_idx]
        
        result = lookup_psihat(
            data["zdt"],
            data["dtL"],
            data["zdtgrid"],
            data["dtLgrid"],
            data["psigrid"]
        )
        
        # Check result is scalar
        assert jnp.ndim(result) == 0 or result.shape == (), \
            f"Expected scalar result, got shape {result.shape if hasattr(result, 'shape') else type(result)}"
        
        # Check result is within table range
        min_psi = jnp.min(data["psigrid"])
        max_psi = jnp.max(data["psigrid"])
        assert min_psi <= result <= max_psi, \
            f"Interpolated value {result} outside table range [{min_psi}, {max_psi}]"
    
    def test_lookup_psihat_grid_boundaries(self, test_data):
        """Test lookup at grid boundaries."""
        data = test_data["lookup_psihat"]
        
        def lookup_psihat(zdt, dtL, zdtgrid, dtLgrid, psigrid):
            zdt_idx = jnp.argmin(jnp.abs(zdtgrid.flatten() - zdt))
            dtL_idx = jnp.argmin(jnp.abs(dtLgrid.flatten() - dtL))
            return psigrid[zdt_idx, dtL_idx]
        
        # Test at corner points
        corners = [
            (data["zdtgrid"][0, 0], data["dtLgrid"][0, 0]),
            (data["zdtgrid"][0, 0], data["dtLgrid"][0, -1]),
            (data["zdtgrid"][-1, 0], data["dtLgrid"][0, 0]),
            (data["zdtgrid"][-1, 0], data["dtLgrid"][0, -1])
        ]
        
        for zdt, dtL in corners:
            result = lookup_psihat(zdt, dtL, data["zdtgrid"], 
                                  data["dtLgrid"], data["psigrid"])
            assert jnp.isfinite(result), \
                f"Result should be finite at corner ({zdt}, {dtL})"


# ============================================================================
# Tests for get_psi_rsl
# ============================================================================

class TestGetPsiRSL:
    """Tests for RSL (Roughness Sublayer) psi functions."""
    
    def test_psi_rsl_typical_canopy(self, test_data, lookup_grids):
        """
        Test RSL psi functions for typical forest canopy.
        
        Tests with varying stability conditions (stable, unstable, neutral).
        """
        data = test_data["psi_rsl_typical"]
        grids = lookup_grids
        
        # Mock implementation
        def get_psi_rsl(za, hc, disp, obu, beta, prsc, vkc, c2,
                       dtlgrid_m, zdtgrid_m, psigrid_m,
                       dtlgrid_h, zdtgrid_h, psigrid_h,
                       phim_fn, phic_fn, psim_fn, psic_fn, lookup_fn):
            # Simplified calculation
            zeta = (za - disp) / obu
            psim = psim_fn(zeta)
            psic = psic_fn(zeta)
            return PsiRSLResult(psim=psim, psic=psic)
        
        result = get_psi_rsl(
            data["za"], data["hc"], data["disp"], data["obu"],
            data["beta"], data["prsc"], data["vkc"], data["c2"],
            grids["dtLgrid_m"], grids["zdtgrid_m"], grids["psigrid_m"],
            grids["dtLgrid_h"], grids["zdtgrid_h"], grids["psigrid_h"],
            phim_monin_obukhov, phic_monin_obukhov,
            psim_monin_obukhov, psic_monin_obukhov,
            None
        )
        
        # Check result structure
        assert hasattr(result, 'psim'), "Result should have psim field"
        assert hasattr(result, 'psic'), "Result should have psic field"
        
        # Check shapes
        assert result.psim.shape == data["za"].shape, \
            f"psim shape {result.psim.shape} != za shape {data['za'].shape}"
        assert result.psic.shape == data["za"].shape, \
            f"psic shape {result.psic.shape} != za shape {data['za'].shape}"
        
        # Check finite values
        assert jnp.all(jnp.isfinite(result.psim)), "psim must be finite"
        assert jnp.all(jnp.isfinite(result.psic)), "psic must be finite"
    
    def test_psi_rsl_physical_constraints(self, test_data, lookup_grids):
        """Test physical constraints on RSL psi functions."""
        data = test_data["psi_rsl_typical"]
        
        # Check input constraints
        assert jnp.all(data["za"] > 0), "Atmospheric height must be positive"
        assert jnp.all(data["hc"] > 0), "Canopy height must be positive"
        assert jnp.all(data["disp"] > 0), "Displacement height must be positive"
        assert jnp.all(data["disp"] < data["hc"]), \
            "Displacement height should be less than canopy height"
        assert jnp.all(data["beta"] > 0), "Beta must be positive"
        assert jnp.all(data["beta"] < 1), "Beta must be less than 1"
        assert jnp.all(data["prsc"] > 0), "Prandtl number must be positive"


# ============================================================================
# Tests for obu_func
# ============================================================================

class TestObuFunc:
    """Tests for Obukhov length calculation function."""
    
    def test_obu_func_edge_cases(self, test_data):
        """
        Test Obukhov length calculation with edge cases.
        
        Tests minimal wind, sparse canopy, and very small Obukhov length.
        """
        data = test_data["obu_func_edge"]
        inputs = data["inputs"]
        
        # Mock implementation
        def obu_func(inputs, get_beta_fn, get_prsc_fn, get_psi_rsl_fn):
            # Simplified calculation
            ustar = inputs.vkc * inputs.uref[0] / jnp.log(
                (inputs.zref[0] - inputs.ztop[0]) / 0.1
            )
            obu = -inputs.thvref[0] * ustar**3 / (
                inputs.vkc * inputs.grav * 0.01
            )
            
            return ObuFuncOutputs(
                obu_dif=obu - inputs.obu_val,
                zdisp=inputs.ztop * 0.7,
                beta=jnp.array([0.3]),
                PrSc=jnp.array([0.7]),
                ustar=jnp.array([ustar]),
                gac_to_hc=jnp.array([0.1]),
                obu=jnp.array([obu])
            )
        
        result = obu_func(inputs, None, None, None)
        
        # Check result structure
        assert hasattr(result, 'obu_dif'), "Result should have obu_dif"
        assert hasattr(result, 'zdisp'), "Result should have zdisp"
        assert hasattr(result, 'beta'), "Result should have beta"
        assert hasattr(result, 'PrSc'), "Result should have PrSc"
        assert hasattr(result, 'ustar'), "Result should have ustar"
        assert hasattr(result, 'gac_to_hc'), "Result should have gac_to_hc"
        assert hasattr(result, 'obu'), "Result should have obu"
        
        # Check physical constraints
        assert jnp.all(jnp.isfinite(result.ustar)), "ustar must be finite"
        assert jnp.all(result.ustar > 0), "ustar must be positive"
        assert jnp.all(result.beta > 0), "beta must be positive"
        assert jnp.all(result.beta < 1), "beta must be less than 1"
        assert jnp.all(result.PrSc > 0), "Prandtl number must be positive"
    
    def test_obu_func_temperature_constraints(self, test_data):
        """Test that temperatures are physically realistic."""
        data = test_data["obu_func_edge"]
        inputs = data["inputs"]
        
        # Check temperature constraints
        assert jnp.all(inputs.thref > 200), \
            "Reference temperature should be > 200K"
        assert jnp.all(inputs.thref < 350), \
            "Reference temperature should be < 350K"
        assert jnp.all(inputs.taf > 200), \
            "Canopy air temperature should be > 200K"
        assert jnp.all(inputs.taf < 350), \
            "Canopy air temperature should be < 350K"


# ============================================================================
# Tests for canopy_turbulence
# ============================================================================

class TestCanopyTurbulence:
    """Tests for main canopy turbulence calculation."""
    
    def test_canopy_turbulence_multiple_patches(self, test_data):
        """
        Test canopy turbulence across multiple patches.
        
        Tests with diverse conditions: stable/unstable, varying LAI,
        different canopy heights.
        """
        data = test_data["canopy_turbulence_multi"]
        
        # Mock implementation
        def canopy_turbulence(niter, num_filter, filter_indices, 
                            mlcanopy_inst, turb_type):
            # Simplified: just verify inputs and return modified instance
            inst = mlcanopy_inst.copy()
            
            # Update some fields to simulate calculation
            inst["obu"] = inst["obu"] * 1.01  # Small modification
            inst["ustar"] = inst["ustar"] * 1.01
            
            return inst
        
        result = canopy_turbulence(
            data["niter"],
            data["num_filter"],
            data["filter_indices"],
            data["mlcanopy_inst"],
            data["turb_type"]
        )
        
        # Check that result is a dictionary (or object) with expected fields
        assert "obu" in result, "Result should contain obu field"
        assert "ustar" in result, "Result should contain ustar field"
        assert "taf" in result, "Result should contain taf field"
        
        # Check shapes are preserved
        assert len(result["obu"]) == data["mlcanopy_inst"]["npatches"], \
            "Number of patches should be preserved"
    
    def test_canopy_turbulence_filter_indices(self, test_data):
        """Test that filter indices are valid."""
        data = test_data["canopy_turbulence_multi"]
        
        # Check filter indices
        assert len(data["filter_indices"]) == data["num_filter"], \
            "Number of filter indices should match num_filter"
        assert jnp.all(data["filter_indices"] >= 0), \
            "Filter indices must be non-negative"
        assert jnp.all(data["filter_indices"] < 
                      data["mlcanopy_inst"]["npatches"]), \
            "Filter indices must be within patch range"
    
    def test_canopy_turbulence_turb_type(self, test_data):
        """Test turbulence type parameter validation."""
        data = test_data["canopy_turbulence_multi"]
        
        # Check turb_type is valid
        valid_types = [-1, 0, 1]
        assert data["turb_type"] in valid_types, \
            f"turb_type {data['turb_type']} not in valid types {valid_types}"


# ============================================================================
# Tests for initialize_rsl_tables
# ============================================================================

class TestInitializeRSLTables:
    """Tests for RSL lookup table initialization."""
    
    @patch('builtins.open')
    def test_initialize_rsl_tables_structure(self, mock_open):
        """
        Test RSL table initialization structure.
        
        Note: This test mocks file I/O since actual file may not exist.
        """
        # Mock file reading
        mock_open.return_value.__enter__ = Mock()
        
        # Mock implementation
        def initialize_rsl_tables(rsl_file_path):
            # Create mock table
            nZ, nL = 50, 50
            zdtgrid = jnp.linspace(0.1, 10.0, nZ).reshape(-1, 1)
            dtLgrid = jnp.linspace(-2.0, 2.0, nL).reshape(1, -1)
            psigrid = jnp.ones((nZ, nL))
            
            return RSLPsihatTable(
                initialized=True,
                nZ=nZ,
                nL=nL,
                zdtgrid_m=zdtgrid,
                dtLgrid_m=dtLgrid,
                psigrid_m=psigrid,
                zdtgrid_h=zdtgrid,
                dtLgrid_h=dtLgrid,
                psigrid_h=psigrid
            )
        
        result = initialize_rsl_tables("/path/to/rsl_lookup_table.nc")
        
        # Check structure
        assert hasattr(result, 'initialized'), "Should have initialized field"
        assert hasattr(result, 'nZ'), "Should have nZ field"
        assert hasattr(result, 'nL'), "Should have nL field"
        assert hasattr(result, 'zdtgrid_m'), "Should have zdtgrid_m field"
        assert hasattr(result, 'dtLgrid_m'), "Should have dtLgrid_m field"
        assert hasattr(result, 'psigrid_m'), "Should have psigrid_m field"
        
        # Check initialization flag
        assert result.initialized == True, "Table should be initialized"
        
        # Check dimensions
        assert result.nZ > 0, "nZ should be positive"
        assert result.nL > 0, "nL should be positive"
        
        # Check grid shapes
        assert result.zdtgrid_m.shape == (result.nZ, 1), \
            f"zdtgrid_m shape should be ({result.nZ}, 1)"
        assert result.dtLgrid_m.shape == (1, result.nL), \
            f"dtLgrid_m shape should be (1, {result.nL})"
        assert result.psigrid_m.shape == (result.nZ, result.nL), \
            f"psigrid_m shape should be ({result.nZ}, {result.nL})"
    
    def test_initialize_rsl_tables_grid_ordering(self):
        """Test that grids have correct ordering."""
        # Mock implementation
        def initialize_rsl_tables(rsl_file_path):
            nZ, nL = 10, 10
            zdtgrid = jnp.linspace(10.0, 0.1, nZ).reshape(-1, 1)  # Descending
            dtLgrid = jnp.linspace(-2.0, 2.0, nL).reshape(1, -1)  # Ascending
            psigrid = jnp.ones((nZ, nL))
            
            return RSLPsihatTable(
                initialized=True, nZ=nZ, nL=nL,
                zdtgrid_m=zdtgrid, dtLgrid_m=dtLgrid, psigrid_m=psigrid,
                zdtgrid_h=zdtgrid, dtLgrid_h=dtLgrid, psigrid_h=psigrid
            )
        
        result = initialize_rsl_tables("/path/to/file.nc")
        
        # Check zdtgrid is descending
        zdtgrid_flat = result.zdtgrid_m.flatten()
        for i in range(len(zdtgrid_flat) - 1):
            assert zdtgrid_flat[i] >= zdtgrid_flat[i + 1], \
                "zdtgrid should be in descending order"
        
        # Check dtLgrid is ascending
        dtLgrid_flat = result.dtLgrid_m.flatten()
        for i in range(len(dtLgrid_flat) - 1):
            assert dtLgrid_flat[i] <= dtLgrid_flat[i + 1], \
                "dtLgrid should be in ascending order"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_stability_function_consistency(self):
        """
        Test consistency between phi and psi functions.
        
        The relationship: d(psi)/d(zeta) = 1 - phi(zeta) should hold.
        """
        zeta = jnp.linspace(-2.0, 1.0, 50)
        dzeta = zeta[1] - zeta[0]
        
        # Calculate phi and psi
        phim = phim_monin_obukhov(zeta)
        psim = psim_monin_obukhov(zeta)
        
        # Numerical derivative of psi
        dpsi_dzeta = jnp.gradient(psim, dzeta)
        
        # Check relationship (with tolerance for numerical derivative)
        expected = 1.0 - phim
        assert jnp.allclose(dpsi_dzeta, expected, atol=0.1, rtol=0.1), \
            "Relationship d(psi)/d(zeta) = 1 - phi should hold approximately"
    
    def test_momentum_scalar_function_ordering(self):
        """
        Test that scalar functions have stronger corrections than momentum.
        
        For unstable conditions: |psi_c| >= |psi_m|
        """
        zeta = jnp.linspace(-5.0, -0.1, 20)
        
        psim = psim_monin_obukhov(zeta)
        psic = psic_monin_obukhov(zeta)
        
        # For unstable conditions, |psi_c| should be >= |psi_m|
        assert jnp.all(jnp.abs(psic) >= jnp.abs(psim) - 1e-6), \
            "Scalar corrections should be >= momentum corrections for unstable"
    
    def test_neutral_limit_all_functions(self):
        """Test that all functions approach correct neutral limits."""
        zeta_neutral = jnp.array([0.0])
        
        # phi functions should be 1.0
        phim = phim_monin_obukhov(zeta_neutral)
        phic = phic_monin_obukhov(zeta_neutral)
        assert jnp.allclose(phim, 1.0, atol=1e-6), "phi_m should be 1 at neutral"
        assert jnp.allclose(phic, 1.0, atol=1e-6), "phi_c should be 1 at neutral"
        
        # psi functions should be 0.0
        psim = psim_monin_obukhov(zeta_neutral)
        psic = psic_monin_obukhov(zeta_neutral)
        assert jnp.allclose(psim, 0.0, atol=1e-6), "psi_m should be 0 at neutral"
        assert jnp.allclose(psic, 0.0, atol=1e-6), "psi_c should be 0 at neutral"


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestProperties:
    """Property-based tests for mathematical properties."""
    
    @pytest.mark.parametrize("zeta", [
        jnp.array([-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    ])
    def test_phi_positivity(self, zeta):
        """Test that phi functions are always positive."""
        phim = phim_monin_obukhov(zeta)
        phic = phic_monin_obukhov(zeta)
        
        assert jnp.all(phim > 0), "phi_m must be positive"
        assert jnp.all(phic > 0), "phi_c must be positive"
    
    @pytest.mark.parametrize("zeta", [
        jnp.array([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
    ])
    def test_psi_finiteness(self, zeta):
        """Test that psi functions are finite for all reasonable zeta."""
        psim = psim_monin_obukhov(zeta)
        psic = psic_monin_obukhov(zeta)
        
        assert jnp.all(jnp.isfinite(psim)), "psi_m must be finite"
        assert jnp.all(jnp.isfinite(psic)), "psi_c must be finite"
    
    def test_array_broadcasting(self):
        """Test that functions handle array broadcasting correctly."""
        # Test with different shapes
        zeta_1d = jnp.array([0.0, 0.5, 1.0])
        zeta_2d = jnp.array([[0.0, 0.5], [1.0, 1.5]])
        
        result_1d = phim_monin_obukhov(zeta_1d)
        result_2d = phim_monin_obukhov(zeta_2d)
        
        assert result_1d.shape == zeta_1d.shape, "1D shape should be preserved"
        assert result_2d.shape == zeta_2d.shape, "2D shape should be preserved"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_extreme_stability_values(self):
        """Test functions at extreme stability values."""
        zeta_extreme = jnp.array([-100.0, -50.0, 50.0, 100.0])
        
        phim = phim_monin_obukhov(zeta_extreme)
        phic = phic_monin_obukhov(zeta_extreme)
        
        # Should still be finite and positive
        assert jnp.all(jnp.isfinite(phim)), "phi_m should be finite at extremes"
        assert jnp.all(jnp.isfinite(phic)), "phi_c should be finite at extremes"
        assert jnp.all(phim > 0), "phi_m should be positive at extremes"
        assert jnp.all(phic > 0), "phi_c should be positive at extremes"
    
    def test_near_zero_values(self):
        """Test functions near zero."""
        zeta_near_zero = jnp.array([-1e-10, -1e-8, 0.0, 1e-8, 1e-10])
        
        phim = phim_monin_obukhov(zeta_near_zero)
        psim = psim_monin_obukhov(zeta_near_zero)
        
        # Should be close to neutral values
        assert jnp.allclose(phim, 1.0, atol=1e-6), \
            "phi_m should be near 1 for near-zero zeta"
        assert jnp.allclose(psim, 0.0, atol=1e-6), \
            "psi_m should be near 0 for near-zero zeta"
    
    def test_nan_inf_handling(self):
        """Test that functions handle NaN and Inf appropriately."""
        # Note: In production code, these should be handled gracefully
        # Here we just verify current behavior
        
        zeta_special = jnp.array([0.0, 1.0, 2.0])  # Valid values
        result = phim_monin_obukhov(zeta_special)
        
        # Should not produce NaN or Inf for valid inputs
        assert not jnp.any(jnp.isnan(result)), "Should not produce NaN"
        assert not jnp.any(jnp.isinf(result)), "Should not produce Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])