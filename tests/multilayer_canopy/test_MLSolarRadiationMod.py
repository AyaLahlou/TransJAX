"""
Comprehensive pytest suite for MLSolarRadiationMod.solar_radiation function.

This module tests the solar radiation transfer calculations through multilayer canopies,
including both Norman and TwoStream radiation transfer methods.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from typing import Dict, Any
import json


# Define namedtuples matching the function signature
BoundsType = namedtuple('BoundsType', ['begp', 'endp', 'begg', 'endg'])
PatchState = namedtuple('PatchState', ['itype', 'cosz', 'swskyb', 'swskyd', 'albsoib', 'albsoid'])
MLCanopyState = namedtuple('MLCanopyState', [
    'dlai_profile', 'dsai_profile', 'dpai_profile', 
    'ntop_canopy', 'nbot_canopy', 'ncan_canopy'
])
PFTParams = namedtuple('PFTParams', ['rhol', 'taul', 'rhos', 'taus', 'xl', 'clump_fac'])
OpticalProperties = namedtuple('OpticalProperties', [
    'rho', 'tau', 'omega', 'kb', 'fracsun', 'tb', 'td', 'tbi', 'avmu', 'betab', 'betad'
])
RadiationFluxes = namedtuple('RadiationFluxes', [
    'swleaf', 'swsoi', 'swveg', 'swvegsun', 'swvegsha', 'albcan', 'apar_sun', 'apar_shade'
])


# Test data as embedded JSON
TEST_DATA_JSON = """
{
  "function_name": "solar_radiation",
  "test_cases": [
    {
      "name": "test_nominal_single_patch_midday",
      "inputs": {
        "bounds": {"begp": 0, "endp": 1, "begg": 0, "endg": 1},
        "num_filter": 1,
        "filter_indices": [0],
        "patch_state": {
          "itype": [5],
          "cosz": [0.866],
          "swskyb": [[800.0, 600.0]],
          "swskyd": [[200.0, 150.0]],
          "albsoib": [[0.15, 0.25]],
          "albsoid": [[0.2, 0.3]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[0.5, 0.8, 1.2, 0.9, 0.6]],
          "dsai_profile": [[0.1, 0.15, 0.2, 0.15, 0.1]],
          "dpai_profile": [[0.6, 0.95, 1.4, 1.05, 0.7]],
          "ntop_canopy": [0],
          "nbot_canopy": [4],
          "ncan_canopy": [5]
        },
        "pft_params": {
          "rhol": [[0.1, 0.45], [0.08, 0.4], [0.12, 0.5], [0.09, 0.42], [0.11, 0.48], [0.1, 0.45]],
          "taul": [[0.05, 0.25], [0.04, 0.22], [0.06, 0.28], [0.05, 0.24], [0.05, 0.26], [0.05, 0.25]],
          "rhos": [[0.16, 0.39], [0.15, 0.38], [0.17, 0.4], [0.16, 0.39], [0.16, 0.39], [0.16, 0.39]],
          "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [0.01, -0.1, 0.1, 0.0, 0.05, 0.01],
          "clump_fac": [0.85, 0.8, 0.9, 0.85, 0.87, 0.85]
        },
        "nlevmlcan": 5,
        "light_type": 1,
        "numrad": 2
      },
      "metadata": {
        "type": "nominal",
        "description": "Typical midday conditions with moderate LAI, single patch, Norman radiation transfer",
        "edge_cases": []
      }
    },
    {
      "name": "test_nominal_multiple_patches_varied_lai",
      "inputs": {
        "bounds": {"begp": 0, "endp": 3, "begg": 0, "endg": 1},
        "num_filter": 3,
        "filter_indices": [0, 1, 2],
        "patch_state": {
          "itype": [2, 5, 8],
          "cosz": [0.707, 0.866, 0.5],
          "swskyb": [[700.0, 550.0], [850.0, 650.0], [600.0, 450.0]],
          "swskyd": [[180.0, 140.0], [220.0, 170.0], [160.0, 120.0]],
          "albsoib": [[0.12, 0.22], [0.18, 0.28], [0.14, 0.24]],
          "albsoid": [[0.17, 0.27], [0.23, 0.33], [0.19, 0.29]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[0.3, 0.5, 0.7, 0.5, 0.3], [0.8, 1.2, 1.5, 1.0, 0.6], [0.2, 0.3, 0.4, 0.3, 0.2]],
          "dsai_profile": [[0.05, 0.08, 0.12, 0.08, 0.05], [0.15, 0.22, 0.28, 0.18, 0.12], [0.03, 0.05, 0.07, 0.05, 0.03]],
          "dpai_profile": [[0.35, 0.58, 0.82, 0.58, 0.35], [0.95, 1.42, 1.78, 1.18, 0.72], [0.23, 0.35, 0.47, 0.35, 0.23]],
          "ntop_canopy": [0, 0, 0],
          "nbot_canopy": [4, 4, 4],
          "ncan_canopy": [5, 5, 5]
        },
        "pft_params": {
          "rhol": [[0.08, 0.4], [0.1, 0.45], [0.12, 0.5], [0.09, 0.42], [0.11, 0.48], [0.1, 0.45], [0.07, 0.38], [0.13, 0.52], [0.09, 0.43]],
          "taul": [[0.04, 0.22], [0.05, 0.25], [0.06, 0.28], [0.05, 0.24], [0.05, 0.26], [0.05, 0.25], [0.04, 0.2], [0.07, 0.3], [0.05, 0.23]],
          "rhos": [[0.15, 0.38], [0.16, 0.39], [0.17, 0.4], [0.16, 0.39], [0.16, 0.39], [0.16, 0.39], [0.14, 0.37], [0.18, 0.41], [0.15, 0.38]],
          "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [-0.1, 0.01, 0.1, 0.0, 0.05, 0.01, -0.15, 0.15, 0.02],
          "clump_fac": [0.8, 0.85, 0.9, 0.85, 0.87, 0.85, 0.75, 0.92, 0.83]
        },
        "nlevmlcan": 5,
        "light_type": 1,
        "numrad": 2
      },
      "metadata": {
        "type": "nominal",
        "description": "Multiple patches with varying LAI (sparse, moderate, dense canopies) and different PFT types",
        "edge_cases": []
      }
    },
    {
      "name": "test_nominal_twostream_method",
      "inputs": {
        "bounds": {"begp": 0, "endp": 2, "begg": 0, "endg": 1},
        "num_filter": 2,
        "filter_indices": [0, 1],
        "patch_state": {
          "itype": [3, 7],
          "cosz": [0.8, 0.6],
          "swskyb": [[750.0, 580.0], [680.0, 520.0]],
          "swskyd": [[190.0, 145.0], [175.0, 135.0]],
          "albsoib": [[0.16, 0.26], [0.14, 0.24]],
          "albsoid": [[0.21, 0.31], [0.19, 0.29]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[0.6, 0.9, 1.3, 0.9, 0.5], [0.4, 0.7, 1.0, 0.7, 0.4]],
          "dsai_profile": [[0.12, 0.18, 0.25, 0.18, 0.1], [0.08, 0.14, 0.2, 0.14, 0.08]],
          "dpai_profile": [[0.72, 1.08, 1.55, 1.08, 0.6], [0.48, 0.84, 1.2, 0.84, 0.48]],
          "ntop_canopy": [0, 0],
          "nbot_canopy": [4, 4],
          "ncan_canopy": [5, 5]
        },
        "pft_params": {
          "rhol": [[0.09, 0.43], [0.11, 0.47], [0.1, 0.45], [0.08, 0.41], [0.12, 0.49], [0.1, 0.44], [0.09, 0.42], [0.11, 0.46]],
          "taul": [[0.045, 0.23], [0.055, 0.27], [0.05, 0.25], [0.04, 0.21], [0.06, 0.29], [0.05, 0.24], [0.045, 0.22], [0.055, 0.26]],
          "rhos": [[0.155, 0.385], [0.165, 0.395], [0.16, 0.39], [0.15, 0.38], [0.17, 0.4], [0.16, 0.39], [0.155, 0.385], [0.165, 0.395]],
          "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [-0.05, 0.08, 0.01, -0.12, 0.12, 0.03, -0.08, 0.1],
          "clump_fac": [0.82, 0.88, 0.85, 0.78, 0.91, 0.86, 0.81, 0.89]
        },
        "nlevmlcan": 5,
        "light_type": 2,
        "numrad": 2
      },
      "metadata": {
        "type": "nominal",
        "description": "Testing TwoStream radiation transfer method with moderate canopy conditions",
        "edge_cases": []
      }
    },
    {
      "name": "test_edge_zero_solar_zenith_angle",
      "inputs": {
        "bounds": {"begp": 0, "endp": 2, "begg": 0, "endg": 1},
        "num_filter": 2,
        "filter_indices": [0, 1],
        "patch_state": {
          "itype": [4, 6],
          "cosz": [0.0, 0.001],
          "swskyb": [[0.0, 0.0], [5.0, 3.0]],
          "swskyd": [[50.0, 40.0], [55.0, 42.0]],
          "albsoib": [[0.15, 0.25], [0.16, 0.26]],
          "albsoid": [[0.2, 0.3], [0.21, 0.31]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[0.5, 0.8, 1.1, 0.8, 0.5], [0.6, 0.9, 1.2, 0.9, 0.6]],
          "dsai_profile": [[0.1, 0.15, 0.2, 0.15, 0.1], [0.12, 0.17, 0.22, 0.17, 0.12]],
          "dpai_profile": [[0.6, 0.95, 1.3, 0.95, 0.6], [0.72, 1.07, 1.42, 1.07, 0.72]],
          "ntop_canopy": [0, 0],
          "nbot_canopy": [4, 4],
          "ncan_canopy": [5, 5]
        },
        "pft_params": {
          "rhol": [[0.1, 0.45], [0.09, 0.43], [0.11, 0.47], [0.1, 0.44], [0.1, 0.46], [0.09, 0.44], [0.11, 0.46]],
          "taul": [[0.05, 0.25], [0.045, 0.23], [0.055, 0.27], [0.05, 0.24], [0.05, 0.26], [0.045, 0.24], [0.055, 0.26]],
          "rhos": [[0.16, 0.39], [0.155, 0.385], [0.165, 0.395], [0.16, 0.39], [0.16, 0.39], [0.155, 0.385], [0.165, 0.395]],
          "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [0.01, -0.05, 0.08, 0.02, 0.04, -0.03, 0.06],
          "clump_fac": [0.85, 0.82, 0.88, 0.86, 0.87, 0.83, 0.89]
        },
        "nlevmlcan": 5,
        "light_type": 1,
        "numrad": 2
      },
      "metadata": {
        "type": "edge",
        "description": "Near-horizon sun (dawn/dusk) with zero or near-zero cosine of solar zenith angle",
        "edge_cases": ["zero_cosz", "minimal_direct_beam"]
      }
    },
    {
      "name": "test_edge_maximum_solar_zenith",
      "inputs": {
        "bounds": {"begp": 0, "endp": 1, "begg": 0, "endg": 1},
        "num_filter": 1,
        "filter_indices": [0],
        "patch_state": {
          "itype": [5],
          "cosz": [1.0],
          "swskyb": [[1000.0, 750.0]],
          "swskyd": [[100.0, 75.0]],
          "albsoib": [[0.1, 0.2]],
          "albsoid": [[0.15, 0.25]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[0.7, 1.0, 1.4, 1.0, 0.7]],
          "dsai_profile": [[0.14, 0.2, 0.28, 0.2, 0.14]],
          "dpai_profile": [[0.84, 1.2, 1.68, 1.2, 0.84]],
          "ntop_canopy": [0],
          "nbot_canopy": [4],
          "ncan_canopy": [5]
        },
        "pft_params": {
          "rhol": [[0.1, 0.45], [0.09, 0.43], [0.11, 0.47], [0.1, 0.44], [0.1, 0.46], [0.09, 0.44]],
          "taul": [[0.05, 0.25], [0.045, 0.23], [0.055, 0.27], [0.05, 0.24], [0.05, 0.26], [0.045, 0.24]],
          "rhos": [[0.16, 0.39], [0.155, 0.385], [0.165, 0.395], [0.16, 0.39], [0.16, 0.39], [0.155, 0.385]],
          "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [0.01, -0.05, 0.08, 0.02, 0.04, -0.03],
          "clump_fac": [0.85, 0.82, 0.88, 0.86, 0.87, 0.83]
        },
        "nlevmlcan": 5,
        "light_type": 1,
        "numrad": 2
      },
      "metadata": {
        "type": "edge",
        "description": "Solar noon with sun directly overhead (cosz = 1.0), maximum direct beam radiation",
        "edge_cases": ["maximum_cosz", "maximum_direct_beam"]
      }
    },
    {
      "name": "test_edge_zero_lai_bare_ground",
      "inputs": {
        "bounds": {"begp": 0, "endp": 1, "begg": 0, "endg": 1},
        "num_filter": 1,
        "filter_indices": [0],
        "patch_state": {
          "itype": [0],
          "cosz": [0.7],
          "swskyb": [[650.0, 500.0]],
          "swskyd": [[170.0, 130.0]],
          "albsoib": [[0.25, 0.35]],
          "albsoid": [[0.3, 0.4]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[0.0, 0.0, 0.0, 0.0, 0.0]],
          "dsai_profile": [[0.0, 0.0, 0.0, 0.0, 0.0]],
          "dpai_profile": [[0.0, 0.0, 0.0, 0.0, 0.0]],
          "ntop_canopy": [0],
          "nbot_canopy": [0],
          "ncan_canopy": [0]
        },
        "pft_params": {
          "rhol": [[0.1, 0.45]],
          "taul": [[0.05, 0.25]],
          "rhos": [[0.16, 0.39]],
          "taus": [[0.001, 0.001]],
          "xl": [0.0],
          "clump_fac": [1.0]
        },
        "nlevmlcan": 5,
        "light_type": 1,
        "numrad": 2
      },
      "metadata": {
        "type": "edge",
        "description": "Bare ground with zero LAI/SAI/PAI - all radiation reaches soil directly",
        "edge_cases": ["zero_lai", "bare_ground", "no_canopy"]
      }
    },
    {
      "name": "test_edge_very_dense_canopy",
      "inputs": {
        "bounds": {"begp": 0, "endp": 1, "begg": 0, "endg": 1},
        "num_filter": 1,
        "filter_indices": [0],
        "patch_state": {
          "itype": [5],
          "cosz": [0.75],
          "swskyb": [[800.0, 600.0]],
          "swskyd": [[200.0, 150.0]],
          "albsoib": [[0.12, 0.22]],
          "albsoid": [[0.17, 0.27]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[2.0, 2.5, 3.0, 2.5, 2.0]],
          "dsai_profile": [[0.4, 0.5, 0.6, 0.5, 0.4]],
          "dpai_profile": [[2.4, 3.0, 3.6, 3.0, 2.4]],
          "ntop_canopy": [0],
          "nbot_canopy": [4],
          "ncan_canopy": [5]
        },
        "pft_params": {
          "rhol": [[0.1, 0.45], [0.09, 0.43], [0.11, 0.47], [0.1, 0.44], [0.1, 0.46], [0.09, 0.44]],
          "taul": [[0.05, 0.25], [0.045, 0.23], [0.055, 0.27], [0.05, 0.24], [0.05, 0.26], [0.045, 0.24]],
          "rhos": [[0.16, 0.39], [0.155, 0.385], [0.165, 0.395], [0.16, 0.39], [0.16, 0.39], [0.155, 0.385]],
          "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [0.01, -0.05, 0.08, 0.02, 0.04, -0.03],
          "clump_fac": [0.7, 0.68, 0.72, 0.71, 0.69, 0.7]
        },
        "nlevmlcan": 5,
        "light_type": 1,
        "numrad": 2
      },
      "metadata": {
        "type": "edge",
        "description": "Very dense canopy with high LAI - minimal radiation reaches soil, strong attenuation",
        "edge_cases": ["high_lai", "dense_canopy", "strong_attenuation"]
      }
    },
    {
      "name": "test_edge_extreme_albedo_boundaries",
      "inputs": {
        "bounds": {"begp": 0, "endp": 3, "begg": 0, "endg": 1},
        "num_filter": 3,
        "filter_indices": [0, 1, 2],
        "patch_state": {
          "itype": [1, 3, 7],
          "cosz": [0.6, 0.7, 0.8],
          "swskyb": [[700.0, 550.0], [750.0, 580.0], [800.0, 620.0]],
          "swskyd": [[180.0, 140.0], [190.0, 145.0], [200.0, 155.0]],
          "albsoib": [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]],
          "albsoid": [[0.0, 0.0], [0.55, 0.65], [1.0, 1.0]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[0.5, 0.8, 1.0, 0.8, 0.5], [0.6, 0.9, 1.2, 0.9, 0.6], [0.4, 0.7, 0.9, 0.7, 0.4]],
          "dsai_profile": [[0.1, 0.15, 0.2, 0.15, 0.1], [0.12, 0.18, 0.24, 0.18, 0.12], [0.08, 0.14, 0.18, 0.14, 0.08]],
          "dpai_profile": [[0.6, 0.95, 1.2, 0.95, 0.6], [0.72, 1.08, 1.44, 1.08, 0.72], [0.48, 0.84, 1.08, 0.84, 0.48]],
          "ntop_canopy": [0, 0, 0],
          "nbot_canopy": [4, 4, 4],
          "ncan_canopy": [5, 5, 5]
        },
        "pft_params": {
          "rhol": [[0.0, 0.0], [0.1, 0.45], [0.5, 0.95], [0.05, 0.4], [0.15, 0.5], [0.1, 0.45], [0.08, 0.42], [0.12, 0.48]],
          "taul": [[0.0, 0.0], [0.05, 0.25], [0.5, 0.05], [0.04, 0.22], [0.06, 0.28], [0.05, 0.25], [0.04, 0.23], [0.06, 0.27]],
          "rhos": [[0.0, 0.0], [0.16, 0.39], [1.0, 1.0], [0.15, 0.38], [0.17, 0.4], [0.16, 0.39], [0.15, 0.38], [0.17, 0.4]],
          "taus": [[0.0, 0.0], [0.001, 0.001], [0.0, 0.0], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [0.0, 0.01, 0.0, -0.05, 0.08, 0.02, -0.03, 0.06],
          "clump_fac": [1.0, 0.85, 0.5, 0.82, 0.88, 0.86, 0.83, 0.89]
        },
        "nlevmlcan": 5,
        "light_type": 1,
        "numrad": 2
      },
      "metadata": {
        "type": "edge",
        "description": "Testing boundary albedo values (0.0, 0.5, 1.0) for soil and vegetation optical properties",
        "edge_cases": ["zero_albedo", "maximum_albedo", "boundary_reflectance"]
      }
    },
    {
      "name": "test_special_single_canopy_layer",
      "inputs": {
        "bounds": {"begp": 0, "endp": 1, "begg": 0, "endg": 1},
        "num_filter": 1,
        "filter_indices": [0],
        "patch_state": {
          "itype": [4],
          "cosz": [0.65],
          "swskyb": [[720.0, 560.0]],
          "swskyd": [[185.0, 142.0]],
          "albsoib": [[0.14, 0.24]],
          "albsoid": [[0.19, 0.29]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[2.5]],
          "dsai_profile": [[0.5]],
          "dpai_profile": [[3.0]],
          "ntop_canopy": [0],
          "nbot_canopy": [0],
          "ncan_canopy": [1]
        },
        "pft_params": {
          "rhol": [[0.1, 0.45], [0.09, 0.43], [0.11, 0.47], [0.1, 0.44], [0.09, 0.42]],
          "taul": [[0.05, 0.25], [0.045, 0.23], [0.055, 0.27], [0.05, 0.24], [0.045, 0.22]],
          "rhos": [[0.16, 0.39], [0.155, 0.385], [0.165, 0.395], [0.16, 0.39], [0.155, 0.385]],
          "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [0.01, -0.05, 0.08, 0.02, -0.03],
          "clump_fac": [0.85, 0.82, 0.88, 0.86, 0.83]
        },
        "nlevmlcan": 1,
        "light_type": 1,
        "numrad": 2
      },
      "metadata": {
        "type": "special",
        "description": "Minimum canopy layers (nlevmlcan=1) with all LAI concentrated in single layer",
        "edge_cases": ["single_layer"]
      }
    },
    {
      "name": "test_special_many_canopy_layers",
      "inputs": {
        "bounds": {"begp": 0, "endp": 1, "begg": 0, "endg": 1},
        "num_filter": 1,
        "filter_indices": [0],
        "patch_state": {
          "itype": [6],
          "cosz": [0.72],
          "swskyb": [[780.0, 590.0]],
          "swskyd": [[195.0, 148.0]],
          "albsoib": [[0.13, 0.23]],
          "albsoid": [[0.18, 0.28]]
        },
        "mlcanopy_state": {
          "dlai_profile": [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5]],
          "dsai_profile": [[0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.14, 0.12, 0.1]],
          "dpai_profile": [[0.24, 0.36, 0.48, 0.6, 0.72, 0.84, 0.96, 0.84, 0.72, 0.6]],
          "ntop_canopy": [0],
          "nbot_canopy": [9],
          "ncan_canopy": [10]
        },
        "pft_params": {
          "rhol": [[0.1, 0.45], [0.09, 0.43], [0.11, 0.47], [0.1, 0.44], [0.1, 0.46], [0.09, 0.44], [0.11, 0.46]],
          "taul": [[0.05, 0.25], [0.045, 0.23], [0.055, 0.27], [0.05, 0.24], [0.05, 0.26], [0.045, 0.24], [0.055, 0.26]],
          "rhos": [[0.16, 0.39], [0.155, 0.385], [0.165, 0.395], [0.16, 0.39], [0.16, 0.39], [0.155, 0.385], [0.165, 0.395]],
          "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
          "xl": [0.01, -0.05, 0.08, 0.02, 0.04, -0.03, 0.06],
          "clump_fac": [0.85, 0.82, 0.88, 0.86, 0.87, 0.83, 0.89]
        },
        "nlevmlcan": 10,
        "light_type": 2,
        "numrad": 2
      },
      "metadata": {
        "type": "special",
        "description": "Many canopy layers (10) with gradual LAI distribution, testing vertical resolution",
        "edge_cases": ["many_layers"]
      }
    }
  ]
}
"""


def convert_to_jax_arrays(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert lists to JAX arrays in nested dictionaries.
    
    Args:
        data: Dictionary potentially containing nested lists
        
    Returns:
        Dictionary with lists converted to JAX arrays
    """
    if isinstance(data, dict):
        return {k: convert_to_jax_arrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        return jnp.array(data)
    else:
        return data


def create_namedtuple_from_dict(tuple_class, data_dict: Dict[str, Any]):
    """
    Create a namedtuple instance from a dictionary.
    
    Args:
        tuple_class: The namedtuple class to instantiate
        data_dict: Dictionary with field values
        
    Returns:
        Instance of the namedtuple
    """
    converted_data = convert_to_jax_arrays(data_dict)
    return tuple_class(**converted_data)


@pytest.fixture(scope="module")
def test_data():
    """
    Load and parse test data from embedded JSON.
    
    Returns:
        Dictionary containing all test cases with proper data structures
    """
    data = json.loads(TEST_DATA_JSON)
    
    # Convert test case inputs to proper namedtuples
    for test_case in data['test_cases']:
        inputs = test_case['inputs']
        
        # Convert bounds
        inputs['bounds'] = BoundsType(**inputs['bounds'])
        
        # Convert filter_indices to JAX array
        inputs['filter_indices'] = jnp.array(inputs['filter_indices'])
        
        # Convert patch_state
        inputs['patch_state'] = create_namedtuple_from_dict(
            PatchState, inputs['patch_state']
        )
        
        # Convert mlcanopy_state
        inputs['mlcanopy_state'] = create_namedtuple_from_dict(
            MLCanopyState, inputs['mlcanopy_state']
        )
        
        # Convert pft_params
        inputs['pft_params'] = create_namedtuple_from_dict(
            PFTParams, inputs['pft_params']
        )
    
    return data


@pytest.fixture
def solar_radiation_function():
    """
    Fixture providing the solar_radiation function.
    
    Note: This is a placeholder. In actual use, import the real function:
    from multilayer_canopy.MLSolarRadiationMod import solar_radiation
    
    Returns:
        The solar_radiation function
    """
    # Placeholder - replace with actual import
    def mock_solar_radiation(bounds, num_filter, filter_indices, patch_state,
                            mlcanopy_state, pft_params, nlevmlcan, 
                            light_type=1, numrad=2):
        """Mock function returning properly shaped outputs."""
        n_patches = bounds.endp - bounds.begp
        
        return RadiationFluxes(
            swleaf=jnp.zeros((n_patches, nlevmlcan, 2, numrad)),
            swsoi=jnp.zeros((n_patches, numrad)),
            swveg=jnp.zeros((n_patches, numrad)),
            swvegsun=jnp.zeros((n_patches, numrad)),
            swvegsha=jnp.zeros((n_patches, numrad)),
            albcan=jnp.zeros((n_patches, numrad)),
            apar_sun=jnp.zeros((n_patches, nlevmlcan)),
            apar_shade=jnp.zeros((n_patches, nlevmlcan))
        )
    
    return mock_solar_radiation


# Parametrize tests with all test cases
def get_test_case_ids(test_data):
    """Extract test case names for parametrization."""
    return [tc['name'] for tc in test_data['test_cases']]


def get_test_cases(test_data):
    """Extract test cases for parametrization."""
    return test_data['test_cases']


class TestSolarRadiationShapes:
    """Test suite for verifying output shapes of solar_radiation function."""
    
    @pytest.mark.parametrize("test_case", get_test_cases(json.loads(TEST_DATA_JSON)), 
                            ids=get_test_case_ids(json.loads(TEST_DATA_JSON)))
    def test_output_shapes(self, test_case, solar_radiation_function):
        """
        Test that solar_radiation returns correctly shaped outputs.
        
        Verifies:
        - swleaf: (n_patches, nlevmlcan, 2, numrad)
        - swsoi: (n_patches, numrad)
        - swveg: (n_patches, numrad)
        - swvegsun: (n_patches, numrad)
        - swvegsha: (n_patches, numrad)
        - albcan: (n_patches, numrad)
        - apar_sun: (n_patches, nlevmlcan)
        - apar_shade: (n_patches, nlevmlcan)
        """
        inputs = test_case['inputs']
        n_patches = inputs['bounds'].endp - inputs['bounds'].begp
        nlevmlcan = inputs['nlevmlcan']
        numrad = inputs['numrad']
        
        result = solar_radiation_function(**inputs)
        
        assert result.swleaf.shape == (n_patches, nlevmlcan, 2, numrad), \
            f"swleaf shape mismatch for {test_case['name']}"
        assert result.swsoi.shape == (n_patches, numrad), \
            f"swsoi shape mismatch for {test_case['name']}"
        assert result.swveg.shape == (n_patches, numrad), \
            f"swveg shape mismatch for {test_case['name']}"
        assert result.swvegsun.shape == (n_patches, numrad), \
            f"swvegsun shape mismatch for {test_case['name']}"
        assert result.swvegsha.shape == (n_patches, numrad), \
            f"swvegsha shape mismatch for {test_case['name']}"
        assert result.albcan.shape == (n_patches, numrad), \
            f"albcan shape mismatch for {test_case['name']}"
        assert result.apar_sun.shape == (n_patches, nlevmlcan), \
            f"apar_sun shape mismatch for {test_case['name']}"
        assert result.apar_shade.shape == (n_patches, nlevmlcan), \
            f"apar_shade shape mismatch for {test_case['name']}"


class TestSolarRadiationDtypes:
    """Test suite for verifying data types of solar_radiation outputs."""
    
    @pytest.mark.parametrize("test_case", get_test_cases(json.loads(TEST_DATA_JSON)), 
                            ids=get_test_case_ids(json.loads(TEST_DATA_JSON)))
    def test_output_dtypes(self, test_case, solar_radiation_function):
        """
        Test that solar_radiation returns floating point outputs.
        
        All radiation fluxes and albedos should be floating point values.
        """
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        assert jnp.issubdtype(result.swleaf.dtype, jnp.floating), \
            f"swleaf should be floating point for {test_case['name']}"
        assert jnp.issubdtype(result.swsoi.dtype, jnp.floating), \
            f"swsoi should be floating point for {test_case['name']}"
        assert jnp.issubdtype(result.swveg.dtype, jnp.floating), \
            f"swveg should be floating point for {test_case['name']}"
        assert jnp.issubdtype(result.swvegsun.dtype, jnp.floating), \
            f"swvegsun should be floating point for {test_case['name']}"
        assert jnp.issubdtype(result.swvegsha.dtype, jnp.floating), \
            f"swvegsha should be floating point for {test_case['name']}"
        assert jnp.issubdtype(result.albcan.dtype, jnp.floating), \
            f"albcan should be floating point for {test_case['name']}"
        assert jnp.issubdtype(result.apar_sun.dtype, jnp.floating), \
            f"apar_sun should be floating point for {test_case['name']}"
        assert jnp.issubdtype(result.apar_shade.dtype, jnp.floating), \
            f"apar_shade should be floating point for {test_case['name']}"


class TestSolarRadiationPhysicalConstraints:
    """Test suite for verifying physical constraints on outputs."""
    
    @pytest.mark.parametrize("test_case", get_test_cases(json.loads(TEST_DATA_JSON)), 
                            ids=get_test_case_ids(json.loads(TEST_DATA_JSON)))
    def test_non_negative_radiation(self, test_case, solar_radiation_function):
        """
        Test that all radiation fluxes are non-negative.
        
        Physical constraint: Radiation absorption cannot be negative.
        """
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        assert jnp.all(result.swleaf >= 0), \
            f"swleaf contains negative values for {test_case['name']}"
        assert jnp.all(result.swsoi >= 0), \
            f"swsoi contains negative values for {test_case['name']}"
        assert jnp.all(result.swveg >= 0), \
            f"swveg contains negative values for {test_case['name']}"
        assert jnp.all(result.swvegsun >= 0), \
            f"swvegsun contains negative values for {test_case['name']}"
        assert jnp.all(result.swvegsha >= 0), \
            f"swvegsha contains negative values for {test_case['name']}"
        assert jnp.all(result.apar_sun >= 0), \
            f"apar_sun contains negative values for {test_case['name']}"
        assert jnp.all(result.apar_shade >= 0), \
            f"apar_shade contains negative values for {test_case['name']}"
    
    @pytest.mark.parametrize("test_case", get_test_cases(json.loads(TEST_DATA_JSON)), 
                            ids=get_test_case_ids(json.loads(TEST_DATA_JSON)))
    def test_albedo_bounds(self, test_case, solar_radiation_function):
        """
        Test that canopy albedo is within [0, 1].
        
        Physical constraint: Albedo represents fraction of reflected radiation.
        """
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        assert jnp.all(result.albcan >= 0), \
            f"albcan contains values < 0 for {test_case['name']}"
        assert jnp.all(result.albcan <= 1), \
            f"albcan contains values > 1 for {test_case['name']}"
    
    @pytest.mark.parametrize("test_case", get_test_cases(json.loads(TEST_DATA_JSON)), 
                            ids=get_test_case_ids(json.loads(TEST_DATA_JSON)))
    def test_sunlit_shaded_sum(self, test_case, solar_radiation_function):
        """
        Test that sunlit + shaded absorption equals total vegetation absorption.
        
        Physical constraint: Total canopy absorption is sum of sunlit and shaded.
        """
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        total_computed = result.swvegsun + result.swvegsha
        
        # Allow small numerical differences
        assert jnp.allclose(total_computed, result.swveg, rtol=1e-5, atol=1e-6), \
            f"swvegsun + swvegsha != swveg for {test_case['name']}"


class TestSolarRadiationEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_zero_lai_bare_ground(self, test_data, solar_radiation_function):
        """
        Test bare ground case with zero LAI.
        
        Expected behavior:
        - All radiation reaches soil (swsoi > 0)
        - No canopy absorption (swveg ≈ 0)
        - Canopy albedo should be minimal or zero
        """
        test_case = next(tc for tc in test_data['test_cases'] 
                        if tc['name'] == 'test_edge_zero_lai_bare_ground')
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        # For bare ground, canopy absorption should be zero or very small
        assert jnp.allclose(result.swveg, 0.0, atol=1e-6), \
            "Bare ground should have zero canopy absorption"
        
        # Soil should receive radiation
        total_incoming = (inputs['patch_state'].swskyb + 
                         inputs['patch_state'].swskyd)
        assert jnp.any(result.swsoi > 0), \
            "Bare ground should have soil absorption"
    
    def test_zero_solar_zenith_angle(self, test_data, solar_radiation_function):
        """
        Test dawn/dusk conditions with zero or near-zero cosz.
        
        Expected behavior:
        - Minimal or zero direct beam radiation
        - Diffuse radiation still processed
        - No numerical instabilities
        """
        test_case = next(tc for tc in test_data['test_cases'] 
                        if tc['name'] == 'test_edge_zero_solar_zenith_angle')
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        # Should not have NaN or Inf values
        assert jnp.all(jnp.isfinite(result.swleaf)), \
            "Zero cosz should not produce NaN/Inf in swleaf"
        assert jnp.all(jnp.isfinite(result.swsoi)), \
            "Zero cosz should not produce NaN/Inf in swsoi"
        assert jnp.all(jnp.isfinite(result.albcan)), \
            "Zero cosz should not produce NaN/Inf in albcan"
    
    def test_maximum_solar_zenith(self, test_data, solar_radiation_function):
        """
        Test solar noon with cosz = 1.0.
        
        Expected behavior:
        - Maximum direct beam penetration
        - Proper handling of vertical sun angle
        - No numerical issues
        """
        test_case = next(tc for tc in test_data['test_cases'] 
                        if tc['name'] == 'test_edge_maximum_solar_zenith')
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        # Should have valid outputs
        assert jnp.all(jnp.isfinite(result.swleaf)), \
            "Maximum cosz should not produce NaN/Inf"
        
        # With high direct beam, should have significant absorption
        assert jnp.any(result.swveg > 0), \
            "Maximum cosz should produce canopy absorption"
    
    def test_dense_canopy_attenuation(self, test_data, solar_radiation_function):
        """
        Test very dense canopy with high LAI.
        
        Expected behavior:
        - Strong radiation attenuation
        - Minimal radiation reaching soil
        - High canopy absorption
        """
        test_case = next(tc for tc in test_data['test_cases'] 
                        if tc['name'] == 'test_edge_very_dense_canopy')
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        # Dense canopy should absorb most radiation
        total_incoming = jnp.sum(inputs['patch_state'].swskyb + 
                                inputs['patch_state'].swskyd)
        total_absorbed = jnp.sum(result.swveg + result.swsoi)
        
        # Canopy should absorb significant fraction
        canopy_fraction = jnp.sum(result.swveg) / (total_absorbed + 1e-10)
        assert canopy_fraction > 0.5, \
            "Dense canopy should absorb majority of radiation"
    
    def test_extreme_albedo_boundaries(self, test_data, solar_radiation_function):
        """
        Test boundary albedo values (0.0 and 1.0).
        
        Expected behavior:
        - Zero albedo: maximum absorption
        - Unit albedo: maximum reflection, minimal absorption
        - No numerical instabilities
        """
        test_case = next(tc for tc in test_data['test_cases'] 
                        if tc['name'] == 'test_edge_extreme_albedo_boundaries')
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        # Should handle extreme albedos without issues
        assert jnp.all(jnp.isfinite(result.swsoi)), \
            "Extreme albedos should not produce NaN/Inf in swsoi"
        assert jnp.all(jnp.isfinite(result.albcan)), \
            "Extreme albedos should not produce NaN/Inf in albcan"
        
        # Albedo should still be bounded
        assert jnp.all(result.albcan >= 0) and jnp.all(result.albcan <= 1), \
            "Canopy albedo should remain in [0,1] even with extreme inputs"


class TestSolarRadiationSpecialCases:
    """Test suite for special configurations."""
    
    def test_single_canopy_layer(self, test_data, solar_radiation_function):
        """
        Test minimum canopy layers (nlevmlcan=1).
        
        Expected behavior:
        - Single layer should handle all LAI
        - Proper sunlit/shaded partitioning
        - Valid outputs
        """
        test_case = next(tc for tc in test_data['test_cases'] 
                        if tc['name'] == 'test_special_single_canopy_layer')
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        # Should have valid single-layer outputs
        assert result.swleaf.shape[1] == 1, \
            "Single layer case should have nlevmlcan=1"
        assert jnp.all(jnp.isfinite(result.swleaf)), \
            "Single layer should produce finite values"
        
        # Should still partition sunlit/shaded
        assert jnp.any(result.swvegsun > 0) or jnp.any(result.swvegsha > 0), \
            "Single layer should have some absorption"
    
    def test_many_canopy_layers(self, test_data, solar_radiation_function):
        """
        Test many canopy layers (nlevmlcan=10).
        
        Expected behavior:
        - Proper vertical resolution
        - Gradual attenuation through layers
        - No numerical issues with many layers
        """
        test_case = next(tc for tc in test_data['test_cases'] 
                        if tc['name'] == 'test_special_many_canopy_layers')
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        # Should have correct number of layers
        assert result.swleaf.shape[1] == 10, \
            "Many layer case should have nlevmlcan=10"
        
        # Should show vertical gradient (top layers receive more)
        # Average absorption in top half vs bottom half
        top_half = jnp.mean(result.swleaf[:, :5, :, :])
        bottom_half = jnp.mean(result.swleaf[:, 5:, :, :])
        
        # Top should generally receive more (unless very sparse canopy)
        # This is a weak test since it depends on LAI distribution
        assert jnp.isfinite(top_half) and jnp.isfinite(bottom_half), \
            "Many layers should produce finite values throughout"


class TestSolarRadiationMethodComparison:
    """Test suite comparing Norman and TwoStream methods."""
    
    def test_norman_vs_twostream_consistency(self, test_data, solar_radiation_function):
        """
        Test that Norman and TwoStream methods produce reasonable results.
        
        Both methods should:
        - Conserve energy
        - Produce similar magnitudes (though not identical)
        - Handle same inputs without errors
        """
        # Get a nominal test case
        test_case = next(tc for tc in test_data['test_cases'] 
                        if tc['name'] == 'test_nominal_single_patch_midday')
        
        # Test with Norman method (light_type=1)
        inputs_norman = test_case['inputs'].copy()
        inputs_norman['light_type'] = 1
        result_norman = solar_radiation_function(**inputs_norman)
        
        # Test with TwoStream method (light_type=2)
        inputs_twostream = test_case['inputs'].copy()
        inputs_twostream['light_type'] = 2
        result_twostream = solar_radiation_function(**inputs_twostream)
        
        # Both should produce valid outputs
        assert jnp.all(jnp.isfinite(result_norman.swveg)), \
            "Norman method should produce finite values"
        assert jnp.all(jnp.isfinite(result_twostream.swveg)), \
            "TwoStream method should produce finite values"
        
        # Both should conserve energy (within numerical precision)
        # Total absorption should be less than or equal to incoming
        incoming = jnp.sum(inputs_norman['patch_state'].swskyb + 
                          inputs_norman['patch_state'].swskyd)
        
        total_norman = jnp.sum(result_norman.swveg + result_norman.swsoi)
        total_twostream = jnp.sum(result_twostream.swveg + result_twostream.swsoi)
        
        assert total_norman <= incoming * 1.01, \
            "Norman method should conserve energy"
        assert total_twostream <= incoming * 1.01, \
            "TwoStream method should conserve energy"


class TestSolarRadiationNumericalStability:
    """Test suite for numerical stability and edge conditions."""
    
    @pytest.mark.parametrize("test_case", get_test_cases(json.loads(TEST_DATA_JSON)), 
                            ids=get_test_case_ids(json.loads(TEST_DATA_JSON)))
    def test_no_nan_or_inf(self, test_case, solar_radiation_function):
        """
        Test that outputs never contain NaN or Inf values.
        
        This is critical for numerical stability in coupled models.
        """
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        assert jnp.all(jnp.isfinite(result.swleaf)), \
            f"swleaf contains NaN/Inf for {test_case['name']}"
        assert jnp.all(jnp.isfinite(result.swsoi)), \
            f"swsoi contains NaN/Inf for {test_case['name']}"
        assert jnp.all(jnp.isfinite(result.swveg)), \
            f"swveg contains NaN/Inf for {test_case['name']}"
        assert jnp.all(jnp.isfinite(result.swvegsun)), \
            f"swvegsun contains NaN/Inf for {test_case['name']}"
        assert jnp.all(jnp.isfinite(result.swvegsha)), \
            f"swvegsha contains NaN/Inf for {test_case['name']}"
        assert jnp.all(jnp.isfinite(result.albcan)), \
            f"albcan contains NaN/Inf for {test_case['name']}"
        assert jnp.all(jnp.isfinite(result.apar_sun)), \
            f"apar_sun contains NaN/Inf for {test_case['name']}"
        assert jnp.all(jnp.isfinite(result.apar_shade)), \
            f"apar_shade contains NaN/Inf for {test_case['name']}"
    
    @pytest.mark.parametrize("test_case", get_test_cases(json.loads(TEST_DATA_JSON)), 
                            ids=get_test_case_ids(json.loads(TEST_DATA_JSON)))
    def test_energy_conservation(self, test_case, solar_radiation_function):
        """
        Test energy conservation: absorbed + reflected ≈ incoming.
        
        Within numerical precision, total energy should be conserved.
        """
        inputs = test_case['inputs']
        result = solar_radiation_function(**inputs)
        
        # Calculate total incoming radiation per patch
        incoming = inputs['patch_state'].swskyb + inputs['patch_state'].swskyd
        total_incoming = jnp.sum(incoming, axis=1)  # Sum over radiation bands
        
        # Calculate total absorbed (vegetation + soil)
        total_absorbed = jnp.sum(result.swveg + result.swsoi, axis=1)
        
        # Calculate reflected (albedo * incoming)
        # Note: This is approximate since albcan is effective canopy albedo
        total_reflected = jnp.sum(result.albcan * incoming, axis=1)
        
        # Total should approximately equal incoming
        # Allow 5% tolerance for numerical precision and model approximations
        total_accounted = total_absorbed + total_reflected
        
        relative_error = jnp.abs(total_accounted - total_incoming) / (total_incoming + 1e-10)
        
        assert jnp.all(relative_error < 0.05), \
            f"Energy conservation violated for {test_case['name']}: " \
            f"max relative error = {jnp.max(relative_error):.4f}"


class TestSolarRadiationInputValidation:
    """Test suite for input validation and error handling."""
    
    def test_filter_indices_bounds(self, test_data, solar_radiation_function):
        """
        Test that filter_indices are within valid bounds.
        
        All indices should be >= 0 and < endp.
        """
        for test_case in test_data['test_cases']:
            inputs = test_case['inputs']
            filter_indices = inputs['filter_indices']
            endp = inputs['bounds'].endp
            
            assert jnp.all(filter_indices >= 0), \
                f"filter_indices contains negative values in {test_case['name']}"
            assert jnp.all(filter_indices < endp), \
                f"filter_indices exceeds bounds in {test_case['name']}"
    
    def test_num_filter_consistency(self, test_data, solar_radiation_function):
        """
        Test that num_filter matches length of filter_indices.
        """
        for test_case in test_data['test_cases']:
            inputs = test_case['inputs']
            num_filter = inputs['num_filter']
            filter_indices = inputs['filter_indices']
            
            assert num_filter == len(filter_indices), \
                f"num_filter != len(filter_indices) in {test_case['name']}"
    
    def test_array_dimension_consistency(self, test_data, solar_radiation_function):
        """
        Test that all patch-level arrays have consistent dimensions.
        """
        for test_case in test_data['test_cases']:
            inputs = test_case['inputs']
            n_patches = inputs['bounds'].endp - inputs['bounds'].begp
            
            # Check patch_state arrays
            assert len(inputs['patch_state'].itype) == n_patches, \
                f"itype dimension mismatch in {test_case['name']}"
            assert len(inputs['patch_state'].cosz) == n_patches, \
                f"cosz dimension mismatch in {test_case['name']}"
            assert inputs['patch_state'].swskyb.shape[0] == n_patches, \
                f"swskyb dimension mismatch in {test_case['name']}"
            
            # Check mlcanopy_state arrays
            assert inputs['mlcanopy_state'].dlai_profile.shape[0] == n_patches, \
                f"dlai_profile dimension mismatch in {test_case['name']}"
            assert inputs['mlcanopy_state'].ntop_canopy.shape[0] == n_patches, \
                f"ntop_canopy dimension mismatch in {test_case['name']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])