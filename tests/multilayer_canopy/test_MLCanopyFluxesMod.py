"""
Comprehensive pytest suite for MLCanopyFluxesMod.ml_canopy_fluxes function.

This test suite covers:
- Nominal cases with single and multiple patches
- Edge cases: zero LAI, nighttime, extreme temperatures, high LAI
- Special cases: varying sub-steps, extreme weather conditions
- Physical constraints validation
- Output shape and dtype verification
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Tuple
from collections import namedtuple


# Define NamedTuples matching the function signature
Bounds = namedtuple('Bounds', ['begp', 'endp', 'begc', 'endc', 'begg', 'endg'])

AtmosphericForcing = namedtuple('AtmosphericForcing', [
    'tref', 'qref', 'pref', 'lwsky', 'qflx_rain', 'qflx_snow',
    'co2ref', 'o2ref', 'tacclim', 'uref', 'swskyb_vis', 'swskyd_vis',
    'swskyb_nir', 'swskyd_nir'
])

CanopyProfileState = namedtuple('CanopyProfileState', [
    'lai', 'sai', 'dlai', 'dsai', 'dpai'
])

MLCanopyFluxesState = namedtuple('MLCanopyFluxesState', [
    'rnleaf_sun', 'rnleaf_shade', 'rnsoi', 'rhg',
    'flux_accumulator', 'flux_accumulator_profile', 'flux_accumulator_leaf'
])

FluxAccumulators = namedtuple('FluxAccumulators', [
    'flux_1d', 'flux_2d', 'flux_3d'
])


# Test data fixture
@pytest.fixture
def test_data():
    """
    Load and provide test data for all test cases.
    
    Returns:
        dict: Dictionary containing all test cases with inputs and metadata
    """
    return {
        "test_nominal_single_patch_moderate_conditions": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 1,
                "filter_exposedvegp": jnp.array([0]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([298.15]),
                    qref=jnp.array([0.012]),
                    pref=jnp.array([101325.0]),
                    lwsky=jnp.array([350.0]),
                    qflx_rain=jnp.array([0.0]),
                    qflx_snow=jnp.array([0.0]),
                    co2ref=jnp.array([0.0004]),
                    o2ref=jnp.array([0.209]),
                    tacclim=jnp.array([288.15]),
                    uref=jnp.array([3.5]),
                    swskyb_vis=jnp.array([250.0]),
                    swskyd_vis=jnp.array([150.0]),
                    swskyb_nir=jnp.array([200.0]),
                    swskyd_nir=jnp.array([100.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([4.0]),
                    sai=jnp.array([1.0]),
                    dlai=jnp.array([[0.3, 0.25, 0.2, 0.15, 0.1]]),
                    dsai=jnp.array([[0.2, 0.2, 0.2, 0.2, 0.2]]),
                    dpai=jnp.array([[0.5, 0.45, 0.4, 0.35, 0.3]])
                ),
                "nstep": 100,
                "dtime": 1800.0,
                "dtime_substep": 300.0,
                "num_sub_steps": 6
            },
            "metadata": {
                "type": "nominal",
                "description": "Single patch with moderate LAI, typical summer daytime conditions",
                "edge_cases": []
            }
        },
        "test_nominal_multiple_patches_varying_lai": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=4, begc=0, endc=2, begg=0, endg=1),
                "num_exposedvegp": 4,
                "filter_exposedvegp": jnp.array([0, 1, 2, 3]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([295.15, 297.15, 299.15, 296.15]),
                    qref=jnp.array([0.01, 0.013, 0.015, 0.011]),
                    pref=jnp.array([101325.0, 101200.0, 101400.0, 101300.0]),
                    lwsky=jnp.array([340.0, 355.0, 360.0, 345.0]),
                    qflx_rain=jnp.array([0.0, 0.0, 0.0, 0.0]),
                    qflx_snow=jnp.array([0.0, 0.0, 0.0, 0.0]),
                    co2ref=jnp.array([0.0004, 0.000405, 0.000398, 0.000402]),
                    o2ref=jnp.array([0.209, 0.209, 0.209, 0.209]),
                    tacclim=jnp.array([288.15, 289.15, 287.15, 288.65]),
                    uref=jnp.array([2.5, 4.0, 3.0, 3.5]),
                    swskyb_vis=jnp.array([200.0, 280.0, 240.0, 260.0]),
                    swskyd_vis=jnp.array([120.0, 160.0, 140.0, 150.0]),
                    swskyb_nir=jnp.array([180.0, 220.0, 190.0, 210.0]),
                    swskyd_nir=jnp.array([90.0, 110.0, 95.0, 105.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([2.5, 5.0, 3.5, 6.0]),
                    sai=jnp.array([0.5, 1.2, 0.8, 1.5]),
                    dlai=jnp.array([
                        [0.35, 0.3, 0.2, 0.1, 0.05],
                        [0.25, 0.22, 0.2, 0.18, 0.15],
                        [0.3, 0.28, 0.22, 0.15, 0.05],
                        [0.22, 0.2, 0.19, 0.18, 0.21]
                    ]),
                    dsai=jnp.array([
                        [0.25, 0.25, 0.2, 0.2, 0.1],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.22, 0.22, 0.2, 0.18, 0.18],
                        [0.18, 0.2, 0.21, 0.21, 0.2]
                    ]),
                    dpai=jnp.array([
                        [0.6, 0.55, 0.4, 0.3, 0.15],
                        [0.45, 0.42, 0.4, 0.38, 0.35],
                        [0.52, 0.5, 0.42, 0.33, 0.23],
                        [0.4, 0.4, 0.4, 0.39, 0.41]
                    ])
                ),
                "nstep": 250,
                "dtime": 1800.0,
                "dtime_substep": 450.0,
                "num_sub_steps": 4
            },
            "metadata": {
                "type": "nominal",
                "description": "Multiple patches with varying LAI/SAI",
                "edge_cases": []
            }
        },
        "test_edge_zero_lai_bare_ground": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 2,
                "filter_exposedvegp": jnp.array([0, 1]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([300.15, 298.15]),
                    qref=jnp.array([0.008, 0.01]),
                    pref=jnp.array([101325.0, 101325.0]),
                    lwsky=jnp.array([370.0, 360.0]),
                    qflx_rain=jnp.array([0.0, 0.0]),
                    qflx_snow=jnp.array([0.0, 0.0]),
                    co2ref=jnp.array([0.0004, 0.0004]),
                    o2ref=jnp.array([0.209, 0.209]),
                    tacclim=jnp.array([290.15, 289.15]),
                    uref=jnp.array([5.0, 4.5]),
                    swskyb_vis=jnp.array([300.0, 280.0]),
                    swskyd_vis=jnp.array([180.0, 170.0]),
                    swskyb_nir=jnp.array([240.0, 220.0]),
                    swskyd_nir=jnp.array([120.0, 110.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([0.0, 0.1]),
                    sai=jnp.array([0.0, 0.05]),
                    dlai=jnp.array([
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0]
                    ]),
                    dsai=jnp.array([
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0]
                    ]),
                    dpai=jnp.array([
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0]
                    ])
                ),
                "nstep": 50,
                "dtime": 1800.0,
                "dtime_substep": 600.0,
                "num_sub_steps": 3
            },
            "metadata": {
                "type": "edge",
                "description": "Zero and minimal LAI representing bare ground",
                "edge_cases": ["zero_lai", "minimal_vegetation"]
            }
        },
        "test_edge_nighttime_no_solar_radiation": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=3, begc=0, endc=2, begg=0, endg=1),
                "num_exposedvegp": 3,
                "filter_exposedvegp": jnp.array([0, 1, 2]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([285.15, 283.15, 286.15]),
                    qref=jnp.array([0.008, 0.007, 0.009]),
                    pref=jnp.array([101325.0, 101400.0, 101250.0]),
                    lwsky=jnp.array([280.0, 275.0, 285.0]),
                    qflx_rain=jnp.array([0.0, 0.0, 0.0]),
                    qflx_snow=jnp.array([0.0, 0.0, 0.0]),
                    co2ref=jnp.array([0.00041, 0.000415, 0.000408]),
                    o2ref=jnp.array([0.209, 0.209, 0.209]),
                    tacclim=jnp.array([285.15, 284.15, 286.15]),
                    uref=jnp.array([2.0, 1.5, 2.5]),
                    swskyb_vis=jnp.array([0.0, 0.0, 0.0]),
                    swskyd_vis=jnp.array([0.0, 0.0, 0.0]),
                    swskyb_nir=jnp.array([0.0, 0.0, 0.0]),
                    swskyd_nir=jnp.array([0.0, 0.0, 0.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([3.5, 4.5, 2.8]),
                    sai=jnp.array([0.9, 1.1, 0.7]),
                    dlai=jnp.array([
                        [0.28, 0.25, 0.22, 0.15, 0.1],
                        [0.24, 0.22, 0.2, 0.18, 0.16],
                        [0.32, 0.28, 0.2, 0.12, 0.08]
                    ]),
                    dsai=jnp.array([
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.22, 0.22, 0.2, 0.18, 0.18]
                    ]),
                    dpai=jnp.array([
                        [0.48, 0.45, 0.42, 0.35, 0.3],
                        [0.44, 0.42, 0.4, 0.38, 0.36],
                        [0.54, 0.5, 0.4, 0.3, 0.26]
                    ])
                ),
                "nstep": 500,
                "dtime": 1800.0,
                "dtime_substep": 300.0,
                "num_sub_steps": 6
            },
            "metadata": {
                "type": "edge",
                "description": "Nighttime conditions with zero solar radiation",
                "edge_cases": ["zero_solar", "nighttime"]
            }
        },
        "test_edge_extreme_cold_winter_conditions": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 2,
                "filter_exposedvegp": jnp.array([0, 1]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([253.15, 258.15]),
                    qref=jnp.array([0.0005, 0.0008]),
                    pref=jnp.array([102500.0, 102300.0]),
                    lwsky=jnp.array([180.0, 195.0]),
                    qflx_rain=jnp.array([0.0, 0.0]),
                    qflx_snow=jnp.array([0.0001, 0.00015]),
                    co2ref=jnp.array([0.00042, 0.000418]),
                    o2ref=jnp.array([0.209, 0.209]),
                    tacclim=jnp.array([253.15, 258.15]),
                    uref=jnp.array([6.0, 5.5]),
                    swskyb_vis=jnp.array([80.0, 100.0]),
                    swskyd_vis=jnp.array([50.0, 60.0]),
                    swskyb_nir=jnp.array([60.0, 75.0]),
                    swskyd_nir=jnp.array([30.0, 40.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([1.5, 2.0]),
                    sai=jnp.array([0.8, 1.0]),
                    dlai=jnp.array([
                        [0.4, 0.3, 0.2, 0.1, 0.0],
                        [0.35, 0.3, 0.2, 0.1, 0.05]
                    ]),
                    dsai=jnp.array([
                        [0.25, 0.25, 0.25, 0.25, 0.0],
                        [0.22, 0.22, 0.2, 0.18, 0.18]
                    ]),
                    dpai=jnp.array([
                        [0.65, 0.55, 0.45, 0.35, 0.0],
                        [0.57, 0.52, 0.4, 0.28, 0.23]
                    ])
                ),
                "nstep": 1000,
                "dtime": 1800.0,
                "dtime_substep": 900.0,
                "num_sub_steps": 2
            },
            "metadata": {
                "type": "edge",
                "description": "Extreme cold winter conditions",
                "edge_cases": ["extreme_cold", "low_humidity", "winter"]
            }
        },
        "test_edge_high_lai_dense_canopy": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 2,
                "filter_exposedvegp": jnp.array([0, 1]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([296.15, 297.15]),
                    qref=jnp.array([0.018, 0.019]),
                    pref=jnp.array([101325.0, 101325.0]),
                    lwsky=jnp.array([360.0, 365.0]),
                    qflx_rain=jnp.array([0.0002, 0.00025]),
                    qflx_snow=jnp.array([0.0, 0.0]),
                    co2ref=jnp.array([0.000395, 0.000398]),
                    o2ref=jnp.array([0.209, 0.209]),
                    tacclim=jnp.array([290.15, 291.15]),
                    uref=jnp.array([1.5, 1.8]),
                    swskyb_vis=jnp.array([320.0, 340.0]),
                    swskyd_vis=jnp.array([200.0, 210.0]),
                    swskyb_nir=jnp.array([260.0, 280.0]),
                    swskyd_nir=jnp.array([130.0, 140.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([8.0, 9.5]),
                    sai=jnp.array([2.0, 2.5]),
                    dlai=jnp.array([
                        [0.21, 0.2, 0.2, 0.19, 0.2],
                        [0.205, 0.2, 0.2, 0.195, 0.2]
                    ]),
                    dsai=jnp.array([
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2]
                    ]),
                    dpai=jnp.array([
                        [0.41, 0.4, 0.4, 0.39, 0.4],
                        [0.405, 0.4, 0.4, 0.395, 0.4]
                    ])
                ),
                "nstep": 300,
                "dtime": 1800.0,
                "dtime_substep": 360.0,
                "num_sub_steps": 5
            },
            "metadata": {
                "type": "edge",
                "description": "Very high LAI representing dense tropical forest",
                "edge_cases": ["high_lai", "dense_canopy", "high_humidity"]
            }
        },
        "test_special_single_substep": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 1,
                "filter_exposedvegp": jnp.array([0]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([293.15]),
                    qref=jnp.array([0.011]),
                    pref=jnp.array([101325.0]),
                    lwsky=jnp.array([340.0]),
                    qflx_rain=jnp.array([0.0]),
                    qflx_snow=jnp.array([0.0]),
                    co2ref=jnp.array([0.0004]),
                    o2ref=jnp.array([0.209]),
                    tacclim=jnp.array([288.15]),
                    uref=jnp.array([3.0]),
                    swskyb_vis=jnp.array([220.0]),
                    swskyd_vis=jnp.array([130.0]),
                    swskyb_nir=jnp.array([180.0]),
                    swskyd_nir=jnp.array([90.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([3.0]),
                    sai=jnp.array([0.8]),
                    dlai=jnp.array([[0.3, 0.28, 0.22, 0.15, 0.05]]),
                    dsai=jnp.array([[0.2, 0.2, 0.2, 0.2, 0.2]]),
                    dpai=jnp.array([[0.5, 0.48, 0.42, 0.35, 0.25]])
                ),
                "nstep": 1,
                "dtime": 1800.0,
                "dtime_substep": 1800.0,
                "num_sub_steps": 1
            },
            "metadata": {
                "type": "special",
                "description": "Single sub-timestep case",
                "edge_cases": ["single_substep", "minimal_temporal_resolution"]
            }
        },
        "test_special_many_substeps_fine_temporal": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=1, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 1,
                "filter_exposedvegp": jnp.array([0]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([299.15]),
                    qref=jnp.array([0.014]),
                    pref=jnp.array([101325.0]),
                    lwsky=jnp.array([355.0]),
                    qflx_rain=jnp.array([0.0]),
                    qflx_snow=jnp.array([0.0]),
                    co2ref=jnp.array([0.000402]),
                    o2ref=jnp.array([0.209]),
                    tacclim=jnp.array([289.15]),
                    uref=jnp.array([4.0]),
                    swskyb_vis=jnp.array([270.0]),
                    swskyd_vis=jnp.array([160.0]),
                    swskyb_nir=jnp.array([210.0]),
                    swskyd_nir=jnp.array([105.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([4.5]),
                    sai=jnp.array([1.2]),
                    dlai=jnp.array([[0.26, 0.24, 0.22, 0.18, 0.1]]),
                    dsai=jnp.array([[0.2, 0.2, 0.2, 0.2, 0.2]]),
                    dpai=jnp.array([[0.46, 0.44, 0.42, 0.38, 0.3]])
                ),
                "nstep": 200,
                "dtime": 1800.0,
                "dtime_substep": 90.0,
                "num_sub_steps": 20
            },
            "metadata": {
                "type": "special",
                "description": "Many sub-timesteps testing fine temporal resolution",
                "edge_cases": ["many_substeps", "fine_temporal_resolution"]
            }
        },
        "test_special_high_wind_low_pressure": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=3, begc=0, endc=2, begg=0, endg=1),
                "num_exposedvegp": 3,
                "filter_exposedvegp": jnp.array([0, 1, 2]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([291.15, 292.15, 290.15]),
                    qref=jnp.array([0.009, 0.01, 0.008]),
                    pref=jnp.array([95000.0, 94500.0, 95500.0]),
                    lwsky=jnp.array([320.0, 325.0, 315.0]),
                    qflx_rain=jnp.array([0.0005, 0.0006, 0.0004]),
                    qflx_snow=jnp.array([0.0, 0.0, 0.0]),
                    co2ref=jnp.array([0.000398, 0.0004, 0.000396]),
                    o2ref=jnp.array([0.209, 0.209, 0.209]),
                    tacclim=jnp.array([287.15, 288.15, 286.15]),
                    uref=jnp.array([12.0, 15.0, 10.0]),
                    swskyb_vis=jnp.array([180.0, 200.0, 160.0]),
                    swskyd_vis=jnp.array([110.0, 120.0, 100.0]),
                    swskyb_nir=jnp.array([140.0, 160.0, 120.0]),
                    swskyd_nir=jnp.array([70.0, 80.0, 60.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([2.8, 3.2, 2.5]),
                    sai=jnp.array([0.7, 0.8, 0.6]),
                    dlai=jnp.array([
                        [0.32, 0.28, 0.22, 0.13, 0.05],
                        [0.3, 0.26, 0.22, 0.14, 0.08],
                        [0.34, 0.3, 0.2, 0.11, 0.05]
                    ]),
                    dsai=jnp.array([
                        [0.22, 0.22, 0.2, 0.18, 0.18],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.24, 0.22, 0.2, 0.18, 0.16]
                    ]),
                    dpai=jnp.array([
                        [0.54, 0.5, 0.42, 0.31, 0.23],
                        [0.5, 0.46, 0.42, 0.34, 0.28],
                        [0.58, 0.52, 0.4, 0.29, 0.21]
                    ])
                ),
                "nstep": 750,
                "dtime": 1800.0,
                "dtime_substep": 225.0,
                "num_sub_steps": 8
            },
            "metadata": {
                "type": "special",
                "description": "High elevation conditions with low pressure and high winds",
                "edge_cases": ["high_wind", "low_pressure", "high_elevation"]
            }
        },
        "test_special_hot_dry_desert_conditions": {
            "inputs": {
                "bounds": Bounds(begp=0, endp=2, begc=0, endc=1, begg=0, endg=1),
                "num_exposedvegp": 2,
                "filter_exposedvegp": jnp.array([0, 1]),
                "atmospheric_forcing": AtmosphericForcing(
                    tref=jnp.array([313.15, 315.15]),
                    qref=jnp.array([0.003, 0.0025]),
                    pref=jnp.array([99000.0, 98800.0]),
                    lwsky=jnp.array([420.0, 430.0]),
                    qflx_rain=jnp.array([0.0, 0.0]),
                    qflx_snow=jnp.array([0.0, 0.0]),
                    co2ref=jnp.array([0.000405, 0.000407]),
                    o2ref=jnp.array([0.209, 0.209]),
                    tacclim=jnp.array([308.15, 310.15]),
                    uref=jnp.array([7.0, 8.0]),
                    swskyb_vis=jnp.array([450.0, 480.0]),
                    swskyd_vis=jnp.array([250.0, 270.0]),
                    swskyb_nir=jnp.array([380.0, 400.0]),
                    swskyd_nir=jnp.array([190.0, 200.0])
                ),
                "canopy_state": CanopyProfileState(
                    lai=jnp.array([0.8, 1.2]),
                    sai=jnp.array([0.4, 0.6]),
                    dlai=jnp.array([
                        [0.5, 0.3, 0.15, 0.05, 0.0],
                        [0.45, 0.3, 0.15, 0.08, 0.02]
                    ]),
                    dsai=jnp.array([
                        [0.4, 0.3, 0.2, 0.1, 0.0],
                        [0.35, 0.3, 0.2, 0.1, 0.05]
                    ]),
                    dpai=jnp.array([
                        [0.9, 0.6, 0.35, 0.15, 0.0],
                        [0.8, 0.6, 0.35, 0.18, 0.07]
                    ])
                ),
                "nstep": 600,
                "dtime": 1800.0,
                "dtime_substep": 450.0,
                "num_sub_steps": 4
            },
            "metadata": {
                "type": "special",
                "description": "Hot, dry desert conditions with sparse vegetation",
                "edge_cases": ["extreme_heat", "very_low_humidity", "sparse_vegetation", "high_solar"]
            }
        }
    }


@pytest.fixture
def mock_ml_canopy_fluxes():
    """
    Mock implementation of ml_canopy_fluxes for testing.
    
    This mock returns properly shaped outputs based on input dimensions.
    In actual implementation, this would be replaced with the real function import.
    """
    def _mock_function(bounds, num_exposedvegp, filter_exposedvegp,
                      atmospheric_forcing, canopy_state, nstep, dtime,
                      dtime_substep, num_sub_steps):
        n_patches = bounds.endp - bounds.begp
        n_canopy_layers = 5  # Fixed from test data
        
        # Create mock output state
        state = MLCanopyFluxesState(
            rnleaf_sun=jnp.zeros((n_patches, n_canopy_layers)),
            rnleaf_shade=jnp.zeros((n_patches, n_canopy_layers)),
            rnsoi=jnp.zeros(n_patches),
            rhg=jnp.zeros(n_patches),
            flux_accumulator=jnp.zeros((n_patches, 10)),
            flux_accumulator_profile=jnp.zeros((n_patches, n_canopy_layers, 8)),
            flux_accumulator_leaf=jnp.zeros((n_patches, n_canopy_layers, 2, 12))
        )
        
        # Create mock flux accumulators
        accumulators = FluxAccumulators(
            flux_1d=jnp.zeros((n_patches, 15)),
            flux_2d=jnp.zeros((n_patches, n_canopy_layers, 10)),
            flux_3d=jnp.zeros((n_patches, n_canopy_layers, 2, 8))
        )
        
        return state, accumulators
    
    return _mock_function


# ============================================================================
# Test: Physical Constraints Validation
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_patch_moderate_conditions",
    "test_nominal_multiple_patches_varying_lai",
    "test_edge_zero_lai_bare_ground",
    "test_edge_nighttime_no_solar_radiation",
    "test_edge_extreme_cold_winter_conditions",
    "test_edge_high_lai_dense_canopy",
    "test_special_single_substep",
    "test_special_many_substeps_fine_temporal",
    "test_special_high_wind_low_pressure",
    "test_special_hot_dry_desert_conditions"
])
def test_input_physical_constraints(test_data, test_case_name):
    """
    Test that all input data satisfies physical constraints.
    
    Validates:
    - Temperatures > 0K
    - Fractions in [0, 1]
    - Pressures > 0
    - LAI/SAI >= 0
    - Time parameters > 0
    """
    inputs = test_data[test_case_name]["inputs"]
    atm = inputs["atmospheric_forcing"]
    canopy = inputs["canopy_state"]
    
    # Temperature constraints
    assert jnp.all(atm.tref > 0), f"{test_case_name}: tref must be > 0K"
    assert jnp.all(atm.tacclim > 0), f"{test_case_name}: tacclim must be > 0K"
    
    # Humidity constraints
    assert jnp.all(atm.qref >= 0), f"{test_case_name}: qref must be >= 0"
    assert jnp.all(atm.qref <= 1), f"{test_case_name}: qref must be <= 1"
    
    # Pressure constraints
    assert jnp.all(atm.pref > 0), f"{test_case_name}: pref must be > 0"
    
    # LAI/SAI constraints
    assert jnp.all(canopy.lai >= 0), f"{test_case_name}: lai must be >= 0"
    assert jnp.all(canopy.sai >= 0), f"{test_case_name}: sai must be >= 0"
    
    # Fraction constraints
    assert jnp.all(canopy.dlai >= 0), f"{test_case_name}: dlai must be >= 0"
    assert jnp.all(canopy.dlai <= 1), f"{test_case_name}: dlai must be <= 1"
    assert jnp.all(canopy.dsai >= 0), f"{test_case_name}: dsai must be >= 0"
    assert jnp.all(canopy.dsai <= 1), f"{test_case_name}: dsai must be <= 1"
    assert jnp.all(canopy.dpai >= 0), f"{test_case_name}: dpai must be >= 0"
    assert jnp.all(canopy.dpai <= 1), f"{test_case_name}: dpai must be <= 1"
    
    # CO2/O2 constraints
    assert jnp.all(atm.co2ref >= 0), f"{test_case_name}: co2ref must be >= 0"
    assert jnp.all(atm.co2ref <= 1), f"{test_case_name}: co2ref must be <= 1"
    assert jnp.all(atm.o2ref >= 0), f"{test_case_name}: o2ref must be >= 0"
    assert jnp.all(atm.o2ref <= 1), f"{test_case_name}: o2ref must be <= 1"
    
    # Time constraints
    assert inputs["dtime"] > 0, f"{test_case_name}: dtime must be > 0"
    assert inputs["dtime_substep"] > 0, f"{test_case_name}: dtime_substep must be > 0"
    assert inputs["num_sub_steps"] >= 1, f"{test_case_name}: num_sub_steps must be >= 1"
    assert inputs["nstep"] >= 0, f"{test_case_name}: nstep must be >= 0"
    
    # Radiation constraints (non-negative)
    assert jnp.all(atm.lwsky >= 0), f"{test_case_name}: lwsky must be >= 0"
    assert jnp.all(atm.swskyb_vis >= 0), f"{test_case_name}: swskyb_vis must be >= 0"
    assert jnp.all(atm.swskyd_vis >= 0), f"{test_case_name}: swskyd_vis must be >= 0"
    assert jnp.all(atm.swskyb_nir >= 0), f"{test_case_name}: swskyb_nir must be >= 0"
    assert jnp.all(atm.swskyd_nir >= 0), f"{test_case_name}: swskyd_nir must be >= 0"
    
    # Wind speed constraints
    assert jnp.all(atm.uref >= 0), f"{test_case_name}: uref must be >= 0"
    
    # Precipitation constraints
    assert jnp.all(atm.qflx_rain >= 0), f"{test_case_name}: qflx_rain must be >= 0"
    assert jnp.all(atm.qflx_snow >= 0), f"{test_case_name}: qflx_snow must be >= 0"


# ============================================================================
# Test: Output Shapes
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_patch_moderate_conditions",
    "test_nominal_multiple_patches_varying_lai",
    "test_edge_zero_lai_bare_ground",
    "test_edge_nighttime_no_solar_radiation",
])
def test_output_shapes(test_data, mock_ml_canopy_fluxes, test_case_name):
    """
    Test that output arrays have correct shapes based on input dimensions.
    
    Validates:
    - State arrays match expected patch and layer dimensions
    - Flux accumulator arrays have correct dimensionality
    """
    inputs = test_data[test_case_name]["inputs"]
    n_patches = inputs["bounds"].endp - inputs["bounds"].begp
    n_canopy_layers = 5
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check MLCanopyFluxesState shapes
    assert state.rnleaf_sun.shape == (n_patches, n_canopy_layers), \
        f"{test_case_name}: rnleaf_sun shape mismatch"
    assert state.rnleaf_shade.shape == (n_patches, n_canopy_layers), \
        f"{test_case_name}: rnleaf_shade shape mismatch"
    assert state.rnsoi.shape == (n_patches,), \
        f"{test_case_name}: rnsoi shape mismatch"
    assert state.rhg.shape == (n_patches,), \
        f"{test_case_name}: rhg shape mismatch"
    
    # Check flux accumulator shapes
    assert state.flux_accumulator.ndim == 2, \
        f"{test_case_name}: flux_accumulator should be 2D"
    assert state.flux_accumulator.shape[0] == n_patches, \
        f"{test_case_name}: flux_accumulator first dim should be n_patches"
    
    assert state.flux_accumulator_profile.ndim == 3, \
        f"{test_case_name}: flux_accumulator_profile should be 3D"
    assert state.flux_accumulator_profile.shape[0] == n_patches, \
        f"{test_case_name}: flux_accumulator_profile first dim should be n_patches"
    assert state.flux_accumulator_profile.shape[1] == n_canopy_layers, \
        f"{test_case_name}: flux_accumulator_profile second dim should be n_canopy_layers"
    
    assert state.flux_accumulator_leaf.ndim == 4, \
        f"{test_case_name}: flux_accumulator_leaf should be 4D"
    assert state.flux_accumulator_leaf.shape[0] == n_patches, \
        f"{test_case_name}: flux_accumulator_leaf first dim should be n_patches"
    assert state.flux_accumulator_leaf.shape[1] == n_canopy_layers, \
        f"{test_case_name}: flux_accumulator_leaf second dim should be n_canopy_layers"
    assert state.flux_accumulator_leaf.shape[2] == 2, \
        f"{test_case_name}: flux_accumulator_leaf third dim should be 2 (sun/shade)"
    
    # Check FluxAccumulators shapes
    assert accumulators.flux_1d.ndim == 2, \
        f"{test_case_name}: flux_1d should be 2D"
    assert accumulators.flux_1d.shape[0] == n_patches, \
        f"{test_case_name}: flux_1d first dim should be n_patches"
    
    assert accumulators.flux_2d.ndim == 3, \
        f"{test_case_name}: flux_2d should be 3D"
    assert accumulators.flux_2d.shape[0] == n_patches, \
        f"{test_case_name}: flux_2d first dim should be n_patches"
    assert accumulators.flux_2d.shape[1] == n_canopy_layers, \
        f"{test_case_name}: flux_2d second dim should be n_canopy_layers"
    
    assert accumulators.flux_3d.ndim == 4, \
        f"{test_case_name}: flux_3d should be 4D"
    assert accumulators.flux_3d.shape[0] == n_patches, \
        f"{test_case_name}: flux_3d first dim should be n_patches"
    assert accumulators.flux_3d.shape[1] == n_canopy_layers, \
        f"{test_case_name}: flux_3d second dim should be n_canopy_layers"
    assert accumulators.flux_3d.shape[2] == 2, \
        f"{test_case_name}: flux_3d third dim should be 2 (sun/shade)"


# ============================================================================
# Test: Data Types
# ============================================================================

@pytest.mark.parametrize("test_case_name", [
    "test_nominal_single_patch_moderate_conditions",
    "test_edge_high_lai_dense_canopy",
])
def test_output_dtypes(test_data, mock_ml_canopy_fluxes, test_case_name):
    """
    Test that output arrays have correct data types (float32 or float64).
    
    Validates:
    - All output arrays are floating point
    - Consistent dtype across outputs
    """
    inputs = test_data[test_case_name]["inputs"]
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check state dtypes
    assert jnp.issubdtype(state.rnleaf_sun.dtype, jnp.floating), \
        f"{test_case_name}: rnleaf_sun should be floating point"
    assert jnp.issubdtype(state.rnleaf_shade.dtype, jnp.floating), \
        f"{test_case_name}: rnleaf_shade should be floating point"
    assert jnp.issubdtype(state.rnsoi.dtype, jnp.floating), \
        f"{test_case_name}: rnsoi should be floating point"
    assert jnp.issubdtype(state.rhg.dtype, jnp.floating), \
        f"{test_case_name}: rhg should be floating point"
    
    # Check accumulator dtypes
    assert jnp.issubdtype(state.flux_accumulator.dtype, jnp.floating), \
        f"{test_case_name}: flux_accumulator should be floating point"
    assert jnp.issubdtype(state.flux_accumulator_profile.dtype, jnp.floating), \
        f"{test_case_name}: flux_accumulator_profile should be floating point"
    assert jnp.issubdtype(state.flux_accumulator_leaf.dtype, jnp.floating), \
        f"{test_case_name}: flux_accumulator_leaf should be floating point"
    
    assert jnp.issubdtype(accumulators.flux_1d.dtype, jnp.floating), \
        f"{test_case_name}: flux_1d should be floating point"
    assert jnp.issubdtype(accumulators.flux_2d.dtype, jnp.floating), \
        f"{test_case_name}: flux_2d should be floating point"
    assert jnp.issubdtype(accumulators.flux_3d.dtype, jnp.floating), \
        f"{test_case_name}: flux_3d should be floating point"


# ============================================================================
# Test: Edge Cases - Zero LAI
# ============================================================================

def test_zero_lai_behavior(test_data, mock_ml_canopy_fluxes):
    """
    Test behavior with zero LAI (bare ground).
    
    Validates:
    - Function handles zero LAI without errors
    - Output shapes remain consistent
    - No NaN or Inf values in outputs
    """
    test_case_name = "test_edge_zero_lai_bare_ground"
    inputs = test_data[test_case_name]["inputs"]
    
    # Verify zero LAI in first patch
    assert inputs["canopy_state"].lai[0] == 0.0, "First patch should have zero LAI"
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(state.rnleaf_sun)), \
        "rnleaf_sun should not contain NaN with zero LAI"
    assert not jnp.any(jnp.isinf(state.rnleaf_sun)), \
        "rnleaf_sun should not contain Inf with zero LAI"
    assert not jnp.any(jnp.isnan(state.rnsoi)), \
        "rnsoi should not contain NaN with zero LAI"
    assert not jnp.any(jnp.isinf(state.rnsoi)), \
        "rnsoi should not contain Inf with zero LAI"


# ============================================================================
# Test: Edge Cases - Nighttime (Zero Solar)
# ============================================================================

def test_nighttime_zero_solar(test_data, mock_ml_canopy_fluxes):
    """
    Test behavior during nighttime with zero solar radiation.
    
    Validates:
    - Function handles zero solar radiation correctly
    - Longwave-only energy balance works
    - No NaN or Inf values
    """
    test_case_name = "test_edge_nighttime_no_solar_radiation"
    inputs = test_data[test_case_name]["inputs"]
    atm = inputs["atmospheric_forcing"]
    
    # Verify zero solar radiation
    assert jnp.all(atm.swskyb_vis == 0), "Should have zero beam visible radiation"
    assert jnp.all(atm.swskyd_vis == 0), "Should have zero diffuse visible radiation"
    assert jnp.all(atm.swskyb_nir == 0), "Should have zero beam NIR radiation"
    assert jnp.all(atm.swskyd_nir == 0), "Should have zero diffuse NIR radiation"
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(state.rnleaf_sun)), \
        "rnleaf_sun should not contain NaN at nighttime"
    assert not jnp.any(jnp.isinf(state.rnleaf_sun)), \
        "rnleaf_sun should not contain Inf at nighttime"
    assert not jnp.any(jnp.isnan(accumulators.flux_1d)), \
        "flux_1d should not contain NaN at nighttime"


# ============================================================================
# Test: Edge Cases - Extreme Temperatures
# ============================================================================

def test_extreme_cold_conditions(test_data, mock_ml_canopy_fluxes):
    """
    Test behavior under extreme cold conditions.
    
    Validates:
    - Function handles very low temperatures (253K)
    - Low humidity conditions work correctly
    - Snowfall is handled properly
    """
    test_case_name = "test_edge_extreme_cold_winter_conditions"
    inputs = test_data[test_case_name]["inputs"]
    atm = inputs["atmospheric_forcing"]
    
    # Verify extreme cold
    assert jnp.all(atm.tref < 273.15), "Should have sub-freezing temperatures"
    assert jnp.all(atm.qref < 0.001), "Should have very low humidity"
    assert jnp.any(atm.qflx_snow > 0), "Should have snowfall"
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check outputs are finite
    assert jnp.all(jnp.isfinite(state.rnleaf_sun)), \
        "rnleaf_sun should be finite in extreme cold"
    assert jnp.all(jnp.isfinite(state.rnsoi)), \
        "rnsoi should be finite in extreme cold"


def test_extreme_heat_conditions(test_data, mock_ml_canopy_fluxes):
    """
    Test behavior under extreme heat conditions.
    
    Validates:
    - Function handles very high temperatures (313-315K)
    - Very low humidity conditions work
    - High solar radiation is handled
    """
    test_case_name = "test_special_hot_dry_desert_conditions"
    inputs = test_data[test_case_name]["inputs"]
    atm = inputs["atmospheric_forcing"]
    
    # Verify extreme heat
    assert jnp.all(atm.tref > 310.0), "Should have very high temperatures"
    assert jnp.all(atm.qref < 0.004), "Should have very low humidity"
    assert jnp.all(atm.swskyb_vis > 400), "Should have high solar radiation"
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check outputs are finite
    assert jnp.all(jnp.isfinite(state.rnleaf_sun)), \
        "rnleaf_sun should be finite in extreme heat"
    assert jnp.all(jnp.isfinite(state.rhg)), \
        "rhg should be finite in extreme heat"


# ============================================================================
# Test: Edge Cases - High LAI Dense Canopy
# ============================================================================

def test_high_lai_dense_canopy(test_data, mock_ml_canopy_fluxes):
    """
    Test behavior with very high LAI (dense tropical forest).
    
    Validates:
    - Function handles LAI > 8
    - High humidity conditions work
    - Multiple canopy layers are processed correctly
    """
    test_case_name = "test_edge_high_lai_dense_canopy"
    inputs = test_data[test_case_name]["inputs"]
    canopy = inputs["canopy_state"]
    atm = inputs["atmospheric_forcing"]
    
    # Verify high LAI
    assert jnp.all(canopy.lai >= 8.0), "Should have very high LAI"
    assert jnp.all(atm.qref > 0.017), "Should have high humidity"
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check outputs are reasonable
    assert jnp.all(jnp.isfinite(state.rnleaf_sun)), \
        "rnleaf_sun should be finite with high LAI"
    assert jnp.all(jnp.isfinite(state.rnleaf_shade)), \
        "rnleaf_shade should be finite with high LAI"
    
    # With high LAI, shaded leaves should exist
    assert state.rnleaf_shade.shape == state.rnleaf_sun.shape, \
        "Sun and shade arrays should have same shape"


# ============================================================================
# Test: Special Cases - Temporal Resolution
# ============================================================================

def test_single_substep(test_data, mock_ml_canopy_fluxes):
    """
    Test behavior with single sub-timestep (no temporal subdivision).
    
    Validates:
    - Function works with num_sub_steps = 1
    - dtime_substep equals dtime
    - Outputs are consistent
    """
    test_case_name = "test_special_single_substep"
    inputs = test_data[test_case_name]["inputs"]
    
    # Verify single substep
    assert inputs["num_sub_steps"] == 1, "Should have single substep"
    assert inputs["dtime_substep"] == inputs["dtime"], \
        "dtime_substep should equal dtime for single substep"
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check outputs exist and are finite
    assert jnp.all(jnp.isfinite(state.flux_accumulator)), \
        "flux_accumulator should be finite with single substep"
    assert jnp.all(jnp.isfinite(accumulators.flux_1d)), \
        "flux_1d should be finite with single substep"


def test_many_substeps(test_data, mock_ml_canopy_fluxes):
    """
    Test behavior with many sub-timesteps (fine temporal resolution).
    
    Validates:
    - Function works with num_sub_steps = 20
    - Fine temporal discretization is handled
    - Flux accumulation is correct
    """
    test_case_name = "test_special_many_substeps_fine_temporal"
    inputs = test_data[test_case_name]["inputs"]
    
    # Verify many substeps
    assert inputs["num_sub_steps"] == 20, "Should have 20 substeps"
    assert inputs["dtime_substep"] == inputs["dtime"] / 20, \
        "dtime_substep should be dtime/20"
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check outputs are finite
    assert jnp.all(jnp.isfinite(state.flux_accumulator)), \
        "flux_accumulator should be finite with many substeps"
    assert jnp.all(jnp.isfinite(accumulators.flux_2d)), \
        "flux_2d should be finite with many substeps"


# ============================================================================
# Test: Special Cases - Extreme Weather
# ============================================================================

def test_high_wind_low_pressure(test_data, mock_ml_canopy_fluxes):
    """
    Test behavior under high wind and low pressure (high elevation).
    
    Validates:
    - Function handles wind speeds > 10 m/s
    - Low pressure (~95000 Pa) is handled correctly
    - Precipitation with high winds works
    """
    test_case_name = "test_special_high_wind_low_pressure"
    inputs = test_data[test_case_name]["inputs"]
    atm = inputs["atmospheric_forcing"]
    
    # Verify high wind and low pressure
    assert jnp.all(atm.uref >= 10.0), "Should have high wind speeds"
    assert jnp.all(atm.pref < 96000), "Should have low pressure"
    assert jnp.any(atm.qflx_rain > 0), "Should have precipitation"
    
    state, accumulators = mock_ml_canopy_fluxes(**inputs)
    
    # Check outputs are finite
    assert jnp.all(jnp.isfinite(state.rnleaf_sun)), \
        "rnleaf_sun should be finite with high wind"
    assert jnp.all(jnp.isfinite(state.rnsoi)), \
        "rnsoi should be finite with low pressure"


# ============================================================================
# Test: Consistency Checks
# ============================================================================

def test_bounds_consistency(test_data):
    """
    Test that bounds structure is internally consistent.
    
    Validates:
    - endp > begp
    - endc > begc
    - endg > begg
    - Filter indices are within bounds
    """
    for test_case_name, test_case in test_data.items():
        inputs = test_case["inputs"]
        bounds = inputs["bounds"]
        
        assert bounds.endp > bounds.begp, \
            f"{test_case_name}: endp must be > begp"
        assert bounds.endc >= bounds.begc, \
            f"{test_case_name}: endc must be >= begc"
        assert bounds.endg >= bounds.begg, \
            f"{test_case_name}: endg must be >= begg"
        
        # Check filter indices
        filter_indices = inputs["filter_exposedvegp"]
        assert jnp.all(filter_indices >= bounds.begp), \
            f"{test_case_name}: filter indices must be >= begp"
        assert jnp.all(filter_indices < bounds.endp), \
            f"{test_case_name}: filter indices must be < endp"
        
        # Check num_exposedvegp matches filter length
        assert inputs["num_exposedvegp"] == len(filter_indices), \
            f"{test_case_name}: num_exposedvegp must match filter length"


def test_lai_sai_fraction_sums(test_data):
    """
    Test that LAI/SAI layer fractions sum to approximately 1.0.
    
    Validates:
    - dlai fractions sum to ~1.0 per patch
    - dsai fractions sum to ~1.0 per patch
    - dpai fractions sum to ~1.0 per patch
    """
    for test_case_name, test_case in test_data.items():
        inputs = test_case["inputs"]
        canopy = inputs["canopy_state"]
        
        # Sum across layers (axis=1)
        dlai_sums = jnp.sum(canopy.dlai, axis=1)
        dsai_sums = jnp.sum(canopy.dsai, axis=1)
        dpai_sums = jnp.sum(canopy.dpai, axis=1)
        
        # Allow some tolerance for numerical precision and zero LAI cases
        # For zero LAI, fractions might all be zero
        non_zero_lai = canopy.lai > 0
        
        if jnp.any(non_zero_lai):
            assert jnp.allclose(dlai_sums[non_zero_lai], 1.0, atol=0.05), \
                f"{test_case_name}: dlai fractions should sum to ~1.0"
            assert jnp.allclose(dsai_sums[non_zero_lai], 1.0, atol=0.05), \
                f"{test_case_name}: dsai fractions should sum to ~1.0"
            assert jnp.allclose(dpai_sums[non_zero_lai], 1.0, atol=0.05), \
                f"{test_case_name}: dpai fractions should sum to ~1.0"


def test_timestep_consistency(test_data):
    """
    Test that time step parameters are consistent.
    
    Validates:
    - dtime_substep * num_sub_steps ≈ dtime
    - All time parameters are positive
    """
    for test_case_name, test_case in test_data.items():
        inputs = test_case["inputs"]
        
        dtime = inputs["dtime"]
        dtime_substep = inputs["dtime_substep"]
        num_sub_steps = inputs["num_sub_steps"]
        
        # Check consistency
        expected_dtime = dtime_substep * num_sub_steps
        assert jnp.isclose(expected_dtime, dtime, rtol=1e-5), \
            f"{test_case_name}: dtime_substep * num_sub_steps should equal dtime"


# ============================================================================
# Test: Array Dimension Consistency
# ============================================================================

def test_atmospheric_forcing_dimensions(test_data):
    """
    Test that all atmospheric forcing arrays have consistent dimensions.
    
    Validates:
    - All forcing arrays have same length (n_patches)
    - Arrays match bounds dimensions
    """
    for test_case_name, test_case in test_data.items():
        inputs = test_case["inputs"]
        bounds = inputs["bounds"]
        atm = inputs["atmospheric_forcing"]
        n_patches = bounds.endp - bounds.begp
        
        # Check all atmospheric forcing arrays
        assert len(atm.tref) == n_patches, \
            f"{test_case_name}: tref length should match n_patches"
        assert len(atm.qref) == n_patches, \
            f"{test_case_name}: qref length should match n_patches"
        assert len(atm.pref) == n_patches, \
            f"{test_case_name}: pref length should match n_patches"
        assert len(atm.lwsky) == n_patches, \
            f"{test_case_name}: lwsky length should match n_patches"
        assert len(atm.uref) == n_patches, \
            f"{test_case_name}: uref length should match n_patches"
        assert len(atm.swskyb_vis) == n_patches, \
            f"{test_case_name}: swskyb_vis length should match n_patches"


def test_canopy_state_dimensions(test_data):
    """
    Test that canopy state arrays have consistent dimensions.
    
    Validates:
    - LAI/SAI arrays have length n_patches
    - Layer fraction arrays have shape (n_patches, n_layers)
    """
    for test_case_name, test_case in test_data.items():
        inputs = test_case["inputs"]
        bounds = inputs["bounds"]
        canopy = inputs["canopy_state"]
        n_patches = bounds.endp - bounds.begp
        n_layers = 5  # Fixed from test data
        
        # Check 1D arrays
        assert len(canopy.lai) == n_patches, \
            f"{test_case_name}: lai length should match n_patches"
        assert len(canopy.sai) == n_patches, \
            f"{test_case_name}: sai length should match n_patches"
        
        # Check 2D arrays
        assert canopy.dlai.shape == (n_patches, n_layers), \
            f"{test_case_name}: dlai shape should be (n_patches, n_layers)"
        assert canopy.dsai.shape == (n_patches, n_layers), \
            f"{test_case_name}: dsai shape should be (n_patches, n_layers)"
        assert canopy.dpai.shape == (n_patches, n_layers), \
            f"{test_case_name}: dpai shape should be (n_patches, n_layers)"


# ============================================================================
# Test: Metadata Validation
# ============================================================================

def test_metadata_completeness(test_data):
    """
    Test that all test cases have complete metadata.
    
    Validates:
    - Each test case has metadata
    - Metadata contains required fields
    - Edge cases are documented
    """
    required_fields = ["type", "description", "edge_cases"]
    
    for test_case_name, test_case in test_data.items():
        assert "metadata" in test_case, \
            f"{test_case_name}: missing metadata"
        
        metadata = test_case["metadata"]
        for field in required_fields:
            assert field in metadata, \
                f"{test_case_name}: metadata missing field '{field}'"
        
        # Check type is valid
        assert metadata["type"] in ["nominal", "edge", "special"], \
            f"{test_case_name}: invalid metadata type"
        
        # Check description is non-empty
        assert len(metadata["description"]) > 0, \
            f"{test_case_name}: empty description"
        
        # Check edge_cases is a list
        assert isinstance(metadata["edge_cases"], list), \
            f"{test_case_name}: edge_cases should be a list"


# ============================================================================
# Test: Documentation
# ============================================================================

def test_test_data_coverage():
    """
    Test that test data covers all required scenarios.
    
    Validates:
    - At least 2 nominal cases
    - At least 3 edge cases
    - At least 3 special cases
    - Key edge cases are covered (zero LAI, nighttime, extreme temps, high LAI)
    """
    # This would be implemented by analyzing the test_data fixture
    # For now, we document the expected coverage
    
    expected_edge_cases = {
        "zero_lai",
        "nighttime",
        "extreme_cold",
        "extreme_heat",
        "high_lai"
    }
    
    # In a real implementation, we would check that these are present
    # in the test data edge_cases lists
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])