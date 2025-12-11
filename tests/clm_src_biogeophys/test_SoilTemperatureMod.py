"""
Comprehensive pytest suite for compute_soil_temperature function.

This module tests the soil temperature computation including:
- Nominal cases with and without snow layers
- Multiple columns with varying conditions
- Edge cases (zero flux, negative flux, freezing transitions, minimal water)
- Special cases (maximum snow layers, shallow bedrock, extreme timesteps)
- Physical constraint validation
- Array shape and dtype verification
"""

import pytest
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from typing import Dict, Any


# Define namedtuples for structured data
ColumnGeometry = namedtuple('ColumnGeometry', ['dz', 'z', 'zi', 'snl', 'nbedrock'])
WaterState = namedtuple('WaterState', ['h2osoi_liq', 'h2osoi_ice', 'h2osfc', 'h2osno', 'frac_sno_eff'])
SoilState = namedtuple('SoilState', ['tkmg', 'tkdry', 'csol', 'watsat'])
SoilTemperatureParams = namedtuple('SoilTemperatureParams', [
    'thin_sfclayer', 'denh2o', 'denice', 'tfrz', 'tkwat', 'tkice', 'tkair',
    'cpice', 'cpliq', 'thk_bedrock', 'csol_bedrock'
])
SoilTemperatureResult = namedtuple('SoilTemperatureResult', ['t_soisno', 'energy_error'])
ThermalProperties = namedtuple('ThermalProperties', ['tk', 'cv', 'tk_h2osfc', 'thk', 'bw'])


# Mock function - replace with actual import
def compute_soil_temperature(geom, t_soisno, gsoi, water, soil, params, dtime, nlevsno, nlevgrnd):
    """
    Mock implementation for testing purposes.
    Replace with: from SoilTemperatureMod import compute_soil_temperature
    """
    n_cols = t_soisno.shape[0]
    n_levtot = nlevsno + nlevgrnd
    
    # Mock return values with correct shapes
    result = SoilTemperatureResult(
        t_soisno=t_soisno + 0.1,  # Small temperature change
        energy_error=jnp.zeros((n_cols,))
    )
    
    thermal_props = ThermalProperties(
        tk=jnp.ones((n_cols, n_levtot)),
        cv=jnp.ones((n_cols, n_levtot)) * 1e6,
        tk_h2osfc=jnp.ones((n_cols,)) * 0.57,
        thk=jnp.ones((n_cols, n_levtot)) * 2.0,
        bw=jnp.zeros((n_cols, n_levtot))
    )
    
    return result, thermal_props


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load and prepare test data for all test cases.
    
    Returns:
        Dictionary containing all test cases with structured inputs.
    """
    test_cases = {
        "nominal_single_column_no_snow": {
            "geom": ColumnGeometry(
                dz=jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]]),
                z=jnp.array([[0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35, 3.05, 3.85, 4.75, 5.8, 7.1, 8.85, 11.35]]),
                zi=jnp.array([[0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2, 4.0, 5.0, 6.2, 7.7, 9.7, 13.7]]),
                snl=jnp.array([0]),
                nbedrock=jnp.array([15])
            ),
            "t_soisno": jnp.array([[280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0]]),
            "gsoi": jnp.array([50.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]]),
                h2osoi_ice=jnp.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                h2osfc=jnp.array([2.0]),
                h2osno=jnp.array([0.0]),
                frac_sno_eff=jnp.array([0.0])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[2.5] * 15]),
                tkdry=jnp.array([[0.2] * 15]),
                csol=jnp.array([[2000000.0] * 15]),
                watsat=jnp.array([[0.45] * 15])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 1800.0,
            "nlevsno": 5,
            "nlevgrnd": 15,
            "metadata": {
                "type": "nominal",
                "description": "Standard single column case with no snow, moderate temperatures, typical soil properties"
            }
        },
        "nominal_with_snow_layers": {
            "geom": ColumnGeometry(
                dz=jnp.array([[0.05, 0.08, 0.12, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]]),
                z=jnp.array([[0.025, 0.09, 0.19, 0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35, 3.05, 3.85, 4.75, 5.8, 7.1, 8.85, 11.35]]),
                zi=jnp.array([[0.0, 0.05, 0.13, 0.25, 0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2, 4.0, 5.0, 6.2, 7.7, 9.7]]),
                snl=jnp.array([-3]),
                nbedrock=jnp.array([15])
            ),
            "t_soisno": jnp.array([[268.0, 270.0, 272.0, 273.0, 275.0, 277.0, 279.0, 281.0, 283.0, 285.0, 287.0, 289.0, 291.0, 293.0, 295.0, 297.0, 299.0, 301.0]]),
            "gsoi": jnp.array([80.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([[0.5, 1.0, 2.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]]),
                h2osoi_ice=jnp.array([[15.0, 20.0, 25.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                h2osfc=jnp.array([1.5]),
                h2osno=jnp.array([50.0]),
                frac_sno_eff=jnp.array([0.85])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[2.8] * 15]),
                tkdry=jnp.array([[0.25] * 15]),
                csol=jnp.array([[2200000.0] * 15]),
                watsat=jnp.array([[0.5] * 15])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 3600.0,
            "nlevsno": 5,
            "nlevgrnd": 15,
            "metadata": {
                "type": "nominal",
                "description": "Winter conditions with 3 snow layers, temperatures near freezing, significant snow cover"
            }
        },
        "multiple_columns_varying_conditions": {
            "geom": ColumnGeometry(
                dz=jnp.array([
                    [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.08, 0.12, 0.18, 0.22, 0.28, 0.35, 0.45, 0.55, 0.65, 0.75],
                    [0.12, 0.18, 0.22, 0.28, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82]
                ]),
                z=jnp.array([
                    [0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35, 3.05, 3.85],
                    [0.04, 0.14, 0.29, 0.51, 0.79, 1.135, 1.585, 2.135, 2.785, 3.535],
                    [0.06, 0.21, 0.4, 0.65, 0.97, 1.39, 1.91, 2.53, 3.25, 4.07]
                ]),
                zi=jnp.array([
                    [0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2, 4.0],
                    [0.0, 0.08, 0.2, 0.38, 0.6, 0.88, 1.23, 1.68, 2.23, 2.88, 3.63],
                    [0.0, 0.12, 0.3, 0.52, 0.8, 1.12, 1.56, 2.08, 2.7, 3.42, 4.24]
                ]),
                snl=jnp.array([0, -1, -2]),
                nbedrock=jnp.array([10, 10, 10])
            ),
            "t_soisno": jnp.array([
                [295.0, 294.0, 293.0, 292.0, 291.0, 290.0, 289.0, 288.0, 287.0, 286.0],
                [270.0, 275.0, 278.0, 280.0, 282.0, 284.0, 286.0, 288.0, 290.0, 292.0],
                [265.0, 268.0, 273.0, 276.0, 279.0, 282.0, 285.0, 288.0, 291.0, 294.0]
            ]),
            "gsoi": jnp.array([120.0, -50.0, 30.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([
                    [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
                    [2.0, 5.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
                    [1.0, 3.0, 10.0, 18.0, 24.0, 32.0, 38.0, 44.0, 52.0, 58.0]
                ]),
                h2osoi_ice=jnp.array([
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [20.0, 8.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [25.0, 18.0, 5.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                ]),
                h2osfc=jnp.array([3.0, 1.0, 0.5]),
                h2osno=jnp.array([0.0, 15.0, 35.0]),
                frac_sno_eff=jnp.array([0.0, 0.4, 0.7])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[2.2] * 10, [3.0] * 10, [1.8] * 10]),
                tkdry=jnp.array([[0.18] * 10, [0.28] * 10, [0.15] * 10]),
                csol=jnp.array([[1800000.0] * 10, [2400000.0] * 10, [1600000.0] * 10]),
                watsat=jnp.array([[0.4] * 10, [0.52] * 10, [0.35] * 10])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 1200.0,
            "nlevsno": 5,
            "nlevgrnd": 10,
            "metadata": {
                "type": "nominal",
                "description": "Three columns with different conditions: warm/dry, cold/snowy, very cold/snowy"
            }
        },
        "edge_zero_heat_flux": {
            "geom": ColumnGeometry(
                dz=jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]),
                z=jnp.array([[0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35, 3.05, 3.85, 4.75, 5.8]]),
                zi=jnp.array([[0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2, 4.0, 5.0, 6.2]]),
                snl=jnp.array([0]),
                nbedrock=jnp.array([12])
            ),
            "t_soisno": jnp.array([[285.0] * 12]),
            "gsoi": jnp.array([0.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([[20.0] * 12]),
                h2osoi_ice=jnp.array([[0.0] * 12]),
                h2osfc=jnp.array([0.0]),
                h2osno=jnp.array([0.0]),
                frac_sno_eff=jnp.array([0.0])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[2.5] * 12]),
                tkdry=jnp.array([[0.2] * 12]),
                csol=jnp.array([[2000000.0] * 12]),
                watsat=jnp.array([[0.45] * 12])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 1800.0,
            "nlevsno": 5,
            "nlevgrnd": 12,
            "metadata": {
                "type": "edge",
                "description": "Zero ground heat flux with uniform temperature profile - tests equilibrium conditions",
                "edge_cases": ["zero_flux", "uniform_temperature"]
            }
        },
        "edge_negative_heat_flux": {
            "geom": ColumnGeometry(
                dz=jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
                z=jnp.array([[0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35, 3.05, 3.85]]),
                zi=jnp.array([[0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2, 4.0]]),
                snl=jnp.array([0]),
                nbedrock=jnp.array([10])
            ),
            "t_soisno": jnp.array([[300.0, 298.0, 296.0, 294.0, 292.0, 290.0, 288.0, 286.0, 284.0, 282.0]]),
            "gsoi": jnp.array([-450.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([[30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0]]),
                h2osoi_ice=jnp.array([[0.0] * 10]),
                h2osfc=jnp.array([5.0]),
                h2osno=jnp.array([0.0]),
                frac_sno_eff=jnp.array([0.0])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[3.5] * 10]),
                tkdry=jnp.array([[0.3] * 10]),
                csol=jnp.array([[2500000.0] * 10]),
                watsat=jnp.array([[0.48] * 10])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 900.0,
            "nlevsno": 5,
            "nlevgrnd": 10,
            "metadata": {
                "type": "edge",
                "description": "Large negative heat flux (upward) with warm soil - tests cooling scenario",
                "edge_cases": ["negative_flux", "high_temperature"]
            }
        },
        "edge_near_freezing_transition": {
            "geom": ColumnGeometry(
                dz=jnp.array([[0.08, 0.12, 0.16, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]]),
                z=jnp.array([[0.04, 0.14, 0.28, 0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35, 3.05]]),
                zi=jnp.array([[0.0, 0.08, 0.2, 0.36, 0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5]]),
                snl=jnp.array([-3]),
                nbedrock=jnp.array([9])
            ),
            "t_soisno": jnp.array([[271.15, 272.15, 273.05, 273.15, 273.25, 273.5, 274.0, 275.0, 276.0, 277.0, 278.0, 279.0]]),
            "gsoi": jnp.array([25.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([[0.1, 0.5, 1.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]]),
                h2osoi_ice=jnp.array([[18.0, 22.0, 28.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.0, 0.0]]),
                h2osfc=jnp.array([0.8]),
                h2osno=jnp.array([45.0]),
                frac_sno_eff=jnp.array([0.9])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[2.0] * 9]),
                tkdry=jnp.array([[0.18] * 9]),
                csol=jnp.array([[1900000.0] * 9]),
                watsat=jnp.array([[0.47] * 9])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 600.0,
            "nlevsno": 5,
            "nlevgrnd": 9,
            "metadata": {
                "type": "edge",
                "description": "Temperatures near freezing point (273.15K) with phase transition",
                "edge_cases": ["freezing_point", "phase_transition"]
            }
        },
        "edge_minimal_water_content": {
            "geom": ColumnGeometry(
                dz=jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
                z=jnp.array([[0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35, 3.05, 3.85]]),
                zi=jnp.array([[0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2, 4.0]]),
                snl=jnp.array([0]),
                nbedrock=jnp.array([10])
            ),
            "t_soisno": jnp.array([[310.0, 308.0, 306.0, 304.0, 302.0, 300.0, 298.0, 296.0, 294.0, 292.0]]),
            "gsoi": jnp.array([200.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([[0.01] * 10]),
                h2osoi_ice=jnp.array([[0.0] * 10]),
                h2osfc=jnp.array([0.0]),
                h2osno=jnp.array([0.0]),
                frac_sno_eff=jnp.array([0.0])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[1.5] * 10]),
                tkdry=jnp.array([[0.12] * 10]),
                csol=jnp.array([[1500000.0] * 10]),
                watsat=jnp.array([[0.35] * 10])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 1800.0,
            "nlevsno": 5,
            "nlevgrnd": 10,
            "metadata": {
                "type": "edge",
                "description": "Very dry soil with minimal water content and high temperatures - tests arid conditions",
                "edge_cases": ["minimal_water", "high_temperature", "dry_soil"]
            }
        },
        "special_maximum_snow_layers": {
            "geom": ColumnGeometry(
                dz=jnp.array([[0.03, 0.05, 0.08, 0.12, 0.18, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]]),
                z=jnp.array([[0.015, 0.055, 0.12, 0.21, 0.36, 0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35, 3.05, 3.85, 4.75, 5.8, 7.1, 8.85, 11.35]]),
                zi=jnp.array([[0.0, 0.03, 0.08, 0.16, 0.28, 0.46, 0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2, 4.0, 5.0, 6.2, 7.7, 9.7]]),
                snl=jnp.array([-5]),
                nbedrock=jnp.array([15])
            ),
            "t_soisno": jnp.array([[260.0, 262.0, 265.0, 268.0, 271.0, 273.0, 274.0, 275.0, 276.0, 278.0, 280.0, 282.0, 284.0, 286.0, 288.0, 290.0, 292.0, 294.0, 296.0, 298.0]]),
            "gsoi": jnp.array([60.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([[0.05, 0.1, 0.2, 0.5, 1.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0]]),
                h2osoi_ice=jnp.array([[12.0, 16.0, 20.0, 24.0, 28.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                h2osfc=jnp.array([0.3]),
                h2osno=jnp.array([85.0]),
                frac_sno_eff=jnp.array([1.0])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[2.6] * 15]),
                tkdry=jnp.array([[0.22] * 15]),
                csol=jnp.array([[2100000.0] * 15]),
                watsat=jnp.array([[0.46] * 15])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 1800.0,
            "nlevsno": 5,
            "nlevgrnd": 15,
            "metadata": {
                "type": "special",
                "description": "Maximum snow layers (5) with deep snowpack and complete snow cover"
            }
        },
        "special_shallow_bedrock": {
            "geom": ColumnGeometry(
                dz=jnp.array([[0.1, 0.15, 0.2, 0.25, 0.3, 0.4]]),
                z=jnp.array([[0.05, 0.175, 0.35, 0.575, 0.85, 1.25]]),
                zi=jnp.array([[0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4]]),
                snl=jnp.array([0]),
                nbedrock=jnp.array([4])
            ),
            "t_soisno": jnp.array([[288.0, 287.0, 286.0, 285.0, 284.5, 284.0]]),
            "gsoi": jnp.array([35.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([[15.0, 18.0, 20.0, 22.0, 10.0, 5.0]]),
                h2osoi_ice=jnp.array([[2.0, 1.5, 1.0, 0.5, 0.0, 0.0]]),
                h2osfc=jnp.array([1.2]),
                h2osno=jnp.array([0.0]),
                frac_sno_eff=jnp.array([0.0])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[2.3] * 6]),
                tkdry=jnp.array([[0.19] * 6]),
                csol=jnp.array([[1950000.0] * 6]),
                watsat=jnp.array([[0.43] * 6])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 1800.0,
            "nlevsno": 5,
            "nlevgrnd": 6,
            "metadata": {
                "type": "special",
                "description": "Shallow soil profile with bedrock at layer 4 - tests limited soil depth scenarios"
            }
        },
        "special_extreme_timestep_variations": {
            "geom": ColumnGeometry(
                dz=jnp.array([
                    [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6],
                    [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
                ]),
                z=jnp.array([
                    [0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35],
                    [0.05, 0.175, 0.35, 0.575, 0.85, 1.25, 1.75, 2.35]
                ]),
                zi=jnp.array([
                    [0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5],
                    [0.0, 0.1, 0.25, 0.45, 0.7, 1.0, 1.4, 1.9, 2.5]
                ]),
                snl=jnp.array([0, 0]),
                nbedrock=jnp.array([8, 8])
            ),
            "t_soisno": jnp.array([
                [278.0, 279.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0],
                [278.0, 279.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0]
            ]),
            "gsoi": jnp.array([75.0, 75.0]),
            "water": WaterState(
                h2osoi_liq=jnp.array([
                    [18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0],
                    [18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
                ]),
                h2osoi_ice=jnp.array([
                    [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, 0.0],
                    [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, 0.0]
                ]),
                h2osfc=jnp.array([1.8, 1.8]),
                h2osno=jnp.array([0.0, 0.0]),
                frac_sno_eff=jnp.array([0.0, 0.0])
            ),
            "soil": SoilState(
                tkmg=jnp.array([[2.4] * 8, [2.4] * 8]),
                tkdry=jnp.array([[0.21] * 8, [0.21] * 8]),
                csol=jnp.array([[2050000.0] * 8, [2050000.0] * 8]),
                watsat=jnp.array([[0.44] * 8, [0.44] * 8])
            ),
            "params": SoilTemperatureParams(
                thin_sfclayer=1e-6, denh2o=1000.0, denice=917.0, tfrz=273.15,
                tkwat=0.57, tkice=2.29, tkair=0.023, cpice=2117.27,
                cpliq=4188.0, thk_bedrock=3.0, csol_bedrock=2000000.0
            ),
            "dtime": 60.0,
            "nlevsno": 5,
            "nlevgrnd": 8,
            "metadata": {
                "type": "special",
                "description": "Short timestep (60s) to test numerical stability with fine temporal resolution"
            }
        }
    }
    
    return test_cases


# Parametrize test cases
test_case_names = [
    "nominal_single_column_no_snow",
    "nominal_with_snow_layers",
    "multiple_columns_varying_conditions",
    "edge_zero_heat_flux",
    "edge_negative_heat_flux",
    "edge_near_freezing_transition",
    "edge_minimal_water_content",
    "special_maximum_snow_layers",
    "special_shallow_bedrock",
    "special_extreme_timestep_variations"
]


@pytest.mark.parametrize("test_case_name", test_case_names)
def test_compute_soil_temperature_shapes(test_data, test_case_name):
    """
    Test that compute_soil_temperature returns arrays with correct shapes.
    
    Verifies:
    - SoilTemperatureResult.t_soisno has shape (n_cols, nlevgrnd)
    - SoilTemperatureResult.energy_error has shape (n_cols,)
    - ThermalProperties fields have correct shapes
    """
    case = test_data[test_case_name]
    
    result, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    n_cols = case["t_soisno"].shape[0]
    nlevgrnd = case["nlevgrnd"]
    nlevsno = case["nlevsno"]
    n_levtot = nlevsno + nlevgrnd
    
    # Check SoilTemperatureResult shapes
    assert result.t_soisno.shape == (n_cols, nlevgrnd), \
        f"Expected t_soisno shape {(n_cols, nlevgrnd)}, got {result.t_soisno.shape}"
    assert result.energy_error.shape == (n_cols,), \
        f"Expected energy_error shape {(n_cols,)}, got {result.energy_error.shape}"
    
    # Check ThermalProperties shapes
    assert thermal_props.tk.shape == (n_cols, n_levtot), \
        f"Expected tk shape {(n_cols, n_levtot)}, got {thermal_props.tk.shape}"
    assert thermal_props.cv.shape == (n_cols, n_levtot), \
        f"Expected cv shape {(n_cols, n_levtot)}, got {thermal_props.cv.shape}"
    assert thermal_props.tk_h2osfc.shape == (n_cols,), \
        f"Expected tk_h2osfc shape {(n_cols,)}, got {thermal_props.tk_h2osfc.shape}"
    assert thermal_props.thk.shape == (n_cols, n_levtot), \
        f"Expected thk shape {(n_cols, n_levtot)}, got {thermal_props.thk.shape}"
    assert thermal_props.bw.shape == (n_cols, n_levtot), \
        f"Expected bw shape {(n_cols, n_levtot)}, got {thermal_props.bw.shape}"


@pytest.mark.parametrize("test_case_name", test_case_names)
def test_compute_soil_temperature_dtypes(test_data, test_case_name):
    """
    Test that compute_soil_temperature returns arrays with correct data types.
    
    Verifies all output arrays are float32 or float64 (JAX default).
    """
    case = test_data[test_case_name]
    
    result, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Check dtypes are floating point
    assert jnp.issubdtype(result.t_soisno.dtype, jnp.floating), \
        f"t_soisno should be floating point, got {result.t_soisno.dtype}"
    assert jnp.issubdtype(result.energy_error.dtype, jnp.floating), \
        f"energy_error should be floating point, got {result.energy_error.dtype}"
    assert jnp.issubdtype(thermal_props.tk.dtype, jnp.floating), \
        f"tk should be floating point, got {thermal_props.tk.dtype}"
    assert jnp.issubdtype(thermal_props.cv.dtype, jnp.floating), \
        f"cv should be floating point, got {thermal_props.cv.dtype}"


@pytest.mark.parametrize("test_case_name", test_case_names)
def test_compute_soil_temperature_physical_constraints(test_data, test_case_name):
    """
    Test that compute_soil_temperature respects physical constraints.
    
    Verifies:
    - Output temperatures are positive (> 0 K)
    - Output temperatures are in reasonable range (typically 200-350 K)
    - Thermal conductivities are positive
    - Heat capacities are positive
    - No NaN or Inf values in outputs
    """
    case = test_data[test_case_name]
    
    result, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Check temperatures are positive
    assert jnp.all(result.t_soisno > 0), \
        "All output temperatures must be positive (> 0 K)"
    
    # Check temperatures are in reasonable physical range
    assert jnp.all(result.t_soisno >= 200.0), \
        "Output temperatures should be >= 200 K"
    assert jnp.all(result.t_soisno <= 350.0), \
        "Output temperatures should be <= 350 K"
    
    # Check thermal conductivities are positive
    assert jnp.all(thermal_props.tk > 0), \
        "Thermal conductivities must be positive"
    assert jnp.all(thermal_props.thk > 0), \
        "Layer thermal conductivities must be positive"
    assert jnp.all(thermal_props.tk_h2osfc >= 0), \
        "Surface water thermal conductivity must be non-negative"
    
    # Check heat capacities are positive
    assert jnp.all(thermal_props.cv > 0), \
        "Heat capacities must be positive"
    
    # Check for NaN or Inf
    assert jnp.all(jnp.isfinite(result.t_soisno)), \
        "Output temperatures must be finite (no NaN or Inf)"
    assert jnp.all(jnp.isfinite(result.energy_error)), \
        "Energy error must be finite (no NaN or Inf)"
    assert jnp.all(jnp.isfinite(thermal_props.tk)), \
        "Thermal conductivities must be finite (no NaN or Inf)"
    assert jnp.all(jnp.isfinite(thermal_props.cv)), \
        "Heat capacities must be finite (no NaN or Inf)"


def test_compute_soil_temperature_zero_flux_equilibrium(test_data):
    """
    Test that zero heat flux with uniform temperature maintains equilibrium.
    
    With zero ground heat flux and uniform initial temperature, the soil
    temperature should remain nearly constant (small changes due to numerical
    effects are acceptable).
    """
    case = test_data["edge_zero_heat_flux"]
    
    result, _ = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Temperature change should be very small
    initial_temp = case["t_soisno"]
    temp_change = jnp.abs(result.t_soisno - initial_temp)
    
    assert jnp.all(temp_change < 1.0), \
        f"With zero flux, temperature change should be < 1 K, got max change {jnp.max(temp_change):.3f} K"


def test_compute_soil_temperature_negative_flux_cooling(test_data):
    """
    Test that negative (upward) heat flux causes cooling.
    
    With large negative heat flux, surface layers should cool down.
    """
    case = test_data["edge_negative_heat_flux"]
    
    result, _ = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Surface layers should cool (temperature should decrease)
    initial_temp = case["t_soisno"]
    # Check first few layers for cooling
    assert jnp.all(result.t_soisno[:, :3] <= initial_temp[:, :3]), \
        "With negative heat flux, surface layers should cool or stay same"


def test_compute_soil_temperature_freezing_point_handling(test_data):
    """
    Test proper handling of temperatures near freezing point.
    
    Temperatures near 273.15 K should be handled correctly without
    numerical instabilities.
    """
    case = test_data["edge_near_freezing_transition"]
    
    result, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Check that temperatures near freezing are handled properly
    tfrz = case["params"].tfrz
    near_freezing = jnp.abs(result.t_soisno - tfrz) < 5.0
    
    # All temperatures should still be finite and positive
    assert jnp.all(jnp.isfinite(result.t_soisno[near_freezing])), \
        "Temperatures near freezing should be finite"
    assert jnp.all(result.t_soisno[near_freezing] > 0), \
        "Temperatures near freezing should be positive"


def test_compute_soil_temperature_energy_conservation(test_data):
    """
    Test energy conservation by checking energy error magnitude.
    
    Energy error should be small relative to the heat flux magnitude.
    """
    case = test_data["nominal_single_column_no_snow"]
    
    result, _ = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Energy error should be finite
    assert jnp.all(jnp.isfinite(result.energy_error)), \
        "Energy error must be finite"
    
    # Energy error should be relatively small (< 10% of heat flux)
    gsoi_magnitude = jnp.abs(case["gsoi"])
    relative_error = jnp.abs(result.energy_error) / (gsoi_magnitude + 1e-10)
    
    assert jnp.all(relative_error < 0.1), \
        f"Energy error should be < 10% of heat flux, got max {jnp.max(relative_error):.3f}"


def test_compute_soil_temperature_multiple_columns_independence(test_data):
    """
    Test that multiple columns are processed independently.
    
    Different columns with different conditions should produce different results.
    """
    case = test_data["multiple_columns_varying_conditions"]
    
    result, _ = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Results for different columns should be different
    # (since input conditions are different)
    n_cols = result.t_soisno.shape[0]
    
    for i in range(n_cols - 1):
        temp_diff = jnp.abs(result.t_soisno[i] - result.t_soisno[i + 1])
        assert jnp.any(temp_diff > 0.1), \
            f"Columns {i} and {i+1} should have different temperatures"


def test_compute_soil_temperature_snow_layer_effects(test_data):
    """
    Test that snow layers affect thermal properties appropriately.
    
    Cases with snow should have different thermal properties than
    cases without snow.
    """
    case_no_snow = test_data["nominal_single_column_no_snow"]
    case_with_snow = test_data["nominal_with_snow_layers"]
    
    _, thermal_no_snow = compute_soil_temperature(
        geom=case_no_snow["geom"],
        t_soisno=case_no_snow["t_soisno"],
        gsoi=case_no_snow["gsoi"],
        water=case_no_snow["water"],
        soil=case_no_snow["soil"],
        params=case_no_snow["params"],
        dtime=case_no_snow["dtime"],
        nlevsno=case_no_snow["nlevsno"],
        nlevgrnd=case_no_snow["nlevgrnd"]
    )
    
    _, thermal_with_snow = compute_soil_temperature(
        geom=case_with_snow["geom"],
        t_soisno=case_with_snow["t_soisno"],
        gsoi=case_with_snow["gsoi"],
        water=case_with_snow["water"],
        soil=case_with_snow["soil"],
        params=case_with_snow["params"],
        dtime=case_with_snow["dtime"],
        nlevsno=case_with_snow["nlevsno"],
        nlevgrnd=case_with_snow["nlevgrnd"]
    )
    
    # Snow layers should have different thermal conductivity
    # (typically lower than soil)
    snl = int(-case_with_snow["geom"].snl[0])
    if snl > 0:
        snow_tk = thermal_with_snow.tk[0, :snl]
        # Snow thermal conductivity should be relatively low
        assert jnp.all(snow_tk < 1.0), \
            "Snow thermal conductivity should be < 1.0 W/m/K"


def test_compute_soil_temperature_timestep_sensitivity(test_data):
    """
    Test that short timesteps produce stable results.
    
    Very short timesteps (60s) should not cause numerical instabilities.
    """
    case = test_data["special_extreme_timestep_variations"]
    
    result, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Results should be stable (no extreme changes)
    initial_temp = case["t_soisno"]
    temp_change = jnp.abs(result.t_soisno - initial_temp)
    
    # With short timestep, changes should be small
    assert jnp.all(temp_change < 5.0), \
        f"With 60s timestep, temperature change should be < 5 K, got max {jnp.max(temp_change):.3f} K"
    
    # All values should be finite
    assert jnp.all(jnp.isfinite(result.t_soisno)), \
        "Short timestep should produce finite temperatures"


def test_compute_soil_temperature_bedrock_boundary(test_data):
    """
    Test proper handling of bedrock boundary conditions.
    
    Shallow bedrock case should handle thermal properties correctly
    at the bedrock interface.
    """
    case = test_data["special_shallow_bedrock"]
    
    result, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    nbedrock = int(case["geom"].nbedrock[0])
    
    # Bedrock layers should have appropriate thermal properties
    if nbedrock < case["nlevgrnd"]:
        bedrock_tk = thermal_props.thk[0, nbedrock:]
        # Bedrock thermal conductivity should be around thk_bedrock parameter
        expected_tk = case["params"].thk_bedrock
        assert jnp.all(jnp.abs(bedrock_tk - expected_tk) < 1.0), \
            f"Bedrock thermal conductivity should be near {expected_tk} W/m/K"


def test_compute_soil_temperature_dry_soil_limits(test_data):
    """
    Test behavior with minimal water content (dry soil).
    
    Very dry soil should have lower thermal conductivity and
    different heat capacity than wet soil.
    """
    case = test_data["edge_minimal_water_content"]
    
    result, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Dry soil thermal conductivity should be relatively low
    # (closer to tkdry than to wet soil values)
    dry_tk = thermal_props.thk[0, :]
    tkdry = case["soil"].tkdry[0, 0]
    
    # Most values should be closer to dry thermal conductivity
    assert jnp.mean(dry_tk) < 1.0, \
        "Dry soil thermal conductivity should be relatively low (< 1.0 W/m/K)"


def test_compute_soil_temperature_maximum_snow_depth(test_data):
    """
    Test handling of maximum snow layers (5 layers).
    
    Maximum snow configuration should be processed correctly without
    array indexing errors.
    """
    case = test_data["special_maximum_snow_layers"]
    
    result, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Should handle 5 snow layers correctly
    snl = int(-case["geom"].snl[0])
    assert snl == 5, "Test case should have 5 snow layers"
    
    # All outputs should be valid
    assert result.t_soisno.shape[1] == case["nlevgrnd"], \
        "Output should have correct number of ground layers"
    assert jnp.all(jnp.isfinite(result.t_soisno)), \
        "All temperatures should be finite with maximum snow layers"


@pytest.mark.parametrize("test_case_name", test_case_names)
def test_compute_soil_temperature_input_preservation(test_data, test_case_name):
    """
    Test that input arrays are not modified by the function.
    
    JAX functions should not modify input arrays in place.
    """
    case = test_data[test_case_name]
    
    # Make copies of inputs
    t_soisno_original = case["t_soisno"].copy()
    gsoi_original = case["gsoi"].copy()
    
    _ = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Check inputs are unchanged
    assert jnp.allclose(case["t_soisno"], t_soisno_original, atol=1e-10), \
        "Input t_soisno should not be modified"
    assert jnp.allclose(case["gsoi"], gsoi_original, atol=1e-10), \
        "Input gsoi should not be modified"


def test_compute_soil_temperature_thermal_conductivity_bounds(test_data):
    """
    Test that thermal conductivities are within physically reasonable bounds.
    
    Thermal conductivities should be:
    - Air: ~0.023 W/m/K
    - Water: ~0.57 W/m/K
    - Ice: ~2.29 W/m/K
    - Soil minerals: 0.5-5 W/m/K
    - Bedrock: ~3 W/m/K
    """
    case = test_data["nominal_single_column_no_snow"]
    
    _, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Thermal conductivity should be within reasonable bounds
    assert jnp.all(thermal_props.tk >= 0.01), \
        "Thermal conductivity should be >= 0.01 W/m/K (above air)"
    assert jnp.all(thermal_props.tk <= 10.0), \
        "Thermal conductivity should be <= 10.0 W/m/K (reasonable upper bound)"
    
    # Layer thermal conductivity bounds
    assert jnp.all(thermal_props.thk >= 0.01), \
        "Layer thermal conductivity should be >= 0.01 W/m/K"
    assert jnp.all(thermal_props.thk <= 10.0), \
        "Layer thermal conductivity should be <= 10.0 W/m/K"


def test_compute_soil_temperature_heat_capacity_bounds(test_data):
    """
    Test that heat capacities are within physically reasonable bounds.
    
    Volumetric heat capacities should be:
    - Dry soil: ~1-2 MJ/m³/K
    - Wet soil: ~2-4 MJ/m³/K
    - Ice: ~1.9 MJ/m³/K
    - Water: ~4.2 MJ/m³/K
    """
    case = test_data["nominal_single_column_no_snow"]
    
    _, thermal_props = compute_soil_temperature(
        geom=case["geom"],
        t_soisno=case["t_soisno"],
        gsoi=case["gsoi"],
        water=case["water"],
        soil=case["soil"],
        params=case["params"],
        dtime=case["dtime"],
        nlevsno=case["nlevsno"],
        nlevgrnd=case["nlevgrnd"]
    )
    
    # Heat capacity should be within reasonable bounds (J/m²/K)
    # For a layer, this depends on thickness, but should be positive and reasonable
    assert jnp.all(thermal_props.cv > 0), \
        "Heat capacity must be positive"
    assert jnp.all(thermal_props.cv < 1e8), \
        "Heat capacity should be < 1e8 J/m²/K (reasonable upper bound)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])