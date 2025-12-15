from clm_varorb import create_orbital_params, update_orbital_params
from clm_varorb_params import DEFAULT_ECCEN, DEFAULT_OBLIQR

# Initialize with defaults
params = create_orbital_params(
    eccen=DEFAULT_ECCEN,
    obliqr=DEFAULT_OBLIQR
)

# Use in JIT-compiled function
@jax.jit
def compute_solar_declination(params, day_of_year):
    # Would use params.obliqr, params.eccen, etc.
    pass

# Update for sensitivity study
new_params = update_orbital_params(params, eccen=0.02)