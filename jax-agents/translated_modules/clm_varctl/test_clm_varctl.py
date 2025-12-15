from clm_varctl import create_clm_varctl, update_clm_varctl, DEFAULT_CLM_VARCTL

# Use default configuration
ctl = DEFAULT_CLM_VARCTL

# Create custom configuration
custom_ctl = create_clm_varctl(iulog=10)

# Update configuration
new_ctl = update_clm_varctl(ctl, iulog=7)

# Access values
log_unit = ctl.iulog