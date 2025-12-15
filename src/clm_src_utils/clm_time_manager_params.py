from clm_time_manager import (
    TimeManagerState,
    create_time_manager_state,
    advance_timestep,
    get_curr_date,
    get_curr_calday,
)

# Initialize
state = create_time_manager_state(
    dtstep=1800,  # 30 minutes
    start_date_ymd=19790101,
    start_date_tod=0,
)

# Advance simulation
for step in range(num_steps):
    state = advance_timestep(state)
    year, month, day, tod = get_curr_date(state)
    calday, error = get_curr_calday(state)