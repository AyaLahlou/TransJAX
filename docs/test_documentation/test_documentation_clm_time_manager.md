# Test Documentation: clm_time_manager Module

## Test Suite Overview

This test suite provides comprehensive coverage for the `clm_time_manager` module, which handles time management and calendar operations for the Community Land Model (CLM). The module includes functions for date/time arithmetic, leap year detection, and timestep management across different calendar types.

### Functions Tested

The test suite covers **14 functions** across three main categories:

1. **State Query Functions** (5 functions)
   - `get_step_size()` - Retrieves timestep size in seconds
   - `get_nstep()` - Gets current timestep number
   - `get_curr_date()` - Returns current date components
   - `get_prev_date()` - Returns previous timestep date
   - `get_curr_time()` - Returns current time as days and seconds

2. **Calendar Functions** (7 functions)
   - `isleap()` - Checks if year is leap year (Python version)
   - `isleap_jax()` - Checks if year is leap year (JAX version)
   - `get_curr_calday()` - Calculates calendar day with optional offset
   - `get_prev_calday()` - Gets calendar day at timestep start
   - `get_curr_date_tuple()` - Returns CurrentDate NamedTuple
   - `is_end_curr_day()` - Checks if at end of day
   - `is_end_curr_month()` - Checks if at end of month

3. **State Management Functions** (2 functions)
   - `create_time_manager_state()` - Initializes time manager state
   - `advance_timestep()` - Advances simulation by one timestep

### Test Statistics

- **Total Test Cases**: 10 parametrized test scenarios
- **Test Types**:
  - **Nominal Cases** (40%): Standard operational scenarios
  - **Edge Cases** (40%): Boundary conditions and special dates
  - **Special Cases** (20%): Calendar-specific behavior (leap years, month/year boundaries)
- **Calendar Coverage**: Both GREGORIAN and NOLEAP calendars
- **Temporal Coverage**: Years 1900-2100, all months, various times of day

### Coverage Areas

The test suite ensures coverage of:

- ✅ **Leap year detection** (divisible by 4, 100, 400 rules)
- ✅ **Calendar arithmetic** (day/month/year transitions)
- ✅ **Time-of-day handling** (0-86399 seconds)
- ✅ **Timestep advancement** (single and multiple steps)
- ✅ **Date format conversions** (yyyymmdd ↔ year/month/day)
- ✅ **Calendar day calculations** (1.0-366.0 range)
- ✅ **Boundary detection** (end of day/month)
- ✅ **JAX array operations** (vectorized leap year checks)
- ✅ **Error handling** (invalid offsets, date ranges)

---

## Running the Tests

### Basic Execution

```bash
# Run all tests
pytest test_clm_time_manager.py

# Run with verbose output
pytest test_clm_time_manager.py -v

# Run specific test function
pytest test_clm_time_manager.py::test_get_step_size -v

# Run tests matching pattern
pytest test_clm_time_manager.py -k "leap" -v
```

### Coverage Analysis

```bash
# Generate coverage report
pytest test_clm_time_manager.py --cov=clm_time_manager

# Generate HTML coverage report
pytest test_clm_time_manager.py --cov=clm_time_manager --cov-report=html

# Show missing lines
pytest test_clm_time_manager.py --cov=clm_time_manager --cov-report=term-missing
```

### Advanced Options

```bash
# Run with detailed output and stop on first failure
pytest test_clm_time_manager.py -vv -x

# Run only failed tests from last run
pytest test_clm_time_manager.py --lf

# Run tests in parallel (requires pytest-xdist)
pytest test_clm_time_manager.py -n auto

# Generate JUnit XML report
pytest test_clm_time_manager.py --junitxml=test-results.xml
```

---

## Test Cases

### Nominal Cases (Standard Operations)

**Purpose**: Verify correct behavior under typical operating conditions

1. **Mid-year, mid-month dates**
   - Date: 2000-06-15 12:00:00
   - Tests: Standard date arithmetic, calendar day calculation
   - Expected: Smooth progression through timesteps

2. **Start of year**
   - Date: 2000-01-01 00:00:00
   - Tests: Year boundary handling, calendar day = 1.0
   - Expected: Correct initialization and advancement

3. **Standard timestep progression**
   - Timestep: 1800 seconds (30 minutes)
   - Tests: Multiple timestep advancement, time accumulation
   - Expected: Accurate time tracking over multiple steps

4. **Non-leap year operations**
   - Year: 1999 (GREGORIAN calendar)
   - Tests: February has 28 days, year has 365 days
   - Expected: Correct month lengths and year transitions

### Edge Cases (Boundary Conditions)

**Purpose**: Test behavior at critical boundaries and limits

1. **Leap year boundaries**
   - Years: 1900 (not leap), 2000 (leap), 2004 (leap), 2100 (not leap)
   - Tests: Century rule (divisible by 100 but not 400)
   - Expected: Correct leap year detection per Gregorian rules

2. **End of month transitions**
   - Dates: Jan 31, Feb 28/29, Apr 30, Dec 31
   - Tests: Month rollover, varying month lengths
   - Expected: Correct advancement to next month

3. **End of day transitions**
   - Time: 23:59:30 with 30-second timestep
   - Tests: Day boundary crossing, time-of-day reset
   - Expected: Correct day increment and TOD = 0

4. **Midnight (time-of-day = 0)**
   - TOD: 0 seconds
   - Tests: Start-of-day calculations, calendar day fractional part
   - Expected: Integer calendar day values

5. **End of day (time-of-day = 86399)**
   - TOD: 86399 seconds (23:59:59)
   - Tests: Maximum valid TOD, next-second rollover
   - Expected: Correct boundary detection

6. **Small timesteps**
   - Timestep: 1 second
   - Tests: Fine-grained time progression, accumulation accuracy
   - Expected: No rounding errors over many steps

7. **Large timesteps**
   - Timestep: 3600 seconds (1 hour)
   - Tests: Coarse time progression, multiple-hour jumps
   - Expected: Correct multi-hour advancement

### Special Cases (Calendar-Specific Behavior)

**Purpose**: Verify calendar-dependent logic and special scenarios

1. **NOLEAP calendar**
   - Calendar: "NOLEAP"
   - Tests: All years have 365 days, February always 28 days
   - Expected: `isleap()` always returns False

2. **GREGORIAN calendar leap years**
   - Years: 2000, 2004, 2020, 2024
   - Tests: Standard leap year detection
   - Expected: `isleap()` returns True for these years

3. **February 29 in leap years**
   - Date: 2000-02-29
   - Tests: Valid date in leap year, calendar day calculation
   - Expected: Accepted as valid date, calday ≈ 60.0

4. **Year transitions**
   - Date: 2000-12-31 23:59:00 with 60-second timestep
   - Tests: Year rollover, date format updates
   - Expected: Correct transition to 2001-01-01

5. **Calendar day with negative offset**
   - Offset: -3600 (1 hour in past)
   - Tests: Historical time calculation
   - Expected: Correct calendar day for past time

6. **JAX array leap year detection**
   - Input: Array of years [1900, 2000, 2004, 2100]
   - Tests: Vectorized operations, array shape preservation
   - Expected: Boolean array [False, True, True, False]

---

## Test Data

### Generation Strategy

Test data was systematically generated to cover:

1. **Temporal Diversity**
   - Years: 1900-2100 (spanning 3 centuries)
   - Months: All 12 months
   - Days: 1st, 15th, 28th-31st (month boundaries)
   - Times: 0:00, 12:00, 23:59 (start, middle, end of day)

2. **Timestep Variations**
   - 1 second (fine-grained)
   - 30 seconds (sub-minute)
   - 1800 seconds (30 minutes, common CLM timestep)
   - 3600 seconds (1 hour, coarse timestep)

3. **Calendar Configurations**
   - GREGORIAN: Standard calendar with leap years
   - NOLEAP: Perpetual 365-day years

4. **Date Format Coverage**
   - yyyymmdd integers: 19000101, 20000615, 21001231
   - Time-of-day seconds: 0, 43200, 86399
   - Calendar days: 1.0 (Jan 1) to 366.0 (Dec 31 leap year)

### Physical Realism

All test data respects physical constraints:

- ✅ Years ≥ 1 (valid calendar years)
- ✅ Months in [1, 12]
- ✅ Days in [1, 31] (respecting month lengths)
- ✅ Time-of-day in [0, 86399] seconds
- ✅ Timesteps > 0 seconds
- ✅ Calendar days in [1.0, 366.0]

### Fixtures

The test suite uses pytest fixtures for reusable test data:

```python
@pytest.fixture
def sample_states():
    """Provides 10 diverse TimeManagerState instances"""
    # Returns list of states covering various scenarios
```

This fixture is parametrized across all test functions, ensuring consistent coverage.

---

## Expected Behavior

### Passing Tests

Tests **should pass** when:

1. **Leap year detection is correct**
   - 2000: leap (divisible by 400)
   - 2004: leap (divisible by 4, not by 100)
   - 1900: not leap (divisible by 100, not by 400)
   - 2100: not leap (divisible by 100, not by 400)

2. **Date arithmetic is accurate**
   - Advancing from Jan 31 → Feb 1
   - Advancing from Feb 28 (non-leap) → Mar 1
   - Advancing from Feb 29 (leap) → Mar 1
   - Advancing from Dec 31 → Jan 1 (next year)

3. **Time-of-day handling is correct**
   - TOD + timestep < 86400: same day, updated TOD
   - TOD + timestep ≥ 86400: next day, TOD wraps

4. **Calendar day calculations are precise**
   - Jan 1 at midnight: calday = 1.0
   - Jan 1 at noon: calday = 1.5
   - Dec 31 at midnight (non-leap): calday = 365.0
   - Dec 31 at midnight (leap): calday = 366.0

5. **State management is consistent**
   - `advance_timestep()` increments `itim` by 1
   - `get_nstep()` returns current `itim`
   - `get_step_size()` returns `dtstep`

### Failing Tests

Tests **should fail** if:

1. **Leap year logic is incorrect**
   - Treating 1900 or 2100 as leap years
   - Not recognizing 2000 as a leap year
   - NOLEAP calendar returns True for any year

2. **Date boundaries are mishandled**
   - Month doesn't advance on last day
   - Year doesn't advance on Dec 31
   - February 29 accepted in non-leap years

3. **Time accumulation has errors**
   - Rounding errors in calendar day calculation
   - TOD exceeds 86399
   - Negative time values

4. **JAX operations fail**
   - `isleap_jax()` doesn't preserve array shape
   - Non-boolean return type
   - Incorrect vectorization

5. **State updates are inconsistent**
   - `curr_date_ymd` doesn't match computed date
   - `itim` doesn't increment
   - Calendar type changes unexpectedly

### Assertion Details

Tests use descriptive assertions:

```python
assert result == expected, (
    f"Expected {expected} for year {year} "
    f"with {calendar} calendar, got {result}"
)
```

This provides clear failure messages for debugging.

---

## Extending Tests

### Adding New Test Cases

1. **Add to fixture data**:

```python
@pytest.fixture
def sample_states():
    states = [
        # ... existing states ...
        create_time_manager_state(
            dtstep=900,  # 15-minute timestep
            start_date_ymd=20500701,  # July 1, 2050
            start_date_tod=0,
            calendar="GREGORIAN"
        ),
    ]
    return states
```

2. **Create specialized test**:

```python
def test_quarter_hour_timestep():
    """Test 15-minute timestep progression."""
    state = create_time_manager_state(
        dtstep=900,
        start_date_ymd=20000101,
        start_date_tod=0
    )
    
    # Advance 96 steps (24 hours)
    for _ in range(96):
        state = advance_timestep(state)
    
    year, month, day, tod = get_curr_date(state)
    assert (year, month, day) == (2000, 1, 2)
    assert tod == 0
```

### Testing New Functions

When adding new functions to `clm_time_manager`:

1. **Analyze function signature**:
   - Input types and constraints
   - Return type and range
   - Edge cases and error conditions

2. **Design test cases**:
   - Nominal: typical inputs
   - Edge: boundaries, limits
   - Error: invalid inputs (if applicable)

3. **Use parametrize for variations**:

```python
@pytest.mark.parametrize("year,expected", [
    (2000, True),
    (2001, False),
    (2004, True),
])
def test_new_leap_function(year, expected):
    result = new_leap_function(year)
    assert result == expected
```

### Property-Based Testing

For complex functions, consider property-based testing with Hypothesis:

```python
from hypothesis import given, strategies as st

@given(
    year=st.integers(min_value=1, max_value=9999),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28)  # Safe for all months
)
def test_date_roundtrip(year, month, day):
    """Test that date encoding/decoding is reversible."""
    ymd = year * 10000 + month * 100 + day
    state = create_time_manager_state(
        dtstep=1800,
        start_date_ymd=ymd,
        start_date_tod=0
    )
    y, m, d, _ = get_curr_date(state)
    assert (y, m, d) == (year, month, day)
```

---

## Common Issues

### Issue 1: JAX Array Comparison Failures

**Symptom**: Tests fail with "Arrays are not equal" despite values appearing identical.

**Cause**: JAX arrays require special comparison methods.

**Solution**:
```python
import jax.numpy as jnp

# ❌ Don't use: assert jax_array == expected
# ✅ Use:
assert jnp.allclose(jax_array, expected)
# or
assert jnp.array_equal(jax_array, expected)
```

### Issue 2: Leap Year Century Rule Confusion

**Symptom**: Tests fail for years 1900, 2100.

**Cause**: Forgetting that century years must be divisible by 400 to be leap years.

**Solution**: Remember the rule:
- Divisible by 4: **maybe** leap
- Divisible by 100: **not** leap (unless...)
- Divisible by 400: **is** leap

```python
# 1900: divisible by 100 but not 400 → NOT leap
# 2000: divisible by 400 → IS leap
# 2100: divisible by 100 but not 400 → NOT leap
```

### Issue 3: Time-of-Day Overflow

**Symptom**: TOD values exceed 86399 or become negative.

**Cause**: Not handling day boundary correctly.

**Solution**: Ensure wraparound logic:
```python
new_tod = (old_tod + dtstep) % 86400
days_advanced = (old_tod + dtstep) // 86400
```

### Issue 4: Calendar Day Precision

**Symptom**: Calendar day calculations are slightly off (e.g., 1.0000001 instead of 1.0).

**Cause**: Floating-point arithmetic accumulation.

**Solution**: Use appropriate tolerance in assertions:
```python
assert abs(calday - expected) < 1e-6
# or
assert pytest.approx(calday) == expected
```

### Issue 5: NOLEAP Calendar Confusion

**Symptom**: Tests expect leap years in NOLEAP calendar.

**Cause**: Not checking calendar type before leap year logic.

**Solution**: Always verify calendar:
```python
if state.calkindflag == "NOLEAP":
    assert not isleap(year, calendar="NOLEAP")
else:
    # Apply Gregorian rules
    assert isleap(year, calendar="GREGORIAN") == expected
```

### Issue 6: Month Length Variations

**Symptom**: Tests fail when advancing from month-end dates.

**Cause**: Not accounting for different month lengths (28/29/30/31 days).

**Solution**: Use month-aware logic:
```python
# February in leap year: 29 days
# February in non-leap: 28 days
# April, June, Sept, Nov: 30 days
# Others: 31 days
```

### Issue 7: Fixture Scope Issues

**Symptom**: State modifications persist across tests.

**Cause**: Fixture scope is too broad or state is mutable.

**Solution**: Use function-scoped fixtures:
```python
@pytest.fixture(scope="function")  # New instance per test
def sample_states():
    return [create_time_manager_state(...)]
```

### Issue 8: Parametrize ID Collisions

**Symptom**: Pytest shows duplicate test IDs.

**Cause**: Multiple test cases with identical parameters.

**Solution**: Add explicit IDs:
```python
@pytest.mark.parametrize("state,expected", [
    (state1, val1),
    (state2, val2),
], ids=["case1", "case2"])
```

---

## Performance Considerations

### JAX JIT Compilation

First test run may be slower due to JIT compilation:

```bash
# First run: ~2-3 seconds (includes compilation)
pytest test_clm_time_manager.py

# Subsequent runs: ~0.5-1 second (cached)
pytest test_clm_time_manager.py
```

### Parallel Execution

For large test suites, use parallel execution:

```bash
pip install pytest-xdist
pytest test_clm_time_manager.py -n auto
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test CLM Time Manager

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest test_clm_time_manager.py --cov=clm_time_manager --cov-report=xml
      - uses: codecov/codecov-action@v2
```

---

## References

- **pytest documentation**: https://docs.pytest.org/
- **JAX testing guide**: https://jax.readthedocs.io/en/latest/notebooks/testing_cookbook.html
- **CLM documentation**: Community Land Model technical documentation
- **Calendar algorithms**: Dershowitz & Reingold, "Calendrical Calculations"

---

## Maintenance

**Last Updated**: 2024
**Test Suite Version**: 1.0
**Maintainer**: Scientific Computing Team

For questions or issues, please open a GitHub issue or contact the development team.