# tests/test_motor_reaction_time.py

import pytest
from force_analyses import motor_reaction_time

@pytest.mark.parametrize(
    "_test_id, stim_time, onset_time, expected_result",
    [
        # Happy Path: Onset is after stimulus, expects an integer
        ("Normal case", 1.5, 2.0, int(500)),
        # Bad Path: Onset was not detected, expects None
        ("Onset is None", 1.5, None, None),
        # Edge Case: Onset occurs before stimulus, expects None
        ("False start", 1.5, 1.2, None),
    ]
)
def test_motor_reaction_time(_test_id, stim_time, onset_time, expected_result):
    """
    Tests the motor_reaction_time wrapper function for happy and bad paths.
    """
    # Call the function under test with direct inputs
    result = motor_reaction_time(stim_time, onset_time)
    
    # --- Assertions ---
    # -------------------
    if expected_result is not None:
        # For the happy path, check the type and value
        assert isinstance(result, int)
        assert result == expected_result
    else:
        # For bad paths, assert the result is None
        assert result is None