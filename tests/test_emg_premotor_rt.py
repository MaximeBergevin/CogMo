import pytest
from typing import Optional
from emg_analyses import premotor_reaction_time

@pytest.mark.parametrize("stim_time, emg_onset, expected", [
    (1.0, 1.050, 50),     # 50ms latency
    (0.0, 0.1234, 123),   # Rounding check (123.4 -> 123)
    (0.0, 0.1236, 124),   # Rounding check (123.6 -> 124)
    (10.5, 10.600, 100),  # Late stimulus time
    (0.0, 0.0, 0),        # Instantaneous (theoretical limit)
])
def test_premotor_rt_happy_path(stim_time, emg_onset, expected):
    """Test standard valid latency calculations and rounding."""
    result = premotor_reaction_time(stim_time, emg_onset)
    assert result == expected
    assert isinstance(result, int)

def test_premotor_rt_none_onset():
    """Ensure None is returned if no EMG onset was detected."""
    assert premotor_reaction_time(1.0, None) is None

def test_premotor_rt_invalid_sequence():
    """
    Ensure None is returned if EMG onset is recorded BEFORE stimulus.
    Physiologically impossible for reaction time, likely noise/artifact.
    """
    # Stimulus at 1.0s, EMG 'onset' at 0.9s
    assert premotor_reaction_time(1.0, 0.9) is None

def test_premotor_rt_large_values():
    """Test with large timestamps to ensure precision remains."""
    stim = 3600.0  # 1 hour into recording
    onset = 3600.2505
    # (0.2505 * 1000) = 250.5 -> rounded to 251
    assert premotor_reaction_time(stim, onset) == 251