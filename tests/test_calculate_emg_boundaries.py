# tests/test_find_emg_boundaries.py

import pytest
import numpy as np
import pandas as pd
from emg_analyses import find_emg_boundaries

@pytest.mark.parametrize(
    "_test_id, snr_db, emd_s, fs, expected_valid",
    [
        ("Happy Path: Standard EMD", 25.0, 0.050, 1000, True),
        ("Happy Path: Short EMD", 20.0, 0.030, 1000, True),
    ]
)
def test_find_emg_boundaries_parametrized(
    _test_id, snr_db, emd_s, fs, expected_valid,
    mock_signal_data_factory,
    threshold_on = 0.001, threshold_off = 0.0005
):
    """
    Tests EMG onset/offset detection using a sampling rate that 
    actually supports the bandpass filter frequencies.
    """
    mock_df, expected = mock_signal_data_factory(
        include_emg=True,
        snr_db=snr_db,
        emd_s=emd_s,
        sampling_rate_hz=fs,
        dominant_force="right",
        burst_time_s=1
    )
    
    channel_map = {"emg_right": "emg_right", "emg_left": "emg_left"}
    
    # 2. Call the function
    # Note: the function internally calculates fs from the time column
    onset, offset, _ = find_emg_boundaries(
        signal_df=mock_df,
        channel_map=channel_map,
        response_hand="right",
        stim_time=expected['stim_time_exact'],
        force_onset_time=expected['expected_force_onset'],
        force_offset_time=expected['expected_offset_time'],
        min_burst_ms=20,
        threshold_on=0.005,
        threshold_off=0.002
    )

    # 3. Assertions
    if expected_valid:
        assert onset is not None
        # High tolerance because of stochastic nature of EMG + imperfect syntethic data processing
        # Most simulation appear to be in the range of ~70-100 ms error, ocassionaly more
        # Outside simulation with non-simulated truncadated EMG burst managed to show ~30ms delayed
        assert onset == pytest.approx(expected['expected_emg_onset'], abs=0.2)
        assert offset == pytest.approx(expected['expected_offset_time'], abs=0.3)
    else:
        assert onset is None