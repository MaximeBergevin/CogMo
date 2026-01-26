# Third-Party Dependencies
import pytest
import numpy as np
import pandas as pd

# Local Application Imports
from emg_analyses import calculate_emg_rms

@pytest.mark.parametrize(
    "_test_id, snr_db, emd_s, expected_rms",
    [
        ("High SNR - Clean", 20.0, 0.05, 0.1),    # sigma = 0.1
        ("Moderate SNR", 10.0, 0.03, 0.0316),     # sigma = sqrt(0.0001 * 10^1)
        ("Low SNR - Noisy", 6.0, 0.08, 0.02),     # sigma = sqrt(0.0001 * 10^0.6)
    ]
)
def test_calculate_emg_rms_accuracy(
    _test_id, snr_db, emd_s, expected_rms, 
    mock_signal_data_factory
):
    """
    Checks the RMS accuracy across different signal-to-noise ratios and delays.
    """
    # Setup mock data using parameters
    mock_df, expected = mock_signal_data_factory(
        include_emg=True,
        snr_db=snr_db,
        emd_s=emd_s,
        dominant_force="right"
    )
    
    channel_map = {"emg_right": "emg_right", "emg_left": "emg_left"}

    # 2. Call function under test
    rms_value = calculate_emg_rms(
        full_df=mock_df,
        channel_map=channel_map,
        response_hand="right",
        onset_time=expected['expected_emg_onset'],
        offset_time=expected['expected_offset_time']
    )

    # 3. Accuracy Assertions
    assert rms_value is not None, f"Failed for {_test_id}"
    assert rms_value == pytest.approx(expected_rms, abs = 0.15)


def test_calculate_emg_rms_invalid_inputs():
    """Tests that the function handles None or empty segments gracefully."""
    df = pd.DataFrame({
        "time": [0.0, 0.1], 
        "emg_right": [0.01, 0.02]
    })
    channel_map = {"emg_right": "emg_right"}
    
    # Check for None inputs or out-of-bounds segments
    assert calculate_emg_rms(df, channel_map, "right", None, 0.1) is None
    assert calculate_emg_rms(df, channel_map, "right", 5.0, 6.0) is None