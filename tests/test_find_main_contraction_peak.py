import pytest
import pandas as pd
import numpy as np
from force_analyses import find_main_contraction_peak


def test_hand_specific_threshold_and_relaxation_tail_rejection():
    """
    Simulates a trial where:
    - Pre-stimulus has a decaying tail on the RIGHT hand starting at 200 N (from trial N-1).
    - Post-stimulus (at t = 1.4s, stim_time = 1.0s) has an active voluntary LEFT contraction peaking at 140 N.
    - Right threshold is 175 N (30% of 583 N).
    - Left threshold is 118 N (30% of 393 N).
    
    The detector must:
    1. Reject the right-hand decay tail (which has no positive prominence and is pre-stimulus).
    2. Detect the left contraction at 140 N.
    3. Benchmark 140 N against the LEFT threshold (118 N), resulting in status='valid'.
    """
    time = np.linspace(0.0, 3.0, 3000)
    stim_time = 1.0

    # Right hand: relaxation tail from 0.0 to 0.8s (monotonically decreasing from 220 N to 0 N)
    force_right = np.zeros_like(time)
    tail_mask = time < 0.8
    force_right[tail_mask] = 220.0 * np.exp(-time[tail_mask] * 6.0)

    # Left hand: voluntary contraction peaking at t = 1.4s (amplitude 140 N, duration ~400ms)
    force_left = np.zeros_like(time)
    contraction_mask = (time >= 1.2) & (time <= 1.6)
    force_left[contraction_mask] = 140.0 * np.sin(np.pi * (time[contraction_mask] - 1.2) / 0.4)

    df = pd.DataFrame({
        'time': time,
        'force_right': force_right,
        'force_left': force_left
    })

    thresh_r = 175.0
    thresh_l = 118.0

    result = find_main_contraction_peak(
        full_df=df,
        stim_time=stim_time,
        threshold_left=thresh_l,
        threshold_right=thresh_r,
        min_valid_rt_s=0.100,
        min_prominence_n=5.0,
        search_window_pre_s=0.5,
        search_window_post_s=2.0
    )

    assert result['status'] == 'valid'
    assert result['response_hand'] == 'left'
    assert result['peak_value'] == pytest.approx(140.0, rel=1e-2)
    assert result['threshold'] == pytest.approx(thresh_l, rel=1e-2)


def test_motor_error_evaluated_against_responding_hand():
    """
    Tests that a weak voluntary contraction (below hand threshold) is categorized
    as low_force_candidate against that hand's threshold.
    """
    time = np.linspace(0.0, 3.0, 3000)
    stim_time = 1.0

    force_right = np.zeros_like(time)
    force_left = np.zeros_like(time)

    # Squeeze right hand to only 100 N (threshold is 175 N)
    contraction_mask = (time >= 1.3) & (time <= 1.7)
    force_right[contraction_mask] = 100.0 * np.sin(np.pi * (time[contraction_mask] - 1.3) / 0.4)

    df = pd.DataFrame({
        'time': time,
        'force_right': force_right,
        'force_left': force_left
    })

    result = find_main_contraction_peak(
        full_df=df,
        stim_time=stim_time,
        threshold_left=118.0,
        threshold_right=175.0,
        min_valid_rt_s=0.100,
        min_prominence_n=5.0
    )

    assert result['status'] == 'low_force_candidate'
    assert result['response_hand'] == 'right'
    assert result['peak_value'] == pytest.approx(100.0, rel=1e-2)
    assert result['threshold'] == pytest.approx(175.0, rel=1e-2)
