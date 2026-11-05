import pytest
import numpy as np
import pandas as pd
from force_analyses import calculate_rfd, DEFAULT_RFD_WINDOWS


def test_calculate_rfd_linear_ramp():
    """
    Tests calculate_rfd with a predictable linear slope.
    Slope = 1000 N/s from t=1.0 to t=1.3s.
    """
    # 1000 Hz sampling (dt = 0.001s)
    time = np.linspace(0.0, 2.0, 2001)
    onset_time = 1.0
    peak_time = 1.300
    baseline_force = 10.0
    mvc_val = 200.0

    # Linear increase: force starts at baseline (10 N) at t=1.0 and rises by 1000 N/s
    force = np.full_like(time, baseline_force)
    ramp_mask = (time >= onset_time) & (time <= peak_time)
    force[ramp_mask] = baseline_force + 1000.0 * (time[ramp_mask] - onset_time)

    df = pd.DataFrame({
        'time': time,
        'force_right': force,
        'force_left': np.zeros_like(time)
    })

    metrics = calculate_rfd(
        signal_df=df,
        onset_time=onset_time,
        peak_time=peak_time,
        baseline_force=baseline_force,
        response_hand='right',
        mvc_value=mvc_val
    )

    # Check all canonical battery windows
    for start, end in DEFAULT_RFD_WINDOWS:
        key = f"rfd_{start}_{end}"
        pct_key = f"{key}_pct"
        assert metrics[key] is not None, f"Missing {key}"
        assert np.isclose(metrics[key], 1000.0, atol=1.0), f"Incorrect {key}"
        assert np.isclose(metrics[pct_key], (1000.0 / mvc_val) * 100.0, atol=0.5), f"Incorrect {pct_key}"

    # Custom windows 1 and 2 (default: 0-50, 100-200)
    assert np.isclose(metrics['custom_rfd_1'], 1000.0, atol=1.0)
    assert np.isclose(metrics['custom_rfd_2'], 1000.0, atol=1.0)
    assert metrics['custom_rfd_1_window'] == "0-50"
    assert metrics['custom_rfd_2_window'] == "100-200"

    # Backward compatibility aliases
    assert np.isclose(metrics['early_rfd'], 1000.0, atol=1.0)
    assert np.isclose(metrics['late_rfd'], 1000.0, atol=1.0)

    # Peak RFD
    assert metrics['peak_rfd'] is not None
    assert np.isclose(metrics['peak_rfd'], 1000.0, atol=1.0)
    assert np.isclose(metrics['peak_rfd_pct'], (1000.0 / mvc_val) * 100.0, atol=0.5)


def test_calculate_rfd_distinct_slopes():
    """
    Tests calculate_rfd with distinct slopes across:
    0-50ms: slope = 500 N/s
    50-100ms: slope = 1200 N/s
    100-200ms: slope = 2000 N/s
    """
    time = np.linspace(0.0, 2.0, 2001) # 1000 Hz
    onset_time = 0.500
    peak_time = 0.750
    baseline_force = 5.0
    mvc_val = 100.0

    force = np.full_like(time, baseline_force)

    # 0-50ms (0.500 to 0.550): rise by 500 N/s -> Delta F = 25 N
    mask_early = (time >= 0.500) & (time <= 0.550)
    force[mask_early] = baseline_force + 500.0 * (time[mask_early] - 0.500)

    # 50-100ms (0.550 to 0.600): rise by 1200 N/s -> Delta F = 60 N
    force_at_50 = baseline_force + 25.0
    mask_mid = (time > 0.550) & (time <= 0.600)
    force[mask_mid] = force_at_50 + 1200.0 * (time[mask_mid] - 0.550)

    # 100-200ms (0.600 to 0.700): rise by 2000 N/s -> Delta F = 200 N
    force_at_100 = force_at_50 + 60.0
    mask_late = (time > 0.600) & (time <= 0.700)
    force[mask_late] = force_at_100 + 2000.0 * (time[mask_late] - 0.600)

    df = pd.DataFrame({
        'time': time,
        'force_left': force,
        'force_right': np.zeros_like(time)
    })

    metrics = calculate_rfd(
        signal_df=df,
        onset_time=onset_time,
        peak_time=peak_time,
        baseline_force=baseline_force,
        response_hand='left',
        mvc_value=mvc_val
    )

    # 0-50ms should be ~500 N/s
    assert np.isclose(metrics['rfd_0_50'], 500.0, atol=2.0)
    assert np.isclose(metrics['rfd_0_50_pct'], (500.0 / mvc_val) * 100.0, atol=1.0)

    # 50-100ms should be ~1200 N/s
    assert np.isclose(metrics['rfd_50_100'], 1200.0, atol=2.0)

    # 100-200ms should be ~2000 N/s
    assert np.isclose(metrics['rfd_100_200'], 2000.0, atol=2.0)
    assert np.isclose(metrics['rfd_100_200_pct'], (2000.0 / mvc_val) * 100.0, atol=1.0)

    # Peak RFD should capture the steepest slope (~2000 N/s)
    assert np.isclose(metrics['peak_rfd'], 2000.0, atol=10.0)


def test_calculate_rfd_short_contraction():
    """
    If peak occurs at 70ms, windows extending past 70ms:
    - 0-50, 0-25 are valid
    - 25-75 computes over available 25-70ms
    - 100-200 starts after peak -> returns None
    """
    time = np.linspace(0.0, 1.0, 1001)
    onset_time = 0.200
    peak_time_short = 0.270 # 70ms rise
    baseline_force = 0.0

    force_short = np.zeros_like(time)
    mask = (time >= onset_time) & (time <= peak_time_short)
    force_short[mask] = 1000.0 * (time[mask] - onset_time)

    df_short = pd.DataFrame({'time': time, 'force_right': force_short})

    metrics_short = calculate_rfd(
        signal_df=df_short,
        onset_time=onset_time,
        peak_time=peak_time_short,
        baseline_force=baseline_force,
        response_hand='right'
    )

    assert metrics_short['rfd_0_25'] is not None
    assert np.isclose(metrics_short['rfd_0_25'], 1000.0, atol=1.0)

    assert metrics_short['rfd_0_50'] is not None
    assert np.isclose(metrics_short['rfd_0_50'], 1000.0, atol=1.0)

    # 100-200ms starts after contraction ended -> None
    assert metrics_short['rfd_100_200'] is None
    assert metrics_short['custom_rfd_2'] is None

    # Contraction peaking at 150ms (0.350s)
    peak_time_mid = 0.350
    force_mid = np.zeros_like(time)
    mask_mid = (time >= onset_time) & (time <= peak_time_mid)
    force_mid[mask_mid] = 800.0 * (time[mask_mid] - onset_time)
    df_mid = pd.DataFrame({'time': time, 'force_right': force_mid})

    metrics_mid = calculate_rfd(
        signal_df=df_mid,
        onset_time=onset_time,
        peak_time=peak_time_mid,
        baseline_force=baseline_force,
        response_hand='right'
    )

    assert metrics_mid['rfd_0_50'] is not None
    assert np.isclose(metrics_mid['rfd_0_50'], 800.0, atol=1.0)
    assert metrics_mid['rfd_100_200'] is not None
    assert np.isclose(metrics_mid['rfd_100_200'], 800.0, atol=1.0)


def test_calculate_rfd_custom_windows():
    """
    Tests calculate_rfd with user-specified custom window 1 and 2.
    """
    time = np.linspace(0.0, 1.0, 1001)
    onset_time = 0.100
    peak_time = 0.400
    baseline_force = 0.0

    force = np.zeros_like(time)
    mask = (time >= onset_time) & (time <= peak_time)
    force[mask] = 1500.0 * (time[mask] - onset_time)

    df = pd.DataFrame({'time': time, 'force_right': force})

    metrics = calculate_rfd(
        signal_df=df,
        onset_time=onset_time,
        peak_time=peak_time,
        baseline_force=baseline_force,
        response_hand='right',
        custom_window_1=(0, 25),
        custom_window_2=(50, 150)
    )

    assert metrics['custom_rfd_1'] is not None
    assert np.isclose(metrics['custom_rfd_1'], 1500.0, atol=1.0)
    assert metrics['custom_rfd_1_window'] == "0-25"

    assert metrics['custom_rfd_2'] is not None
    assert np.isclose(metrics['custom_rfd_2'], 1500.0, atol=1.0)
    assert metrics['custom_rfd_2_window'] == "50-150"


def test_calculate_rfd_arbitrary_window_list():
    """
    Tests passing an arbitrary list of windows: [(10, 40), (80, 120)].
    """
    time = np.linspace(0.0, 1.0, 1001)
    onset_time = 0.100
    peak_time = 0.400
    baseline_force = 0.0

    force = np.zeros_like(time)
    mask = (time >= onset_time) & (time <= peak_time)
    force[mask] = 2500.0 * (time[mask] - onset_time)

    df = pd.DataFrame({'time': time, 'force_right': force})

    custom_list = [(10, 40), (80, 120)]
    metrics = calculate_rfd(
        signal_df=df,
        onset_time=onset_time,
        peak_time=peak_time,
        baseline_force=baseline_force,
        response_hand='right',
        windows=custom_list
    )

    assert 'rfd_10_40' in metrics
    assert 'rfd_80_120' in metrics
    assert np.isclose(metrics['rfd_10_40'], 2500.0, atol=1.0)
    assert np.isclose(metrics['rfd_80_120'], 2500.0, atol=1.0)


def test_calculate_rfd_invalid_inputs():
    """
    Ensures robust handling of None, invalid timestamps, or inverted peak/onset.
    """
    time = np.linspace(0.0, 1.0, 100)
    df = pd.DataFrame({'time': time, 'force_right': np.zeros_like(time)})

    # None onset
    res1 = calculate_rfd(df, None, 0.5, 0.0, 'right')
    assert res1['custom_rfd_1'] is None
    assert res1['rfd_0_50'] is None
    assert res1['peak_rfd'] is None

    # peak_time <= onset_time
    res2 = calculate_rfd(df, 0.5, 0.4, 0.0, 'right')
    assert res2['custom_rfd_1'] is None
    assert res2['rfd_0_50'] is None
    assert res2['peak_rfd'] is None

    # None baseline
    res3 = calculate_rfd(df, 0.2, 0.5, None, 'right')
    assert res3['custom_rfd_1'] is None
    assert res3['rfd_0_50'] is None
    assert res3['peak_rfd'] is None
