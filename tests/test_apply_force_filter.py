import pytest
import numpy as np
import pandas as pd
from force_analyses import apply_force_filter

def test_apply_force_filter_attenuates_high_frequency():
    """
    Verify that a 10 Hz cutoff Butterworth filter attenuates a 40 Hz noise signal
    while preserving a 2 Hz physiological signal.
    """
    fs = 1000.0
    t = np.linspace(0, 2.0, int(2.0 * fs), endpoint=False)
    # 2 Hz physiological contraction + 40 Hz gradient artifact
    sig_low = np.sin(2 * np.pi * 2.0 * t)
    sig_high = 0.5 * np.sin(2 * np.pi * 40.0 * t)
    raw = sig_low + sig_high

    filtered = apply_force_filter(raw, fs=fs, cutoff=10.0, order=4)

    # Low frequency amplitude should be preserved (~1.0)
    assert np.max(filtered) == pytest.approx(1.0, abs=0.08)
    # The high frequency component should be substantially attenuated
    diff = np.abs(filtered - sig_low)
    assert np.mean(diff) < 0.05

def test_apply_force_filter_zero_phase_alignment():
    """
    Verify zero-phase property: peak location does not shift in time.
    """
    fs = 1000.0
    t = np.linspace(0, 1.0, 1000, endpoint=False)
    # Gaussian pulse centered at t = 0.5s
    center_idx = 500
    pulse = np.exp(-((t - 0.5) ** 2) / (2 * (0.05 ** 2)))
    # Add noise
    noise = 0.1 * np.sin(2 * np.pi * 60.0 * t)
    raw = pulse + noise

    filtered = apply_force_filter(raw, fs=fs, cutoff=15.0, order=4)

    peak_idx_raw = np.argmax(pulse)
    peak_idx_filt = np.argmax(filtered)

    # Zero phase means peak should stay within 1 sample of the original center
    assert abs(peak_idx_filt - peak_idx_raw) <= 1

def test_apply_force_filter_series_and_array_types():
    """
    Verify that Series input returns Series with identical index, and ndarray returns ndarray.
    """
    fs = 500.0
    t = np.linspace(0, 1.0, 500, endpoint=False)
    raw_arr = np.sin(2 * np.pi * 3.0 * t)
    
    # NumPy array input
    res_arr = apply_force_filter(raw_arr, fs=fs, cutoff=15.0)
    assert isinstance(res_arr, np.ndarray)

    # Pandas Series input with time index
    ser = pd.Series(raw_arr, index=t, name="force_left")
    res_ser = apply_force_filter(ser, cutoff=15.0)
    assert isinstance(res_ser, pd.Series)
    assert res_ser.name == "force_left"
    assert (res_ser.index == ser.index).all()

def test_apply_force_filter_edge_cases():
    """
    Verify edge case handling: short sequences, None, invalid cutoff >= nyquist.
    """
    # None
    assert apply_force_filter(None, fs=100.0) is None
    # Short array (< 27 samples)
    short = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(apply_force_filter(short, fs=100.0), short)
    # Cutoff >= nyquist
    data = np.ones(50)
    assert np.array_equal(apply_force_filter(data, fs=100.0, cutoff=60.0), data)
