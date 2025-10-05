# src/force_analyses.py
# Standard Library Imports
from typing import Optional
# Third-party dependencies
import pandas as pd
import numpy as np


def motor_response_time(
    signal_df: pd.DataFrame,
    stim_time: float,
    peak_time: float,
    peak_force: float,
    threshold: float,
    response_hand: str
) -> Optional[int]:
    """
    Computes motor response time (force onset to peak force).

    Finds the last time point at or below the force threshold before the peak.

    Args:
        signal_df: DataFrame of the trial segment.
        stim_time: Stimulus onset time (s).
        peak_time: Time of peak force (s), pre-calculated.
        peak_force: Value of peak force (N), pre-calculated.
        threshold: Force threshold (N), pre-calculated.
        response_hand: "left" or "right".

    Returns:
        Motor response time in milliseconds, or None if not found.
    """
    if response_hand not in ['left', 'right'] or threshold is None:
        return None

    force_col = f"force_{response_hand}"

    # Early exit if the peak force never even reached the threshold
    if peak_force < threshold:
        return None

    # Filter the signal to the window between stimulus and peak force
    response_window_df = signal_df[
        (signal_df['time'] >= stim_time) & (signal_df['time'] <= peak_time)
    ]
    
    # Find all points at or below the threshold within that window
    onset_candidates = response_window_df[response_window_df[force_col] <= threshold]

    if onset_candidates.empty:
        return None # Should not happen if peak > threshold, but a safe check

    # The force onset is the last time point (max time) at or below the threshold
    force_onset_time = onset_candidates['time'].max()
    
    # Calculate duration in milliseconds
    mrspt_ms = int(round((force_onset_time - stim_time) * 1000))

    return mrspt_ms


def peak_force_metrics(
        signal_df: pd.DataFrame,
        stim_time: float,
        response_hand: str,
        threshold: float,
        mvc_left: float,
        mvc_right: float
) -> dict:
    """
    Calculate peak force metrics from a trial segment.
    Finds the post-stimulus peak, its time-to-peak, normalization to %MVC and
    the signed error vs. the target threshold (overshoot/undershoot).

    Args:
    

    Parameters:
    - signal_df: DataFrame containing 'time', 'left_force', and 'right_force' columns.
    - stim_time: Time of stimulus in seconds.
    - response_hand: 'left' or 'right' indicating which hand responded.
    - threshold: Target force for this trial.
    - mvc_left: Maximum voluntary contraction for the left hand.
    - mvc_right: Maximum voluntary contraction for the right hand.

    Returns:
    A dictionary with peak force metrics.
    """
    
    if response_hand not in ['left', 'right']:
        raise ValueError("response_hand must be 'left' or 'right'")
    
    # Determine the correct force column to use
    force_col = f"force_{response_hand}"
    mvc_value = mvc_left if response_hand == 'left' else mvc_right

    # Filter for the post-stimulus window
    post_stim_df = signal_df[signal_df['time'] >= stim_time].copy()

    if post_stim_df.empty:
        return {} # Return empty dict if no post-stimulus data
    
    # Find the peak force and its time of occurence
    peak_index = post_stim_df[force_col].idxmax()
    peak_value = post_stim_df.loc[peak_index, force_col]
    peak_time = post_stim_df.loc[peak_index, 'time']

    # Metrics calculations
    # ----------------------

    # 1. Time to peak
    time_to_peak = peak_time - stim_time

    #2. Peak force as %MVC
    peak_force_pct_mvc = (peak_value / mvc_value) * 100 if mvc_value > 0 else 0

    # 3. Overshoot / Undershoot relative to threshold
    delta_threshold = peak_value - threshold

    # Handle potential division by zero if threshold is 0 (unlikely in practice)
    if threshold > 0:
        delta_threshold_pct = (delta_threshold / threshold) * 100
        # Check if the error is within a small tolerance of the target
        if abs(delta_threshold) <= 1.0:
            threshold_direction = "on-target"
        elif delta_threshold > 0:
            threshold_direction = "overshoot"
        else:
            threshold_direction = "undershoot"
    else:
        delta_threshold_pct = np.nan
        threshold_direction = "N/A"

    # Package into dict
    # ------------------
    metrics = {
        "peak_force": peak_value,
        "peak_force_pct_mvc": peak_force_pct_mvc,
        "time_to_peak": time_to_peak,
        "delta_threshold": delta_threshold,
        "delta_threshold_pct": delta_threshold_pct,
        "threshold_direction": threshold_direction
    }

    return metrics