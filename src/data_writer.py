# Third-Party Dependencies
# ----------------------------------------------------
import pandas as pd

# Local Application Imports
#---------------------------

from get_condition_lookup import get_condition_lookup
from data_loader import load_signal, find_best_match
from trial_segmentation import get_trial_data_and_metrics, get_trial_segment, create_trial_lookup
import force_analyses as fa
import emg_analyses as ea

import pandas as pd
import numpy as np
import force_analyses as fa
import emg_analyses as ea

def export_trial_metrics(
    full_df, trial_lookup, condition_data, channel_map,
    mvc_left, mvc_right, pre_window, post_window, 
    min_valid_rt_s, min_prominence_n, pre_stim_search_s, post_stim_search_s,
    run_peak_force, run_mrspt, run_mrt, run_fti, run_mean_force, run_rfd, rfd_window_ms,
    run_pmrt, run_emg_rms, emg_min_duration_ms, emg_h_onset, emg_h_offset,
    current_discards
):
    # Mapping internal DataFrames
    lookup_df = pd.DataFrame(trial_lookup)
    cond_df = pd.DataFrame(condition_data)
    all_final_metrics = []
    
    for _, trial_row in lookup_df.iterrows():
        global_idx = trial_row['global_index']
        
        try:
            # Extract Trial Data
            trial_view_df, base_metrics = get_trial_data_and_metrics(
                full_df=full_df, trial_lookup=lookup_df, condition_data=cond_df,
                trial_index=global_idx, channel_map=channel_map, mvc_left=mvc_left, mvc_right=mvc_right,
                pre_window=pre_window, post_window=post_window
            )
            base_metrics['discarded_flag'] = 1 if global_idx in current_discards else 0

            # Peak Detection
            peak_info = fa.find_main_contraction_peak(
                full_df=full_df, stim_time=base_metrics['stim_time'], channel_map=channel_map,
                threshold=base_metrics['threshold'], min_valid_rt_s=min_valid_rt_s,
                min_prominence_n=min_prominence_n, search_window_pre_s=pre_stim_search_s,
                search_window_post_s=post_stim_search_s
            )
            
            # Extraction of variables
            peak_time = peak_info.get('peak_time')
            peak_value = peak_info.get('peak_value')
            analysis_df = peak_info.get('analysis_df')
            
            # Only proceed with math if we have a numeric peak
            if analysis_df is not None and isinstance(peak_time, (int, float)):
                response_hand = peak_info['response_hand']
                mvc_val = mvc_right if response_hand == 'right' else mvc_left
                stim_t = base_metrics['stim_time']
                
                # Baseline detection (Returns dict with mean and sd)
                baseline_data = fa.find_baseline_force(analysis_df, stim_t, response_hand)
                baseline_val = baseline_data.get('mean')
                
                # Onset and offset detection
                onset_t = fa.find_contraction_onset(analysis_df, stim_t, peak_time, peak_value, response_hand, mvc_val)
                offset_t = fa.find_contraction_offset(analysis_df, peak_time, peak_value, response_hand, mvc_val)
                
                base_metrics.update({
                    'peak_time': peak_time,
                    'peak_value': peak_value,
                    'force_onset_time': onset_t,
                    'force_offset_time': offset_t,
                    'baseline_force': baseline_val
                })

                # --- Metric Calculations ---
                if run_peak_force:
                    base_metrics.update(fa.peak_force_metrics(peak_value, peak_time, stim_t, base_metrics['threshold'], mvc_val))
                
                if run_mrt:
                    base_metrics['motor_reaction_time_ms'] = fa.motor_reaction_time(stim_t, onset_t)
                
                if run_mrspt:
                    base_metrics['motor_response_time_ms'] = fa.motor_response_time(analysis_df, stim_t, peak_time, peak_value, base_metrics['threshold'], response_hand)

                if run_rfd:
                    base_metrics.update(fa.calculate_rfd(analysis_df, onset_t, peak_time, baseline_val, response_hand, rfd_window_ms))

                # Impulse & Mean Force ---
                if run_fti:
                    base_metrics.update(fa.calculate_impulse(analysis_df, onset_t, offset_t, baseline_val, mvc_val, response_hand))
                
                if run_mean_force:
                    base_metrics.update(fa.calculate_mean_force(analysis_df, onset_t, offset_t, baseline_val, mvc_val, response_hand))

                # --- EMG Analysis  ---
                if (channel_map.get('emg_left') and channel_map.get('emg_right')) and (run_pmrt or run_emg_rms):
                    try:
                        # Dynamic Thresholds
                        on_thresh = ea.calculate_dynamic_threshold(analysis_df, channel_map, response_hand, 0.1, emg_h_onset)
                        off_thresh = ea.calculate_dynamic_threshold(analysis_df, channel_map, response_hand, 0.1, emg_h_offset)
                        
                        # Burst Detection
                        emg_on, emg_off, _ = ea.find_emg_boundaries(analysis_df, channel_map, response_hand, stim_t, onset_t, offset_t, emg_min_duration_ms, on_thresh, off_thresh)
                        
                        if emg_on:
                            if run_pmrt:
                                base_metrics['premotor_reaction_time_ms'] = ea.premotor_reaction_time(stim_t, emg_on)
                            if run_emg_rms:
                                base_metrics['emg_rms_volts'] = ea.calculate_emg_rms(analysis_df, channel_map, response_hand, emg_on, emg_off)
                    except:
                        pass
            
            all_final_metrics.append(base_metrics)

        except Exception as e:
            # If the whole row fails, still append it so the CSV indices match lookup
            all_final_metrics.append({'global_index': global_idx, 'error': str(e)})

    return pd.DataFrame(all_final_metrics)