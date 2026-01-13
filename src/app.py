# ==============================================================================
# CogMo Toolkit - Main Application
# This Dash application provides an interface for uploading and analyzing
# psychophysiological signal data and associated condition order files.
# ==============================================================================

# ==============================================================================
# ---- IMPORTS AND APP INITIALIZATION ----
# ==============================================================================
# Standard Library Imports
# ----------------------------------------------------
import base64
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Third-Party Dependencies
# ----------------------------------------------------
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html, no_update, ctx, Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL, Input, Output, State
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import PyInstaller
import plotly.graph_objs as go
import webbrowser
from threading import Timer

# Local Application Imports
#---------------------------
from get_condition_lookup import get_condition_lookup
from data_loader import load_signal, find_best_match
from trial_segmentation import get_trial_data_and_metrics, get_trial_segment, create_trial_lookup
import force_analyses as fa
import emg_analyses as ea

# DEPENDENCIES FILE MANAGEMENT:
# Requirements.in & requirements.txt (Windows & MacOS)
# 👉requirements.in contains the main dependencies
# 👉requirements.txt contains the OS-specific versions for deployment
# Call those to update/freeze .txt files after adding new dependencies
# MacOS:
#  Update/freeze: pip-compile requirements.in -o requirements_mac.txt
#  Install:       pip install -r requirements_mac.txt
# Windows:
#  Update/freeze: pip-compile requirements.in -o requirements_windows.txt
#  Install:       pip install -r requirements_windows.txt

# TO DEPLOY WITH PYINSTALLER:
# pyinstaller CogMo.spec --clean --log-level DEBUG
# ----------------------------------------------------

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__,
                 external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME],
                 prevent_initial_callbacks = True)


# ==============================================================================
# --- UPLOAD BUTTONS ---
# ==============================================================================
# We define these styles as dictionaries to allow for easy manipulation via 
# Dash Callbacks. By switching the 'style' property of a dcc.Upload component, 
# provide immediate visual confirmation of file processing success or failure.


# BASELINE STYLE: default dashed box look for the app
DEFAULT_UPLOAD_STYLE = {
    'height' : '60px',
    'lineHeight' : '60px',
    'borderWidth' : '1px',
    'borderStyle' : 'dashed',
    'borderRadius' : '5px',
    'textAlign' : 'center',
}

# SUCCESS STATE: Applied when the CSV is parsed and trial_lookup is built
SUCCESS_UPLOAD_STYLE = DEFAULT_UPLOAD_STYLE.copy()
SUCCESS_UPLOAD_STYLE['borderColor'] = 'green'
SUCCESS_UPLOAD_STYLE['backgroundColor'] = '#F0FFF0'

# ERROR STATE: Applied if parsing fails (e.g., missing headers or bad encoding)
ERROR_UPLOAD_STYLE = DEFAULT_UPLOAD_STYLE.copy()
ERROR_UPLOAD_STYLE = DEFAULT_UPLOAD_STYLE.copy()
ERROR_UPLOAD_STYLE['borderColor'] = 'red'

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def create_trial_figure(
    trial_segment_df, channel_map, mvc_left, mvc_right, trial_metrics,
    run_peak_force: bool = False,
    run_motor_response_time: bool = False,
    run_motor_reaction_time: bool = False,
    run_force_time_integral: bool = False,
    run_pmrt: bool = False,
    run_emg_rms: bool = False
):
    """
    Constructs a synchronized Force and EMG plot using Plotly Subplots.
    
    Visualization Strategy:
    1. Layered Traces: Full trial data is drawn in light gray (background context), 
       while the focus window (+/- 200ms around contraction) is in high-contrast colors.
    2. Shared X-Axes: Force and EMG subplots are linked for synchronized zooming/panning.
    3. Analytical Overlays: Vertical markers for reaction times and shaded area for impulse.
    """
    # --- Signal & Mapping Setup ---
    time_col = channel_map.get('time')
    force_r_col = channel_map.get('force_right')
    force_l_col = channel_map.get('force_left')
    emg_r_col = channel_map.get('emg_right')
    emg_l_col = channel_map.get('emg_left')
    include_emg = bool(emg_r_col and emg_l_col)

    # Styling constants for consistency across the app
    COLOR_R, COLOR_L, COLOR_GRAY = 'blue', 'goldenrod', '#D3D3D3'
    max_mvc = max(mvc_left, mvc_right) if mvc_left and mvc_right else 1
    
    # Metadata for annotations and boundaries
    threshold_value = trial_metrics.get('threshold')
    selected_trial = trial_metrics.get('block_trial_str', 'Trial')
    stim_time = trial_metrics.get('stim_time')
    onset_time = trial_metrics.get('force_onset_time')
    offset_time = trial_metrics.get('force_offset_time')
    has_analyzed_segment = onset_time is not None and offset_time is not None

    # --- Subplot Initialization ---
    fig = make_subplots(rows=2 if include_emg else 1, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    # --- Trace Plotting (Context vs. Focus) ---
    if has_analyzed_segment:
        # Define the window to 'highlight' in color
        viz_start_time = onset_time - 0.200
        viz_end_time = offset_time + 0.200
        viz_df = trial_segment_df[
            (trial_segment_df[time_col] >= viz_start_time) & (trial_segment_df[time_col] <= viz_end_time)
        ]
        
        # Plot gray background (entire trial)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_r_col], name = "Out of bound", line=dict(color=COLOR_GRAY), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_l_col], name = "Out of bound", line=dict(color=COLOR_GRAY), showlegend=False), row=1, col=1)
        
        # Plot colored foreground (active segment only)
        fig.add_trace(go.Scatter(x=viz_df[time_col], y=viz_df[force_r_col], name='Right Hand', line=dict(color=COLOR_R)), row=1, col=1)
        fig.add_trace(go.Scatter(x=viz_df[time_col], y=viz_df[force_l_col], name='Left Hand', line=dict(color=COLOR_L)), row=1, col=1)

        if include_emg:
            # Synchronized EMG context/focus layering
            fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[emg_r_col], name = "Out of bound", line=dict(width=0.5, color=COLOR_GRAY), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[emg_l_col], name = "Out of bound", line=dict(width=0.5, color=COLOR_GRAY), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=viz_df[time_col], y=viz_df[emg_r_col], name = "EMG Right", line=dict(width=0.5, color=COLOR_R), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=viz_df[time_col], y=viz_df[emg_l_col], name = "EMG Left", line=dict(width=0.5, color=COLOR_L), showlegend=False), row=2, col=1)
    else:
        # Default view when analysis hasn't run or failed
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_r_col], name='Right Hand', line=dict(color=COLOR_R)), row=1, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_l_col], name='Left Hand', line=dict(color=COLOR_L)), row=1, col=1)
        if include_emg:
            fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[emg_r_col], name='EMG Right', showlegend=False, line=dict(width=0.5, color=COLOR_R)), row=2, col=1)
            fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[emg_l_col], name='EMG Left', showlegend=False, line=dict(width=0.5, color=COLOR_L)), row=2, col=1)

    # --- Subplot Axes Configuration ---
    fig.update_yaxes(title_text="Force", range=[0, max_mvc], row=1, col=1)
    if include_emg:
        fig.update_yaxes(title_text="EMG", row=2, col=1)

    # Global Force Threshold Line (Visual sanity check for MRT/MRsT)
    if threshold_value is not None:
        fig.add_hline(y=threshold_value, line_dash="dash", line_color="grey", row=1, col=1)

    # --- Visualization for Impulse (AUC) ---
    if run_force_time_integral and has_analyzed_segment:
        baseline_force = trial_metrics['baseline_force']
        response_hand = trial_metrics.get('response_hand')
        force_col = f"force_{response_hand}"
        
        auc_df = trial_segment_df[
            (trial_segment_df[time_col] >= onset_time) & (trial_segment_df[time_col] <= offset_time)
        ].copy()

        # Invisible Baseline Trace (floor for the fill)
        fig.add_trace(go.Scatter(
            x=auc_df[time_col], y=[baseline_force] * len(auc_df),
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

        # Fill from the Force curve down to the baseline trace
        fig.add_trace(go.Scatter(
            x=auc_df[time_col], y=auc_df[force_col],
            fill='tonexty', mode='lines', line=dict(width=0),
            fillcolor='rgba(40, 167, 69, 0.3)', showlegend=False, name='Impulse (AUC)'
        ), row=1, col=1)

    # --- Force Analysis Markers ---
    peak_force = trial_metrics.get('peak_force')
    time_to_peak = trial_metrics.get('time_to_peak')
    if run_peak_force and all(v is not None for v in [peak_force, time_to_peak, stim_time, threshold_value]):
        peak_time = stim_time + time_to_peak
        fig.add_trace(go.Scatter(
            x=[peak_time, peak_time], y=[threshold_value, peak_force], name = "Δ threshold",
            mode='lines', line=dict(color='red', dash='dash', width=1),
            showlegend=False
        ), row=1, col=1)

    # MRT vs MRsT: Different algorithms for identifying the start of force production
    mrt = trial_metrics.get('motor_reaction_time')
    if run_motor_reaction_time and all(v is not None for v in [mrt, stim_time, threshold_value]):
        f_onset_mrt = stim_time + (mrt / 1000.0)
        fig.add_trace(go.Scatter(
            x=[f_onset_mrt, f_onset_mrt], y=[0, threshold_value],
            mode='lines', line=dict(color='orange', dash='dash', width=1),
            name='Reaction time', showlegend=False
        ), row=1, col=1)

    mrspt = trial_metrics.get('motor_response_time')
    if run_motor_response_time and all(v is not None for v in [mrspt, stim_time, threshold_value]):
        f_onset_mrspt = stim_time + (mrspt / 1000.0)
        fig.add_trace(go.Scatter(
            x=[f_onset_mrspt, f_onset_mrspt], y=[0, threshold_value],
            mode='lines', line=dict(color='green', dash='dash', width=1),
            name='Response time', showlegend=False
        ), row=1, col=1)

    # --- EMG Onset/Offset Markers (PMRT Validation) ---
    if include_emg:
        emg_onset_time = trial_metrics.get('emg_onset_time')
        emg_offset_time = trial_metrics.get('emg_offset_time')
        
        # Determine shared height for EMG markers based on current subplot range
        emg_min = trial_segment_df[[emg_r_col, emg_l_col]].min().min()
        emg_max = trial_segment_df[[emg_r_col, emg_l_col]].max().max()

        if run_pmrt and emg_onset_time is not None:
            fig.add_trace(go.Scatter(
                x=[emg_onset_time, emg_onset_time], y=[emg_min, emg_max],
                mode='lines', line=dict(color='purple', dash='dash', width=1.5),
                name='EMG Onset', showlegend=False
            ), row=2, col=1)

        if run_emg_rms and emg_onset_time is not None and emg_offset_time is not None:
            # Adding offset boundary for RMS window
            fig.add_trace(go.Scatter(
                x=[emg_offset_time, emg_offset_time], y=[emg_min, emg_max],
                mode='lines', line=dict(color='purple', dash='dot', width=1.5),
                name='EMG Offset', showlegend=False
            ), row=2, col=1)

    # --- Final Layout & Annotations ---
    annotations = []
    if stim_time is not None:
        # Arrow indicating the moment of Stimulus (T=0 for PMRT, MRT, etc.)
        annotations.append(
            dict(x=stim_time, y=0, xref="x", yref="y", text="",
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1.5,
                arrowcolor="black", ax=0, ay=-40)
        )

    fig.update_layout(
        title=selected_trial,
        margin=dict(t=30, b=0, l=0, r=0),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255, 255, 255, 0.75)"),
        annotations=annotations
    )
    
    return fig


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return Path(os.path.join(base_path, relative_path))


# ==============================================================================
# --- APP LAYOUT (UI) ---
# ==============================================================================
# Tab 1 = Data ingestion (file uploads, channel mapping, reference values)
# Tab 2 = Analysis options (checkboxes, parameters)
# Tab 3 = Trial viewer (trial selection, plotting, metrics display)
app.layout = dbc.Container([
    html.H1("CogMo toolkit", className="my-3"),
    dbc.Tabs([
        # Tab 1 : Data Upload Tab
        # ---------------------------------------------------
        # Hanndles file ingestion for both time-series signal and experimental condition order.
        # Also captures participant-sepcific reference values (MVC, RFD) for normalization.
        dbc.Tab(
            label="Upload Data",
            children=[
                html.Div([
                    dbc.Row(className="g-2", children=[
                        # Signal data upload: Accepts raw formats (.csv, .txt. .tsv)
                        dbc.Col(
                            html.Div([
                                html.H4("Upload your data file"),
                                dcc.Upload(
                                    id='upload-signal-data',
                                    children=html.Div([
                                        'Your raw data here (.csv, .tsv, .txt, .xlsx)'
                                    ]),
                                    style={ # Style toggled by callback on success/fail
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                    },
                                    multiple=False
                                ),
                                dcc.Loading(
                                    id="data-upload-throbber",
                                    type="dot",
                                    children=html.Div(id='data-upload-output-message',
                                                      className="mt-1",
                                                      style={'min-height': '25px'}),
                                )
                            ]),
                            width=6,
                        ),
                        # Condition order upload: Map trials to experimental blocks
                        dbc.Col(
                            html.Div([
                                html.H4(html.Br()),
                                dcc.Upload(
                                    id='upload-condition-order',
                                    children=html.Div([
                                        'Your condition file here (.xlsx or .csv)'
                                    ]),
                                    style={
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                    },
                                    multiple=False
                                ),
                                dcc.Loading(
                                    id = 'condition-upload-throbber',
                                    type = 'dot',
                                    children = html.Div(id = 'condition-upload-output-message',
                                                          className='mt-1',
                                                          style={'min-height': '25px'})
                                )
                            ]),
                            width=6,
                        ),
                    ]),
                    
                    # Mapping containiner is populated dynamically once headers are parsed
                    html.Div(id='channel-mapping-container', className="mt-2"),
                    
                    # Static reference inputs for normalization across trials
                    html.H4("Baseline Reference values", className="mt-3"),
                    dbc.Row([
                        dbc.Col(
                            dbc.FormFloating([
                                dbc.Input(type="number", id="input-mvc-left", placeholder="Left"),
                                dbc.Label("Maximum voluntary force (Left)")
                            ]),
                            width=6
                        ),
                        dbc.Col(
                            dbc.FormFloating([
                                dbc.Input(type="number", id="input-mvc-right", placeholder="Right"),
                                dbc.Label("Maximum voluntary force (Right)")
                            ]),
                            width=6
                        ),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dbc.FormFloating([
                                dbc.Input(type="number", id="input-rfd-left", placeholder="Left"),
                                dbc.Label("Peak rate of force development (Left)")
                            ]),
                            width=6
                        ),
                        dbc.Col(
                            dbc.FormFloating([
                                dbc.Input(type="number", id="input-rfd-right", placeholder="Right"),
                                dbc.Label("Peak rate of force development (Right)")
                            ]),
                            width=6
                        ),
                    ]), 
                ], className="p-3")
            ]
        ),
        # Tab 2 : Analyses options & signal processing settings
        # ---------------------------------------------------
        # Hthis tab defines the global parameters for the session
        # Allows the user to toggle specific force/EMG metrics and fine-tune detection settings.
        dbc.Tab(
            label="Analyses Option",
            children=[
                html.Div([
                    # --- Force Analyses ---
                    dbc.Row([
                        dbc.Col([
                            html.H4("Force Signal Analyses"),
                            html.Hr(),
                            html.H6("Latencies", className="mt-3"),
                            dbc.Checkbox(id="analysis-mrt-checkbox", label="Motor Reaction Time"),
                            dbc.Checkbox(id="analysis-mrspt-checkbox", label="Motor Response Time"),
                            
                            html.H6("Rate of Force", className="mt-4"),
                            dbc.Checkbox(id="analysis-rfd-checkbox", label="Rate of Force Development (RFD)", value=True),
                            dbc.Input(id="analysis-rfd-window-input", type="number", value=75, placeholder="e.g., 75", disabled=False, className="mt-2 w-50"),

                            html.H6("Force Magnitudes", className="mt-4"),
                            dbc.Checkbox(id="analysis-peak-force-checkbox", label="Peak Force & Overshoot"),
                            dbc.Checkbox(id="analysis-mean-force-checkbox", label="Mean Force"),
                            dbc.Checkbox(id="analysis-fti-checkbox", label="Force-Time Integral"),
                        ], width=6),

                        # --- EMG Analyses ---
                        dbc.Col([
                            html.H4("EMG Signal Analyses"),
                            html.Hr(),
                            html.H6("Latency", className="mt-3"),
                            dbc.Checkbox(id="analysis-pmrt-checkbox", label="Premotor Reaction Time"),
                            html.H6("EMG Activity", className="mt-4"),
                            dbc.Checkbox(id="analysis-rms-checkbox", label="Root Mean Square (RMS)"),
                        ], width=6),
                    ]),
                    
                    html.Hr(className="my-4"),

                    # --- Peak Detection Settings ---
                    html.H4("Force Peak Detection Settings"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Min. Valid Reaction Time (s)"),
                            dbc.Input(id="min-valid-rt-input", type="number", value=0.250, step=0.01),
                            dbc.Tooltip("The earliest a peak can occur after the stimulus to be considered a valid response.", target="min-valid-rt-input"),
                        ], width=6, md=3),
                        dbc.Col([
                            dbc.Label("Min. Peak Prominence (N)"),
                            dbc.Input(id="min-prominence-input", type="number", value=5, step=1),
                            dbc.Tooltip("How much a peak must 'stick out' from the surrounding signal to be considered a new event.", target="min-prominence-input"),
                        ], width=6, md=3),
                        dbc.Col([
                            dbc.Label("Pre-Stimulus Search (s)"),
                            dbc.Input(id="pre-stim-search-input", type="number", value=1.0, step=0.1),
                            dbc.Tooltip("How far before the stimulus to look for peaks.", target="pre-stim-search-input"),
                        ], width=6, md=3),
                        dbc.Col([
                            dbc.Label("Post-Stimulus Search (s)"),
                            dbc.Input(id="post-stim-search-input", type="number", value=2.0, step=0.1),
                             dbc.Tooltip("How far after the stimulus to look for peaks.", target="post-stim-search-input"),
                        ], width=6, md=3),
                    ]),
                    
                    html.Hr(className="my-4"),

                    # --- Signal Processing Settings ---
                    html.H4("Force Signal Processing"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Apply Low-Pass Filter"),
                            dbc.Checklist(
                                options=[{"label": "Enable Zero-Phase Butterworth", "value": 1}],
                                value=[],
                                id="force-filter-check",
                                switch=True,
                            ),
                            dbc.Tooltip("Filters high-frequency noise without shifting the signal in time (non-causal).", target="force-filter-check"),
                        ], width=6, md=4),
                        dbc.Col([
                            dbc.Label("Cutoff Frequency (Hz)"),
                            dbc.Input(
                                id="force-cutoff-input", 
                                type="number", 
                                value=50, 
                                step=1,
                                disabled=True # Greayed out by default
                            ),
                            dbc.Tooltip("Low pass frequency threshold", target="force-cutoff-input"),
                        ], width=6, md=2),
                    ]),

                    html.Hr(className="my-4"),
                    
                    # --- EMG Detection Settings ---
                    html.H4("EMG Burst Detection Settings"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Onset SD (h-on)"),
                            dbc.Input(id="emg-h-onset-input", type="number", value=15.0, step=1),
                            dbc.Tooltip(
                                "Multiplier for onset threshold (mean + σ from local quietest noise). Lower values are more sensitive (earlier onset).",
                                target="emg-h-onset-input",
                            ),
                        ], width=6, md=3),
                        dbc.Col([
                            dbc.Label("Offset SD (h-off)"),
                            dbc.Input(id="emg-h-offset-input", type="number", value=30.0, step=1),
                            dbc.Tooltip(
                                "Multiplier for offset threshold (mean + σ from local quietest noise). Higher values prevent offset 'bleeding' into post-contraction noise.",
                                target="emg-h-offset-input",
                            ),
                        ], width=6, md=3),
                        dbc.Col([
                            dbc.Label("Min. Duration (ms)"),
                            dbc.Input(id="emg-min-duration-input", type="number", value=10, step=1),
                            dbc.Tooltip(
                                "Minimal duration for a valid EMG burst.",
                                target="emg-min-duration-input",
                            ),
                        ], width=6, md=3),
                    ]),

                    # --- Tooltips for all checkboxes ---
                    dbc.Tooltip("Time from stimulus onset to force onset.", target="analysis-mrt-checkbox"),
                    dbc.Tooltip("Time from force onset to peak force.", target="analysis-mrspt-checkbox"),
                    dbc.Tooltip("The steepest slope of the force curve after force onset.", target="analysis-rfd-checkbox"),
                    dbc.Tooltip("Time from stimulus onset to EMG onset.", target="analysis-pmrt-checkbox"),
                    dbc.Tooltip("The peak force achieved and the amount it exceeds the target.", target="analysis-peak-force-checkbox"),
                    dbc.Tooltip("The average force during a specified window.", target="analysis-mean-force-checkbox"),
                    dbc.Tooltip("The area under the force-time curve.", target="analysis-fti-checkbox"),
                    dbc.Tooltip("A measure of the magnitude of the EMG signal.", target="analysis-rms-checkbox"),

                ], className="p-3")
            ]
        ),

         # Tab 3 : Trial viewer
        # ---------------------------------------------------
        # Validation core. Uses a compact layout to maximize graph area. Allow for frame-by-frame verification
        # of trial segmentations and analysis results.
        dbc.Tab(
            label="Trial Viewer",
            children=[
                html.Div([
                    dcc.Download(id="download-trial-csv"),
                    # Compact navigation: Single-row for block and trial selectors
                    html.Div([
                        html.Div([
                            html.H4("Trial Controls", className="fw-bold mb-0 me-4", style={'white-space': 'nowrap'}),
                            html.Span("Block:", className="me-2 small fw-bold"),
                            html.Div(
                                dcc.Dropdown(
                                    id='block-selector-dropdown', 
                                    style={'width': '140px'},
                                    placeholder="Select..."
                                ), 
                                className="me-5"
                            ),
                            html.Span("Trial:", className="me-2 small fw-bold"),
                            dbc.InputGroup([
                                dbc.Button(
                                    html.I(className="fas fa-chevron-left"), 
                                    id='prev-trial-button', 
                                    size="sm", 
                                    color="secondary", 
                                    outline=True
                                ),
                                dcc.Dropdown(
                                    id='trial-selector-dropdown', 
                                    style={'width': '120px'},
                                    className="flex-grow-1"
                                ),
                                dbc.Button(
                                    html.I(className="fas fa-chevron-right"), 
                                    id='next-trial-button', 
                                    size="sm", 
                                    color="secondary", 
                                    outline=True
                                )
                            ], size="sm", style={'width': '220px'})
                        ], className="d-flex align-items-center mb-3 p-2 border-bottom"),
                    ]),
                    # View parameters: adjust the x-axis window
                    html.Div([
                        html.H4("View Parameters", className="fw-bold mb-0 me-3", style={'white-space': 'nowrap'}),
                        html.Span("Pre-Stim (s):", className="me-2 small fw-bold"),
                        dbc.Input(id='pre-stim-window-input', type='number', value=1, step=0.05, size="sm", style={'width': '80px'}, className="me-4"),
                        html.Span("Post-Stim (s):", className="me-2 small fw-bold"),
                        dbc.Input(id='post-stim-window-input', type='number', value=2, step=0.05, size="sm", style={'width': '80px'})
                    ], className="d-flex align-items-center mb-3"),
                    html.Hr(className="my-2"),
                    # Trial viewer graph
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='trial-graph', style={'height': '50vh'})
                        ], width=12) 
                    ]),
                    # Results panel: displays calculated metrics for the current trial
                    html.Div([
                        html.Div([
                            html.H4("Trial Metrics", className="fw-bold mb-0"),
                            dbc.Button([
                                html.I(className="fas fa-download me-2"),
                                "Download Trial"
                            ], id="btn-download-trial", color="primary", outline=True, size="sm")
                        ], className="d-flex justify-content-between align-items-center mb-2"),
                        
                        html.Div(id='trial-metrics-display'), 
                    ], className="mt-4")
                ], className="p-3")
            ]
        ),
    ])
 ])


# Add dcc.Store components to save the identified comment types and reference values
# File storage (as data frame) and channel mapping
# ---------------------------------------------------
app.layout.children.append(dcc.Store(id = 'signal-data-store')),
app.layout.children.append(dcc.Store(id = 'condition-data-store')),
app.layout.children.append(dcc.Store(id = 'channel-map-store')),
app.layout.children.append(dcc.Store(id = 'trial-lookup-store')),
# Comment types stored as a list of strings
# ---------------------------------------------------
app.layout.children.append(dcc.Store(id = 'block-comments-store'))
app.layout.children.append(dcc.Store(id = 'stimulus-comments-store'))
# Reference values stored as floats
# ---------------------------------------------------
app.layout.children.append(dcc.Store(id = 'mvc-left-store'))
app.layout.children.append(dcc.Store(id = 'mvc-right-store'))
app.layout.children.append(dcc.Store(id = 'rfd-left-store'))
app.layout.children.append(dcc.Store(id = 'rfd-right-store'))
app.layout.children.append(dcc.Store(id = 'force-channels-store'))
app.layout.children.append(dcc.Store(id = 'emg-channels-store'))
THRESHOLD_CACHE = {} # EMG threshold cache
# Navigation system
# ---------------------------------------------------
app.layout.children.append(dcc.Store(id = 'ui-generator-signal-store'))
app.layout.children.append(dcc.Store(id = 'current-stim-time-store'))
app.layout.children.append(dcc.Store(id = 'current-trial-metrics-store'))


# ==============================================================================
# --- CALLBACKS (Backend Logic) ---
# ==============================================================================

# Callback for Signal Data Upload
# --------------------------------
@app.callback(
    Output('signal-data-store', 'data', allow_duplicate=True),
    Output('block-comments-store', 'data', allow_duplicate=True),
    Output('stimulus-comments-store', 'data', allow_duplicate=True),
    Output('data-upload-output-message', 'children', allow_duplicate=True),
    Output('upload-signal-data', 'style', allow_duplicate=True),
    Input('upload-signal-data', 'contents'),
    State('upload-signal-data', 'filename'),
    prevent_initial_callbacks = True
)
def upload_signal_data_callback(signal_contents, signal_filename):
    message = "" # Initialize to avoid potential NameError 
    if signal_contents is None:
        raise dash.exceptions.PreventUpdate

    content_type, content_string=signal_contents.split(',')
    
    # Save the uploaded file temporarily
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_filepath = temp_dir / signal_filename
    
    try:
        decoded = base64.b64decode(content_string)
        with open(temp_filepath, 'wb') as f:
            f.write(decoded)
        
        # Load and process the signal data
        df, comment_summary = load_signal(temp_filepath)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # --- SUCCESS PATH ---
            # Create session ID and dedicated temporary directory for app's session
            session_id = str(uuid.uuid4())
            app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
            app_temp_dir.mkdir(exist_ok = True)
            filepath = app_temp_dir / f"{session_id}.feather"
            df.to_feather(filepath)

            # Message not printed, but stored in dcc.Store linked to dcc.Loading for throbber 
            if not df.empty and not comment_summary:
                message = f"File uploaded but there was an issue with processing"
            else:
                message = ""

            # Find the lowest block count and total stimulus count, and capture comment types
            block_comments = []
            stimulus_comments = []

            if comment_summary:
                for comment_type, count in comment_summary.items():
                    comment_lower = comment_type.lower()
                    if 'block' in comment_lower:
                        if comment_type not in block_comments:
                            block_comments.append(comment_type)
                    elif 'stimulus' in comment_lower:
                        if comment_type not in stimulus_comments:
                            stimulus_comments.append(comment_type)

            # TODO: Comment out for deployment, this is for debugging/testing purposes
            print(f"Data's head:\n {df.head()}")
            print(f"Session ID: {session_id}")
            print(f"Comment's count: {comment_summary}")
            print(f"Block comments: {block_comments}")
            print(f"Stimulus comments: {stimulus_comments}")
            print(f"Total rows tagged as True for block start: {df['is_block_start'].sum()}")
            print(message)

            return session_id, block_comments, stimulus_comments, message,  SUCCESS_UPLOAD_STYLE
        else:
            # --- FAILURE PATH ---
            message = ""
            return None, None, None, message, ERROR_UPLOAD_STYLE
        
    except Exception as e:
        block_comments = None
        stimulus_comments = None
        df = None
        session_id = None
        message = f' Error processing file: {str(e)}'

        return session_id, block_comments, stimulus_comments, message, ERROR_UPLOAD_STYLE
        
    finally:
        # Clean up the temporary file and directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# Callback for Channel Selection
# ------------------------------
@app.callback(
        Output('channel-mapping-container', 'children'),
        Output('ui-generator-signal-store', 'data'),
        Input('signal-data-store', 'data')
)
def update_channel_mapping_ui(session_id):
    # --- FAILURE PATH ----
    # guard clause:  Ensure session_id exists
    if not session_id:
        #TODO: Message to display? See what I might want to do.
        return None

    # --- HAPPY PATH ---
    # Recreate path to temp data folder & full path, then read pd.DataFrame from feather file
    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    app_temp_dir.mkdir(exist_ok = True)
    filepath = app_temp_dir /f"{session_id}.feather"
    df = pd.read_feather(filepath)

    # Define auto-detection patterns, based on regex keywords
    time_patterns       = ["(?i)^time$", "(?i)^timestamp$", "(?i)^ts$", "(?i)^t$"]
    force_right_pattern = ["(?i)^force[-_\\s]?right$", "(?i)^right[-_\\s]?force$", "(?i)^fr$", "(?i)^force\\s*\\(r\\)", "(?i)grip[-_\\s]?r"]
    force_left_pattern  = ["(?i)^force[-_\\s]?left$", "(?i)^left[-_\\s]?force$", "(?i)^fl$", "(?i)^force\\s*\\(l\\)", "(?i)grip[-_\\s]?l"]
    emg_right_pattern   = ["(?i)^emg[-_\\s]?right$", "(?i)^right[-_\\s]?emg$", "(?i)^emg\\s*\\(r\\)", "(?i)^er$", "(?i)fcr[-_\\s]?r$"]
    emg_left_pattern    = ["(?i)^emg[-_\\s]?left$", "(?i)^left[-_\\s]?emg$", "(?i)^emg\\s*\\(l\\)", "(?i)^el$", "(?i)fcr[-_\\s]?l$"]

    all_channel_names = df.columns.tolist() # Get channel names from full DataFrame
    internal_columns_to_exclude = ['comments', 'is_block_start', 'block_number', 'is_trial_start', 'trial_number']
    selectable_channels = [
        name for name in all_channel_names
        if name not in internal_columns_to_exclude
    ]
    dropdown_options = [{'label': name, 'value' :name} for name in selectable_channels]

    detected_channels = {
        'time'         : find_best_match(selectable_channels, time_patterns),
        'force_right'  : find_best_match(selectable_channels, force_right_pattern),
        'force_left'   : find_best_match(selectable_channels, force_left_pattern),
        'emg_right'    : find_best_match(selectable_channels, emg_right_pattern),
        'emg_left'     : find_best_match(selectable_channels, emg_left_pattern)
    }
    #TODO: Comment out for deployment, this is for debugging/testing purposes
    print(f"Channel names: {selectable_channels}")

    # Sets the initial state of the EMG checkbox based on auto-detection
    emg_detected = bool(detected_channels['emg_right'] and detected_channels['emg_left'])

    # Build layout components to return
    ui_layout = html.Div([
    html.H4("Channel Mapping"),
    html.P("Review the detected channels or make a manual selection."),
    
    # --- Row 1 : Time channel ---
    dbc.Row([
        dbc.Col([
            html.H5("Time Channel"),
            dcc.Dropdown(
                id='time-channel-dropdown',
                options=dropdown_options,
                value=detected_channels['time'],
                clearable=False
            )
        ], width=6)
    ], className="mt-2"),
    
    # --- Row 2: Force Channels ---
    dbc.Row([
        dbc.Col(html.H5("Force Channels"), width=12),
        dbc.Col([
            dbc.Label("Force Right Channel:"),
            dcc.Dropdown(
                id='force-right-channel-dropdown',
                options=dropdown_options,
                value=detected_channels['force_right'],
                clearable=False
            )
        ], width=6),
        dbc.Col([
            dbc.Label("Force Left Channel:"),
            dcc.Dropdown(
                id='force-left-channel-dropdown',
                options=dropdown_options,
                value=detected_channels['force_left'],
                clearable=False
            )
        ], width=6)
    ], className="mt-2"),
    
    # --- Row 3: EMG Channels ---
    dbc.Row([
        dbc.Col(html.H5("EMG Channels"), width=12),
        dbc.Col(
            dbc.Checkbox(
                id='include-emg-checkbox',
                label="File includes EMG channels",
                value=emg_detected, # Sets the initial state
            ),
            width=12,
        ),
        html.Div(
            id='emg-dropdown-container',
            children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("EMG Right Channel:"),
                        dcc.Dropdown(
                            id='emg-right-channel-dropdown',
                            options=dropdown_options,
                            value=detected_channels['emg_right'],
                            clearable=True
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("EMG Left Channel:"),
                        dcc.Dropdown(
                            id='emg-left-channel-dropdown',
                            options=dropdown_options,
                            value=detected_channels['emg_left'],
                            clearable=True
                        )
                    ], width=6)
                ])
            ],
            style={'display': 'block' if emg_detected else 'none'}
        )
    ], className="mt-2")
    ])

    return ui_layout, time.time() # return current time as a dummy data to trigger the store update


# Callback to control visibility of EMG Channel Dropdowns
# --------------------------------------------------------
@app.callback(
        Output('emg-dropdown-container', 'style'),
        Input('include-emg-checkbox', 'value')
)
def toggle_emg_visibility(is_checked):
    if is_checked:
        return {'display': 'block'} # Shown dropdowns
    else:
        return {'display': 'none'} # Hide dropdowns


# Callback to control channel name storage
# -----------------------------------------
@app.callback(
        Output('channel-map-store', 'data', allow_duplicate=True),
        Input('time-channel-dropdown', 'value'),
        Input('force-right-channel-dropdown', 'value'),
        Input('force-left-channel-dropdown', 'value'),
        Input('emg-right-channel-dropdown', 'value'),
        Input('emg-left-channel-dropdown', 'value'),
        prevent_initial_call=True
)
def save_channel_mapping(time_col, fr_col, fl_col, er_col, el_col):
    # Guard clause: Ensure required channels are selected
    if not all([time_col, fr_col, fl_col]):
        raise PreventUpdate
    
    # It's inefficient to rewrite the entire dataFrame everytime we make a change to the dropdown menu
    # Better to store a channel mapping as a dict, and use that to reference the correct columns in the DataFrame
    channel_map = {
        'time' :time_col,
        'force_right' : fr_col,
        'force_left' : fl_col,
        'emg_right' : er_col,
        'emg_left' : el_col
    }

    #TODO: Comment out for deployment, this is for debugging/testing purposes
    print(f"Channel mapping updated:\n {channel_map}")

    return channel_map


# Callback to prevent channel duplicates
# ---------------------------------------
@app.callback(
    Output('time-channel-dropdown', 'value', allow_duplicate=True),
    Output('force-right-channel-dropdown', 'value', allow_duplicate=True),
    Output('force-left-channel-dropdown', 'value', allow_duplicate=True),
    Output('emg-right-channel-dropdown', 'value', allow_duplicate=True),
    Output('emg-left-channel-dropdown', 'value', allow_duplicate=True),
    Input('time-channel-dropdown', 'value'),
    Input('force-right-channel-dropdown', 'value'),
    Input('force-left-channel-dropdown', 'value'),
    Input('emg-right-channel-dropdown', 'value'),
    Input('emg-left-channel-dropdown', 'value'),
    prevent_initial_call=True  
)
def prevent_duplicate_channels (time_val, fr_val, fl_val, er_val, el_val):
    trigger_id = ctx.triggered_id
    if not trigger_id:
        raise PreventUpdate
    
    all_values = {
        'time-channel-dropdown': time_val,
        'force-right-channel-dropdown': fr_val,
        'force-left-channel-dropdown': fl_val,
        'emg-right-channel-dropdown': er_val,
        'emg-left-channel-dropdown': el_val   
    }
    newly_selected_value = all_values[trigger_id]

    output_values = { # Assume nothing will change (no_update)
        'time-channel-dropdown': no_update,
        'force-right-channel-dropdown': no_update,
        'force-left-channel-dropdown': no_update,
        'emg-right-channel-dropdown': no_update,
        'emg-left-channel-dropdown': no_update
    }
    
    for dropdown_id, value in all_values.items():
        # Check every dropdown except the one that was just changed
        if dropdown_id != trigger_id:
            if value == newly_selected_value: # → Duplicate found, Reset old dropdown value to None
                output_values[dropdown_id] = None

    return (
        output_values['time-channel-dropdown'],
        output_values['force-right-channel-dropdown'],
        output_values['force-left-channel-dropdown'],
        output_values['emg-right-channel-dropdown'],
        output_values['emg-left-channel-dropdown']
    )


# Callback for Condition Order File Upload
# -----------------------------------------
@app.callback(
        Output('condition-data-store', 'data', allow_duplicate = True),
        Output('condition-upload-output-message', 'children', allow_duplicate = True),
        Output('upload-condition-order', 'style', allow_duplicate=True),
        Input('upload-condition-order', 'contents'),
        State('upload-condition-order', 'filename'),
        prevent_initial_callbacks = True
    
)
def upload_condition_callback(condition_contents, condition_filename):
    message = "" # Initialize to avoid potential NameError
    if condition_contents is None:
        raise dash.exceptions.PreventUpdate
    content_type, content_string = condition_contents.split(',')

    try:
        decoded = base64.b64decode(content_string)
        # Read file based on its extension
        if condition_filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded))
        elif condition_filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, message, ERROR_UPLOAD_STYLE
        # Message not printed, but stored in dcc.Store linked to dcc.Loading for throbber 
        message = ""
        # TODO: Comment out for deployment, this is for debugging/testing purposes
        print(f"Condition file's head:\n {df.head()}")
        return df.to_dict('records'), message, SUCCESS_UPLOAD_STYLE
    
    except Exception as e:
        return None, message, ERROR_UPLOAD_STYLE


# Callback for Baseline Reference Value Updates
# ----------------------------------------------
@app.callback(
    Output('mvc-left-store', 'data'),
    Output('mvc-right-store', 'data'),
    Output('rfd-left-store', 'data'),
    Output('rfd-right-store', 'data'),
    Input('input-mvc-left', 'value'),
    Input('input-mvc-right', 'value'),
    Input('input-rfd-left', 'value'),
    Input('input-rfd-right', 'value'),
    prevent_initial_callbacks=True
)
def update_reference_values(mvc_left, mvc_right, rfd_left, rfd_right):
    return mvc_left, mvc_right, rfd_left, rfd_right


# Callback to enable/disable the RFD window input
# ------------------------------------------------
@app.callback(
    Output('analysis-rfd-window-input', 'disabled'),
    Input('analysis-rfd-checkbox', 'value')
)
def toggle_rfd_input_disabled(is_checked):
    """
    Disables the RFD window input if the RFD analysis checkbox is unchecked.
    """
    return not is_checked

# Callback to enable/disable the Force Cutoff Frequency input
# --------------------------------------------------------------
@app.callback(
    Output("force-cutoff-input", "disabled"),
    Input("force-filter-check", "value")
)
def toggle_cutoff_sensitivity(filter_value):
    # If the list is empty ([]) or None, return True to keep it disabled
    # If it has [1], return False to enable it
    return not bool(filter_value)


# Callback for block navigation
# ------------------------------
@app.callback(
    Output('block-selector-dropdown', 'options'),
    Output('block-selector-dropdown', 'value'),
    Output('trial-lookup-store', 'data'),
    Input('ui-generator-signal-store', 'data'),
    State('signal-data-store', 'data')
)
def update_block_dropdown(signal, session_id):
    # --- FAILURE PATH ---
    # Guard clause: Ensure required inputs exist
    if not signal:
        raise PreventUpdate
    
    # Load data from temp feather file
    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    app_temp_dir.mkdir(exist_ok = True)
    filepath = app_temp_dir /f"{session_id}.feather"
    df = pd.read_feather(filepath)

    # Create lookup table to fetch block numbers
    trial_lookup = create_trial_lookup(df)
    if trial_lookup is None or trial_lookup.empty:
        print("Trial lookup table is empty or could not be created.")
        return [], None, None # Clears the dropdown if inputs are missing
    
    blocks = sorted(trial_lookup['block_number'].unique())
    block_options = [{'label': f'Block {b}', 'value': b} for b in blocks]
    default_block = blocks[0] if blocks else None

    #TODO: Comment out for deployment, this is for debugging/testing purposes
    print(f"Available blocks: {blocks}")

    return block_options, default_block, trial_lookup.to_dict('records')

# Callback for trial navigation
# ------------------------------
@app.callback(
    Output('trial-selector-dropdown', 'options'),
    Output('trial-selector-dropdown', 'value'),
    Output('block-selector-dropdown', 'value', allow_duplicate=True),
    Input('block-selector-dropdown', 'value'),
    Input('prev-trial-button', 'n_clicks'),
    Input('next-trial-button', 'n_clicks'),
    State('trial-selector-dropdown', 'value'),
    State('trial-lookup-store', 'data'),
    prevent_initial_call=True
)
def handle_trial_navigation(
    selected_block, prev_clicks, next_clicks, 
    current_trial, trial_lookup_data
):
    triggered_id = ctx.triggered_id
    if not triggered_id or not trial_lookup_data:
        raise PreventUpdate

    trial_lookup = pd.DataFrame(trial_lookup_data)

    # --- BRANCH 1: User manually changes the block dropdown ---
    if triggered_id == 'block-selector-dropdown':
        if not selected_block:
            return [], None, no_update
        
        # Standard behavior: populate trials and reset to the first one
        trials_in_block = sorted(
            trial_lookup[trial_lookup['block_number'] == selected_block]['trial_number'].unique()
        )
        trial_options = [{'label': f'Trial {t}', 'value': t} for t in trials_in_block]
        default_trial = trials_in_block[0] if trials_in_block else None
        
        # Don't update the block dropdown, as it was the trigger
        return trial_options, default_trial, no_update

    # --- BRANCH 2: User clicks a navigation button ---
    if triggered_id in ['prev-trial-button', 'next-trial-button']:
        if not selected_block or not current_trial:
            raise PreventUpdate

        # Get the min and max global indices for circular navigation (for circular navigation)
        min_global_index = trial_lookup['global_index'].min()
        max_global_index = trial_lookup['global_index'].max()

        try:
            current_row = trial_lookup.query(
                f"block_number == @selected_block and trial_number == @current_trial"
            )
            current_global_index = current_row['global_index'].iloc[0]

            # Determine the target index
            if triggered_id == 'next-trial-button':
                target_global_index = current_global_index + 1
            else: # 'prev-trial-button'
                target_global_index = current_global_index - 1

            # Implement circular navigation
            if target_global_index > max_global_index:
                target_global_index = min_global_index
            elif target_global_index < min_global_index:
                target_global_index = max_global_index
            
            # Find the new trial's info
            new_row = trial_lookup.query(f"global_index == @target_global_index")
            new_block = int(new_row['block_number'].iloc[0])
            new_trial = int(new_row['trial_number'].iloc[0])

            # If the block has changed, also update the trial options
            if new_block != selected_block:
                trials_in_new_block = sorted(
                    trial_lookup[trial_lookup['block_number'] == new_block]['trial_number'].unique()
                )
                new_options = [{'label': f'Trial {t}', 'value': t} for t in trials_in_new_block]
                return new_options, new_trial, new_block
            else:
                # If block is the same, only update the trial value
                return no_update, new_trial, no_update

        except (IndexError, KeyError):
            raise PreventUpdate

    # Default case
    return no_update, no_update, no_update


# Callback for trial segmentation initilization
# ----------------------------------------------
@app.callback(
    # --- Outputs ---
    Output('trial-graph', 'figure'),
    Output('trial-metrics-display', 'children'),
    Output('current-stim-time-store', 'data'),
    Output('current-trial-metrics-store', 'data'),
    
    # --- Triggers ---
    Input('block-selector-dropdown', 'value'),
    Input('trial-selector-dropdown', 'value'),
       # Force signal processing ---
    Input('force-filter-check', 'value'),
    Input('force-cutoff-input', 'value'),
    # --- Data Sources ---
    State('condition-data-store', 'data'),
    State('signal-data-store', 'data'),
    State('channel-map-store', 'data'),
    State('trial-lookup-store', 'data'),
    State('pre-stim-window-input', 'value'),
    State('post-stim-window-input', 'value'),
    State('mvc-left-store', 'data'),
    State('mvc-right-store', 'data'),
    
    # --- Peak Detection Settings ---
    State('min-valid-rt-input', 'value'),
    State('min-prominence-input', 'value'),
    State('pre-stim-search-input', 'value'),
    State('post-stim-search-input', 'value'),
    
    # --- Analysis Checkboxes ---
    State('analysis-peak-force-checkbox', 'value'),
    State('analysis-mrspt-checkbox', 'value'),
    State('analysis-mrt-checkbox', 'value'),
    State('analysis-fti-checkbox', 'value'),
    State('analysis-mean-force-checkbox', 'value'),
    State('analysis-rfd-checkbox', 'value'),
    State('analysis-rfd-window-input', 'value'),
    State('input-rfd-left', 'value'),
    State('input-rfd-right', 'value'),

    # --- EMG Analysis Checkboxes ---
    State('analysis-pmrt-checkbox', 'value'),
    State('analysis-rms-checkbox', 'value'),
    # --- EMG Settings ---
    State('emg-min-duration-input', 'value'),
    State('emg-h-onset-input', 'value'),
    State('emg-h-offset-input', 'value'),
)
def update_trial_data(
    selected_block, selected_trial, force_filter_val, force_cutoff_hz, condition_data_dict, session_id,
    channel_map, trial_lookup_dict, pre_window, post_window,
    mvc_left, mvc_right,
    min_valid_rt_s, min_prominence_n, pre_stim_search_s, post_stim_search_s,
    run_peak_force, run_motor_response_time, run_motor_reaction_time, run_force_time_integral,
    run_mean_force, run_rfd, rfd_window_ms,
    rfd_baseline_left, rfd_baseline_right,
    run_pmrt, run_emg_rms,
    emg_min_duration_ms, emg_h_onset, emg_h_offset
):
    """
    This is the main "controller" callback for the Trial Viewer.
    It runs the full, robust analysis pipeline when the selected trial changes.
    """
    if not all([session_id, channel_map, trial_lookup_dict, selected_block, selected_trial]):
        raise PreventUpdate

    #  Initial Data Loading
    # ----------------------
    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    filepath = app_temp_dir / f"{session_id}.feather"
    full_df = pd.read_feather(filepath)
    trial_lookup = pd.DataFrame(trial_lookup_dict)

    matching_trials = trial_lookup.query(
        f"block_number == @selected_block and trial_number == @selected_trial"
    )
    if matching_trials.empty:
        raise PreventUpdate
    global_index_to_use = matching_trials['global_index'].iloc[0]

    # Call function to get the base metrics (threshold, stim_time, etc.)
    # and the DataFrame for the user's visualization window.
    trial_view_df, base_metrics = get_trial_data_and_metrics(
        full_df=full_df,
        trial_lookup=trial_lookup,
        condition_data=pd.DataFrame(condition_data_dict),
        trial_index=global_index_to_use,
        channel_map=channel_map,
        mvc_left=mvc_left,
        mvc_right=mvc_right,
        pre_window=pre_window,
        post_window=post_window
    )
    # Determine filterering state
    is_filter_enabled = 1 in (force_filter_val or [])
    # Process both full dataframe and trial view dataframe
    if is_filter_enabled and force_cutoff_hz:
        # Determine which hand's force to filter based on available channels
        for hand in ['right', 'left']:
            force_col = channel_map.get(f'force_{hand}')
            if force_col and force_col in full_df.columns:
                # Apply zero-phase Butterworth filter to the full dataset
                full_df[force_col] = fa.apply_force_filter(
                    full_df[force_col].values,
                    cutoff = force_cutoff_hz
                )
                # Also update the view dataframe for consistent plotting
                trial_view_df[force_col] = fa.apply_force_filter(
                    trial_view_df[force_col].values,
                    cutoff= force_cutoff_hz
                )

    #  Analysis Pipeline
    # --------------------------
    
    # Find the main contraction event, regardless of hand.
    peak_info = fa.find_main_contraction_peak(
        full_df=full_df,
        stim_time=base_metrics['stim_time'],
        channel_map=channel_map,
        threshold=base_metrics['threshold'],
        min_valid_rt_s=min_valid_rt_s,
        min_prominence_n=min_prominence_n,
        search_window_pre_s=pre_stim_search_s,
        search_window_post_s=post_stim_search_s
    )
    
    # Set the initial status and the true responding hand from the peak finder
    base_metrics['trial_status'] = peak_info['status']
    base_metrics['response_hand'] = peak_info['response_hand']
    
    # Run all subsequent analyses only if a valid peak was found.
    if peak_info['analysis_df'] is not None and base_metrics['response_hand'] is not None:
        
        # Use the dynamically centered analysis window from the peak finder
        analysis_df = peak_info['analysis_df'].copy()
        
        # Store the found peak info
        base_metrics['peak_time'] = peak_info['peak_time']
        base_metrics['peak_value'] = peak_info['peak_value']
        base_metrics['time_to_peak'] = peak_info['peak_time'] - base_metrics['stim_time']

        # Calculate foundational metrics
        if (run_peak_force or run_motor_reaction_time or run_motor_response_time or 
            run_force_time_integral or run_mean_force or run_rfd):
            
            mvc_val = mvc_right if base_metrics.get('response_hand') == 'right' else mvc_left
            peak_time = base_metrics['stim_time'] + base_metrics['time_to_peak']
            
            # Find onset time (needed for MRT, FTI, Mean Force, RFD)
            if run_motor_reaction_time or run_force_time_integral or run_mean_force or run_rfd:
                onset_time = fa.find_contraction_onset(
                    signal_df=analysis_df,
                    stim_time=base_metrics['stim_time'],
                    peak_time=peak_time,
                    response_hand=base_metrics['response_hand']
                )
                base_metrics['force_onset_time'] = onset_time

            # Find offset time & baseline (needed for FTI, Mean Force, RFD)
            if run_force_time_integral or run_mean_force or run_rfd:
                offset_time = fa.find_contraction_offset(
                    signal_df=analysis_df,
                    peak_time=peak_time,
                    peak_value=base_metrics['peak_value'],
                    response_hand=base_metrics['response_hand']
                )
                base_metrics['force_offset_time'] = offset_time
                
                baseline = fa.find_baseline_force(
                    signal_df=analysis_df,
                    peak_time=peak_time,
                    response_hand=base_metrics['response_hand']
                )
                base_metrics['baseline_force'] = baseline

        # Calculate final leaf metrics
        
        if run_peak_force:
            mvc_val = mvc_right if base_metrics.get('response_hand') == 'right' else mvc_left
            derived_peak_metrics = fa.peak_force_metrics(
                peak_value=base_metrics['peak_value'],
                peak_time=base_metrics['peak_time'],
                stim_time=base_metrics['stim_time'],
                threshold=base_metrics['threshold'],
                mvc_value=mvc_val
            )
            base_metrics.update(derived_peak_metrics)

        if run_motor_reaction_time:
            base_metrics['motor_reaction_time'] = fa.motor_reaction_time(
                stim_time=base_metrics.get('stim_time'), 
                onset_time=base_metrics.get('force_onset_time')
            )
        
        if run_motor_response_time:
            if all(k in base_metrics for k in ['force_onset_time', 'peak_time']):
                mrspt_val = fa.motor_response_time(
                    signal_df=analysis_df,
                    stim_time=base_metrics.get('stim_time'),
                    peak_time=base_metrics.get('peak_time'),
                    peak_force=base_metrics.get('peak_value'),
                    threshold=base_metrics.get('threshold'),
                    response_hand=base_metrics.get('response_hand')
                )
                base_metrics['motor_response_time'] = mrspt_val

        if run_force_time_integral:
            if all(k in base_metrics for k in ['force_onset_time', 'force_offset_time', 'baseline_force']):
                mvc_val = mvc_right if base_metrics.get('response_hand') == 'right' else mvc_left
                impulse_metrics = fa.calculate_impulse(
                    signal_df=analysis_df,
                    onset_time=base_metrics['force_onset_time'],
                    offset_time=base_metrics['force_offset_time'],
                    baseline_force=base_metrics['baseline_force'],
                    mvc_value=mvc_val,
                    response_hand=base_metrics['response_hand']
                )
                base_metrics.update(impulse_metrics)
        
        if run_mean_force:
            if all(k in base_metrics for k in ['force_onset_time', 'force_offset_time', 'baseline_force']):
                mvc_val = mvc_right if base_metrics.get('response_hand') == 'right' else mvc_left
                mean_force_metrics = fa.calculate_mean_force(
                    signal_df=analysis_df,
                    onset_time=base_metrics['force_onset_time'],
                    offset_time=base_metrics['force_offset_time'],
                    baseline_force=base_metrics['baseline_force'],
                    mvc_value=mvc_val,
                    response_hand=base_metrics['response_hand']
                )
                base_metrics.update(mean_force_metrics)
        
        if run_rfd:
            if all(k in base_metrics for k in ['force_onset_time', 'peak_time', 'baseline_force']):
                rfd_metrics = fa.calculate_rfd(
                    signal_df=analysis_df,
                    onset_time=base_metrics['force_onset_time'],
                    peak_time=base_metrics['peak_time'],
                    baseline_force=base_metrics['baseline_force'],
                    response_hand=base_metrics['response_hand'],
                    early_rfd_window_ms=rfd_window_ms
                )
                base_metrics.update(rfd_metrics)
                
                # Perform normalization for RFD
                rfd_baseline = rfd_baseline_right if base_metrics.get('response_hand') == 'right' else rfd_baseline_left
                if rfd_baseline and rfd_baseline > 0:
                    if 'early_rfd' in base_metrics and base_metrics['early_rfd'] is not None:
                        base_metrics['early_rfd_pct'] = (base_metrics['early_rfd'] / rfd_baseline) * 100
                    if 'peak_rfd' in base_metrics and base_metrics['peak_rfd'] is not None:
                        base_metrics['peak_rfd_pct'] = (base_metrics['peak_rfd'] / rfd_baseline) * 100

        # EMG Analysis Section
        # ---------------------
        
        if (channel_map.get('emg_left') and channel_map.get('emg_right')) and (run_pmrt or run_emg_rms):
            try:
                # Calculate local dynamic thresholds for BOTH onset and offset
                onset_threshold = ea.calculate_dynamic_threshold(
                    full_df=analysis_df,
                    channel_map=channel_map,
                    response_hand=base_metrics['response_hand'],
                    duration_sec=0.1,  
                    h_multiplier=emg_h_onset if emg_h_onset is not None else 15.0 # From UI
                )
                
                offset_threshold = ea.calculate_dynamic_threshold(
                    full_df=analysis_df,
                    channel_map=channel_map,
                    response_hand=base_metrics['response_hand'],
                    duration_sec=0.1,  
                    h_multiplier=emg_h_offset if emg_h_offset is not None else 30.0 # From UI
                    # Default is higher to avoid premature offsets, as EMG activity often trails
                )

                # Define the search window based on force metrics
                if base_metrics['force_onset_time'] is None or base_metrics['force_offset_time'] is None:
                    viz_df = analysis_df
                else:                                 
                    viz_start_time = base_metrics['force_onset_time'] - 0.200
                    viz_end_time   = base_metrics['force_offset_time'] + 0.200
                    time_col = channel_map['time']
                    viz_df = analysis_df[
                        (analysis_df[time_col] >= viz_start_time) &
                        (analysis_df[time_col] <= viz_end_time)
                    ].copy()

                # Detect EMG boundaries using dual thresholds
                onset_time, offset_time, active_threshold = ea.find_emg_boundaries(
                    signal_df=viz_df,
                    channel_map=channel_map,
                    response_hand=base_metrics['response_hand'],
                    stim_time=base_metrics['stim_time'],
                    force_onset_time=base_metrics.get('force_onset_time'),
                    force_offset_time=base_metrics.get('force_offset_time'),
                    min_burst_ms=emg_min_duration_ms if emg_min_duration_ms is not None else 10,
                    threshold_on=onset_threshold,  # Pass onset threshold
                    threshold_off=offset_threshold # Pass offset threshold
                )

                if onset_time and offset_time:
                    base_metrics['emg_onset_time'] = onset_time
                    base_metrics['emg_offset_time'] = offset_time
                    # Store onset threshold for the graph visualization
                    base_metrics['emg_threshold'] = onset_threshold 

                    # 4. Compute pre-motor RT (Stimulus -> EMG Onset)
                    if run_pmrt:
                        base_metrics['premotor_reaction_time'] = (onset_time - base_metrics['stim_time']) * 1000

                    # 5. Compute RMS amplitude
                    if run_emg_rms:
                        base_metrics['emg_rms'] = ea.calculate_emg_rms(
                            full_df=analysis_df,
                            channel_map=channel_map,
                            response_hand=base_metrics['response_hand'],
                            onset_time=onset_time,
                            offset_time=offset_time
                        )
                else:
                    base_metrics['emg_onset_time'] = None
                    base_metrics['emg_offset_time'] = None
                    base_metrics['premotor_reaction_time'] = None
                    base_metrics['emg_rms'] = None

            except Exception as e:
                print(f"⚠️ TKEO EMG detection error: {e}")
                base_metrics['emg_onset_time'] = None
                base_metrics['emg_offset_time'] = None
                base_metrics['premotor_reaction_time'] = None
                base_metrics['emg_rms'] = None

    # Final trial status interpretation
    # ----------------------------------
    LATE_RESPONSE_THRESHOLD_S = 1.0
    if base_metrics['trial_status'] == 'valid':
        mrt_s = base_metrics.get('motor_reaction_time', float('inf')) / 1000.0 if base_metrics.get('motor_reaction_time') is not None else float('inf')
        if mrt_s > LATE_RESPONSE_THRESHOLD_S:
            base_metrics['trial_status'] = 'late_response'
        elif base_metrics.get('force_offset_time') is not None and base_metrics.get('force_offset_time') >= trial_view_df['time'].iloc[-1]:
            base_metrics['trial_status'] = 'late_offset'

    # Plotting and Metrics Display
    # -----------------------------
    base_metrics['block_trial_str'] = f"Block {selected_block}, Trial {selected_trial}"
    
    fig = create_trial_figure(
    trial_view_df, channel_map, mvc_left, mvc_right, base_metrics,
    run_peak_force=run_peak_force,
    run_motor_response_time=run_motor_response_time,
    run_motor_reaction_time=run_motor_reaction_time,
    run_force_time_integral=run_force_time_integral,
    run_pmrt=run_pmrt,
    run_emg_rms=run_emg_rms
)
    
    def create_metric_p(label, key, unit=""):
        value = base_metrics.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return html.P(f"{label}: N/A")
        if isinstance(value, float):
            return html.P(f"{label}: {value:.2f} {unit}")
        return html.P(f"{label}: {value} {unit}")

    key_info_metrics = [
        create_metric_p("Participant Id", "participant_id"),
        create_metric_p("Global Index", "global_index"),
        create_metric_p("Block", "block"),
        create_metric_p("Trial Status", "trial_status"),
        create_metric_p("Response Hand", "response_hand"),
    ]
    latency_metrics = [
        create_metric_p("Motor Reaction Time", "motor_reaction_time", "ms"),
        create_metric_p("Motor Response Time", "motor_response_time", "ms"),
    ]
    magnitude_metrics = [
        create_metric_p("Peak Force", "peak_value", "N"),
        create_metric_p("Mean Force", "mean_force", "N"),
        create_metric_p("Overshoot/Undershoot", "delta_threshold", "N"),
    ]
    rate_metrics = [
        create_metric_p(f"Early RFD (0-{rfd_window_ms}ms)", "early_rfd", "N/s"),
        create_metric_p("Early RFD (% Max)", "early_rfd_pct", "%"),
        create_metric_p("Peak RFD", "peak_rfd", "N/s"),
        create_metric_p("Peak RFD (% Max)", "peak_rfd_pct", "%"),
    ]
    integral_metrics = [
        create_metric_p("Impulse (AUC)", "impulse_auc", "N*s"),
        create_metric_p("Mean Force as %MVC", "impulse_auc_percent_mvc", "%"),
    ]
    emg_metrics = [
        create_metric_p("Premotor Reaction Time", "premotor_reaction_time", "ms"),
        create_metric_p("EMG RMS", "emg_rms", "a.u."),
    ]

    metrics_layout = html.Div([
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    dbc.Card(dbc.CardBody(key_info_metrics)),
                    title="Key Trial Info",
                    item_id="item-key-info"
                ),
                dbc.AccordionItem(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(latency_metrics, title="Latencies", item_id="sub-latencies"),
                            dbc.AccordionItem(magnitude_metrics, title="Force Magnitudes", item_id="sub-magnitudes"),
                            dbc.AccordionItem(rate_metrics, title="Rates of Force", item_id="sub-rates"),
                            dbc.AccordionItem(integral_metrics, title="Force-Time Integral", item_id="sub-integral"),
                        ],
                        always_open=True,
                        active_item=["sub-latencies", "sub-magnitudes", "sub-rates"]
                    ),
                    title="Force Metrics",
                    item_id="item-force-metrics"
                ),
                dbc.AccordionItem(
                    emg_metrics,
                    title="EMG Metrics",
                    item_id="item-emg-metrics"
                ),
            ],
            always_open=True,
            active_item="item-key-info"
        )
    ])

    stim_time = base_metrics.get('stim_time')

    return fig, metrics_layout, stim_time, base_metrics


# Callback for updating the trial viewer
# ---------------------------------------
@app.callback(
    Output('trial-graph', 'figure', allow_duplicate=True),
    # --- TRIGGERS ---
    Input('pre-stim-window-input', 'value'),
    Input('post-stim-window-input', 'value'),
    # --- DATA SOURCES (as State) ---
    State('signal-data-store', 'data'),
    State('channel-map-store', 'data'),
    State('mvc-left-store', 'data'),
    State('mvc-right-store', 'data'),
    State('current-stim-time-store', 'data'),
    State('current-trial-metrics-store', 'data'),
    # Add states for the analysis checkboxes
    State('analysis-peak-force-checkbox', 'value'),
    State('analysis-mrspt-checkbox', 'value'),
    State('analysis-mrt-checkbox', 'value'),
    State('analysis-fti-checkbox', 'value'),
    prevent_initial_call=True
)
def update_graph_view(
    pre_window, post_window, session_id, channel_map, 
    mvc_left, mvc_right, stim_time, trial_metrics,
    run_peak_force, run_motor_response_time, run_motor_reaction_time, run_force_time_integral
):
    """
    This callback runs the "Partial Update" when the view window changes.
    It ONLY re-slices the data and redraws the plot. It does not calculate metrics.
    """
    if not all([session_id, channel_map, stim_time, trial_metrics]):
        raise PreventUpdate

    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    filepath = app_temp_dir / f"{session_id}.feather"
    df = pd.read_feather(filepath)

    time_col = channel_map.get('time')
    trial_segment_df = get_trial_segment(
        full_df=df,
        stim_time=stim_time,
        time_col=time_col,
        pre_window=pre_window,
        post_window=post_window
    )

    # Call the helper function with the new boolean flags
    fig = create_trial_figure(
        trial_segment_df, channel_map, mvc_left, mvc_right, trial_metrics,
        run_peak_force=run_peak_force, 
        run_motor_response_time=run_motor_response_time,
        run_motor_reaction_time=run_motor_reaction_time,
        run_force_time_integral=run_force_time_integral
    )
    
    return fig


# Callback for trial data download
# ---------------------------------------
@app.callback(
    Output("download-trial-csv", "data"),
    Input("btn-download-trial", "n_clicks"),
    # --- Identifiers ---
    State('signal-data-store', 'data'),      # session_id
    State('trial-lookup-store', 'data'),     # list of all trials
    State('condition-data-store', 'data'),   
    State('channel-map-store', 'data'),
    # --- Analysis Settings (to match the UI) ---
    State('mvc-left-store', 'data'),
    State('mvc-right-store', 'data'),
    State('pre-stim-window-input', 'value'),
    State('post-stim-window-input', 'value'),
    State('min-valid-rt-input', 'value'),
    State('min-prominence-input', 'value'),
    State('pre-stim-search-input', 'value'),
    State('post-stim-search-input', 'value'),
    # --- Analysis Toggles ---
    State('analysis-peak-force-checkbox', 'value'),
    State('analysis-mrspt-checkbox', 'value'),
    State('analysis-mrt-checkbox', 'value'),
    State('analysis-fti-checkbox', 'value'),
    State('analysis-mean-force-checkbox', 'value'),
    State('analysis-rfd-checkbox', 'value'),
    State('analysis-rfd-window-input', 'value'),
    # --- EMG Settings ---
    State('analysis-pmrt-checkbox', 'value'),
    State('analysis-rms-checkbox', 'value'),
    State('emg-min-duration-input', 'value'),
    State('emg-h-onset-input', 'value'), 
    State('emg-h-offset-input', 'value'),
    prevent_initial_call=True,
)
def handle_bulk_metrics_download(
    n_clicks, session_id, lookup_dict, condition_dict, channel_map,
    mvc_left, mvc_right, pre_window, post_window, 
    min_valid_rt_s, min_prominence_n, pre_stim_search_s, post_stim_search_s,
    run_peak_force, run_motor_response_time, run_motor_reaction_time, run_force_time_integral,
    run_mean_force, run_rfd, rfd_window_ms,
    run_pmrt, run_emg_rms, emg_min_duration_ms, emg_h_onset, emg_h_offset
):
    if not n_clicks:
        raise PreventUpdate

    # Setup data
    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    filepath = app_temp_dir / f"{session_id}.feather"
    full_df = pd.read_feather(filepath)
    trial_lookup = pd.DataFrame(lookup_dict)
    condition_data = pd.DataFrame(condition_dict)
    
    all_final_metrics = []

    # Start the loop
    for _, trial_row in trial_lookup.iterrows():
        global_idx = trial_row['global_index']
        
        # Base data & thresholds
        trial_view_df, base_metrics = get_trial_data_and_metrics(
            full_df=full_df, trial_lookup=trial_lookup, condition_data=condition_data,
            trial_index=global_idx, channel_map=channel_map, mvc_left=mvc_left, mvc_right=mvc_right,
            pre_window=pre_window, post_window=post_window
        )

        # Peak finder
        peak_info = fa.find_main_contraction_peak(
            full_df=full_df, stim_time=base_metrics['stim_time'], channel_map=channel_map,
            threshold=base_metrics['threshold'], min_valid_rt_s=min_valid_rt_s,
            min_prominence_n=min_prominence_n, search_window_pre_s=pre_stim_search_s,
            search_window_post_s=post_stim_search_s
        )
        
        base_metrics['trial_status'] = peak_info['status']
        base_metrics['response_hand'] = peak_info['response_hand']

        # Metric pipeline
        if peak_info['analysis_df'] is not None and base_metrics['response_hand'] is not None:
            analysis_df = peak_info['analysis_df']
            peak_time = peak_info['peak_time']
            
            # Peak & Timing
            # ---------------
            base_metrics.update({
                'peak_time': peak_info['peak_time'],
                'peak_value': peak_info['peak_value'],
                'time_to_peak': peak_info['peak_time'] - base_metrics['stim_time']
            })

            # Onset/Offset for Force
            # ------------------------
            onset_time = fa.find_contraction_onset(analysis_df, base_metrics['stim_time'], peak_time, base_metrics['response_hand'])
            offset_time = fa.find_contraction_offset(analysis_df, peak_time, peak_info['peak_value'], base_metrics['response_hand'])
            baseline = fa.find_baseline_force(analysis_df, peak_time, base_metrics['response_hand'])
            
            base_metrics['force_onset_time'] = onset_time
            base_metrics['force_offset_time'] = offset_time

            # Calculate Individual Metrics
            # ------------------------------
            if run_peak_force:
                mvc_val = mvc_right if base_metrics['response_hand'] == 'right' else mvc_left
                base_metrics.update(fa.peak_force_metrics(base_metrics['peak_value'], peak_time, base_metrics['stim_time'], base_metrics['threshold'], mvc_val))
            
            if run_motor_reaction_time:
                base_metrics['motor_reaction_time'] = fa.motor_reaction_time(base_metrics['stim_time'], onset_time)
            
            if run_rfd:
                base_metrics.update(fa.calculate_rfd(analysis_df, onset_time, peak_time, baseline, base_metrics['response_hand'], rfd_window_ms))

            #  EMG analyses
            # ---------------- 
            if (channel_map.get('emg_left') and channel_map.get('emg_right')) and (run_pmrt or run_emg_rms):
                try:
                    # Dual Threshold Detection
                    on_thresh = ea.calculate_dynamic_threshold(analysis_df, channel_map, base_metrics['response_hand'], 0.1, emg_h_onset)
                    off_thresh = ea.calculate_dynamic_threshold(analysis_df, channel_map, base_metrics['response_hand'], 0.1, emg_h_offset)
                    
                    emg_on, emg_off, _ = ea.find_emg_boundaries(analysis_df, channel_map, base_metrics['response_hand'], base_metrics['stim_time'], onset_time, offset_time, emg_min_duration_ms, on_thresh, off_thresh)
                    
                    if emg_on:
                        base_metrics['premotor_reaction_time'] = (emg_on - base_metrics['stim_time']) * 1000
                        if run_emg_rms:
                            base_metrics['emg_rms'] = ea.calculate_emg_rms(analysis_df, channel_map, base_metrics['response_hand'], emg_on, emg_off)
                except: pass

        all_final_metrics.append(base_metrics)

    # Save to CSV
    # --------------
    final_df = pd.DataFrame(all_final_metrics)
    
    print(f"✅ Bulk Export Complete: {len(final_df)} trials analyzed.")
    
    return dcc.send_data_frame(
        final_df.to_csv, 
        f"CogMo_Bulk_Export_{session_id}.csv", 
        index=False
    )

# ==============================================================================
# --- MAIN APP EXECUTION ---
# ==============================================================================
def clean_temp_dir():
    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    if app_temp_dir.exists():
        shutil.rmtree(app_temp_dir, ignore_errors=True)
    app_temp_dir.mkdir(parents=True, exist_ok=True)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")

if __name__ == '__main__':
    clean_temp_dir()
    # Timer ensures the server is actually up before the browser tries to hit it
    Timer(1.5, open_browser).start()
    app.run(debug=False, host='127.0.0.1', port=8050)