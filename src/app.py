# ==============================================================================
# CogMo Toolkit - Main Application
# This Dash application provides an interface for uploading and analyzing
# psychophysiological signal data and associated condition order files.
# ==============================================================================

# ==============================================================================
# --- IMPORTS AND APP INITIALIZATION ---
# ==============================================================================
# Standard Library Imports
# ----------------------------------------------------
import base64
import io
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path

# Third-Party Dependencies
# ----------------------------------------------------
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html, no_update, ctx
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL, Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import webview

# Local Application Imports
#---------------------------
from get_condition_lookup import get_condition_lookup
from data_loader import load_signal, find_best_match
from trial_segmentation import get_trial_data_and_metrics, get_trial_segment, create_trial_lookup
from force_analyses import calculate_impulse, find_baseline_force, find_contraction_offset, find_contraction_onset, motor_reaction_time, motor_response_time, peak_force_metrics

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
# ----------------------------------------------------

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__,
                 external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME],
                 prevent_initial_callbacks = True)


# ==============================================================================
# --- UPLOAD BUTTONS ---
# ==============================================================================
DEFAULT_UPLOAD_STYLE = {
    'height' : '60px',
    'lineHeight' : '60px',
    'borderWidth' : '1px',
    'borderStyle' : 'dashed',
    'borderRadius' : '5px',
    'textAlign' : 'center',
}

# New style for successful uploads
SUCCESS_UPLOAD_STYLE = DEFAULT_UPLOAD_STYLE.copy()
SUCCESS_UPLOAD_STYLE['borderColor'] = 'green'
SUCCESS_UPLOAD_STYLE['backgroundColor'] = '#F0FFF0'

# New style for failed uploads
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
    run_force_time_integral: bool = False
):
    """Creates the Plotly figure for the trial viewer, with conditional visualizations."""
    time_col = channel_map.get('time')
    force_r_col = channel_map.get('force_right')
    force_l_col = channel_map.get('force_left')
    emg_r_col = channel_map.get('emg_right')
    emg_l_col = channel_map.get('emg_left')
    include_emg = bool(emg_r_col and emg_l_col)

    COLOR_R = 'blue'
    COLOR_L = 'goldenrod'
    max_mvc = max(mvc_left, mvc_right) if mvc_left and mvc_right else 1
    threshold_value = trial_metrics.get('threshold')
    selected_trial = trial_metrics.get('block_trial_str', 'Trial')
    stim_time = trial_metrics.get('stim_time')

    if include_emg:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_r_col],
                                name='Right Hand', legendgroup='right', line=dict(color=COLOR_R)), row=1, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_l_col],
                                name='Left Hand', legendgroup='left', line=dict(color=COLOR_L)), row=1, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[emg_r_col],
                                name='EMG Right', legendgroup='right', showlegend=False, line=dict(width=0.5, color=COLOR_R)), row=2, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[emg_l_col],
                                name='EMG Left', legendgroup='left', showlegend=False, line=dict(width=0.5, color=COLOR_L)), row=2, col=1)
        fig.update_yaxes(title_text="EMG", row=2, col=1)
    else:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_r_col],
                                name='Right Hand', line=dict(color=COLOR_R)), row=1, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_l_col],
                                name='Left Hand', line=dict(color=COLOR_L)), row=1, col=1)

    fig.update_yaxes(title_text="Force", range=[0, max_mvc], row=1, col=1)

    if threshold_value is not None:
        fig.add_hline(y=threshold_value, line_dash="dash", line_color="grey", row=1, col=1)

    # --- Visualization for Impulse (AUC) ---
    if run_force_time_integral and all(k in trial_metrics for k in ['force_onset_time', 'force_offset_time', 'baseline_force']):
        onset_time = trial_metrics['force_onset_time']
        offset_time = trial_metrics['force_offset_time']
        baseline_force = trial_metrics['baseline_force']
        response_hand = trial_metrics.get('response_hand')
        force_col = f"force_{response_hand}"
        
        auc_df = trial_segment_df[
            (trial_segment_df['time'] >= onset_time) & (trial_segment_df['time'] <= offset_time)
        ].copy()
        
        fig.add_trace(go.Scatter(
            x=auc_df[time_col],
            y=auc_df[force_col] - baseline_force,
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(40, 167, 69, 0.3)',
            showlegend=False,
            name='Impulse (AUC)'
        ), row=1, col=1)

    # --- Visualization for Peak Force ---
    peak_force = trial_metrics.get('peak_force')
    time_to_peak = trial_metrics.get('time_to_peak')
    if run_peak_force and all(v is not None for v in [peak_force, time_to_peak, stim_time, threshold_value]):
        peak_time = stim_time + time_to_peak
        fig.add_trace(go.Scatter(
            x=[peak_time, peak_time], y=[threshold_value, peak_force],
            mode='lines', line=dict(color='red', dash='dash', width=1),
            showlegend=False
        ), row=1, col=1)

    # --- Visualization for Motor Reaction Time ---
    mrt = trial_metrics.get('motor_reaction_time')
    if run_motor_reaction_time and all(v is not None for v in [mrt, stim_time, threshold_value]):
        force_onset_time = stim_time + (mrt / 1000.0)
        fig.add_trace(go.Scatter(
            x=[force_onset_time, force_onset_time],
            y=[0, threshold_value],
            mode='lines',
            line=dict(color='orange', dash='dash', width=1),
            name='Force Onset (MRT)',
            showlegend=False
        ), row=1, col=1)

    # --- Visualization for Motor Response Time ---
    mrspt = trial_metrics.get('motor_response_time')
    if run_motor_response_time and all(v is not None for v in [mrspt, stim_time, threshold_value]):
        force_onset_time = stim_time + (mrspt / 1000.0)
        fig.add_trace(go.Scatter(
            x=[force_onset_time, force_onset_time],
            y=[0, threshold_value],
            mode='lines',
            line=dict(color='green', dash='dash', width=1),
            name='Force Onset (MRsT)',
            showlegend=False
        ), row=1, col=1)
        
    # --- Layout with Stimulus Arrow Annotation ---
    annotations = []
    if stim_time is not None:
        annotations.append(
            dict(
                x=stim_time, y=0, xref="x", yref="y", text="",
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1.5,
                arrowcolor="black", ax=0, ay=-40
            )
        )

    fig.update_layout(
        title=selected_trial,
        margin=dict(t=30, b=0, l=0, r=0),
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor="rgba(255, 255, 255, 0.75)"
        ),
        annotations=annotations
    )
    
    return fig

# ==============================================================================
# --- APP LAYOUT (UI) ---
# ==============================================================================
app.layout = dbc.Container([
    html.H1("CogMo toolkit", className="my-3"),
    dbc.Tabs([
        # First Tab: Upload Data
        dbc.Tab(
            label="Upload Data",
            children=[
                html.Div([
                    dbc.Row(className="g-2", children=[
                        dbc.Col(
                            html.Div([
                                html.H4("Upload your data file"),
                                dcc.Upload(
                                    id='upload-signal-data',
                                    children=html.Div([
                                        'Your raw data here (.csv, .tsv, .txt, .xlsx)'
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
                                    id="data-upload-throbber",
                                    type="dot",
                                    children=html.Div(id='data-upload-output-message',
                                                      className="mt-1",
                                                      style={'min-height': '25px'}),
                                )
                            ]),
                            width=6,
                        ),
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
                    
                    # --- Channel Mapping Section ---
                    html.Div(id='channel-mapping-container', className="mt-2"),
                    
                    # --- Baseline Reference values ---
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
        

        # --- Second tab: Analyses Option Tab ---
        dbc.Tab(
            label="Analyses Option",
            children=[
                html.Div([
                    dbc.Row([
                        # --- Left Column: Force Analyses ---
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

                        # --- Right Column: EMG Analyses ---
                        dbc.Col([
                            html.H4("EMG Signal Analyses"),
                            html.Hr(),

                            html.H6("Latency", className="mt-3"),
                            dbc.Checkbox(id="analysis-pmrt-checkbox", label="Premotor Reaction Time"),
                            
                            html.H6("EMG Activity", className="mt-4"),
                            dbc.Checkbox(id="analysis-rms-checkbox", label="Root Mean Square (RMS)"),
                        ], width=6),
                    ]),
                    
                    # --- Tooltips for all checkboxes ---
                    dbc.Tooltip("Time from stimulus onset to force onset.", target="analysis-mrt-checkbox"),
                    dbc.Tooltip("Time from force onset to peak force.", target="analysis-mrspt-checkbox"),
                    dbc.Tooltip("The steepest slope of the force curve after force onset.", target="analysis-rfd-checkbox"),
                    dbc.Tooltip("Time from EMG onset to force onset.", target="analysis-pmrt-checkbox"),
                    dbc.Tooltip("The peak force achieved and the amount it exceeds the target.", target="analysis-peak-force-checkbox"),
                    dbc.Tooltip("The average force during a specified window.", target="analysis-mean-force-checkbox"),
                    dbc.Tooltip("The area under the force-time curve.", target="analysis-fti-checkbox"),
                    dbc.Tooltip("A measure of the magnitude of the EMG signal.", target="analysis-rms-checkbox"),

                ], className="p-3")
            ]
        ),

        # Third Tab: Trial Viewer
        dbc.Tab(
            label="Trial Viewer",
            children=[
                html.Div([
                    # --- Section 1: Trial Controls ---
                    html.H4("Trial Controls", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Block:"),
                            dcc.Dropdown(id='block-selector-dropdown')
                        ], width=6, lg=4), # Takes less space on large screens
                        dbc.Col([
                            dbc.Label("Select Trial:"),
                            dbc.InputGroup([
                                dbc.Button(
                                    html.I(className="fas fa-chevron-left"),
                                    id='prev-trial-button', n_clicks=0, color="secondary", outline=True
                                ),
                                dcc.Dropdown(id='trial-selector-dropdown', style={'flex': '1'}),
                                dbc.Button(
                                    html.I(className="fas fa-chevron-right"),
                                    id='next-trial-button', n_clicks=0, color="secondary", outline=True
                                )
                            ])
                        ], width=6, lg=4)
                    ]),

                    html.Hr(),

                    # --- Section 2: View Parameters ---
                    html.H4("View Parameters", className="mt-3"),
                    dbc.Row([
                         dbc.Col([
                            dbc.Label("Pre-Stimulus Window (s):"),
                            dbc.Input(id='pre-stim-window-input', type='number', value=1, step=0.05),
                         ], width=6, lg=4),
                         dbc.Col([
                            dbc.Label("Post-Stimulus Window (s):"),
                            dbc.Input(id='post-stim-window-input', type='number', value=2, step=0.05),
                         ], width=6, lg=4)
                    ]),
                    
                    html.Hr(),

                    # --- Section 3: Trial Plot (now full-width) ---
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='trial-graph', style={'height': '40vh'})
                        ], width=12) 
                    ]),

                    html.Hr(),

                    # --- Section 4: Trial Metrics (now full-width) ---
                    dbc.Row([
                        dbc.Col([
                            html.H4("Trial Metrics"),
                            html.Div(id='trial-metrics-display'), 
                        ], width=12)
                    ])

                ], className="p-3")
            ]
        ),
    ])
], fluid=False, style={'width': '1000px', 'overflow-x': 'hidden'})


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
    Output('trial-graph', 'figure'),
    Output('trial-metrics-display', 'children'),
    Output('current-stim-time-store', 'data'),
    Output('current-trial-metrics-store', 'data'),
    # Triggers
    Input('block-selector-dropdown', 'value'),
    Input('trial-selector-dropdown', 'value'),
    # Data Sources (as State)
    State('condition-data-store', 'data'),
    State('signal-data-store', 'data'),
    State('channel-map-store', 'data'),
    State('trial-lookup-store', 'data'),
    State('pre-stim-window-input', 'value'),
    State('post-stim-window-input', 'value'),
    State('mvc-left-store', 'data'),
    State('mvc-right-store', 'data'),
    # Analysis States
    State('analysis-peak-force-checkbox', 'value'),
    State('analysis-mrspt-checkbox', 'value'),
    State('analysis-mrt-checkbox', 'value'),
    State('analysis-fti-checkbox', 'value')    
)
def update_trial_data(
    selected_block, selected_trial, condition_data_dict, session_id,
    channel_map, trial_lookup_dict, pre_window, post_window,
    mvc_left, mvc_right,
    run_peak_force, run_motor_response_time, run_motor_reaction_time, run_force_time_integral
):
    """
    This callback runs a "Full Update" when the selected trial changes,
    including any selected single-trial analyses in a dependency chain.
    """
    if not all([session_id, channel_map, trial_lookup_dict, selected_block, selected_trial]):
        raise PreventUpdate

    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    filepath = app_temp_dir / f"{session_id}.feather"
    df = pd.read_feather(filepath)
    trial_lookup = pd.DataFrame(trial_lookup_dict)

    matching_trials = trial_lookup.query(
        f"block_number == @selected_block and trial_number == @selected_trial"
    )
    if matching_trials.empty:
        raise PreventUpdate
    global_index_to_use = matching_trials['global_index'].iloc[0]

    # Call the main "contractor" function to get the base data and metrics
    trial_segment_df, base_metrics = get_trial_data_and_metrics(
        full_df=df,
        trial_lookup=trial_lookup,
        condition_data=pd.DataFrame(condition_data_dict),
        trial_index=global_index_to_use,
        channel_map=channel_map,
        mvc_left=mvc_left,
        mvc_right=mvc_right,
        pre_window=pre_window,
        post_window=post_window
    )

    # --- Analysis Dependency Chain ---
    # ----------------------------------

     # 1. Calculate Foundational Metrics if needed
    if run_peak_force or run_motor_reaction_time or run_motor_response_time or run_force_time_integral:
        peak_metrics = peak_force_metrics(
            signal_df=trial_segment_df,
            stim_time=base_metrics['stim_time'],
            response_hand=base_metrics['response_hand'],
            threshold=base_metrics['threshold'],
            mvc_left=mvc_left,
            mvc_right=mvc_right
        )
        base_metrics.update(peak_metrics)
        
        if 'time_to_peak' in base_metrics:
            peak_time = base_metrics['stim_time'] + base_metrics['time_to_peak']
            
            if run_motor_reaction_time or run_force_time_integral:
                onset_time = find_contraction_onset(
                    signal_df=trial_segment_df,
                    stim_time=base_metrics['stim_time'],
                    peak_time=peak_time,
                    response_hand=base_metrics['response_hand']
                )
                base_metrics['force_onset_time'] = onset_time

            if run_force_time_integral:
                offset_time = find_contraction_offset(
                    signal_df=trial_segment_df,
                    peak_time=peak_time,
                    peak_value=base_metrics['peak_force'],
                    response_hand=base_metrics['response_hand']
                )
                base_metrics['force_offset_time'] = offset_time
                
                baseline = find_baseline_force(
                    signal_df=trial_segment_df,
                    peak_time=peak_time,
                    response_hand=base_metrics['response_hand']
                )
                base_metrics['baseline_force'] = baseline

    # 2. Calculate Final "Leaf" Metrics
    if run_motor_reaction_time:
        mrt_val = motor_reaction_time(
            stim_time=base_metrics.get('stim_time'),
            onset_time=base_metrics.get('force_onset_time')
        )
        base_metrics['motor_reaction_time'] = mrt_val
    
    if run_motor_response_time:
        if all(k in base_metrics for k in ['time_to_peak', 'peak_force']):
            mrspt_val = motor_response_time(
                signal_df=trial_segment_df,
                stim_time=base_metrics['stim_time'],
                peak_time=base_metrics['stim_time'] + base_metrics['time_to_peak'],
                peak_force=base_metrics['peak_force'],
                threshold=base_metrics['threshold'],
                response_hand=base_metrics['response_hand']
            )
            base_metrics['motor_response_time'] = mrspt_val
            
    if run_force_time_integral:
        if all(k in base_metrics for k in ['force_onset_time', 'force_offset_time', 'baseline_force']):
            mvc_val = mvc_right if base_metrics.get('response_hand') == 'right' else mvc_left
            
            impulse_metrics = calculate_impulse(
                signal_df=trial_segment_df,
                onset_time=base_metrics.get('force_onset_time'),
                offset_time=base_metrics.get('force_offset_time'),
                baseline_force=base_metrics.get('baseline_force'),
                mvc_value=mvc_val,
                response_hand=base_metrics.get('response_hand')
            )
            base_metrics.update(impulse_metrics)
 

    # --- Plotting and display logic ---
    base_metrics['block_trial_str'] = f"Block {selected_block}, Trial {selected_trial}"
    fig = create_trial_figure(
        trial_segment_df, channel_map, mvc_left, mvc_right, base_metrics,
        run_peak_force=run_peak_force, 
        run_motor_response_time=run_motor_response_time,
        run_motor_reaction_time=run_motor_reaction_time,
        run_force_time_integral=run_force_time_integral
    )
    
    metrics_layout = dbc.Card(dbc.CardBody([
        html.P(f"{key.replace('_', ' ').title()}: {value}")
        for key, value in base_metrics.items() if key != 'block_trial_str'
    ]))

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


# ==============================================================================
# --- MAIN APP EXECUTION ---
# ==============================================================================
def run_app_server():
    # Note: debug=False is important for packaged apps
    app.run(debug=False) 

def on_closing():
    # Clean up temporary files and directories when the app is closed
    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    if app_temp_dir.exists():
        shutil.rmtree(app_temp_dir, ignore_errors=True)

if __name__ == '__main__':
    # Run the Dash server in a separate thread
    server_thread = threading.Thread(target=run_app_server)
    server_thread.daemon = True
    server_thread.start()

    # Create and start the pywebview window
    window = webview.create_window(
        'CogMo Toolkit', 
        'http://127.0.0.1:8050/',
        width=1000,
        height=750
    )

    window.events.closing += on_closing

    webview.start()
