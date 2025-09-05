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
from trial_segmentation import get_trial_segment, create_trial_lookup

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
                 external_stylesheets=[dbc.themes.MINTY],
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
                                # A placeholder to align the button with the other one
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
                                                         className='"mt-1")',
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
        # Second Tab: Baseline Noise
        dbc.Tab(
            label="Baseline Noise",
            children=[
                html.Div([
                    html.H4("Content for Baseline Noise Tab"),
                ], className="p-3")
            ]
        ),
        # Third Tab: Analyses Option
        dbc.Tab(
            label="Analyses Option",
            children=[
                html.Div([
                    html.H4("Content for Analyses Option Tab"),
                ], className="p-3")
            ]
        ),
        # Fourth Tab: Trial Viewer
        dbc.Tab(
            label="Trial Viewer",
            children=[
                html.Div([
                    dbc.Row([
                        # --- Sidebar for Controls ---
                dbc.Col([
                    html.H4("Trial Controls"),
                    
                    # Trial Navigator (Hybrid Slider + Input)
                    dbc.Label("Select Global Trial:"),
                    dbc.InputGroup([
                        dbc.Input(id='trial-selector-input', type='number', value=1, min=1, step=1),
                        dcc.Slider(
                            id='trial-selector-slider',
                            min=1,
                            max=100,  # This will be updated dynamically later
                            step=1,
                            value=1,
                            className="flex-grow-1 mx-2"
                        ),
                    ]),
                    
                    # Parameter Controls
                    dbc.Label("Pre-Stimulus Window (s):", className="mt-3"),
                    dbc.Input(id='pre-stim-window-input', type='number', value=0.125, step=0.05),
    
                    dbc.Label("Post-Stimulus Window (s):", className="mt-3"),
                    dbc.Input(id='post-stim-window-input', type='number', value=1.25, step=0.05),
                    
                    html.Hr(),
                    
                    # Display area for calculated metrics
                    html.H4("Trial Metrics"),
                    html.Div(id='trial-metrics-display')
                    
                ], width=4),  # End of sidebar column
                
                # --- Main Area for the Graph ---
                dbc.Col([
                    dcc.Graph(id='trial-graph', style={'height': '80vh'})
                ], width=8)  # End of graph column
            ])
        ], className="p-3") # Add some padding around the content
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
       
    return ui_layout


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
    Output('mvc-left-store', 'data', allow_duplicate=True),
    Output('mvc-right-store', 'data', allow_duplicate=True),
    Output('rfd-left-store', 'data', allow_duplicate=True),
    Output('rfd-right-store', 'data', allow_duplicate=True),
    Input('input-mvc-left', 'value'),
    Input('input-mvc-right', 'value'),
    Input('input-rfd-left', 'value'),
    Input('input-rfd-right', 'value'),
    prevent_initial_callbacks=True
)
def update_reference_values(mvc_left, mvc_right, rfd_left, rfd_right):
    return mvc_left, mvc_right, rfd_left, rfd_right


# Callback for trial segmentation
# --------------------------------
@app.callback(
        Output('trial-graph', 'figure', allow_duplicate=True),
        Output('trial-metrics-display', 'children', allow_duplicate=True),
        Output('trial-selector-slider', 'max', allow_duplicate=True),
        Output('trial-selector-input', 'value', allow_duplicate=True),
        Input('trial-selector-slider', 'value'),
        State('condition-data-store', 'data'),     # Condition
        State('signal-data-store', 'data'),        # Session id
        State('channel-map-store', 'data'),        # Channel Map
        State('pre-stim-window-input', 'value'),   # Adjustable pre-stimulus window
        State('post-stim-window-input', 'value'),  # Adjustable post-stimulus window
        State('mvc-left-store', 'data'),           # Reference values
        State('mvc-right-store', 'data'),          # Reference values
        prevent_initial_callbacks=True
    )
def update_trial_viewer(
    selected_trial,
    condition_data_dict, session_id,
    channel_map,
    pre_window, post_window,
    mvc_left, mvc_right
):
    # --- FAILURE PATH ---
    # Guard clause: Ensure required inputs exist
    if not all([session_id, channel_map, condition_data_dict, mvc_left, mvc_right]):
        raise PreventUpdate
    
    # Recreate path to temp data folder & full path, then read pd.DataFrame from feather file
    app_temp_dir = Path(tempfile.gettempdir()) / "CogMo-App"
    app_temp_dir.mkdir(exist_ok = True)
    print(f"Session ID in trial viewer callback: {session_id}")
    filepath = app_temp_dir /f"{session_id}.feather"
    df = pd.read_feather(filepath)

    # Total number of trials
    trial_lookup = create_trial_lookup(df)
    total_trials = len(trial_lookup)

    # Get condition data as pd.DataFrame
    condition_data = pd.DataFrame(condition_data_dict)

    # Segment trial based on selected trial index
    trial_segment_df, trial_metrics = get_trial_segment(
        full_df = df,
        trial_lookup = trial_lookup,
        condition_data = condition_data,
        trial_index = selected_trial,
        channel_map = channel_map,
        mvc_left = mvc_left,
        mvc_right = mvc_right,
        pre_window = pre_window,
        post_window = post_window
    )

    #TODO: Comment out for deployment, this is for debugging/testing purposes
    print(f"Trial segment's head:\n {trial_segment_df.head()}")
    print(f"Trial metrics:\n {trial_metrics}")

    # Channel mapping
    time_col = channel_map.get('time')
    force_r_col = channel_map.get('force_right')
    force_l_col = channel_map.get('force_left')
    emg_r_col = channel_map.get('emg_right')
    emg_l_col = channel_map.get('emg_left')
    include_emg = bool(emg_r_col and emg_l_col)

    # Figures
    COLOR_R = 'blue'
    COLOR_L = 'goldenrod'
    max_mvc = max(mvc_left, mvc_right)
    threshold_value = trial_metrics.get('threshold')

    # --- Create Figure ---
    if include_emg:
        # Create a figure with 2 subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        # Add force traces to the first subplot
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_r_col], 
                                 name='Right Hand', legendgroup='right', line=dict(color=COLOR_R)), row=1, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_l_col], 
                                 name='Left Hand', legendgroup='left', line=dict(color=COLOR_L)), row=1, col=1)
        # Add EMG traces to the second subplot
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[emg_r_col], 
                                 name='EMG Right', legendgroup='right', showlegend=False, line=dict(color=COLOR_R)), row=2, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[emg_l_col], 
                                 name='EMG Left', legendgroup='left', showlegend=False, line=dict(color=COLOR_L)), row=2, col=1)
    else:
        # FIX: Also use make_subplots for the single-plot case
        fig = make_subplots(rows=1, cols=1)
        # Add force traces, specifying row=1, col=1
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_r_col], 
                                 name='Right Hand', line=dict(color=COLOR_R)), row=1, col=1)
        fig.add_trace(go.Scatter(x=trial_segment_df[time_col], y=trial_segment_df[force_l_col], 
                                 name='Left Hand', line=dict(color=COLOR_L)), row=1, col=1)

    # --- Apply Customizations to the Figure (this code now works for both cases) ---
    fig.update_yaxes(title_text="Force", range=[0, max_mvc], row=1, col=1)

    if include_emg:
        fig.update_yaxes(title_text="EMG", row=2, col=1)

    if threshold_value is not None:
        fig.add_hline(y=threshold_value, line_dash="dash", line_color="grey", row=1, col=1)

    fig.update_layout(title=f"Trial #{selected_trial}", margin=dict(t=30, b=0, l=0, r=0))

    metrics_layout = dbc.Card(dbc.CardBody([
        html.P(f"{key.replace('_', ' ').title()}: {value}")
        for key, value in trial_metrics.items()
    ]))

    return fig, metrics_layout, total_trials, selected_trial



# ==============================================================================
# --- MAIN APP EXECUTION ---
# ==============================================================================
def run_app_server():
    # Note: debug=False is important for packaged apps
    app.run(debug=False) 

if __name__ == '__main__':
    # Run the Dash server in a separate thread
    server_thread = threading.Thread(target=run_app_server)
    server_thread.daemon = True
    server_thread.start()

    # Create and start the pywebview window
    webview.create_window(
        'CogMo Toolkit', 
        'http://127.0.0.1:8050/',
        width=1000,
        height=750
    )
    webview.start()
