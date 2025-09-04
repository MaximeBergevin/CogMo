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
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
import webview
# Local Application Imports
from get_condition_lookup import get_condition_lookup
from data_loader import load_signal, find_best_match

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
# --- HELPER FUNCTIONS ---
# ==============================================================================
def find_best_match(channel_names: list, patterns: list) -> str | None:
    """
    Finds the first channel name that matches any of the given patterns.
    """
    for channel in channel_names:
        for pattern in patterns:
            if re.match(pattern, channel):
                return channel
    return None

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
                                                      className="mt-3",
                                                      style={'min-height': '50px'}),
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
                                                         className='"mt-2")',
                                                         style={'min-height': '50px'})
                                )
                            ]),
                            width=6,
                        ),
                    ]),
                    
                    # --- Channel Mapping Section ---
                    html.Div(id='channel-mapping-container', className="mt-4"),
                    
                    # --- Baseline Reference values ---
                    html.H4("Baseline Reference values", className="mt-4"),
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
                    html.H4("Content for Trial Viewer Tab"),
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

# 1. Callback for Signal Data Upload
# ----------------------------------
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
        
    #finally:
        # Clean up the temporary file and directory
        #shutil.rmtree(temp_dir, ignore_errors=True)


# 2. Callback for Channel Selection
# ---------------------------------
@app.callback(
        Output('channel-mapping-container', 'children'),
        Input('signal-data-store', 'data')
)
def update_channel_mapping_ui(session_id):
    # --- FAILURE PATH ----
    # guard clause in case session_id does not exist
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

    channel_names = df.columns.tolist()
    detected_channels = {
        'time'         : find_best_match(channel_names, time_patterns),
        'force_right'  : find_best_match(channel_names, force_right_pattern),
        'force_left'   : find_best_match(channel_names, force_left_pattern),
        'emg_right'    : find_best_match(channel_names, emg_right_pattern),
        'emg_left'     :find_best_match(channel_names, emg_left_pattern)
    }
    #TODO: Comment out for deployment, this is for debugging/testing purposes
    print(f"Channel names: {channel_names}")

    # Format channel names for dropwon's options' property
    dropdown_options = [{'labels': name, 'value': name} for name in channel_names]
    # Build layout components to return
    ui_layout = html.Div({
        html.H4("Channel Mapping"),
        html.P("Review the detected channels or make a manual selection."),

        dbc.Row([
            # Column for Time Channel
            dbc.Col([
                dbc.Label("Time Channel:"),
                dcc.Dropdown(
                    id='time-channel-dropdown',
                    options=dropdown_options,
                    value=detected_channels['time'], # Set the default value
                    clearable=False # A time channel is mandatory
                )
            ], width=6),
            # Column for Force Right
            dbc.Col([
                dbc.Label("Force Right Channel:"),
                dcc.Dropdown(
                    id='force-right-channel-dropdown',
                    options=dropdown_options,
                    value=detected_channels['force_right'],
                    clearable=False # Force channels are mandatory
                )
            ], width=6),
            dbc.Col([
                dbc.Label("Force Left Channel:"),
                dcc.Dropdown(
                    id='force-left-channel-dropdown',
                    options=dropdown_options,
                    value=detected_channels['force_left'],
                    clearable=False # Force channels are mandatory
                )
            ], width=6),
            dbc.Col([
                dbc.Label("EMG Right Channel:"),
                dcc.Dropdown(
                    id='force-left-channel-dropdown',
                    options=dropdown_options,
                    value=detected_channels['emg_right'],
                    clearable=False # Force channels are mandatory
                )
            ], width=6),
            dbc.Col([
                dbc.Label("EMG Left Channel:"),
                dcc.Dropdown(
                    id='force-left-channel-dropdown',
                    options=dropdown_options,
                    value=detected_channels['emg_left'],
                    clearable=False # Force channels are mandatory
                )
            ], width=6)
        ]),
    })
    return ui_layout
    

# 3. Callback for Condition Order File Upload
# -------------------------------------------
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

# 3. Callback for Baseline Reference Value Updates
# ------------------------------------------------
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
