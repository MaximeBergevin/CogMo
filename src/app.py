# ==============================================================================
# CogMo Toolkit - Main Application
# This Dash application provides an interface for uploading and analyzing
# psychophysiological signal data and associated condition order files.
# ==============================================================================

# ==============================================================================
# --- IMPORTS AND APP INITIALIZATION ---
# ==============================================================================
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import io
import base64
from pathlib import Path
import os
import shutil
import threading
import webview
import pandas as pd
from get_condition_lookup import get_condition_lookup
from load_signal import load_signal
import time
import re
import logging

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
                                        'Drag and Drop or ',
                                        html.A('Select Signal Data File')
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
                                        'Drag and Drop or ',
                                        html.A('Select Condition Order File')
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
                    # --- Output Message ---
                    dcc.Loading(
                        id="loading-output-message",
                        type="dot",
                        children=html.Div(id='upload-output-message', className="mt-3"),
                    )
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
app.layout.children.append(dcc.Store(id = 'condition-data-store')),
app.layout.children.append(dcc.Store(id = 'signal-data-store')),
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
    Output('channel-mapping-container', 'children', allow_duplicate=True),
    Input('upload-signal-data', 'contents'),
    State('upload-signal-data', 'filename'),
    prevent_initial_callbacks = True
)
def upload_signal_data_callback(signal_contents, signal_filename):
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
        
        # For now, we return a simple placeholder for the channel mapping UI,
        # as we will build this out in a separate step.
        channel_mapping_ui = html.Div([
            html.H4("Channel Mapping"),
            html.P("This section will contain channel selection dropdowns.")
        ])

    except Exception as e:
        block_comments_data = None
        stimulus_comments_data = None
        channel_mapping_ui = html.Div(f'There was an error processing this file: {e}', className="text-danger")
        df = None
    finally:
        # Clean up the temporary file and directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    return df.to_dict('records'), block_comments, stimulus_comments, channel_mapping_ui


# 2. Callback for Condition Order File Upload
# -------------------------------------------
@app.callback(
        Output('condition-data-store', 'data', allow_duplicate = True),
        Input('upload-condition-order', 'contents'),
        State('upload-condition-order', 'filename'),
        prevent_initial_callbacks = True
    
)
def upload_condition_callback(condition_contents, condition_filename):
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
            return None
        
        #TODO: Comment out for deployment, this is for debugging/testing purposes
        #logging.info(f"Head of condition order data frame:\n{df.head()}")

        return df.to_dict('records')
    
    except Exception as e:
        return None

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
