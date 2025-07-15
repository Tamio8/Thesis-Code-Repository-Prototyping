'''
Prototype 1: Dynamic Summary

Purpose:
This prototype explores how patient data summaries could be made more dynamic and user-specific in the ICU context.
It allows clinicians to select their role (e.g., nursing, medical) and view a tailored summary of key patient information.
The goal is to support fast, relevant overviews and note-taking during ward rounds or handovers.

Key Features:
    Displays structured clinical data in an organised layout (vital signs, lab results, fluid balance, etc.).
    Offers clickable source links for traceability to original records.
    Provides an editable checklist and note-taking area for nurses.
    Allows adding custom nursing steps and free-text notes.
    Includes a basic "search chat" field to mock search-like interaction.

Technology Stack:
    Dash (by Plotly) for the interactive front-end.
    Dash Bootstrap Components for layout and styling.
    Mock data is used for prototyping; no real patient data is loaded.

How to Run:
    Install required packages: dash, dash-bootstrap-components.
    Run the script in a Python environment.
    The app will launch automatically in the browser at http://127.0.0.1:8055.

Notes:
    This is a prototype - backend functionality (e.g., real-time data updates, persistent storage, authentication) is not implemented.
    Intended for exploration and feedback from clinicians regarding layout, content relevance, and interactivity.

'''


import dash
from dash import dcc, html, Input, Output
import threading
import webbrowser
import dash_bootstrap_components as dbc

# Mock data for the prototype
data = {
    "Patient ID": "12345",
    "Patient Name": "John Doe",
    "Weight": "75 kg",
    "Age": 65,
    "Diagnosis": "Pneumonia",
    "Allergies": "Shellfish",
    "Vital Signs": "BP: 120/80, HR: 80 bpm, SpO2: 95%",
    "Medications": "Antibiotics, Pain Relief, Vasopressors",
    "Lab Results": "WBC: Elevated, CRP: High, Creatinine: 1.2",
    "Fluid Balance": "+500 mL (last 24h)",
    "Ventilation": "FiO2: 40%, PEEP: 5, RR: 18",
    "Hemodynamics": "MAP: 75, Lactate: 1.2",
    "Recent Changes": "Increased oxygen requirement, Adjusted antibiotics",
    "Skin Integrity": "No pressure injuries",
    "Lines & Devices": "Central line, Foley catheter",
    "Treatment Plan": "Continue antibiotics, Monitor fluid balance, Adjust oxygen therapy",
    "Nursing Steps": {
        "Past": "Administered antibiotics, Monitored vital signs",
        "Next": "Monitor oxygen levels, Reassess fluid balance"
    },
    "Source Links": {
        "Vital Signs": "https://example.com/vitals",
        "Lab Results": "https://example.com/labs",
        "Ventilation": "https://example.com/ventilation",
        "Fluid Balance": "https://example.com/fluid-balance",
        "Treatment Plan": "https://example.com/treatment-plan"
    }
}

# Initialize the Dash app
app_note = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app_note.layout = html.Div([
    # Patient Info Line
    html.Div([
        html.P(f"Patient Name: {data['Patient Name']} | Patient ID: {data['Patient ID']} | Weight: {data['Weight']} | Allergies: {data['Allergies']}")
    ], style={'textAlign': 'center', 'borderBottom': '2px solid #ddd', 'paddingBottom': '10px', 'marginBottom': '10px'}),
    
    html.H1("Dynamic Note", className="text-center"),
    
    html.Label("Select Profession:"),
    dcc.Dropdown(
        id='profession-selector',
        options=[
            {'label': "General", 'value': "General"},
            {'label': "Nursing", 'value': "Nursing"},
            {'label': "Medical", 'value': "Medical"}
        ],
        value="General"
    ),
    
    html.Div(id='summary-output')
])

@app_note.callback(
    Output('summary-output', 'children'),
    Input('profession-selector', 'value')
)
def update_summary(profession):
    summary = []
    
    # Left-hand side: Patient Summary
    summary_content = html.Div([
        html.P([html.Strong("Recent Changes: "), html.Span(data['Recent Changes'], style={'color': 'red'})]),
        html.P(["Vital Signs: ", html.A(data['Vital Signs'], href=data['Source Links']['Vital Signs'], target="_blank")]),
        html.P(["Lab Results: ", html.A(data['Lab Results'], href=data['Source Links']['Lab Results'], target="_blank")]),
        html.P(f"Hemodynamics: {data['Hemodynamics']}"),
        html.P(f"Medications: {data['Medications']}"),
        html.P(f"Fluid Balance: {data['Fluid Balance']}"),
        html.P(["Ventilation: ", html.A(data['Ventilation'], href=data['Source Links']['Ventilation'], target="_blank")]),
        html.P(f"Skin Integrity: {data['Skin Integrity']}"),
        html.P(f"Lines & Devices: {data['Lines & Devices']}"),
        html.P([html.Strong("Treatment Plan: "), html.A(data['Treatment Plan'], href=data['Source Links']['Treatment Plan'], target="_blank")])
    ], style={'width': '65%', 'float': 'left', 'padding': '10px'})
    
    # Right-hand side: Nursing steps as checklist
    nursing_steps = html.Div([
        html.H4("Nursing Steps", style={'textAlign': 'center'}),
        dcc.Checklist(
            id='nursing-steps-checklist',
            options=[
                {'label': 'Administer antibiotics', 'value': 'administer_antibiotics'},
                {'label': 'Monitor vital signs', 'value': 'monitor_vital_signs'},
                {'label': 'Monitor oxygen levels', 'value': 'monitor_oxygen_levels'},
                {'label': 'Reassess fluid balance', 'value': 'reassess_fluid_balance'}
            ],
            value=['administer_antibiotics', 'monitor_vital_signs'],  # Example of checked items
            labelStyle={'display': 'block'}
        ),
        
        # Add Custom Nursing Step
        html.Div([
            html.Label("Add Custom Nursing Step:"),
            dcc.Input(id='custom-nursing-step', type='text', placeholder='Enter custom step'),
            html.Button('Add Step', id='add-nursing-step', n_clicks=0)
        ], style={'marginTop': '10px'}),
        
        html.Div(id='added-nursing-steps', style={'marginTop': '20px'}),
        
        # Notes section under nursing steps
        html.Label("Nursing Notes:"),
        dcc.Textarea(
            id='nursing-notes',
            value='Enter your notes here...',
            style={'width': '100%', 'height': '100px'}
        ),
    ], style={'width': '30%', 'float': 'right', 'padding': '10px', 'border': '1px solid #ddd'})

    # Notes section as chat-like field
    notes_section = html.Div([
        html.H4("Search Chat", style={'textAlign': 'center'}),
        dcc.Textarea(
            id='notes-chat',
            value='What are you looking for?',
            style={'width': '100%', 'height': '100px', 'resize': 'vertical'}
        ),
    ], style={'clear': 'both', 'padding': '10px', 'marginTop': '20px', 'border': '1px solid #ddd'})
    
    return [summary_content, nursing_steps, notes_section]

# Callback to add custom nursing steps
@app_note.callback(
    Output('added-nursing-steps', 'children'),
    Input('add-nursing-step', 'n_clicks'),
    Input('custom-nursing-step', 'value'),
    prevent_initial_call=True
)
def add_nursing_step(n_clicks, custom_step):
    if n_clicks > 0 and custom_step:
        return html.Div([html.P(f"- {custom_step}")])
    return ""

def run_app(app, port): 
    threading.Thread(target=app.run_server, kwargs={'port': port, 'debug': False, 'use_reloader': False}).start()
    webbrowser.open_new(f'http://127.0.0.1:{port}')

run_app(app_note, 8055)





