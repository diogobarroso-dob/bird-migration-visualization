import sys
import os
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output,State,ctx # Required to detect which button was clicked
import dash_bootstrap_components as dbc
import plotly.express as px
sys.path.append(os.path.abspath(".."))

df = pd.read_csv('data/bird_migration.csv')

# --- 4. MAP VISUALIZATION ENGINE ---
def create_map(df):
    if df.empty:
        fig = px.scatter_geo()
        fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
        fig.add_annotation(text="No data selected", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Transform data for Line + Marker plotting
    plot_data = []
    for index, row in df.iterrows():
        segment_id = f"{row['bird_name']}_{index}"
        
        # # Get the Reason, Remove Coordinate Strings ---
        # # Ensure your column name matches 'Migration_Reason'
        # reason = row['Migration_Reason'] 
        
        # Start Point
        plot_data.append({
            "Bird_name": row['bird_name'],
            "Latitude": row['latitude'],
            "Longitude": row['longitude'],
            "Segment_ID": segment_id
        })
    
    df_plot = pd.DataFrame(plot_data)


    # Plot
    fig = px.line_geo(
        df_plot,
        lat="Latitude", lon="Longitude", color="Bird_name",
        line_group="Bird_name", 
        hover_name="Bird_name", 
        
        #  Update Hover Data
        hover_data={
            # "bird_name": True, 
            # "Migration Reason": True, 
            # "Position": True, 
            "Bird_name": True, 
            "Latitude": False, 
            "Longitude": False
        },
        
        projection="orthographic", 
        title=f"Tracking {df['bird_name'].nunique()} Unique Birds",
        color_discrete_sequence=px.colors.qualitative.Bold,
        fitbounds="locations"
    )

    # Styling: Lines + Markers
    fig.update_traces(
        mode='lines+markers', 
        line=dict(width=2), 
        marker=dict(size=6, symbol='circle', opacity=1, line=dict(width=0)),
        opacity=0.8
    )
    
    # Map Geos styling
    fig.update_geos(
        visible=True, resolution=50,
        showcountries=True, countrycolor="#bbbbbb",
        showcoastlines=True, coastlinecolor="#bbbbbb",
        showland=True, landcolor="#f0f0f0",      
        showocean=True, oceancolor="#e4edff",   
        projection_rotation=dict(lon=-10, lat=20)
    )
    
    fig.update_layout(
        template="plotly_white",
        margin={"r":0,"t":50,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)", 
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(255,255,255,0.9)")
    )
    return fig

# create_map(df).show()

# filrer for all charts for altitude outliers
df = df[(df["altitude"] >= -100) & (df["altitude"] <= 1000)]
# Aggregate Data
df_bird_stats = df.groupby('bird_name').agg(
    Max_Altitude=('altitude', 'max'),
    Avg_Altitude=('altitude', 'mean'),
    Min_Altitude=('altitude', 'min'),
    Max_Speed=('speed_2d', 'max'),
    Avg_Speed=('speed_2d', 'mean'),
    Min_Speed=('speed_2d', 'min')
).reset_index()

# Bar chart
def build_bar_chart(df_bird_stats, selected_bird, category, selected_stats):

    # Empty State
    if not selected_bird:
        fig = px.bar()
        fig.update_layout(
            title={
                'text': "Select bird from the configuration panel to begin analysis",
                'y': 0.5, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'middle'
            },
            xaxis={'visible': False}, yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18, color="#6c757d")
        )
        return fig

    #  If one bird selected → violin plot 
    if len(selected_bird) == 1:
        bird = selected_bird[0]
        df_single = df[df["bird_name"] == bird]

        if category == "Altitude":
            fig = px.violin(
                df_single,
                y="altitude",
                box=True,
                points=False,
                title=f"Altitude Distribution for {bird}",
                template="plotly_white"
            )
            fig.update_layout(yaxis_title="Altitude (meters)")
        else:
            fig = px.violin(
                df_single,
                y="speed_2d",
                box=True,
                points="all",
                title=f"Speed Distribution for {bird}",
                template="plotly_white"
            )
            fig.update_layout(yaxis_title="Speed (km/h)")

        return fig
    # -------------------------------------------------------------

    # Setup Columns and Labels based on Category
    cols_to_plot = []

    if category == "Altitude":
        y_label = "Altitude (meters)"
        title_prefix = "Altitude Comparison"

        color_map = {
            'Max_Altitude': '#0d6efd',
            'Avg_Altitude': '#aecdf7',
            'Min_Altitude': '#fd7e14'
        }

        for stat in selected_stats:
            cols_to_plot.append(f"{stat}_{category}")

    else:  # Speed
        y_label = "Speed"
        title_prefix = "Speed Comparison"

        color_map = {
            'Max_Speed': '#198754',
            'Avg_Speed': '#a3cfbb',
            'Min_Speed': '#fd7e14'
        }

        for stat in selected_stats:
            cols_to_plot.append(f"{stat}_{category}")

    # Filter Data
    df_filtered = df_bird_stats[df_bird_stats['bird_name'].isin(selected_bird)]

    # Melt Data
    valid_cols = [c for c in cols_to_plot if c in df_filtered.columns]

    df_melted = df_filtered.melt(
        id_vars='bird_name',
        value_vars=valid_cols,
        var_name='Metric',
        value_name='Value'
    )

    if df_melted.empty:
        return px.bar(title="No data available for the selected options.")

    # Build Figure (original bar chart)
    fig = px.bar(
        df_melted,
        x='bird_name',
        y='Value',
        color='Metric',
        barmode='group',
        title=title_prefix,
        template='plotly_white',
        text_auto=',.0f',
        color_discrete_map=color_map
    )

    fig.update_layout(
        legend_title_text="Metric",
        xaxis_title=None,
        margin=dict(t=60, b=40, l=40, r=180),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        yaxis=dict(title=y_label, tickformat=","),
        transition={'duration': 500}
    )

    return fig

# ---- Line chart for Altitude over Time ----
def build_line_chart_altitude(df, selected_bird):
    # Empty State
    if not selected_bird:
        fig = px.line()
        fig.update_layout(
            title={
                'text': "Select bird from the configuration panel to begin analysis",
                'y': 0.5, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'middle'
            },
            xaxis={'visible': False}, yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18, color="#6c757d")
        )
        return fig

    # Filter Data
    df_filtered = df[df['bird_name'].isin(selected_bird)]


    # Build Figure
    fig = px.line(
        df_filtered,
        x='date_time',
        y='altitude',
        color='bird_name',
        title=f"Altitude Over Time for {selected_bird}",
        labels={'date_time': 'Time', 'altitude': 'Altitude (meters)'},
        template='plotly_white'
    )
    
    fig.update_layout(
        margin=dict(t=60, b=40, l=40, r=40),
        yaxis=dict(title='Altitude (meters)', tickformat=","),
        transition={'duration': 500}
    )

    return fig

# ---- Line chart for Speed over Time ----
def build_line_chart_speed(df, selected_bird):
    # Empty State
    if not selected_bird:
        fig = px.line()
        fig.update_layout(
            title={
                'text': "Select bird from the configuration panel to begin analysis",
                'y': 0.5, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'middle'
            },
            xaxis={'visible': False}, yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18, color="#6c757d")
        )
        return fig

    # Filter Data
    df_filtered = df[df['bird_name'].isin(selected_bird)]


    # Build Figure
    fig = px.line(
        df_filtered,
        x='date_time',
        y='speed_2d',
        color='bird_name',
        title=f"Speed Over Time for {selected_bird}",
        labels={'date_time': 'Time', 'speed_2d': 'km/h'},
        template='plotly_white'
    )
    
    fig.update_layout(
        margin=dict(t=60, b=40, l=40, r=40),
        yaxis=dict(title='Speed (km/h)', tickformat=","),
        transition={'duration': 500}
    )

    return fig

def build_animated_map(df, selected_bird):
    # ... (Empty state and initial filtering/cleaning are the same)
    
    # Normalize input to list
    if isinstance(selected_bird, str):
        selected_bird = [selected_bird]

    # Filter data (using .copy() to prevent SettingWithCopyWarning if modifying later)
    df_filtered = df[df["bird_name"].isin(selected_bird)].head(50).copy() 

    # Ensure datetime and drop NaNs (same as original code)
    if df_filtered["date_time"].dtype == "object":
        df_filtered["date_time"] = pd.to_datetime(df_filtered["date_time"], utc=True, errors="coerce")
    df_filtered = df_filtered.dropna(subset=["date_time", "latitude", "longitude"])
    df_filtered = df_filtered.sort_values(["bird_name", "date_time"])
    df_filtered["frame"] = df_filtered["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    
    # --- 1. Create the CUMULATIVE PATH Data (The Fix!) ---
    # This DataFrame will contain the full path for every frame.
    
    # Get all unique frame strings
    unique_frames = df_filtered["frame"].unique()
    
    # Initialize the path DataFrame
    df_path = pd.DataFrame()

    # For each unique frame, include all data points up to that frame
    for frame_name in unique_frames:
        # Filter for all points up to and including the current frame
        temp_df = df_filtered[df_filtered["frame"] <= frame_name].copy()
        
        # Assign the current frame name to ALL rows in this temporary subset
        # This is the key: for "Frame N", the data includes points 1 to N.
        temp_df["current_frame"] = frame_name
        
        # Append to the final path DataFrame
        df_path = pd.concat([df_path, temp_df])

    # --- 2. Line trace (growing path) ---
    # Use the new df_path and map animation_frame to the column we created.
    fig = px.line_geo(
        df_path,
        lat="latitude",
        lon="longitude",
        color="bird_name",
        line_group="bird_name",
        animation_frame="current_frame", # Use the cumulative frame column
        animation_group="bird_name",
        hover_name="bird_name",
        title=f"Animated Movement for {', '.join(selected_bird)}",
        height=650,
        fitbounds="locations"
    )

    # --- 3. Points trace (markers showing current position & speed) ---
    # This trace uses the original df_filtered (per-frame data)
    fig_points = px.scatter_geo(
        df_filtered,
        lat="latitude",
        lon="longitude",
        color="bird_name",
        size="speed_2d",
        animation_frame="frame", # Use the original frame column
        animation_group="bird_name",
        hover_name="bird_name",
        size_max=12,
        fitbounds="locations"
    )

    # ... (Merging traces and layout cleanup are the same)

    # Merge the traces/frame data: append scatter points into the line fig
    # Note: Because 'current_frame' in df_path matches 'frame' in df_filtered, 
    # Plotly's internal animation system aligns them correctly.
    for trace in fig_points.data:
        fig.add_trace(trace)

    # Tidy layout: (rest of the layout code remains the same)
    fig.update_layout(
        geo=dict(
            showcountries=True,
            showland=True,
            landcolor="rgba(240,240,240,1)",
            projection_type="natural earth",
        ),
        margin={"r":0,"t":50,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05)
    )


    return fig

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Global Bird Migration Tracker", className="display-6"), width=12),
        dbc.Col(html.P("Compare bird_name between specific bird IDs.", className="text-muted"), width=12),
    ], className="my-4"),
    
    dbc.Row([
        # ------ SIDEBAR -------
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filter Controls", className="fw-bold"),
                dbc.CardBody([
                    #bird_name SECTION 
                    html.Label("Select Bird", className="mb-2 fw-bold text-primary"),
                    dcc.Dropdown(
                        id='bird-name-filter',
                        options=[{'label': s, 'value': s} for s in sorted(df['bird_name'].unique())],
                        value=sorted(df['bird_name'].unique()), 
                        multi=True, 
                        clearable=True
                    ),
                    #Button to select all bird_name
                    dbc.Button("Select All Birds", id="btn-all-birds", color="light", size="sm", className="mt-1 w-100 border"),

                    html.Hr(),
                    
                    # DATA CATEGORY
                    html.Label("Choose Data Category", className="fw-bold text-primary"),
                    dbc.RadioItems(
                        id='category-selector',
                        options=[
                            {'label': ' Altitude', 'value': 'Altitude'},
                            {'label': ' Speed', 'value': 'Speed'}
                        ],
                        value='Altitude', 
                        className="mb-3",
                        inputClassName="me-2"
                    ),

                    html.Hr(),

                    # STATISTIC CHECKLIST
                    html.Label("Select Statistics", className="fw-bold text-primary"),
                    dbc.Checklist(
                        id='statistic-checklist',
                        options=[
                            {'label': ' Maximum', 'value': 'Max'},
                            {'label': ' Average', 'value': 'Avg'},
                            {'label': ' Minimum', 'value': 'Min'},
                            # Real World is added dynamically, but we can default it here too
                        ],
                        value=['Max', 'Avg', 'Min'], 
                        switch=True, 
                        className="mb-3"
                    ),
                ])
            ], className="mb-4 shadow-sm")
        ], width=12, md=3), 
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='migration-map', style={'height': '75vh'}) 
                ], style={'padding': '0'})
            ], className="shadow-sm")
        ], width=12, md=9),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='main-bar-chart', style={'height': '75vh'}) 
                ], style={'padding': '0'})
            ], className="shadow-sm border-0")
        ], width=12, md=9),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='line-chart-altitude', style={'height': '75vh'}) 
                ], style={'padding': '0'})
            ], className="shadow-sm border-0")
        ], width=12, md=9),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='line-chart-speed', style={'height': '75vh'}) 
                ], style={'padding': '0'})
            ], className="shadow-sm border-0")
        ], width=12, md=9),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='animated-map', style={'height': '75vh'}) 
                ], style={'padding': '0'})
            ], className="shadow-sm")
        ], width=12, md=9)
    ]),
], fluid=True)

@app.callback(
    Output('migration-map', 'figure'),
    Input('bird-name-filter', 'value')
)
def update_map(selected_bird_names):

    # When nothing is selected → return empty map
    if not selected_bird_names:
        return create_map(pd.DataFrame())

    # Filter by bird_name (since that's the dropdown value)
    filtered = df[df['bird_name'].isin(selected_bird_names)]

    return create_map(filtered)

# Callback: Update Options (Real World is now available for BOTH)
@app.callback(
    [Output('statistic-checklist', 'options'),
     Output('statistic-checklist', 'value')],
    Input('category-selector', 'value'),
    State('statistic-checklist', 'value')
)
def update_stat_options(category, current_values):
    options = [
        {'label': ' Maximum', 'value': 'Max'},
        {'label': ' Average', 'value': 'Avg'},
        {'label': ' Minimun', 'value': 'Min'}
    ]

    return options, current_values


@app.callback(
    Output('main-bar-chart', 'figure'),
    [
        Input('bird-name-filter', 'value'),
        Input('category-selector', 'value'),
        Input('statistic-checklist', 'value')
    ]
)
def update_chart(selected_bird, category, selected_stats):
    return build_bar_chart(
        df_bird_stats=df_bird_stats,
        selected_bird=selected_bird,
        category=category,
        selected_stats=selected_stats
    )

@app.callback(
    Output('line-chart-altitude', 'figure'),
    Input('bird-name-filter', 'value')
)
def update_line_chart_altitude(selected_bird):

    selected_bird = selected_bird
    return build_line_chart_altitude(df=df, selected_bird=selected_bird)

@app.callback(
    Output('line-chart-speed', 'figure'),
    Input('bird-name-filter', 'value')
)
def update_line_chart_speed(selected_bird):

    selected_bird = selected_bird
    return build_line_chart_speed(df=df, selected_bird=selected_bird)



@app.callback(
    Output('animated-map', 'figure'),
    Input('bird-name-filter', 'value')
)
def update_animated_map(selected_bird):

    selected_bird = selected_bird
    return build_animated_map(df=df, selected_bird=selected_bird)

@app.callback(
    Output('bird-name-filter', 'value'),
    Input('btn-all-birds', 'n_clicks'),
    State('bird-name-filter', 'options'),
    prevent_initial_call=True
)
def select_all_species(n_clicks, options):
    return [opt['value'] for opt in options]

if __name__ == '__main__':
    app.run(debug=True, port=8060)