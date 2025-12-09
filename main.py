import sys
import os
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

BIRD_COLORS = ['#1F4068', '#DE8918', '#BF3200']

# Ensure project path is correct (from your original code)
sys.path.append(os.path.abspath(".."))

# ======================================================
# 1. DATA LOADING & PRE-PROCESSING
# ======================================================

# Load Data
df = pd.read_csv('data/bird_migration.csv')

# --- PRE-SORT DATA ---
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values(['bird_name', 'date_time'])

# --- ROBUST STOP DETECTION HELPER (From Branch) ---
def count_significant_stops(sub_df, speed_threshold=0.5, min_duration_hours=1):
    # Detect if bird is resting
    is_resting = sub_df['speed_2d'] < speed_threshold
    # Create blocks of continuous rest
    block_ids = (is_resting != is_resting.shift()).cumsum()
    # Calculate duration of each block
    block_durations = sub_df.groupby(block_ids)['date_time'].agg(lambda x: x.max() - x.min())
    # Verify if block is actually a rest period
    block_is_rest = sub_df.groupby(block_ids)['speed_2d'].mean() < speed_threshold
    min_duration = pd.Timedelta(hours=min_duration_hours)
    # Count valid stops
    return ((block_is_rest) & (block_durations > min_duration)).sum()

# ======================================================
# 2. DATA AGGREGATION (MERGED LOGIC)
# ======================================================

# A. Altitude Stats: Use ALL data (absolute max/min)
alt_stats = df.groupby('bird_name').agg(
    Max_Altitude=('altitude', 'max'),
    Avg_Altitude=('altitude', 'mean'),
    Min_Altitude=('altitude', 'min')
)

# B. Speed Stats: Use FILTERED data (Flying only)
# This prevents sitting birds (0 speed) from dragging down the average.
df_flying = df[df['speed_2d'] >= 0.5] 

speed_stats = df_flying.groupby('bird_name').agg(
    Max_Speed=('speed_2d', 'max'),
    Avg_Speed=('speed_2d', 'mean'), 
    Min_Speed=('speed_2d', 'min')
)

# C. Stopover Counts: Apply complex stop detection
stop_counts = df.groupby('bird_name').apply(
    count_significant_stops, 
    include_groups=False
)

# D. Merge All Stats into one DataFrame
df_bird_stats = alt_stats.join(speed_stats).join(stop_counts.rename('Total_Rest')).reset_index()

# ======================================================
# 3. VISUALIZATION ENGINES
# ======================================================

# --- MAP ENGINE (Unchanged from Main) ---
def create_map(df):
    if df.empty:
        fig = px.scatter_geo()
        fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
        fig.add_annotation(text="No data selected", x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_data = []
    for index, row in df.iterrows():
        segment_id = f"{row['bird_name']}_{index}"
        plot_data.append({
            "Bird_name": row['bird_name'],
            "Latitude": row['latitude'],
            "Longitude": row['longitude'],
            "Segment_ID": segment_id
        })
    
    df_plot = pd.DataFrame(plot_data)

    fig = px.line_geo(
        df_plot,
        lat="Latitude", lon="Longitude", color="Bird_name",
        line_group="Bird_name", hover_name="Bird_name", 
        hover_data={"Bird_name": True, "Latitude": False, "Longitude": False},
        projection="orthographic", 
        title=f"Tracking {df['bird_name'].nunique()} Unique Birds",
        color_discrete_sequence=BIRD_COLORS,
        fitbounds="locations"
    )

    fig.update_traces(
        mode='lines+markers', 
        line=dict(width=2), 
        marker=dict(size=6, symbol='circle', opacity=1, line=dict(width=0)),
        opacity=0.8
    )
    
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

# --- BAR CHART ENGINE (Updated from Branch + Width Fix) ---
def build_bar_chart(df_bird_stats, selected_bird, category, selected_stats):

    # Empty State
    if not selected_bird:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "Select bird from the configuration panel to begin analysis", 'x': 0.5},
            xaxis={'visible': False}, yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18, color="#6c757d")
        )
        return fig

    cols_to_plot = []

    # --- ALTITUDE LOGIC ---
    if category == "Altitude":
        y_label = "Altitude (meters)"
        title_prefix = "Altitude Comparison"
        for stat in selected_stats:
            cols_to_plot.append(f"{stat}_{category}")

    # --- SPEED LOGIC ---
    elif category == "Speed":
        y_label = "Speed (km/h)"
        title_prefix = "Speed Comparison"
        for stat in selected_stats:
            cols_to_plot.append(f"{stat}_{category}")

    # --- REST LOGIC ---
    elif category == "Rest":
        y_label = "Number of Recorded Stops"
        title_prefix = "Rest Stop Frequency"
        for stat in selected_stats:
            cols_to_plot.append("Total_Rest")

    # --------------------------------

    # Filter Data
    df_filtered = df_bird_stats[df_bird_stats['bird_name'].isin(selected_bird)]
    valid_cols = [c for c in cols_to_plot if c in df_filtered.columns]
    
    if not valid_cols:
         fig = go.Figure()
         fig.update_layout(title="No data available")
         return fig

    # Melt Data
    df_melted = df_filtered.melt(
        id_vars='bird_name',
        value_vars=valid_cols,
        var_name='Metric',
        value_name='Value'
    )

    if df_melted.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    # --- CUSTOM COLOR LOGIC ---
    unique_birds = sorted(df_melted['bird_name'].unique())
    palette = px.colors.qualitative.Bold 
    
    bird_base_colors = {
        bird: palette[i % len(palette)] 
        for i, bird in enumerate(unique_birds)
    }
    
    # Robust Helper to convert Color to RGBA
    def convert_to_rgba(color_code, opacity):
        if color_code.startswith('#'):
            rgb = pc.hex_to_rgb(color_code)
        elif color_code.startswith('rgb'):
            nums = color_code.replace('rgb(', '').replace(')', '').split(',')
            rgb = [int(n) for n in nums]
        else:
            return color_code
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"
    
    # Create color column based on bird and metric
    def get_color_for_row(row):
        bird = row['bird_name']
        metric = row['Metric']
        base_color = bird_base_colors[bird]
        
        # Determine Opacity based on metric
        if 'Max' in metric:
            opacity = 1.0   
        elif 'Avg' in metric:
            opacity = 0.6   
        elif 'Min' in metric:
            opacity = 0.35  
        else:
            opacity = 1.0   
            
        return convert_to_rgba(base_color, opacity)
    
    df_melted['Color'] = df_melted.apply(get_color_for_row, axis=1)

    # Build Figure using graph_objects
    fig = go.Figure()
    
    for bird in unique_birds:
        bird_data = df_melted[df_melted['bird_name'] == bird]
        
        fig.add_trace(go.Bar(
            x=bird_data['Metric'],
            y=bird_data['Value'],
            name=bird,
            marker_color=bird_data['Color'].tolist(),
            text=bird_data['Value'].round(0).astype(int),
            texttemplate='%{text:,}',
            textposition='outside'
        ))

    fig.update_layout(
        title=title_prefix,
        template='plotly_white',
        legend_title_text="Bird",
        margin=dict(t=60, b=40, l=40, r=180),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        yaxis=dict(title=y_label, tickformat=","),
        xaxis=dict(title="Metric"),
        barmode='group',
        transition={'duration': 500},
        
        # --- FIXED SPACING FOR BARS ---
        bargap=0.5,       # 50% gap between groups (makes bars thinner)
        bargroupgap=0.1   # Small gap between bars of the same bird
    )

    return fig


# --- LINE CHART ENGINES (Unchanged from Main) ---
def build_line_chart_altitude(df, selected_bird):
    if not selected_bird:
        fig = px.line()
        fig.update_layout(title="Select bird to begin", template='plotly_white')
        return fig
    
    if isinstance(selected_bird, str): selected_bird = [selected_bird]
    
    df_filtered = df[df["bird_name"].isin(selected_bird)].copy()
    df_filtered = df_filtered.dropna(subset=["date_time", "latitude", "longitude"])
    df_filtered = df_filtered.sort_values(["bird_name", "date_time"])
    
    df_filtered["frame"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d %H"))
    df_filtered["frame_daily"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))

    df_daily = (df_filtered.groupby(["bird_name", "frame_daily"]).first().reset_index())
    df_daily["avg_altitude"] = (
    df_daily
    .groupby("bird_name")["altitude"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)   # removes group index so it fits back
    )
    
    # Line Plot
    line_plot = px.line(
        df_daily, x='date_time', y='avg_altitude', color='bird_name',
        color_discrete_sequence=BIRD_COLORS,
        title=f"Altitude Over Time",
        labels={'date_time': 'Time', 'altitude': 'm'},
        template='plotly_white'
    )

    # Altitude Scatter Plot with Animation
    scatter_plot = px.scatter(
    df_daily, x='date_time', y='avg_altitude', color='bird_name',
    color_discrete_sequence=BIRD_COLORS,
    animation_frame="frame")

    for trace in line_plot.data:
        scatter_plot.add_trace(trace)
    
    scatter_plot.update_layout(margin=dict(t=60, b=40, l=40, r=40))
    return scatter_plot

# -- Speed Line Chart --
# Data Preprocessing

def build_line_chart_speed(df, selected_bird):
    if not selected_bird:
        fig = px.line()
        fig.update_layout(title="Select bird to begin", template='plotly_white')
        return fig
    
    if isinstance(selected_bird, str): selected_bird = [selected_bird]
    
    df_filtered = df[df["bird_name"].isin(selected_bird)].copy()
    df_filtered = df_filtered.dropna(subset=["date_time", "latitude", "longitude"])
    df_filtered = df_filtered.sort_values(["bird_name", "date_time"])
    
    df_filtered["frame"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d %H"))
    df_filtered["frame_daily"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))

    df_daily = (df_filtered.groupby(["bird_name", "frame_daily"]).first().reset_index())
    df_daily["avg_speed"] = (
    df_daily
    .groupby("bird_name")["speed_2d"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)   # removes group index so it fits back
    )
    
    # Line Plot
    line_plot = px.line(
        df_daily, x='date_time', y='avg_speed', color='bird_name',
        color_discrete_sequence=BIRD_COLORS,
        title=f"Speed Over Time",
        labels={'date_time': 'Time', 'speed_2d': 'km/h'},
        template='plotly_white'
    )

    # 2D Speed Scatter Plot with Animation
    scatter_plot = px.scatter(
    df_daily, x='date_time', y='avg_speed', color='bird_name',
    color_discrete_sequence=BIRD_COLORS,
    animation_frame="frame")

    for trace in line_plot.data:
        scatter_plot.add_trace(trace)
    
    scatter_plot.update_layout(margin=dict(t=60, b=40, l=40, r=40))
    return scatter_plot

# --- ANIMATION ENGINE (Unchanged from Main) ---
def build_animated_map(df, selected_bird):
    if not selected_bird:
        return create_map(pd.DataFrame())

    if isinstance(selected_bird, str): selected_bird = [selected_bird]
    
    df_filtered = df[df["bird_name"].isin(selected_bird)].copy()
    df_filtered = df_filtered.dropna(subset=["date_time", "latitude", "longitude"])
    df_filtered = df_filtered.sort_values(["bird_name", "date_time"])
    
    df_filtered["frame"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d %H"))
    df_filtered["frame_daily"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))

    df_hourly = (df_filtered.groupby(["bird_name", "frame"]).first().reset_index())
    df_daily = (df_filtered.groupby(["bird_name", "frame_daily"]).first().reset_index())
    
    df_hourly["step"] = range(len(df_hourly))
    df_hourly["step"] = df_hourly["frame"].map({f:i for i,f in enumerate(df_hourly["frame"])})
    df_path = df_hourly[df_hourly["step"].notna()].copy()
    df_path = df_path[df_path["step"].apply(lambda s: df_path["step"] <= s).any(axis=0)]

    fig = px.line_geo(
        df_path, lat="latitude", lon="longitude", color="bird_name",
        color_discrete_sequence=BIRD_COLORS,
        line_group="bird_name", hover_name="bird_name",
        title=f"Animated Movement", height=650, fitbounds="locations"
    )
    
    fig_points = px.scatter_geo(
        df_daily, lat="latitude", lon="longitude", color="bird_name",
        color_discrete_sequence=BIRD_COLORS,
        size=np.array([10]*len(df_daily)), animation_frame="frame",
        animation_group="bird_name", hover_name="bird_name"
    )

    for trace in fig.data:
        fig_points.add_trace(trace)

    fig_points.update_layout(
        geo=dict(showcountries=True, showland=True, landcolor="rgba(240,240,240,1)", projection_type="natural earth"),
        margin={"r":0,"t":50,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", template="plotly_white"
    )
    return fig_points

# ======================================================
# 4. APP LAYOUT (Main + New Controls)
# ======================================================

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
                    # BIRD SELECTOR
                    html.Label("Select Bird", className="mb-2 fw-bold text-primary"),
                    dcc.Dropdown(
                        id='bird-name-filter',
                        options=[{'label': s, 'value': s} for s in sorted(df['bird_name'].unique())],
                        value=sorted(df['bird_name'].unique()), 
                        multi=True, 
                        clearable=True
                    ),
                    dbc.Button("Select All Birds", id="btn-all-birds", color="light", size="sm", className="mt-1 w-100 border"),

                    html.Hr(),
                    
                   # DATA CATEGORY (Updated with 'Rest')
                    html.Label("Choose Data Category", className="fw-bold text-primary"),
                    dbc.RadioItems(
                        id='category-selector',
                        options=[
                            {'label': ' Altitude', 'value': 'Altitude'},
                            {'label': ' Speed', 'value': 'Speed'},
                            {'label': ' Rest Stops', 'value': 'Rest'} # <--- Added
                        ],
                        value='Altitude', 
                        className="mb-3",
                        inputClassName="me-2"
                    ),

                    html.Hr(),

                    # STATISTIC CHECKLIST (Wrapped for visibility control)
                    html.Div([
                        html.Label("Select Statistics", className="fw-bold text-primary"),
                        dbc.Checklist(
                            id='statistic-checklist',
                            options=[
                                {'label': ' Maximum', 'value': 'Max'},
                                {'label': ' Average', 'value': 'Avg'},
                                {'label': ' Minimum', 'value': 'Min'},
                            ],
                            value=['Max', 'Avg', 'Min'], 
                            switch=True, 
                            className="mb-3"
                        ),
                    ], id='stats-container')  
                ])
            ], className="mb-4 shadow-sm")
        ], width=12, md=3), 
        
        # ------ GRAPHS -------
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

    dbc.Row([
        # Left: animated map (full height)
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id="animated-map", style={"height": "75vh"},)
                ],style={"padding": "0"},)
            ],className="shadow-sm",
            )
            ],width=12,md=6,
        ),

        # Right: two line charts stacked
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id="line-chart-altitude",style={"height": "35vh"},)
                ],style={"padding": "0"},)
            ],className="shadow-sm border-0 mb-3",
            ),
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id="line-chart-speed",style={"height": "35vh"},)
                ],style={"padding": "0"},)
            ],className="shadow-sm border-0",
            ),
        ],width=12,md=6,
        ),
    ]),

    # dbc.Row([
    #     dbc.Col(
    #         html.Div(
    #             html.Button("Play", id="play-pause", n_clicks=0),
    #             style={"textAlign": "center", "marginTop": "10px"},
    #         ),width=12,),
    # ],className="mb-4",)

    ]),
], fluid=True)


# ======================================================
# 5. CALLBACKS
# ======================================================

# --- MAP UPDATE ---
@app.callback(
    Output('migration-map', 'figure'),
    Input('bird-name-filter', 'value')
)
def update_map(selected_bird_names):
    if not selected_bird_names:
        return create_map(pd.DataFrame())
    filtered = df[df['bird_name'].isin(selected_bird_names)]
    return create_map(filtered)

# --- OPTIONS & VISIBILITY UPDATE (From Branch) ---
@app.callback(
    [Output('statistic-checklist', 'options'),
     Output('statistic-checklist', 'value'),
     Output('stats-container', 'style')], 
    Input('category-selector', 'value'),
    State('statistic-checklist', 'value')
)
def update_stat_options(category, current_values):
    # Rest: Hide Checklist
    if category == 'Rest':
        options = [{'label': ' Total Count', 'value': 'Total'}]
        return options, ['Total'], {'display': 'none'}

    # Speed: Show Checklist (Max/Avg only)
    elif category == 'Speed':
        options = [
            {'label': ' Maximum Speed', 'value': 'Max'},
            {'label': ' Average Speed', 'value': 'Avg'},
        ]
        if not current_values or 'Total' in current_values:
             return options, ['Max', 'Avg'], {'display': 'block'}
        valid_values = [v for v in current_values if v in ['Max', 'Avg']]
        return options, (valid_values if valid_values else ['Max', 'Avg']), {'display': 'block'}

    # Altitude: Show Checklist (All)
    else: 
        options = [
            {'label': ' Maximum', 'value': 'Max'},
            {'label': ' Average', 'value': 'Avg'},
            {'label': ' Minimum', 'value': 'Min'}
        ]
        if not current_values or 'Total' in current_values:
             return options, ['Max', 'Avg', 'Min'], {'display': 'block'}
        valid_values = [v for v in current_values if v in ['Max', 'Avg', 'Min']]
        return options, (valid_values if valid_values else ['Max', 'Avg', 'Min']), {'display': 'block'}

# --- BAR CHART UPDATE ---
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

# --- LINE CHART UPDATES ---
@app.callback(
    Output('line-chart-altitude', 'figure'),
    Input('bird-name-filter', 'value')
)
def update_line_chart_altitude(selected_bird):
    return build_line_chart_altitude(df=df, selected_bird=selected_bird)

@app.callback(
    Output('line-chart-speed', 'figure'),
    Input('bird-name-filter', 'value')
)
def update_line_chart_speed(selected_bird):
    return build_line_chart_speed(df=df, selected_bird=selected_bird)

# --- ANIMATION UPDATE ---
@app.callback(
    Output('animated-map', 'figure'),
    Input('bird-name-filter', 'value')
)
def update_animated_map(selected_bird):
    return build_animated_map(df=df, selected_bird=selected_bird)

# --- SELECT ALL BUTTON ---
@app.callback(
    Output('bird-name-filter', 'value'),
    Input('btn-all-birds', 'n_clicks'),
    State('bird-name-filter', 'options'),
    prevent_initial_call=True
)
def select_all_species(n_clicks, options):
    return [opt['value'] for opt in options]

# --- RUN ---
if __name__ == '__main__':
    app.run(debug=True)