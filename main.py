import sys
import os
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.io as pio

# --- BUG FIX: Force default template globally to avoid 'None' issues ---
pio.templates.default = "plotly_white"

# ======================================================
# CONSISTENT BIRD COLOR MAPPING
# ======================================================
BIRD_COLOR_MAP = {
    'Eric': '#9370DB',    # Purple
    'Nico': '#2ECC71',    # Green  
    'Sanne': '#3498DB'    # Blue
}

# Ensure project path is correct
sys.path.append(os.path.abspath(".."))

# ======================================================
# 1. DATA LOADING & PRE-PROCESSING
# ======================================================

# Load Data
df = pd.read_csv('data/bird_migration.csv')

# --- PRE-SORT DATA ---
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values(['bird_name', 'date_time'])

# --- ROBUST STOP DETECTION HELPER ---
def count_significant_stops(sub_df, speed_threshold=0.5, min_duration_hours=1):
    is_resting = sub_df['speed_2d'] < speed_threshold
    block_ids = (is_resting != is_resting.shift()).cumsum()
    block_durations = sub_df.groupby(block_ids)['date_time'].agg(lambda x: x.max() - x.min())
    block_is_rest = sub_df.groupby(block_ids)['speed_2d'].mean() < speed_threshold
    min_duration = pd.Timedelta(hours=min_duration_hours)
    return ((block_is_rest) & (block_durations > min_duration)).sum()

# ======================================================
# 2. DATA AGGREGATION
# ======================================================

# A. Altitude Stats
alt_stats = df.groupby('bird_name').agg(
    Max_Altitude=('altitude', 'max'),
    Avg_Altitude=('altitude', 'mean'),
    Min_Altitude=('altitude', 'min')
)

# B. Speed Stats (Flying only)
df_flying = df[df['speed_2d'] >= 0.5] 

speed_stats = df_flying.groupby('bird_name').agg(
    Max_Speed=('speed_2d', 'max'),
    Avg_Speed=('speed_2d', 'mean'), 
    Min_Speed=('speed_2d', 'min')
)

# C. Stopover Counts
stop_counts = df.groupby('bird_name').apply(
    count_significant_stops, 
    include_groups=False
)

# D. Merge All Stats
df_bird_stats = alt_stats.join(speed_stats).join(stop_counts.rename('Total_Rest')).reset_index()

# ======================================================
# 3. VISUALIZATION ENGINES
# ======================================================

# --- MAP ENGINE (STATIC - ATLAS STYLE) ---
def create_map(df):
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(text="No data selected", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
        return fig
    
    # Manually add traces for each bird
    unique_birds = df['bird_name'].unique()
    for bird in unique_birds:
        dff = df[df['bird_name'] == bird]
        fig.add_trace(go.Scattergeo(
            lat=dff['latitude'],
            lon=dff['longitude'],
            mode='lines+markers',
            name=bird,
            line=dict(width=2, color=BIRD_COLOR_MAP.get(bird, 'gray')),
            marker=dict(size=6, opacity=0.8),
            hoverinfo='text',
            text=dff['bird_name']
        ))
    
    fig.update_geos(
        visible=True, resolution=50,
        showcountries=True, countrycolor="#A0A0A0",
        showcoastlines=True, coastlinecolor="#888888",
        
        # --- ATLAS STYLE ---
        showland=True, landcolor="#EAE7D6",      
        showocean=True, oceancolor="#D4F1F9",    
        showlakes=True, lakecolor="#D4F1F9",     
        showrivers=True, rivercolor="#D4F1F9",   
        
        projection_rotation=dict(lon=-10, lat=20),
        projection_type="orthographic",
        fitbounds="locations"
    )
    
    fig.update_layout(
        title={'text': "Trajectory Map", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        margin={"r":0,"t":60,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)", 
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(255,255,255,0.9)"),
        template="plotly_white"
    )
    return fig

# --- BAR CHART ENGINE ---
def build_bar_chart(df_bird_stats, selected_bird, category, selected_stats):

    if not selected_bird:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "Select bird from the configuration panel to begin analysis", 'x': 0.5},
            xaxis={'visible': False}, yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18, color="#6c757d"),
            height=500 # --- ADJUSTED HEIGHT (Not too big, not too small) ---
        )
        return fig

    if not selected_stats:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "Please select at least one statistic", 'x': 0.5},
            xaxis={'visible': False}, yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18, color="#6c757d"),
            height=500 
        )
        return fig

    cols_to_plot = []
    title_prefix = ""

    if category == "Altitude":
        y_label = "Altitude (meters)"
        title_prefix = "Altitude Comparison"
        for stat in selected_stats:
            cols_to_plot.append(f"{stat}_{category}")

    elif category == "Speed":
        y_label = "Speed (km/h)"
        title_prefix = "Speed Comparison"
        for stat in selected_stats:
            cols_to_plot.append(f"{stat}_{category}")

    elif category == "Rest":
        y_label = "Number of Rest Stops"
        title_prefix = "Rest Stop Frequency"
        cols_to_plot.append("Total_Rest")

    df_filtered = df_bird_stats[df_bird_stats['bird_name'].isin(selected_bird)].copy()
    valid_cols = [c for c in cols_to_plot if c in df_filtered.columns]
    
    if not valid_cols:
         fig = go.Figure()
         fig.update_layout(title="No data available for the selected options.", height=500)
         return fig

    df_melted = df_filtered.melt(
        id_vars='bird_name',
        value_vars=valid_cols,
        var_name='Metric',
        value_name='Value'
    )

    if df_melted.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for the selected options.", height=500)
        return fig

    unique_birds = sorted(df_melted['bird_name'].unique())
    
    def convert_to_rgba(color_code, opacity):
        if color_code.startswith('#'):
            rgb = pc.hex_to_rgb(color_code)
        elif color_code.startswith('rgb'):
            nums = color_code.replace('rgb(', '').replace(')', '').split(',')
            rgb = [int(n) for n in nums]
        else:
            return color_code
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"
    
    def get_opacity(metric):
        if 'Max' in metric: return 1.0
        elif 'Avg' in metric: return 0.7
        elif 'Min' in metric: return 0.4
        else: return 1.0
    
    fig = go.Figure()
    
    unique_metrics = sorted(df_melted['Metric'].unique(), 
                           key=lambda x: (0 if 'Max' in x else 1 if 'Avg' in x else 2 if 'Min' in x else 3))
    
    for bird in unique_birds:
        bird_data = df_melted[df_melted['bird_name'] == bird].copy()
        bird_data['metric_order'] = bird_data['Metric'].map({m: i for i, m in enumerate(unique_metrics)})
        bird_data = bird_data.sort_values('metric_order')
        
        base_color = BIRD_COLOR_MAP.get(bird, '#808080')
        colors = [convert_to_rgba(base_color, get_opacity(m)) for m in bird_data['Metric']]
        
        fig.add_trace(go.Bar(
            x=bird_data['Metric'],
            y=bird_data['Value'],
            name=bird,
            marker=dict(color=colors, line=dict(color='white', width=1)),
            text=bird_data['Value'].round(0).astype(int),
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate=(f'<b>{bird}</b><br>Metric: %{{x}}<br>Value: %{{y:,.0f}}<br><extra></extra>')
        ))

    fig.update_layout(
        title={'text': title_prefix, 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        template='plotly_white',
        legend=dict(title="Birds", orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02, bgcolor='rgba(255,255,255,0.9)', bordercolor='#e0e0e0', borderwidth=1),
        margin=dict(t=60, b=60, l=60, r=200),
        yaxis=dict(title=y_label, tickformat=",", gridcolor='#e0e0e0', showgrid=True, zeroline=True),
        xaxis=dict(title="Statistics", tickangle=-45 if len(unique_metrics) > 3 else 0, showgrid=False),
        barmode='group',
        bargap=0.5,      
        bargroupgap=0.1,
        height=500, # --- ADJUSTED HEIGHT (Not too big, not too small) ---
        hovermode='closest',
        plot_bgcolor='#fafafa',
    )
    
    if category == "Speed":
        fig.add_annotation(text="Note: Speed statistics calculated from flying segments only (speed ≥ 0.5 km/h)", xref="paper", yref="paper", x=0.5, y=-0.15, showarrow=False, font=dict(size=10, color="#7f8c8d"), xanchor='center')

    return fig

# --- ALTITUDE LINE CHART ---
def build_line_chart_altitude(df, selected_bird):
    if not selected_bird:
        fig = go.Figure()
        fig.update_layout(title="Select bird to begin", template='plotly_white')
        return fig
    
    if isinstance(selected_bird, str): selected_bird = [selected_bird]
    
    df_filtered = df[df["bird_name"].isin(selected_bird)].copy()
    df_filtered = df_filtered.dropna(subset=["date_time", "latitude", "longitude"])
    df_filtered = df_filtered.sort_values(["bird_name", "date_time"])
    
    df_filtered["frame"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))
    df_filtered["frame_daily"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))

    df_daily = (df_filtered.groupby(["bird_name", "frame_daily"]).first().reset_index())
    df_daily["avg_altitude"] = (df_daily.groupby("bird_name")["altitude"].expanding().mean().reset_index(level=0, drop=True))
    
    # --- SAFE PX CALL: Use empty dict for template ---
    line_plot = px.line(
        df_daily, x='date_time', y='avg_altitude', color='bird_name',
        color_discrete_map=BIRD_COLOR_MAP,
        labels={'date_time': 'Date', 'avg_altitude': 'Altitude (m)'},
        template={} 
    )

    scatter_plot = px.scatter(
        df_daily, x='date_time', y='avg_altitude', color='bird_name',
        color_discrete_map=BIRD_COLOR_MAP,
        animation_frame="frame",
        template={} 
    )

    for trace in line_plot.data:
        scatter_plot.add_trace(trace)
    
    scatter_plot.update_layout(
        uirevision=str(selected_bird), 
        title={'text': "Altitude Over Time", 'x': 0.05, 'xanchor': 'left'},
        template='plotly_white',
        margin=dict(t=60, b=50, l=40, r=40), 
        xaxis=dict(title="", showticklabels=False, visible=True),
        yaxis=dict(title="Altitude (m)"),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x': 0, 'y': -0.15, 'xanchor': 'left', 'yanchor': 'top',
            'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 50}}]},
                        {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]}]
        }],
        sliders=[{
            'active': 0, 'yanchor': 'top', 'y': -0.1, 'xanchor': 'left',
            'currentvalue': {'prefix': '', 'visible': False, 'xanchor': 'right'},
            'transition': {'duration': 50}, 'pad': {'b': 5, 't': 5}, 'len': 0.9, 'x': 0.05,
            'steps': [{'args': [[f['name']], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'method': 'animate', 'label': pd.to_datetime(f['name']).strftime('%b %Y')} for f in scatter_plot.frames]
        }]
    )
    return scatter_plot

# --- SPEED LINE CHART ---
def build_line_chart_speed(df, selected_bird):
    if not selected_bird:
        fig = go.Figure()
        fig.update_layout(title="Select bird to begin", template='plotly_white')
        return fig
    
    if isinstance(selected_bird, str): selected_bird = [selected_bird]
    
    df_filtered = df[df["bird_name"].isin(selected_bird)].copy()
    df_filtered = df_filtered.dropna(subset=["date_time", "latitude", "longitude"])
    df_filtered = df_filtered.sort_values(["bird_name", "date_time"])
    
    df_filtered["frame"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))
    df_filtered["frame_daily"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))

    df_daily = (df_filtered.groupby(["bird_name", "frame_daily"]).first().reset_index())
    df_daily["avg_speed"] = (df_daily.groupby("bird_name")["speed_2d"].expanding().mean().reset_index(level=0, drop=True))
    
    # --- SAFE PX CALL ---
    line_plot = px.line(
        df_daily, x='date_time', y='avg_speed', color='bird_name',
        color_discrete_map=BIRD_COLOR_MAP,
        labels={'date_time': 'Date', 'avg_speed': 'Speed (km/h)'},
        template={}
    )

    scatter_plot = px.scatter(
        df_daily, x='date_time', y='avg_speed', color='bird_name',
        color_discrete_map=BIRD_COLOR_MAP,
        animation_frame="frame",
        template={}
    )

    for trace in line_plot.data:
        scatter_plot.add_trace(trace)
    
    scatter_plot.update_layout(
        uirevision=str(selected_bird),
        title={'text': "Speed Over Time", 'x': 0.05, 'xanchor': 'left'},
        template='plotly_white',
        margin=dict(t=60, b=50, l=40, r=40),
        xaxis=dict(title="", showticklabels=False, visible=True),
        yaxis=dict(title="Speed (km/h)"),
        updatemenus=[{
            'type': 'buttons', 'showactive': False, 'x': 0, 'y': -0.15, 'xanchor': 'left', 'yanchor': 'top',
            'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 50}}]},
                        {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]}]
        }],
        sliders=[{
            'active': 0, 'yanchor': 'top', 'y': -0.1, 'xanchor': 'left',
            'currentvalue': {'prefix': '', 'visible': False, 'xanchor': 'right'},
            'transition': {'duration': 50}, 'pad': {'b': 5, 't': 5}, 'len': 0.9, 'x': 0.05,
            'steps': [{'args': [[f['name']], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'method': 'animate', 'label': pd.to_datetime(f['name']).strftime('%b %Y')} for f in scatter_plot.frames]
        }]
    )
    return scatter_plot

# --- ANIMATED MAP ---
def build_animated_map(df, selected_bird):
    if not selected_bird:
        return create_map(pd.DataFrame())

    if isinstance(selected_bird, str): selected_bird = [selected_bird]
    
    df_filtered = df[df["bird_name"].isin(selected_bird)].copy()
    df_filtered = df_filtered.dropna(subset=["date_time", "latitude", "longitude"])
    df_filtered = df_filtered.sort_values(["bird_name", "date_time"])
    
    df_filtered["frame"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))
    df_filtered["frame_daily"] = pd.to_datetime(df_filtered["date_time"].dt.strftime("%Y-%m-%d"))

    df_hourly = (df_filtered.groupby(["bird_name", "frame"]).first().reset_index())
    df_daily = (df_filtered.groupby(["bird_name", "frame_daily"]).first().reset_index())
    
    df_hourly["step"] = range(len(df_hourly))
    df_hourly["step"] = df_hourly["frame"].map({f:i for i,f in enumerate(df_hourly["frame"])})
    df_path = df_hourly[df_hourly["step"].notna()].copy()
    df_path = df_path[df_path["step"].apply(lambda s: df_path["step"] <= s).any(axis=0)]

    # --- SAFE PX CALL ---
    fig = px.line_geo(
        df_path, lat="latitude", lon="longitude", color="bird_name",
        color_discrete_map=BIRD_COLOR_MAP,
        line_group="bird_name", hover_name="bird_name",
        title=f"Animated Movement", height=650,
        template={} 
    )
    
    # --- SAFE PX CALL ---
    fig_points = px.scatter_geo(
        df_daily, lat="latitude", lon="longitude", color="bird_name",
        color_discrete_map=BIRD_COLOR_MAP,
        size=np.array([10]*len(df_daily)), animation_frame="frame",
        animation_group="bird_name", hover_name="bird_name",
        template={} 
    )

    for trace in fig.data:
        fig_points.add_trace(trace)

    fig_points.update_layout(
        geo=dict(
            fitbounds="locations", 
            showcountries=True, countrycolor="#A0A0A0",
            showcoastlines=True, coastlinecolor="#888888",
            
            # --- ATLAS STYLE APPLIED HERE TOO ---
            showland=True, landcolor="#EAE7D6",
            showocean=True, oceancolor="#D4F1F9",
            showlakes=True, lakecolor="#D4F1F9",
            showrivers=True, rivercolor="#D4F1F9",
            
            projection_type="natural earth"
        ),
        uirevision=str(selected_bird),
        title={'text': "Animated Movement", 'x': 0.05, 'xanchor': 'left'},
        margin={"r":0,"t":50,"l":0,"b":80}, 
        paper_bgcolor="rgba(0,0,0,0)", 
        template="plotly_white",
        updatemenus=[{
            'type': 'buttons', 'showactive': False, 'x': 0, 'y': 0, 'xanchor': 'left', 'yanchor': 'bottom',
            'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 150, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 75}}]},
                        {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]}]
        }],
        sliders=[{
            'active': 0, 'yanchor': 'bottom', 'y': 0.1, 'xanchor': 'left',
            'currentvalue': {'prefix': '', 'visible': False, 'xanchor': 'right'},
            'transition': {'duration': 75}, 'pad': {'b': 5, 't': 5}, 'len': 0.9, 'x': 0.05,
            'steps': [{'args': [[f['name']], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'method': 'animate', 'label': pd.to_datetime(f['name']).strftime('%b %Y')} for f in fig_points.frames]
        }]
    )
    return fig_points

# ======================================================
# 4. APP LAYOUT
# ======================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    
    # --- 1. HERO BANNER SECTION (STICKY) ---
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(
                    src=app.get_asset_url('seagull_3.gif'), 
                    style={'width': '100%', 'height': '200px', 'objectFit': 'cover', 'filter': 'blur(4px)', 'position': 'absolute', 'top': '0', 'left': '0', 'zIndex': '1'}
                ),
                html.Div(style={'position': 'absolute', 'top': '0', 'left': '0', 'width': '100%', 'height': '200px', 'backgroundColor': 'rgba(0, 0, 0, 0.4)', 'zIndex': '2'}),
                html.Div([
                    html.H1("Journeys In The Sky", className="display-3 fw-bold text-white"),
                    html.P("Global Bird Migration Tracker", className="lead text-light"),
                    html.P("Compare patterns, altitudes, and speeds across 3 seagulls - Eric, Nico and Sanne.", className="text-white-50")
                ], style={'position': 'relative', 'zIndex': '3', 'paddingTop': '30px', 'textAlign': 'center'})
            ], style={'position': 'relative', 'height': '200px', 'overflow': 'hidden', 'borderRadius': '0 0 15px 15px'})
        ], width=12)
    ], className="mb-4", style={"position": "sticky", "top": "0", "zIndex": "2000", "backgroundColor": "white"}),
    
    # --- MAIN CONTENT ROW ---
    dbc.Row([
        # ---------------------
        # LEFT COLUMN (SIDEBAR - BIRDS ONLY)
        # ---------------------
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Global Filter", className="fw-bold"),
                dbc.CardBody([
                    html.Label("Select Birds", className="mb-2 fw-bold text-primary"),
                    dbc.Checklist(
                        id='bird-name-filter',
                        options=[{'label': s, 'value': s} for s in sorted(df['bird_name'].unique())],
                        value=sorted(df['bird_name'].unique()), 
                        switch=True, 
                        className="mb-3"
                    ),
                    dbc.Button("Select All Birds", id="btn-all-birds", color="light", size="sm", className="mt-1 w-100 border"),
                ])
            ], 
            className="mb-4 shadow-sm",
            # --- STICKY SIDEBAR ADJUSTED FOR HEADER ---
            style={"position": "sticky", "top": "220px", "zIndex": "1000", "height": "fit-content"})
        ], width=12, md=3), 
        
        # ----------------------
        # RIGHT COLUMN (CHARTS)
        # ----------------------
        dbc.Col([
            # ROW 1: STATIC MAP
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(id='migration-map', style={'height': '75vh'}, config={'responsive': True})], style={'padding': '0'})
                    ], className="shadow-sm mb-5") 
                ], width=12)
            ]),
            
            html.Div(style={"height": "50px"}),

            # ROW 2: BAR CHART (SPLIT COLUMN)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        # --- REDUCED HEIGHT HERE ---
                        dbc.CardBody([dcc.Graph(id='main-bar-chart', style={'height': '50vh'}, config={'responsive': True})], style={'padding': '0'})
                    ], className="shadow-sm border-0 mb-5")
                ], width=12, md=9),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Chart Controls", className="fw-bold"),
                        dbc.CardBody([
                            html.Label("Choose Data Category", className="fw-bold text-primary"),
                            dbc.RadioItems(
                                id='category-selector',
                                options=[{'label': ' Altitude', 'value': 'Altitude'}, {'label': ' Speed', 'value': 'Speed'}, {'label': ' Rest Stops', 'value': 'Rest'}],
                                value='Altitude', className="mb-3", inputClassName="me-2"
                            ),
                            html.Hr(),
                            html.Div([
                                html.Label("Select Statistics", className="fw-bold text-primary"),
                                dbc.Checklist(
                                    id='statistic-checklist',
                                    options=[{'label': ' Maximum', 'value': 'Max'}, {'label': ' Average', 'value': 'Avg'}, {'label': ' Minimum', 'value': 'Min'}],
                                    value=['Max', 'Avg', 'Min'], switch=True, className="mb-3"
                                ),
                            ], id='stats-container') 
                        ])
                    ], className="shadow-sm sticky-top", style={"top": "220px"})
                ], width=12, md=3)
            ]),

            html.Div(style={"height": "200px"}),

            # ROW 3: ANIMATED MAP + LINE CHARTS
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(id="animated-map", style={"height": "75vh"}, config={'responsive': True})], style={"padding": "0"})
                    ], className="shadow-sm")
                ], width=12, md=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(id="line-chart-altitude", style={"height": "35vh"}, config={'responsive': True})], style={"padding": "0"})
                    ], className="shadow-sm border-0 mb-4"), 
                    
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(id="line-chart-speed", style={"height": "35vh"}, config={'responsive': True})], style={"padding": "0"})
                    ], className="shadow-sm border-0")
                ], width=12, md=6),
            ])
        ], width=12, md=9), 
    ]), 
], fluid=True)

# ======================================================
# 5. CALLBACKS
# ======================================================

@app.callback(Output('migration-map', 'figure'), Input('bird-name-filter', 'value'))
def update_map(selected_bird_names):
    if not selected_bird_names: return create_map(pd.DataFrame())
    filtered = df[df['bird_name'].isin(selected_bird_names)]
    return create_map(filtered)

@app.callback([Output('statistic-checklist', 'options'), Output('statistic-checklist', 'value'), Output('stats-container', 'style')], Input('category-selector', 'value'), State('statistic-checklist', 'value'))
def update_stat_options(category, current_values):
    if category == 'Rest': return [{'label': ' Total Count', 'value': 'Total'}], ['Total'], {'display': 'none'}
    elif category == 'Speed':
        valid_values = [v for v in current_values if v in ['Max', 'Avg']]
        return [{'label': ' Maximum Speed', 'value': 'Max'}, {'label': ' Average Speed', 'value': 'Avg'}], valid_values if valid_values else ['Max', 'Avg'], {'display': 'block'}
    else: 
        valid_values = [v for v in current_values if v in ['Max', 'Avg', 'Min']]
        return [{'label': ' Maximum', 'value': 'Max'}, {'label': ' Average', 'value': 'Avg'}, {'label': ' Minimum', 'value': 'Min'}], valid_values if valid_values else ['Max', 'Avg', 'Min'], {'display': 'block'}

@app.callback(Output('main-bar-chart', 'figure'), [Input('bird-name-filter', 'value'), Input('category-selector', 'value'), Input('statistic-checklist', 'value')])
def update_chart(selected_bird, category, selected_stats):
    return build_bar_chart(df_bird_stats, selected_bird, category, selected_stats)

@app.callback(Output('line-chart-altitude', 'figure'), Input('bird-name-filter', 'value'))
def update_line_chart_altitude(selected_bird):
    return build_line_chart_altitude(df=df, selected_bird=selected_bird)

@app.callback(Output('line-chart-speed', 'figure'), Input('bird-name-filter', 'value'))
def update_line_chart_speed(selected_bird):
    return build_line_chart_speed(df=df, selected_bird=selected_bird)

@app.callback(Output('animated-map', 'figure'), Input('bird-name-filter', 'value'))
def update_animated_map(selected_bird):
    return build_animated_map(df=df, selected_bird=selected_bird)

@app.callback(Output('bird-name-filter', 'value'), Input('btn-all-birds', 'n_clicks'), State('bird-name-filter', 'options'), prevent_initial_call=True)
def select_all_species(n_clicks, options):
    return [opt['value'] for opt in options]

# --- RUN ---
if __name__ == '__main__':
    app.run(debug=True)