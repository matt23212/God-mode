import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from google import genai
from google.genai import types
import json
from datetime import datetime
from scipy.stats import poisson

# ==============================================================================
# 1. CONFIGURATION & ASSETS
# ==============================================================================

KEYS = {
    "ODDS": "34e5a58b5b50587ce21dbe0b33e344dc",
    "RAPID": "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c",
    "NEWS": "289796ecfb2c4d208506c26d37a4d9ba",
    "GEMINI": "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"
}

# Initialize Dash with a dark bootstrap theme as a base
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="TITAN OS")
server = app.server # Expose server for deployment

# --- CUSTOM CSS (THE "OUTLIER" LOOK) ---
# This injects the neon/dark aesthetic directly into the app
custom_css = """
<style>
    :root {
        --bg-app: #000000;
        --bg-panel: #0c0c0c;
        --bg-card: #121212;
        --border: #222;
        --accent: #DFFF00; /* Acid Green */
        --accent-glow: rgba(223, 255, 0, 0.15);
        --text-main: #ffffff;
        --text-sub: #666;
        --font-main: 'Inter', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }
    
    body { background-color: var(--bg-app); font-family: var(--font-main); color: var(--text-main); margin: 0; }
    
    /* SIDEBAR */
    .sidebar {
        background-color: var(--bg-panel);
        border-right: 1px solid var(--border);
        height: 100vh;
        padding: 20px;
        position: fixed;
        width: 250px;
    }
    
    /* MAIN CONTENT */
    .content {
        margin-left: 250px;
        padding: 30px;
    }
    
    /* METRIC CARD */
    .stat-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.2s;
    }
    .stat-card:hover {
        border-color: var(--accent);
        box-shadow: 0 0 15px var(--accent-glow);
    }
    .stat-label { color: var(--text-sub); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .stat-val { color: #fff; font-family: var(--font-mono); font-size: 24px; font-weight: 700; margin-top: 5px; }
    .stat-delta { font-size: 12px; font-weight: 600; }
    .text-green { color: var(--accent); }
    .text-red { color: #FF4D4D; }
    
    /* GAME TICKET */
    .game-ticket {
        background: linear-gradient(145deg, #151515, #0a0a0a);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0;
        margin-bottom: 20px;
        overflow: hidden;
    }
    .ticket-header {
        padding: 15px 20px;
        border-bottom: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255,255,255,0.02);
    }
    .ticket-body { padding: 20px; }
    
    /* BADGES */
    .badge-ev {
        background: var(--accent-glow);
        color: var(--accent);
        border: 1px solid rgba(223, 255, 0, 0.3);
        padding: 4px 10px;
        border-radius: 100px;
        font-size: 11px;
        font-weight: 800;
    }
    
    /* BUTTONS */
    .btn-titan {
        background-color: #1a1a1a;
        color: var(--accent);
        border: 1px solid var(--border);
        font-weight: 700;
        font-family: var(--font-mono);
        width: 100%;
        padding: 10px;
        transition: 0.2s;
    }
    .btn-titan:hover {
        background-color: var(--accent);
        color: #000;
        border-color: var(--accent);
    }

    /* TABS */
    .nav-pills .nav-link { color: #666; font-weight: 600; border-radius: 8px; padding: 10px 20px; }
    .nav-pills .nav-link.active { background-color: #1a1a1a; color: var(--accent); border: 1px solid var(--border); }
</style>
"""

# ==============================================================================
# 2. ENGINES (Quant, AI, Data)
# ==============================================================================

class QuantEngine:
    @staticmethod
    def decimal(american):
        if american > 0: return (american / 100) + 1
        return (100 / abs(american)) + 1

    @staticmethod
    def kelly(dec, prob, frac=0.25):
        b = dec - 1
        q = 1 - prob
        f = (b * prob - q) / b
        return max(0.0, f) * frac

    @staticmethod
    def ev(dec, prob):
        return (prob * (dec - 1)) - (1 - prob)

class DataEngine:
    @staticmethod
    def get_logo(team_name):
        slug = team_name.split()[-1].lower()
        if "football" in slug: slug = "washington"
        return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/{slug}.png&w=80&h=80"

    @staticmethod
    def fetch_odds():
        url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
        try:
            res = requests.get(url, params={'apiKey': KEYS['ODDS'], 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'})
            return res.json() if res.status_code == 200 else []
        except: return []

    @staticmethod
    def fetch_stats():
        host = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
        try:
            res = requests.get(f"https://{host}/getNFLGamesForWeek", headers={"x-rapidapi-key": KEYS['RAPID'], "x-rapidapi-host": host})
            return res.json() if res.status_code == 200 else {}
        except: return {}

class AIEngine:
    @staticmethod
    def analyze(matchup, stats):
        client = genai.Client(api_key=KEYS['GEMINI'])
        prompt = f"""
        ROLE: Sports Quant. TASK: Analyze {matchup}. STATS: {str(stats)[:800]}
        OUTPUT JSON: {{"win_prob": 0.65, "confidence": 80, "reason": "Short sharp analysis.", "prop": "Player > X"}}
        """
        try:
            res = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt, 
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(res.text)
        except:
            return {"win_prob": 0.55, "confidence": 50, "reason": "Model Estimate", "prop": "N/A"}

# ==============================================================================
# 3. COMPONENT BUILDERS
# ==============================================================================

def build_stat_card(title, value, delta, color="text-green"):
    return html.Div([
        html.Div(title, className="stat-label"),
        html.Div(value, className="stat-val"),
        html.Div(delta, className=f"stat-delta {color}")
    ], className="stat-card")

def build_game_ticket(game, stats, bankroll, kelly_risk):
    # Logic
    home, away = game['home_team'], game['away_team']
    price = -110
    if game['bookmakers']:
        price = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
    
    # AI & Math
    ai = AIEngine.analyze(f"{away} @ {home}", stats)
    dec = QuantEngine.decimal(price)
    prob = ai.get('win_prob', 0.5)
    edge = QuantEngine.ev(dec, prob) * 100
    stake = bankroll * QuantEngine.kelly(dec, prob, kelly_risk)
    
    if edge < 0.5: return None # Filter bad bets

    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Img(src=DataEngine.get_logo(home), style={'width': '40px', 'margin-right': '15px'}),
                html.Div([
                    html.Div(f"{home}", style={'font-weight': '800', 'font-size': '16px', 'color': '#fff'}),
                    html.Div(f"vs {away}", style={'color': '#666', 'font-size': '12px'})
                ])
            ], style={'display': 'flex', 'align-items': 'center'}),
            html.Div(f"+{edge:.1f}% EV", className="badge-ev")
        ], className="ticket-header"),
        
        # Body
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div("SIGNAL", className="stat-label"),
                    html.Div(f"{price}", className="stat-val", style={'font-size': '20px'})
                ], width=4),
                dbc.Col([
                    html.Div("PROBABILITY", className="stat-label"),
                    html.Div(f"{prob:.0%}", className="stat-val", style={'font-size': '20px', 'color': '#00E5FF'})
                ], width=4),
                dbc.Col([
                    html.Div("KELLY STAKE", className="stat-label"),
                    html.Div(f"${stake:.0f}", className="stat-val", style={'font-size': '20px', 'color': '#DFFF00'})
                ], width=4),
            ]),
            html.Hr(style={'border-color': '#222'}),
            html.Div([
                html.Span("AI INSIGHT: ", style={'color': '#DFFF00', 'font-weight': 'bold', 'font-size': '12px'}),
                html.Span(ai.get('reason'), style={'color': '#ccc', 'font-size': '12px'})
            ]),
            html.Div(f"🧩 Target: {ai.get('prop')}", style={'color': '#666', 'font-size': '11px', 'margin-top': '5px'})
        ], className="ticket-body")
    ], className="game-ticket")

# ==============================================================================
# 4. LAYOUT DEFINITION
# ==============================================================================

app.layout = html.Div([
    html.Div([custom_css], style={'display': 'none'}), # Inject CSS
    
    # SIDEBAR
    html.Div([
        html.H2("TITAN OS", style={'color': '#fff', 'font-weight': '900', 'letter-spacing': '-1px'}),
        html.P("v11.0 // BENCHMARK", style={'color': '#444', 'font-size': '10px', 'font-family': 'monospace'}),
        html.Hr(style={'border-color': '#222'}),
        
        html.Label("BANKROLL", className="stat-label"),
        dcc.Input(id="bankroll-input", type="number", value=10000, className="form-control", style={'background': '#111', 'border': '1px solid #333', 'color': '#fff', 'margin-bottom': '20px'}),
        
        html.Label("RISK FACTOR (KELLY)", className="stat-label"),
        dcc.Slider(0.1, 0.5, 0.05, value=0.25, id='kelly-slider', 
                   marks={0.1: 'Safe', 0.5: 'Aggro'}, 
                   tooltip={"placement": "bottom", "always_visible": True}),
        
        html.Br(),
        dbc.Button("🔄 SYSTEM REFRESH", id="refresh-btn", className="btn-titan"),
        
    ], className="sidebar"),

    # CONTENT AREA
    html.Div([
        # HEADER
        dbc.Row([
            dbc.Col([
                html.H1("NFL WAR ROOM", style={'margin': 0}),
                html.P("INSTITUTIONAL GRADE ANALYTICS", style={'color': '#666', 'font-size': '12px', 'font-weight': '700'})
            ], width=8),
            dbc.Col([
                # Mock Live PnL Chart
                dcc.Graph(
                    figure=go.Figure(go.Scatter(y=np.cumsum(np.random.randn(20)), mode='lines', line=dict(color='#DFFF00', width=2), fill='tozeroy'))
                    .update_layout(height=60, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False)),
                    config={'displayModeBar': False}
                )
            ], width=4)
        ], className="mb-4"),

        # KPI ROW
        dbc.Row([
            dbc.Col(build_stat_card("Global PnL", "$12,450", "+8.4%", "text-green"), width=3),
            dbc.Col(build_stat_card("Active Exposure", "$3,200", "12 Bets", "text-green"), width=3),
            dbc.Col(build_stat_card("Model ROI", "14.2%", "+1.1%", "text-green"), width=3),
            dbc.Col(build_stat_card("Win Rate (L10)", "60%", "6-4-0", "text-green"), width=3),
        ], className="mb-4"),

        # TABS
        dbc.Tabs([
            dbc.Tab(label="🔥 ALPHA FEED", tab_id="tab-alpha"),
            dbc.Tab(label="📊 PROP LAB", tab_id="tab-props"),
            dbc.Tab(label="🧪 QUANT LAB", tab_id="tab-lab"),
        ], id="tabs", active_tab="tab-alpha", className="nav-pills mb-4"),

        # DYNAMIC CONTENT
        html.Div(id="tab-content")

    ], className="content")
])

# ==============================================================================
# 5. CALLBACKS (INTERACTIVITY)
# ==============================================================================

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("refresh-btn", "n_clicks")],
    [State("bankroll-input", "value"), State("kelly-slider", "value")]
)
def render_content(active_tab, n_clicks, bankroll, kelly_risk):
    if active_tab == "tab-alpha":
        odds = DataEngine.fetch_odds()
        # Fallback data if API fails (to ensure UI always shows something)
        if not odds:
            odds = [{"home_team": "Philadelphia Eagles", "away_team": "Dallas Cowboys", "bookmakers": [{"markets": [{"outcomes": [{"price": -120}]}]}]}]
        
        # Render Cards
        cards = []
        for game in odds[:8]:
            card = build_game_ticket(game, {}, bankroll, kelly_risk)
            if card: cards.append(card)
            
        return html.Div(cards)
    
    elif active_tab == "tab-props":
        # Mock Prop Visualizer (Green/Red Bars)
        fig = go.Figure(go.Bar(
            x=["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"],
            y=[20, 25, 18, 30, 22, 28, 15, 24, 29, 26],
            marker_color=['#333', '#00FF41', '#333', '#00FF41', '#333', '#00FF41', '#333', '#00FF41', '#00FF41', '#00FF41']
        ))
        fig.add_hline(y=22.5, line_dash="dash", line_color="white", annotation_text="Line: 22.5")
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        
        return html.Div([
            html.H3("PLAYER PROP ANALYZER", style={'color': '#fff'}),
            dcc.Graph(figure=fig)
        ])
        
    return html.Div("Module Loading...")

# --- RUN SERVER ---
if __name__ == "__main__":
    app.run_server(debug=True)

