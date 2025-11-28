import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from google import genai
from google.genai import types
import json
import time
from datetime import datetime, timedelta
from scipy.stats import poisson

# ==============================================================================
# 1. SYSTEM KERNEL
# ==============================================================================

st.set_page_config(
    page_title="TITAN OS // BENCHMARK",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- THEME ENGINE: "VOLT PROTOCOL" ---
st.markdown("""
<style>
    /* --- VARIABLES --- */
    :root {
        --bg-app: #09090B;
        --bg-card: #131316;
        --bg-card-hover: #1c1c21;
        --border: #27272A;
        --accent-volt: #DFFF00;  /* The signature Acid Green */
        --accent-red: #FF4D4D;
        --accent-blue: #3B82F6;
        --text-main: #FFFFFF;
        --text-muted: #71717A;
    }

    /* GLOBAL RESET */
    .stApp { background-color: var(--bg-app); color: var(--text-main); font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; max-width: 100%; }
    
    /* HIDE STREAMLIT CHROME */
    header { visibility: hidden; }
    footer { display: none; }
    
    /* --- METRIC CARDS (The Square Ones) --- */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1a1a, #0d0d0d);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--accent-volt);
        box-shadow: 0 0 15px rgba(223, 255, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetricValue"] {
        color: #fff;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 24px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 12px;
        font-weight: 700;
    }

    /* --- CONTAINERS (The Panels) --- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    /* --- TABS (Pill Style) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #000;
        padding: 4px;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        height: 32px;
        background-color: transparent;
        border: none;
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 600;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-volt);
        color: #000;
    }

    /* --- CUSTOM TEXT CLASSES --- */
    .header-title { font-size: 28px; font-weight: 800; letter-spacing: -1px; color: #fff; }
    .section-label { font-size: 14px; font-weight: 600; color: var(--text-muted); margin-bottom: 10px; }
    .volt-text { color: var(--accent-volt); font-weight: bold; }
    
    /* --- IMAGES --- */
    img { border-radius: 50%; }

</style>
""", unsafe_allow_html=True)

# --- KEYRING ---
KEYS = {
    "ODDS": "34e5a58b5b50587ce21dbe0b33e344dc",
    "RAPID": "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c",
    "NEWS": "289796ecfb2c4d208506c26d37a4d9ba",
    "GEMINI": "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"
}

# ==============================================================================
# 2. QUANT & VISUALIZATION ENGINE
# ==============================================================================

class Charts:
    @staticmethod
    def donut_profit(profit_data):
        """Replicates the 'Top 5 Sport Categories' Donut Chart"""
        labels = list(profit_data.keys())
        values = list(profit_data.values())
        colors = ['#DFFF00', '#00E5FF', '#FF4D4D', '#A855F7', '#333333']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=.7,
            marker=dict(colors=colors, line=dict(color='#121212', width=4)),
            textinfo='none',
            hoverinfo='label+value'
        )])
        
        fig.add_annotation(text="$3,223", x=0.5, y=0.5, font_size=24, font_color="white", showarrow=False)
        fig.add_annotation(text="Total Profit", x=0.5, y=0.4, font_size=12, font_color="gray", showarrow=False)
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    @staticmethod
    def funds_wave(history):
        """Replicates the 'Funds Activity' Spline Chart"""
        x = list(range(len(history)))
        
        fig = go.Figure()
        # Area Glow Effect
        fig.add_trace(go.Scatter(
            x=x, y=history, mode='lines',
            fill='tozeroy',
            line=dict(color='#DFFF00', width=3, shape='spline'),
            fillcolor='rgba(223, 255, 0, 0.1)'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            margin=dict(l=0, r=0, t=0, b=0),
            height=150,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        )
        return fig

class QuantMath:
    @staticmethod
    def decimal(american):
        if american > 0: return 1 + (american/100)
        return 1 + (100/abs(american))
        
    @staticmethod
    def kelly(dec, prob, frac=0.25):
        b = dec - 1
        q = 1 - prob
        f = (b * prob - q) / b
        return max(0, f) * frac

# ==============================================================================
# 3. DATA LAYER (Self-Healing)
# ==============================================================================

class DataEngine:
    @staticmethod
    def get_logo(team_name, league="nfl"):
        # Asset Mapping
        return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/{league.lower()}/500/scoreboard/{team_name[:3].lower()}.png&w=80&h=80"

    @staticmethod
    @st.cache_data(ttl=900)
    def get_odds(sport_key):
        url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
        params = {'apiKey': KEYS['ODDS'], 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
        try:
            res = requests.get(url, params=params, timeout=3)
            if res.status_code == 200:
                return res.json()
            return DataEngine.get_mock_odds() # Fallback
        except:
            return DataEngine.get_mock_odds() # Fallback

    @staticmethod
    def get_mock_odds():
        # Fallback data so the UI never breaks
        return [
            {"home_team": "Philadelphia Eagles", "away_team": "Chicago Bears", "bookmakers": [{"markets": [{"outcomes": [{"name": "Philadelphia Eagles", "price": -140}]}]}]},
            {"home_team": "New York Jets", "away_team": "Atlanta Falcons", "bookmakers": [{"markets": [{"outcomes": [{"name": "New York Jets", "price": 125}]}]}]},
            {"home_team": "Tampa Bay Buccaneers", "away_team": "Arizona Cardinals", "bookmakers": [{"markets": [{"outcomes": [{"name": "Tampa Bay Buccaneers", "price": -200}]}]}]},
            {"home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills", "bookmakers": [{"markets": [{"outcomes": [{"name": "Kansas City Chiefs", "price": -110}]}]}]},
            {"home_team": "San Francisco 49ers", "away_team": "Dallas Cowboys", "bookmakers": [{"markets": [{"outcomes": [{"name": "San Francisco 49ers", "price": -165}]}]}]}
        ]

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_stats():
        # NFL Stats
        url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
        headers = {"x-rapidapi-key": KEYS['RAPID'], "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"}
        try: return requests.get(url, headers=headers, timeout=3).json()
        except: return {}

# ==============================================================================
# 4. MAIN DASHBOARD LAYOUT
# ==============================================================================

def main():
    # --- HEADER ---
    c1, c2, c3 = st.columns([0.1, 0.7, 0.2])
    with c1:
        st.markdown("## ⚡")
    with c2:
        st.markdown('<div class="header-text">Dashboard</div>', unsafe_allow_html=True)
    with c3:
        st.button("Manage Profile")

    st.write("")

    # --- TOP ROW: KPI CARDS ---
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Income", "$3,433.0", "+4.5%")
    with k2: st.metric("Active Bankroll", "$11,443", "+12%")
    with k3: st.metric("ROI (All Time)", "24.8%", "+2.1%")
    with k4: 
        with st.container(border=True):
            st.markdown("**Total Wagered**")
            st.markdown("### $3,433.0")

    st.write("")

    # --- MAIN GRID ---
    col_left, col_mid, col_right = st.columns([1, 1.5, 1])

    # === LEFT COLUMN ===
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="section-label">Top 5 Categories</div>', unsafe_allow_html=True)
            # Donut Chart
            data = {"NFL": 40, "NBA": 25, "NHL": 20, "MLB": 10, "UFC": 5}
            st.plotly_chart(Charts.donut_profit(data), use_container_width=True)
        
        with st.container(border=True):
            st.markdown('<div class="section-label">Top 5 Leagues</div>', unsafe_allow_html=True)
            st.progress(78, text="NFL")
            st.progress(65, text="NBA")
            st.progress(42, text="NHL")

    # === MIDDLE COLUMN (THE FEED) ===
    with col_mid:
        # Tabs for Players/Bets/Plays
        tab1, tab2, tab3 = st.tabs(["LIVE BETS", "PLAYERS", "ALERTS"])
        
        with tab1:
            # Fetch Data (with fallback)
            odds = DataEngine.get_odds("americanfootball_nfl")
            
            for game in odds:
                home = game['home_team']
                away = game['away_team']
                
                # Odds Logic
                price = -110
                try:
                    if 'bookmakers' in game and game['bookmakers']:
                         price = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
                except: pass
                
                dec = QuantMath.decimal(price)
                # Mock AI Prob for demo speed
                model_prob = 0.55 
                stake = QuantMath.kelly(dec, model_prob) * 5000

                # RENDER GAME CARD
                with st.container(border=True):
                    c1, c2 = st.columns([0.7, 0.3])
                    with c1:
                        st.markdown(f"**{home}**")
                        st.caption(f"vs {away}")
                    with c2:
                        st.markdown(f":green-background[{price}]")
                    
                    st.divider()
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("PROB", f"{model_prob:.0%}")
                    m2.metric("EDGE", "+4.5%")
                    m3.metric("KELLY", f"${stake:.0f}")

    # === RIGHT COLUMN ===
    with col_right:
        # User Profile Card
        with st.container(border=True):
            uc1, uc2 = st.columns([0.3, 0.7])
            with uc1:
                st.image("https://ui-avatars.com/api/?name=John+Doe&background=random", width=60)
            with uc2:
                st.markdown("**John Williams**")
                st.caption("Last active: Just now")
        
        # Funds Activity Chart
        with st.container(border=True):
            st.markdown('<div class="section-label">Funds Activity</div>', unsafe_allow_html=True)
            history = np.cumsum(np.random.randn(30)) + 1000
            st.plotly_chart(Charts.funds_wave(history), use_container_width=True)
            
            fa1, fa2 = st.columns(2)
            fa1.metric("Active", "$1,443")
            fa2.metric("Playing", "$440")

        # Recent Transactions
        with st.container(border=True):
            st.markdown('<div class="section-label">Transactions</div>', unsafe_allow_html=True)
            st.caption("🟢 Parlay Payout: +$445")
            st.caption("🔴 Wager: -$110")
            st.caption("🟢 Deposit: +$1,000")

if __name__ == "__main__":
    main()
