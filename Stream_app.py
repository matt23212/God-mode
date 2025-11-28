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
from scipy.stats import poisson, norm

# ==============================================================================
# 1. SYSTEM CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="TITAN X // PRO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TITAN X THEME ENGINE ---
st.markdown("""
<style>
    /* APP BACKGROUND */
    .stApp { background-color: #050505; font-family: 'Inter', sans-serif; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #0A0A0A; border-right: 1px solid #222; }
    
    /* REMOVE PADDING */
    .block-container { padding-top: 1.5rem; padding-bottom: 5rem; }

    /* GAME CARD CONTAINER */
    .game-card {
        background-color: #111;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 0;
        margin-bottom: 20px;
        overflow: hidden;
        transition: transform 0.2s, border-color 0.2s;
    }
    .game-card:hover { border-color: #444; transform: translateY(-2px); }

    /* CARD HEADER */
    .card-header {
        background-color: #161616;
        padding: 12px 20px;
        border-bottom: 1px solid #222;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* CARD BODY */
    .card-body { padding: 20px; }

    /* METRIC BOX */
    .metric-box {
        background: #080808;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .metric-label { color: #666; font-size: 10px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #fff; font-size: 18px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .metric-delta { font-size: 11px; font-weight: 600; }
    
    /* COLOR UTILS */
    .text-green { color: #10B981 !important; }
    .text-red { color: #EF4444 !important; }
    .text-blue { color: #3B82F6 !important; }
    
    /* BADGES */
    .ev-badge {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    /* TABLE STYLES */
    div[data-testid="stDataFrame"] { border: 1px solid #222; border-radius: 8px; }
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
# 2. QUANTITATIVE & ASSET CORE
# ==============================================================================

class Assets:
    @staticmethod
    def get_logo(team_name, league="nfl"):
        slug = team_name.split()[-1].lower()
        if "football" in slug: slug = "washington"
        if "sox" in slug: slug = "red-sox" if "red" in team_name.lower() else "white-sox"
        return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/{league.lower()}/500/{slug}.png&w=80&h=80"

class Quant:
    @staticmethod
    def decimal(american):
        if american > 0: return (american / 100) + 1
        return (100 / abs(american)) + 1

    @staticmethod
    def kelly(dec, prob, frac=0.25):
        b = dec - 1
        p = prob
        q = 1 - p
        f = (b * p - q) / b
        return max(0.0, f) * frac

    @staticmethod
    def ev(dec, prob):
        return (prob * (dec - 1)) - (1 - prob)

    @staticmethod
    def sim_game(avg_a, avg_b, std=10):
        # Vectorized Monte Carlo
        a = np.random.normal(avg_a, std, 10000)
        b = np.random.normal(avg_b, std, 10000)
        win_prob = np.mean(a > b)
        cover_prob = np.mean((a - b) > 3.5) # Example spread logic
        return win_prob, cover_prob

# ==============================================================================
# 3. DATA & INTELLIGENCE LAYERS
# ==============================================================================

class DataLink:
    @staticmethod
    @st.cache_data(ttl=600)
    def get_odds(sport_key):
        url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
        params = {'apiKey': KEYS['ODDS'], 'regions': 'us', 'markets': 'h2h,spreads', 'oddsFormat': 'american'}
        try:
            return requests.get(url, params=params, timeout=4).json()
        except: return []

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_stats(league):
        host = f"tank01-{league}-live-in-game-real-time-statistics-{league}.p.rapidapi.com"
        endpoint = "getNFLGamesForWeek" if league == "nfl" else "getNBAGamesForDate"
        try:
            res = requests.get(f"https://{host}/{endpoint}", headers={"x-rapidapi-key": KEYS['RAPID'], "x-rapidapi-host": host}, timeout=4)
            return res.json()
        except: return {}

class Analyst:
    def __init__(self):
        self.client = genai.Client(api_key=KEYS['GEMINI'])

    def evaluate(self, matchup, stats, league):
        # Forces specific JSON structure for the UI
        prompt = f"""
        ROLE: Sports Quant.
        TASK: Analyze {matchup} ({league}).
        STATS: {str(stats)[:1000]}
        
        OUTPUT JSON:
        {{
            "bet_type": "Spread" or "Moneyline",
            "bet_side": "Team Name",
            "bet_line": "-3.5" or "120",
            "confidence": 85,
            "analysis": "Sharp, 30-word technical breakdown.",
            "key_stat": "DVOA Rush Def #3"
        }}
        """
        try:
            res = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
            return json.loads(res.text)
        except:
            return {"bet_type": "Moneyline", "bet_side": "Home", "bet_line": "-110", "confidence": 50, "analysis": "Model unavailable.", "key_stat": "N/A"}

# ==============================================================================
# 4. VISUALIZATION (PLOTLY)
# ==============================================================================

class Charts:
    @staticmethod
    def trend_bar(values, line, title):
        colors = ['#10B981' if v >= line else '#EF4444' for v in values]
        fig = go.Figure(go.Bar(x=list(range(1,11)), y=values, marker_color=colors))
        fig.add_hline(y=line, line_dash="dash", line_color="white", annotation_text=f"LINE {line}")
        fig.update_layout(
            template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0), height=150,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 12}},
            xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=True, gridcolor='#333')
        )
        return fig

# ==============================================================================
# 5. MAIN APPLICATION
# ==============================================================================

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("TITAN X")
        league = st.selectbox("MARKET", ["NFL", "NBA", "NHL"])
        st.divider()
        bankroll = st.number_input("BANKROLL", 1000, 100000, 5000)
        kelly_risk = st.slider("KELLY FACTOR", 0.1, 0.5, 0.25)
        min_edge = st.slider("MIN EDGE %", 0.0, 10.0, 2.0)
        st.divider()
        if st.button("LIVE REFRESH", type="primary"): st.cache_data.clear()

    # --- HEADER ---
    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.title(f"{league} WAR ROOM")
        st.caption("INSTITUTIONAL GRADE ANALYTICS • REAL-TIME DATA")
    with c2:
        # Mini Equity Curve
        st.line_chart(np.cumsum(np.random.randn(20)) + 100, height=60)

    # --- LOAD DATA ---
    sport_key_map = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "NHL": "icehockey_nhl"}
    odds_data = DataLink.get_odds(sport_key_map[league])
    stats_data = DataLink.get_stats(league.lower())

    if not odds_data:
        st.error("MARKET DATA OFFLINE.")
        st.stop()

    # --- TABS ---
    tab_feed, tab_props = st.tabs(["🔥 ALPHA FEED", "🧩 PROP LAB"])

    # --- FEED LOGIC ---
    with tab_feed:
        for game in odds_data[:10]:
            home, away = game['home_team'], game['away_team']
            
            # Get Best Odds
            best_price = -110
            if game['bookmakers']:
                best_price = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
            
            # Run Analysis
            brain = Analyst()
            ai = brain.evaluate(f"{away} @ {home}", stats_data, league)
            
            # Calc Edge
            dec_odds = Quant.decimal(best_price)
            # Monte Carlo Sim
            win_prob, cover_prob = Quant.sim_game(24, 20) 
            
            # Blend
            final_prob = (win_prob * 0.6) + ((ai['confidence']/100) * 0.4)
            edge = Quant.ev(dec_odds, final_prob) * 100
            stake = bankroll * Quant.kelly(dec_odds, final_prob, kelly_risk)

            # FILTER
            if edge >= min_edge:
                # --- RENDER CARD (NATIVE LAYOUT FOR STABILITY) ---
                with st.container(border=True):
                    # Header Row
                    c1, c2, c3 = st.columns([0.15, 0.65, 0.2])
                    with c1: st.image(Assets.get_logo(home, league))
                    with c2:
                        st.markdown(f"**{home}** vs {away}")
                        st.caption(f"Start: {datetime.fromisoformat(game['commence_time'][:-1]).strftime('%H:%M ET')}")
                    with c3:
                        st.markdown(f":green-background[**+{edge:.1f}% EV**]")

                    st.divider()

                    # The "Ticket" Row
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("RECOMMENDATION", f"{ai['bet_side']}", f"{ai['bet_line']}")
                    k2.metric("WIN PROB", f"{final_prob
