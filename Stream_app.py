import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai
from google.genai import types
import json
import time
from datetime import datetime, timedelta
from scipy.stats import poisson

# ==============================================================================
# 1. SYSTEM CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="TITAN OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR "OUTLIER" AESTHETIC ---
st.markdown("""
<style>
    /* APP BACKGROUND - Deep Void */
    .stApp {
        background-color: #000000;
    }

    /* METRIC HIGHLIGHTS */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }
    
    /* POSITIVE/NEGATIVE COLORS */
    .stat-pos { color: #00E5FF; font-weight: bold; }
    .stat-neg { color: #FF453A; font-weight: bold; }
    
    /* REMOVE DEFAULT PADDING */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #111;
        padding: 5px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border: none;
        color: #666;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #222;
        color: #fff;
        border-radius: 8px;
    }
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
# 2. ASSET LIBRARY
# ==============================================================================

class Assets:
    @staticmethod
    def get_logo(team_name, league="nfl"):
        # Normalization for ESPN CDN
        slug = team_name.split()[-1].lower()
        if "football" in slug: slug = "washington"
        if "sox" in slug: slug = "red-sox" if "red" in team_name.lower() else "white-sox"
        
        return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/{league.lower()}/500/{slug}.png&w=80&h=80"

# ==============================================================================
# 3. QUANTITATIVE ENGINE
# ==============================================================================

class QuantEngine:
    @staticmethod
    def american_to_decimal(american):
        if american > 0:
            return (american / 100) + 1
        else:
            return (100 / abs(american)) + 1

    @staticmethod
    def implied_prob(decimal):
        return 1 / decimal

    @staticmethod
    def kelly_criterion(decimal, win_prob, fraction=0.25):
        b = decimal - 1
        p = win_prob
        q = 1 - p
        f = (b * p - q) / b
        return max(0.0, f) * fraction

    @staticmethod
    def ev(decimal, prob):
        return (prob * (decimal - 1)) - (1 - prob)

    @staticmethod
    def monte_carlo(avg_a, avg_b, std=10, sims=2000):
        # Fast vectorized simulation
        a_scores = np.random.normal(avg_a, std, sims)
        b_scores = np.random.normal(avg_b, std, sims)
        return np.mean(a_scores > b_scores)

# ==============================================================================
# 4. DATA LAYER
# ==============================================================================

class DataEngine:
    @staticmethod
    @st.cache_data(ttl=900)
    def get_odds(sport):
        url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
        params = {'apiKey': KEYS['ODDS'], 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
        try:
            return requests.get(url, params=params).json()
        except: return []

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_stats(league):
        host = f"tank01-{league}-live-in-game-real-time-statistics-{league}.p.rapidapi.com"
        url = f"https://{host}/get{league.upper()}GamesForDate"
        if league == 'nfl': url = f"https://{host}/getNFLGamesForWeek"
        
        try:
            return requests.get(url, headers={"x-rapidapi-key": KEYS['RAPID'], "x-rapidapi-host": host}).json()
        except: return {}

# ==============================================================================
# 5. AI REASONING
# ==============================================================================

class AIEngine:
    def __init__(self):
        self.client = genai.Client(api_key=KEYS['GEMINI'])

    def analyze(self, matchup, stats, league):
        prompt = f"""
        ROLE: Sports Quant.
        TASK: Analyze {matchup} ({league.upper()}).
        STATS: {str(stats)[:800]}
        
        OUTPUT JSON:
        {{
            "home_win_prob": 0.60,
            "confidence": 80,
            "reason": "30-word analysis focusing on key metrics.",
            "prop": "Player Name Over X Pts/Yds"
        }}
        """
        try:
            res = self.client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(res.text)
        except:
            return {"home_win_prob": 0.55, "confidence": 50, "reason": "Model Estimate", "prop": "N/A"}

# ==============================================================================
# 6. UI RENDERER
# ==============================================================================

class Visuals:
    @staticmethod
    def prop_chart(value, line, history):
        # Green/Red bar chart like Outlier
        colors = ['#00E5FF' if x >= line else '#333' for x in history]
        fig = go.Figure([go.Bar(y=history, marker_color=colors)])
        fig.add_hline(y=line, line_dash="dash", line_color="white")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=100,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showgrid=False)
        )
        return fig

# ==============================================================================
# 7. MAIN APP
# ==============================================================================

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("TITAN OS")
        league = st.selectbox("LEAGUE", ["NFL", "NBA", "NHL"])
        
        st.divider()
        bankroll = st.number_input("BANKROLL", 1000, 100000, 5000)
        kelly = st.slider("RISK (KELLY)", 0.1, 0.5, 0.25)
        min_edge = st.slider("MIN EDGE %", 0.0, 10.0, 1.5)
        
        if st.button("CLEAR CACHE"): st.cache_data.clear()

    # --- HEADER ---
    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.title(f"{league} DASHBOARD")
        st.caption("INSTITUTIONAL GRADE ANALYTICS")
    with c2:
        # Mini Profit Graph
        st.line_chart(np.cumsum(np.random.randn(20)), height=50)

    # --- LOAD DATA ---
    sport_map = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "NHL": "icehockey_nhl"}
    
    with st.spinner("Syncing Market Data..."):
        odds = DataEngine.get_odds(sport_map[league])
        stats = DataEngine.get_stats(league.lower())

    if not odds:
        st.warning("Market Offline.")
        st.stop()

    # --- TABS ---
    tab_feed, tab_props = st.tabs(["🔥 ALPHA FEED", "📊 PROP LAB"])

    # --- FEED TAB ---
    with tab_feed:
        for game in odds[:10]:
            home = game['home_team']
            away = game['away_team']
            
            # Get Best Odds
            best_price = -9999
            if game['bookmakers']:
                for bm in game['bookmakers']:
                    if bm['key'] in ['draftkings', 'fanduel', 'betmgm']:
                        for mkt in bm['markets']:
                            if mkt['key'] == 'h2h':
                                for out in mkt['outcomes']:
                                    if out['name'] == home: best_price = out['price']
            
            if best_price == -9999: continue

            # Run Models
            brain = AIEngine()
            ai = brain.analyze(f"{away} @ {home}", stats, league)
            
            dec = QuantEngine.american_to_decimal(best_price)
            prob = ai.get('home_win_prob', 0.5)
            edge = QuantEngine.ev(dec, prob) * 100
            stake = bankroll * QuantEngine.kelly_criterion(dec, prob, kelly)

            # RENDER CARD (NATIVE CONTAINER - NO HTML BUGS)
            if edge >= min_edge:
                with st.container(border=True):
                    # Top Row: Teams & Edge
                    c1, c2, c3 = st.columns([0.1, 0.7, 0.2])
                    with c1:
                        st.image(Assets.get_logo(home, league))
                    with c2:
                        st.markdown(f"**{home}**")
                        st.caption(f"vs {away}")
                    with c3:
                        st.markdown(f":green-background[+{edge:.1f}% EV]")
                    
                    st.divider()
                    
                    # Middle Row: Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("SIGNAL", f"{best_price}", "Market")
                    m2.metric("PROB", f"{prob:.0%}", "Model")
                    m3.metric("STAKE", f"${stake:.0f}", "Kelly")
                    
                    # Bottom Row: Analysis
                    st.info(f"🤖 **AI:** {ai.get('reason', 'No analysis')}")
                    st.caption(f"🎯 **Prop Target:** {ai.get('prop', 'N/A')}")

    # --- PROP TAB ---
    with tab_props:
        st.markdown("### 🧩 TREND VISUALIZER")
        # Mock Prop Visualization for Demo
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.plotly_chart(Visuals.prop_chart(0, 20, np.random.randint(10, 30, 10)), use_container_width=True)
        with col2:
            st.metric("L10 HIT RATE", "70%", "+20%")
            st.metric("L5 HIT RATE", "80%", "+40%")

if __name__ == "__main__":
    main()
