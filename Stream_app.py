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
    page_title="TITAN X PRO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME ENGINE: "VOID TERMINAL" ---
st.markdown("""
<style>
    /* CORE VARIABLES */
    :root {
        --bg: #000000;
        --card-bg: #0A0A0A;
        --border: #1F1F1F;
        --accent: #DFFF00; /* Cyber Yellow */
        --success: #00FF41; /* Matrix Green */
        --danger: #FF2A6D; /* Neon Red */
        --text: #EDEDED;
        --font: 'JetBrains Mono', monospace;
    }

    /* GLOBAL OVERRIDES */
    .stApp { background-color: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
    
    /* METRIC CARDS - The "Outlier" Look */
    div[data-testid="stMetric"] {
        background-color: #111;
        border-left: 2px solid #333;
        padding: 10px 15px;
        border-radius: 0px 8px 8px 0px;
    }
    div[data-testid="stMetric"]:hover {
        border-left: 2px solid var(--accent);
    }
    div[data-testid="stMetricLabel"] {
        color: #666;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        font-family: var(--font);
        font-size: 24px;
        color: #fff;
    }

    /* ACTION BUTTONS */
    .stButton>button {
        background: #161616;
        color: var(--accent);
        border: 1px solid #333;
        border-radius: 4px;
        font-family: var(--font);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: var(--accent);
        color: #000;
        border-color: var(--accent);
        box-shadow: 0 0 15px rgba(223, 255, 0, 0.3);
    }

    /* CUSTOM CONTAINERS */
    .game-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 10px;
        border-bottom: 1px solid #222;
        margin-bottom: 10px;
    }
    .bet-ticket {
        background: rgba(0, 255, 65, 0.05);
        border: 1px solid rgba(0, 255, 65, 0.2);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #222; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background: transparent;
        border: none;
        color: #666;
        font-family: var(--font);
        font-size: 12px;
    }
    .stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent); }
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
# 2. QUANT ENGINE (The Math Core)
# ==============================================================================

class QuantEngine:
    @staticmethod
    def get_implied_prob(odds):
        """Converts American Odds to Implied Probability."""
        if odds > 0: return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def get_decimal_odds(odds):
        """Converts American to Decimal."""
        if odds > 0: return 1 + (odds / 100)
        return 1 + (100 / abs(odds))

    @staticmethod
    def kelly_stake(odds, prob, bankroll, fraction=0.25):
        """Calculates Dollar Stake using Fractional Kelly."""
        decimal = QuantEngine.get_decimal_odds(odds)
        b = decimal - 1
        q = 1 - prob
        f = (b * prob - q) / b
        return max(0, f) * fraction * bankroll

    @staticmethod
    def calculate_edge(odds, true_prob):
        """Returns ROI % based on Model vs Market."""
        implied = QuantEngine.get_implied_prob(odds)
        return (true_prob - implied) * 100

# ==============================================================================
# 3. DATA ENGINE (Ingestion)
# ==============================================================================

class DataEngine:
    @staticmethod
    @st.cache_data(ttl=600)
    def fetch_market_data(sport):
        """Fetch live lines."""
        sport_keys = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "NHL": "icehockey_nhl"}
        url = f'https://api.the-odds-api.com/v4/sports/{sport_keys[sport]}/odds'
        try:
            res = requests.get(url, params={'apiKey': KEYS['ODDS'], 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'})
            return res.json() if res.status_code == 200 else []
        except: return []

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_stats(sport):
        """Fetch team stats."""
        host = f"tank01-{sport.lower()}-live-in-game-real-time-statistics-{sport.lower()}.p.rapidapi.com"
        endpoint = "getNFLGamesForWeek" if sport == "NFL" else f"get{sport}GamesForDate"
        try:
            res = requests.get(f"https://{host}/{endpoint}", headers={"x-rapidapi-key": KEYS['RAPID'], "x-rapidapi-host": host})
            return res.json() if res.status_code == 200 else {}
        except: return {}

# ==============================================================================
# 4. INTELLIGENCE ENGINE (AI)
# ==============================================================================

class AIEngine:
    def __init__(self):
        self.client = genai.Client(api_key=KEYS['GEMINI'])

    def handicap_game(self, matchup, stats, sport):
        """
        Forces a binary decision with reasoning.
        """
        prompt = f"""
        ROLE: Lead Sports Handicapper.
        TASK: Handicap {matchup} ({sport}).
        DATA: {str(stats)[:1500]}
        
        OUTPUT JSON:
        {{
            "pick_team": "Team Name",
            "confidence": 85,
            "reason": "3 bullet points on key mismatch (DVOA, Injury, Rest).",
            "key_stat": "Rush Yds/Att",
            "stat_val": "5.2"
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
            return {"pick_team": "Home Team", "confidence": 50, "reason": "Data Unavailable", "key_stat": "N/A", "stat_val": "0"}

# ==============================================================================
# 5. UI ENGINE (Visuals)
# ==============================================================================

class UI:
    @staticmethod
    def render_logo(team):
        slug = team.split()[-1].lower()
        if "football" in slug: slug = "washington"
        return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/{slug}.png&w=64&h=64"

    @staticmethod
    def trend_chart(values, line):
        colors = ['#00FF41' if v > line else '#333' for v in values]
        fig = go.Figure(go.Bar(x=list(range(1,11)), y=values, marker_color=colors))
        fig.add_hline(y=line, line_color="white", line_dash="dash")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=0,t=0,b=0),
            height=100,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==============================================================================
# 6. MAIN APPLICATION
# ==============================================================================

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("TITAN X")
        st.caption("PRO // v9.2")
        sport = st.selectbox("MARKET", ["NFL", "NBA", "NHL"])
        st.divider()
        bankroll = st.number_input("BANKROLL", value=10000, step=1000)
        kelly = st.slider("RISK FACTOR", 0.1, 0.5, 0.25)
        min_edge = st.slider("MIN EDGE", 0.0, 10.0, 1.5)
        st.divider()
        if st.button("⚡ SYSTEM REBOOT"): st.cache_data.clear()

    # --- DASHBOARD HEADER ---
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        st.title(f"{sport} WAR ROOM")
        st.markdown("INSTITUTIONAL ANALYTICS • LIVE ODDS • AI CONSENSUS")
    with c2:
        # P&L Chart
        chart_data = pd.DataFrame(np.cumsum(np.random.randn(30)) + 100, columns=['PnL'])
        st.line_chart(chart_data, height=80, color="#DFFF00")

    # --- LOAD DATA ---
    with st.spinner("Fetching Global Liquidity..."):
        odds = DataEngine.fetch_market_data(sport)
        stats = DataEngine.fetch_stats(sport)
        
    if not odds:
        st.error("MARKET OFFLINE.")
        st.stop()

    # --- FEED ---
    tab_feed, tab_props = st.tabs(["🔥 ALPHA FEED", "📊 PROP LAB"])

    with tab_feed:
        st.write("")
        
        for game in odds[:10]:
            home = game['home_team']
            away = game['away_team']
            
            # GET ODDS
            best_odds = -110
            if game['bookmakers']:
                # Simple parser for best line
                best_odds = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
            
            # ANALYZE
            brain = AIEngine()
            ai = brain.handicap_game(f"{away} @ {home}", stats, sport)
            
            # MATH
            prob = ai.get('confidence', 50) / 100.0
            edge = QuantEngine.calculate_edge(best_odds, prob)
            stake = QuantEngine.kelly_stake(best_odds, prob, bankroll, kelly)

            if edge >= min_edge:
                # --- THE TICKET CARD ---
                with st.container(border=True):
                    # 1. HEADER
                    c1, c2, c3 = st.columns([0.1, 0.7, 0.2])
                    with c1: st.image(UI.render_logo(home), width=50)
                    with c2:
                        st.subheader(f"{home}")
                        st.caption(f"vs {away}")
                    with c3:
                        st.metric("EDGE", f"+{edge:.1f}%")

                    st.divider()

                    # 2. THE ACTION (The "Play")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("RECOMMENDED BET", f"{ai['pick_team'].upper()}", f"{best_odds}")
                    k2.metric("WIN PROB", f"{prob:.0%}", "Model")
                    k3.metric("KELLY STAKE", f"${stake:.0f}", "Size")

                    # 3. RATIONALE
                    st.success(f"💡 **THESIS:** {ai['reason']}")
                    
                    # 4. KEY STAT
                    st.caption(f"🔑 Key Metric: {ai.get('key_stat','N/A')} ({ai.get('stat_val','0')})")

    # --- PROP LAB ---
    with tab_props:
        st.markdown("### 🧩 PLAYER PERFORMANCE")
        
        # Mock Prop Data for visual demo
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### J. ALLEN (BUF) - PASS YDS")
            # 10 Game Trend
            UI.trend_chart(np.random.randint(200, 350, 10), 265.5)
        with c2:
            st.metric("LINE", "265.5", "-110")
            st.metric("L5 HIT RATE", "80%", "+30%")
            st.button("ADD TO SLIP")

if __name__ == "__main__":
    main()
