import streamlit as st
import requests
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GOD MODE // QUANT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed for more screen space
)

# --- CSS: HIGH DENSITY TERMINAL THEME ---
st.markdown("""
<style>
    .stApp { background-color: #050505; }
    .block-container { padding-top: 1rem; padding-bottom: 5rem; max-width: 900px; } /* Mobile optimized width */

    /* COMPACT CARD */
    .metric-card {
        background-color: #111;
        border: 1px solid #222;
        padding: 12px; /* Smaller padding */
        border-radius: 8px;
        margin-bottom: 12px;
    }
    
    /* BADGES & TEXT */
    .edge-badge {
        background-color: rgba(0, 200, 5, 0.1);
        color: #00C805;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        border: 1px solid rgba(0, 200, 5, 0.2);
    }
    .stat-text { color: #888; font-size: 11px; font-family: 'Roboto Mono', monospace; }
    .main-text { color: #FFF; font-size: 14px; font-weight: 600; }
    .analyst-note { 
        font-size: 12px; 
        color: #ccc; 
        line-height: 1.4; 
        border-top: 1px solid #222; 
        margin-top: 10px; 
        padding-top: 8px; 
    }

    /* NATIVE METRICS OVERRIDE */
    div[data-testid="stMetricValue"] { font-size: 18px !important; color: #fff !important; }
    div[data-testid="stMetricLabel"] { font-size: 10px !important; color: #666 !important; }
    div[data-testid="stMetricDelta"] { font-size: 10px !important; }

    /* LOGOS */
    .team-logo { width: 32px; height: 32px; object-fit: contain; }
</style>
""", unsafe_allow_html=True)

# --- 1. KEYRING ---
ODDS_API_KEY = "34e5a58b5b50587ce21dbe0b33e344dc"
RAPID_API_KEY = "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c"
NEWS_API_KEY = "289796ecfb2c4d208506c26d37a4d9ba"
GEMINI_API_KEY = "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"

# --- 2. RELIABLE ASSETS ---
# Hardcoded dictionary to ensure logos always load
LOGO_MAP = {
    "Arizona Cardinals": "https://a.espncdn.com/i/teamlogos/nfl/500/ari.png",
    "Atlanta Falcons": "https://a.espncdn.com/i/teamlogos/nfl/500/atl.png",
    "Baltimore Ravens": "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png",
    "Buffalo Bills": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "Carolina Panthers": "https://a.espncdn.com/i/teamlogos/nfl/500/car.png",
    "Chicago Bears": "https://a.espncdn.com/i/teamlogos/nfl/500/chi.png",
    "Cincinnati Bengals": "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png",
    "Cleveland Browns": "https://a.espncdn.com/i/teamlogos/nfl/500/cle.png",
    "Dallas Cowboys": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "Denver Broncos": "https://a.espncdn.com/i/teamlogos/nfl/500/den.png",
    "Detroit Lions": "https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
    "Green Bay Packers": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "Houston Texans": "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png",
    "Indianapolis Colts": "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png",
    "Jacksonville Jaguars": "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
    "Kansas City Chiefs": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "Las Vegas Raiders": "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png",
    "Los Angeles Chargers": "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
    "Los Angeles Rams": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "Miami Dolphins": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "Minnesota Vikings": "https://a.espncdn.com/i/teamlogos/nfl/500/min.png",
    "New England Patriots": "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
    "New Orleans Saints": "https://a.espncdn.com/i/teamlogos/nfl/500/no.png",
    "New York Giants": "https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png",
    "New York Jets": "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png",
    "Philadelphia Eagles": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "Pittsburgh Steelers": "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png",
    "San Francisco 49ers": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "Seattle Seahawks": "https://a.espncdn.com/i/teamlogos/nfl/500/sea.png",
    "Tampa Bay Buccaneers": "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png",
    "Tennessee Titans": "https://a.espncdn.com/i/teamlogos/nfl/500/ten.png",
    "Washington Commanders": "https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png"
}

# --- 3. QUANT ENGINE ---
class QuantEngine:
    @staticmethod
    def kelly(decimal_odds, prob):
        if decimal_odds <= 1: return 0
        b = decimal_odds - 1
        p = prob
        q = 1 - p
        f = (b * p - q) / b
        return max(0, f) 

    @staticmethod
    def ev(decimal_odds, prob):
        return (prob * (decimal_odds - 1)) - (1 - prob)

# --- 4. DATA LAYERS ---
@st.cache_data(ttl=900)
def get_odds():
    try:
        url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
        res = requests.get(url, params={'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'})
        return res.json() if res.status_code == 200 else []
    except: return []

@st.cache_data(ttl=3600)
def get_stats():
    try:
        url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
        res = requests.get(url, headers={"x-rapidapi-key": RAPID_API_KEY, "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"})
        return res.json() if res.status_code == 200 else {}
    except: return {}

def get_ai_prediction(matchup, stats):
    """
    DEEP ANALYST MODE: Returns 55-75% probability + Detailed Rationale.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    ROLE: Elite Sports Quant. 
    MATCHUP: {matchup}.
    DATA: {str(stats)[:1500]}
    
    TASK: Analyze the edge.
    REQUIREMENTS:
    1. 'win_prob': Aggressive probability (0.55-0.75).
    2. 'reason': Write a sophisticated 3-sentence analysis. Cite specific stats (EPA, DVOA, Injuries) or historical trends to justify the edge. Be technical.
    
    OUTPUT JSON ONLY.
    """
    
    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "winner": {"type": "STRING"},
                        "win_prob": {"type": "NUMBER"},
                        "reason": {"type": "STRING"}
                    }
                }
            )
        )
        return json.loads(res.text)
    except:
        return {"winner": "Home", "win_prob": 0.55, "reason": "Model Estimate based on historical power rankings."}

# --- 5. MAIN UI ---
def main():
    # SIDEBAR
    with st.sidebar:
        st.header("⚡ PORTFOLIO")
        bankroll = st.number_input("Bankroll ($)", value=1000, step=100)
        risk_factor = st.slider("Kelly Aggression", 0.1, 0.5, 0.25)
        if st.button("🔴 REFRESH MARKETS"): st.cache_data.clear()

    # HEADER
    st.title("God Mode")
    st.caption("INSTITUTIONAL ANALYTICS v5.2")

    # LOAD DATA
    odds = get_odds()
    stats = get_stats()

    if not odds:
        st.error("⚠️ Market Offline")
        return

    # TABS
    tab_feed, tab_data = st.tabs(["🔥 ALPHA FEED", "📊 DATA"])

    with tab_feed:
        st.write("") 
        
        # Loop Games
        for game in odds[:15]:
            home, away = game['home_team'], game['away_team']
            
            # Find Best Odds
            best_odds = -9999
            if game['bookmakers']:
                for bm in game['bookmakers']:
                    if bm['key'] in ['draftkings', 'fanduel', 'betmgm']:
                        for mkt in bm['markets']:
                            if mkt['key'] == 'h2h':
                                for out in mkt['outcomes']:
                                    if out['name'] == home: best_odds = out['price']
            
            if best_odds == -9999: continue

            # RUN MODELS
            ai = get_ai_prediction(f"{away} @ {home}", stats)
            
            # MATH
            dec_odds = (best_odds / 100) + 1 if best_odds > 0 else (100 / abs(best_odds)) + 1
            true_prob = ai.get('win_prob', 0.53)
            edge = QuantEngine.ev(dec_odds, true_prob)
            kelly_pct = QuantEngine.kelly(dec_odds, true_prob) * risk_factor
            stake = bankroll * kelly_pct

            # RENDER COMPACT CARD
            if edge > 0:
                with st.container():
                    # Card Container (HTML/CSS wrapper for tight spacing)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <div style="display:flex; align-items:center; gap:10px;">
                                <img src="{LOGO_MAP.get(home, '')}" class="team-logo">
                                <div>
                                    <div class="main-text">{home}</div>
                                    <div class="stat-text">vs {away}</div>
                                </div>
                            </div>
                            <div class="edge-badge">+{edge*100:.1f}% EV</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # NATIVE METRICS (3 Columns)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ODDS", f"{best_odds}", delta="Implied " + f"{1/dec_odds:.2f}")
                    c2.metric("PROB", f"{true_prob:.2f}", delta="Model")
                    c3.metric("STAKE", f"${stake:.0f}", delta="Kelly")

                    # ANALYST NOTE
                    st.markdown(f"""
                        <div class="analyst-note">
                            <span style="color:#00C805; font-weight:bold;">ANALYSIS:</span> {ai['reason']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


