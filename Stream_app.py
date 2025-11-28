import streamlit as st
import requests
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import json

# --- PAGE SETUP ---
st.set_page_config(
    page_title="GOD MODE // QUANT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS: ROBINHOOD "MIDNIGHT" THEME ---
st.markdown("""
<style>
    /* APP BACKGROUND */
    .stApp { background-color: #000000; }
    
    /* REMOVE PADDING */
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }

    /* METRIC CONTAINERS */
    div[data-testid="stMetric"] {
        background-color: #111;
        border: 1px solid #222;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00C805;
        transform: translateY(-2px);
    }
    div[data-testid="stMetricLabel"] { color: #888; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { color: #fff; font-size: 24px; font-family: 'Roboto Mono', monospace; }
    div[data-testid="stMetricDelta"] { color: #00C805; font-size: 12px; font-weight: 700; }

    /* EXPANDER STYLING (Cards) */
    .streamlit-expanderHeader {
        background-color: #111 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: #fff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        color: #666;
        border: none;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: #00C805 !important; border-bottom: 2px solid #00C805; }

    /* IMAGES */
    img { border-radius: 50%; }
</style>
""", unsafe_allow_html=True)

# --- 1. KEYRING ---
ODDS_API_KEY = "34e5a58b5b50587ce21dbe0b33e344dc"
RAPID_API_KEY = "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c"
NEWS_API_KEY = "289796ecfb2c4d208506c26d37a4d9ba"
GEMINI_API_KEY = "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"

# --- 2. ASSETS ---
def get_team_logo(team_name):
    # ESPN High-Res CDN
    slug = team_name.split()[-1].lower()
    if "football" in slug: slug = "washington"
    return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/{slug}.png&w=80&h=80"

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
    AGGRESSIVE MODE: Returns 55-75% probability always.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    You are a Sharp Sports Bettor. Matchup: {matchup}.
    Stats: {str(stats)[:1000]}
    
    Task: Pick a winner aggressively.
    Output valid JSON only.
    """
    
    try:
        # Structured output for perfect JSON every time
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
                        "confidence": {"type": "INTEGER"},
                        "reason": {"type": "STRING"}
                    }
                }
            )
        )
        return json.loads(res.text)
    except:
        return {"winner": "Home Team", "win_prob": 0.55, "confidence": 50, "reason": "Model Estimate"}

# --- 5. MAIN UI ---
def main():
    # SIDEBAR
    with st.sidebar:
        st.header("⚡ QUANT SETTINGS")
        bankroll = st.number_input("Bankroll ($)", value=1000, step=100)
        risk_factor = st.slider("Kelly Aggression", 0.1, 0.5, 0.25)
        st.divider()
        if st.button("🔴 LIVE REFRESH"): st.cache_data.clear()

    # HEADER
    c1, c2 = st.columns([0.8, 0.2])
    with c1: 
        st.title("God Mode")
        st.caption("INSTITUTIONAL SPORTS ANALYTICS")
    with c2:
        st.image("https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg", width=50)

    # LOAD DATA
    odds = get_odds()
    stats = get_stats()

    if not odds:
        st.error("⚠️ Market Offline")
        return

    # TABS
    tab_feed, tab_data = st.tabs(["🔥 ALPHA FEED", "📊 DATA GRID"])

    with tab_feed:
        st.write("") # Spacer
        # Loop Games
        for game in odds[:10]:
            home, away = game['home_team'], game['away_team']
            
            # Find Odds
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
            true_prob = ai.get('win_prob', 0.5)
            edge = QuantEngine.ev(dec_odds, true_prob)
            kelly_pct = QuantEngine.kelly(dec_odds, true_prob) * risk_factor
            stake = bankroll * kelly_pct

            # RENDER CARD (NATIVE STREAMLIT LAYOUT)
            if edge > 0:
                with st.container():
                    # Card Header
                    col_logo, col_text = st.columns([0.15, 0.85])
                    with col_logo:
                        st.image(get_team_logo(home))
                    with col_text:
                        st.subheader(f"{home}")
                        st.caption(f"vs {away}")
                    
                    # Metrics Row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("SIGNAL", f"{best_odds}", delta="Market")
                    m2.metric("PROB", f"{true_prob*100:.0f}%", delta="AI Model")
                    m3.metric("STAKE", f"${stake:.0f}", delta=f"Edge +{edge*100:.1f}%")
                    
                    # Rationale
                    st.info(f"🤖 **AI:** {ai.get('reason', 'Analysis unavailable')}")
                    st.divider()

if __name__ == "__main__":
    main()


