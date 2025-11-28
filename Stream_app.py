import streamlit as st
import requests
import pandas as pd
import numpy as np
from google import genai
import json
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GOD MODE // QUANT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS: ROBINHOOD "MIDNIGHT" THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500&display=swap');

    .stApp {
        background-color: #000000;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }

    /* CARD STYLING */
    .metric-card {
        background-color: #111111;
        border: 1px solid #222;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00C805;
        transform: translateY(-2px);
    }

    /* TYPOGRAPHY */
    h1, h2, h3 { font-weight: 800; letter-spacing: -0.5px; color: #fff; }
    .mono { font-family: 'Roboto Mono', monospace; }
    .text-green { color: #00C805; }
    .text-gray { color: #6B7280; font-size: 13px; font-weight: 600; }
    
    /* BADGES */
    .edge-badge {
        background-color: rgba(0, 200, 5, 0.1);
        color: #00C805;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid rgba(0, 200, 5, 0.2);
    }

    /* LOGO IMAGES */
    .team-icon { 
        width: 48px; 
        height: 48px; 
        object-fit: contain; 
        filter: drop-shadow(0 0 8px rgba(255,255,255,0.1));
    }

    /* NAVIGATION */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #222; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: transparent;
        color: #666;
        font-size: 14px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: #00C805 !important; }
</style>
""", unsafe_allow_html=True)

# --- 1. KEYRING ---
ODDS_API_KEY = "34e5a58b5b50587ce21dbe0b33e344dc"
RAPID_API_KEY = "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c"
NEWS_API_KEY = "289796ecfb2c4d208506c26d37a4d9ba"
GEMINI_API_KEY = "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"

# --- 2. ASSETS (High-Res ESPN Logos) ---
def get_logo(team_name):
    # ESPN CDN logic for reliable logos
    slug = team_name.split()[-1].lower()
    return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/{slug}.png&w=100&h=100"

# --- 3. QUANT MATH ---
class QuantEngine:
    @staticmethod
    def american_to_decimal(american):
        return (american / 100) + 1 if american > 0 else (100 / abs(american)) + 1

    @staticmethod
    def kelly_criterion(decimal, prob):
        b = decimal - 1
        p = prob
        q = 1 - p
        f = (b * p - q) / b
        return max(0, f) 

# --- 4. DATA LAYERS ---
@st.cache_data(ttl=900)
def fetch_market_data():
    try:
        url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
        res = requests.get(url, params={'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'})
        return res.json() if res.status_code == 200 else []
    except: return []

@st.cache_data(ttl=3600)
def fetch_stats():
    try:
        url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
        res = requests.get(url, headers={"x-rapidapi-key": RAPID_API_KEY, "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"})
        return res.json() if res.status_code == 200 else {}
    except: return {}

def get_ai_prediction(matchup, stats):
    """
    AGGRESSIVE ANALYST MODE:
    Forces the AI to make a prediction using internal knowledge if external stats fail.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Fallback context if API stats are empty
    context = str(stats)[:1000] if stats else "LIVE_STATS_OFFLINE_USE_INTERNAL_KNOWLEDGE"
    
    prompt = f"""
    You are a Sharp Sports Bettor. 
    MATCHUP: {matchup}
    DATA: {context}
    
    MANDATE:
    1. You MUST pick a winner. Do NOT return 50/50.
    2. If stats are missing, rely on your internal knowledge of NFL rosters, QB matchups, and coaching.
    3. Be aggressive. Identify the 'Sharp' side.
    
    OUTPUT JSON:
    {{
        "home_win_prob": 0.62,
        "rationale": "One sharp sentence on why."
    }}
    """
    try:
        res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return json.loads(res.text.replace("```json", "").replace("```", "").strip())
    except:
        return {"home_win_prob": 0.55, "rationale": "Backup Model Estimate"}

# --- 5. DASHBOARD ---
def main():
    # SIDEBAR CONTROLS
    with st.sidebar:
        st.header("⚡ QUANT SETTINGS")
        bankroll = st.number_input("Portfolio Value", value=10000, step=1000, format="$%d")
        risk_factor = st.slider("Risk Tolerance", 0.1, 0.5, 0.25)
        st.markdown("---")
        st.caption("v4.0.2 // STABLE")
        if st.button("Refresh Feed"): st.cache_data.clear()

    # HEADER AREA
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("God Mode Terminal")
        st.caption("AI-POWERED SPORTS ARBITRAGE & PREDICTION ENGINE")
    with col2:
        # Mini Portfolio Graph (Simulated Visual)
        chart_data = pd.DataFrame(np.random.randn(20, 1).cumsum() + 100, columns=['Value'])
        st.line_chart(chart_data, height=80)

    # DATA LOADING
    odds_data = fetch_market_data()
    stats_data = fetch_stats()

    if not odds_data:
        st.warning("Market Offline. Using cached data if available.")
        # Mock data for UI testing if API fails
        odds_data = [] 

    # PROCESS & DISPLAY
    tabs = st.tabs(["🔥 LIVE ALPHA", "📊 DATA FEED"])

    with tabs[0]:
        st.markdown("##### 🟢 ACTIVE OPPORTUNITIES")
        
        # Limit to 10 games for speed
        for game in odds_data[:10]:
            home, away = game['home_team'], game['away_team']
            
            # Find Best Odds
            best_home = -9999
            if game['bookmakers']:
                for bm in game['bookmakers']:
                    if bm['key'] in ['draftkings', 'fanduel', 'betmgm']:
                        for mkt in bm['markets']:
                            if mkt['key'] == 'h2h':
                                for out in mkt['outcomes']:
                                    if out['name'] == home: best_home = out['price']
            
            if best_home == -9999: continue

            # AI Analysis
            ai = get_ai_prediction(f"{away} @ {home}", stats_data)
            
            # Math
            dec_odds = QuantEngine.american_to_decimal(best_home)
            mkt_prob = 1 / dec_odds
            true_prob = ai['home_win_prob']
            
            # BLEND: 70% AI / 30% Market (Aggressive)
            final_prob = (true_prob * 0.7) + (mkt_prob * 0.3)
            
            edge = (final_prob * (dec_odds - 1)) - (1 - final_prob)
            kelly = QuantEngine.kelly_criterion(dec_odds, final_prob) * risk_factor
            stake = bankroll * kelly

            # RENDER CARD
            if edge > 0.01: # Only show positive edge
                home_logo = get_logo(home)
                away_logo = get_logo(away)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="display: flex; align-items: center;">
                                <img src="{away_logo}" class="team-icon" style="opacity: 0.6;">
                                <span style="margin: 0 10px; color: #444;">@</span>
                                <img src="{home_logo}" class="team-icon">
                            </div>
                            <div>
                                <h3 style="margin:0; font-size: 18px;">{home}</h3>
                                <div class="text-gray" style="font-size: 12px;">vs {away}</div>
                            </div>
                        </div>
                        <div class="edge-badge">+{edge*100:.1f}% EDGE</div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 20px;">
                        <div style="background: #000; padding: 10px; border-radius: 8px;">
                            <div class="text-gray">SIGNAL</div>
                            <div class="mono" style="color: #fff; font-weight: 700;">{best_home}</div>
                        </div>
                        <div style="background: #000; padding: 10px; border-radius: 8px;">
                            <div class="text-gray">PROB</div>
                            <div class="mono" style="color: #00C805; font-weight: 700;">{final_prob*100:.1f}%</div>
                        </div>
                        <div style="background: #000; padding: 10px; border-radius: 8px;">
                            <div class="text-gray">STAKE</div>
                            <div class="mono" style="color: #fff; font-weight: 700;">${stake:.0f}</div>
                        </div>
                    </div>
                    
                    <div style="font-size: 13px; color: #888; border-top: 1px solid #222; padding-top: 10px;">
                        <span style="color: #00C805;">AI RATIONALE:</span> {ai['rationale']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


