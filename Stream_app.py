import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# --- CUSTOM CSS: PREMIUM SAAS THEME ---
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background-color: #0A0C10; /* Deep Midnight */
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #11141D;
        border-right: 1px solid #1E293B;
    }

    /* CARDS & CONTAINERS */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* DATAFRAMES */
    div[data-testid="stDataFrame"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        font-family: 'JetBrains Mono', monospace;
    }

    /* TEXT STYLES */
    h1, h2, h3 {
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .highlight-green {
        color: #4ADE80; /* Neon Green */
        font-weight: 700;
    }
    .highlight-red {
        color: #F87171; /* Soft Red */
        font-weight: 600;
    }
    
    /* BUTTONS */
    .stButton>button {
        background-color: #238636; /* GitHub Green */
        color: white;
        border: none;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2EA043;
        box-shadow: 0 0 8px rgba(46, 160, 67, 0.4);
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        color: #94A3B8;
        font-weight: 600;
        border-radius: 6px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B;
        color: #4ADE80;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. KEYRING (Hardcoded for your deployment) ---
ODDS_API_KEY = "34e5a58b5b50587ce21dbe0b33e344dc"
RAPID_API_KEY = "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c"
NEWS_API_KEY = "289796ecfb2c4d208506c26d37a4d9ba"
GEMINI_API_KEY = "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"

# --- 2. QUANTITATIVE CORE ---

class QuantEngine:
    @staticmethod
    def american_to_decimal(american_odds):
        """Converts American odds (-110) to Decimal (1.91)"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_implied(decimal_odds):
        """Converts Decimal (2.0) to Probability (0.50)"""
        return 1 / decimal_odds

    @staticmethod
    def kelly_criterion(decimal_odds, true_win_prob):
        """
        Calculates Optimal Kelly Stake %
        f* = (bp - q) / b
        """
        b = decimal_odds - 1 # Net odds
        p = true_win_prob
        q = 1 - p
        
        f_star = (b * p - q) / b
        return max(0, f_star) # No negative bets

    @staticmethod
    def calculate_ev(decimal_odds, true_win_prob):
        """Expected Value %: (Prob_Win * Profit) - (Prob_Lose * Stake)"""
        return (true_win_prob * (decimal_odds - 1)) - (1 - true_win_prob)

# --- 3. DATA LAYERS ---

@st.cache_data(ttl=600) # 10 min cache
def fetch_odds():
    """Ingests live lines from Vegas."""
    url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
    try:
        return requests.get(url, params=params).json()
    except:
        return []

@st.cache_data(ttl=3600)
def fetch_stats():
    """Ingests Team Records & Stats."""
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
    headers = {"x-rapidapi-key": RAPID_API_KEY, "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers)
        return res.json() if res.status_code == 200 else {}
    except:
        return {}

def get_ai_analysis(matchup, stats):
    """
    The Brain: Gemini 2.0 Flash
    Generates a probabilistic model for the game based on available data.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    ROLE: Elite Sports Handicapper & Data Scientist.
    TASK: Analyze this NFL matchup and output a JSON probability model.
    MATCHUP: {matchup}
    STATS DUMP: {str(stats)[:800]}
    
    REQUIREMENTS:
    1. Calculate 'True Win Probability' for the HOME TEAM (0.00 to 1.00).
    2. Identify the single best Player Prop (e.g. 'Mahomes Over 250.5 yds').
    3. Keep rationale extremely concise and data-driven.
    
    OUTPUT JSON ONLY:
    {{
        "home_win_prob": 0.65,
        "rationale": "Chiefs passing DVOA ranks #1 vs weak secondary.",
        "prop_bet": "Player Name - Over/Under X Stat"
    }}
    """
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return {"home_win_prob": 0.50, "rationale": "Data insufficient", "prop_bet": "N/A"}

# --- 4. MAIN APPLICATION ---

def main():
    # SIDEBAR CONTROLS
    with st.sidebar:
        st.header("⚡ QUANT CONTROLS")
        st.markdown("---")
        bankroll = st.number_input("Total Bankroll ($)", value=5000, step=500)
        kelly_fraction = st.slider("Kelly Multiplier", 0.1, 1.0, 0.25, help="Standard: 0.25. Aggressive: 0.5.")
        min_edge = st.slider("Min Edge %", 0.0, 10.0, 1.5, help="Only show bets with this much positive EV.")
        
        st.markdown("---")
        st.caption(f"Status: **ONLINE**")
        st.caption(f"Model: **Gemini 2.0 Flash**")
        if st.button("↻ Flush Cache"):
            st.cache_data.clear()

    # MAIN DASHBOARD
    st.title("GOD MODE // TERMINAL")
    st.markdown("##### 🚀 INSTITUTIONAL GRADE SPORTS ANALYTICS")
    st.markdown("---")

    # LOAD DATA
    with st.spinner("Initializing Quant Engine..."):
        odds_data = fetch_odds()
        stats_data = fetch_stats()

    if not odds_data:
        st.error("Market Data Offline. Check API Connection.")
        return

    # PROCESSING LOOP
    market_rows = []
    
    # Analyze Top 8 Games (For speed)
    for game in odds_data[:8]:
        home = game['home_team']
        away = game['away_team']
        
        # 1. Get Best Market Odds
        best_home = -9999
        if game['bookmakers']:
            for bm in game['bookmakers']:
                if bm['key'] in ['draftkings', 'fanduel']:
                    for mkt in bm['markets']:
                        if mkt['key'] == 'h2h':
                            for outcome in mkt['outcomes']:
                                if outcome['name'] == home: best_home = outcome['price']
        
        if best_home == -9999: continue

        # 2. Run AI Model
        ai_res = get_ai_analysis(f"{away} @ {home}", stats_data)
        
        # 3. Run Quant Math
        dec_odds = QuantEngine.american_to_decimal(best_home)
        true_prob = ai_res['home_win_prob']
        ev = QuantEngine.calculate_ev(dec_odds, true_prob)
        kelly = QuantEngine.kelly_criterion(dec_odds, true_prob) * kelly_fraction
        
        market_rows.append({
            "Matchup": f"{away} @ {home}",
            "Odds": best_home,
            "Model Prob": true_prob,
            "Edge": ev * 100, # Percentage
            "Kelly": kelly,
            "Stake": bankroll * kelly,
            "Rationale": ai_res['rationale'],
            "Prop": ai_res['prop_bet']
        })

    df = pd.DataFrame(market_rows)

    # TABS FOR DIFFERENT VIEWS
    tab_alpha, tab_props, tab_parlay = st.tabs(["🔥 ALPHA FEED", "🧩 PROP ENGINE", "🔗 PARLAY LAB"])

    # --- TAB 1: ALPHA FEED (High Value Bets) ---
    with tab_alpha:
        # Filter by Min Edge
        opportunities = df[df['Edge'] >= min_edge].copy()
        
        if not opportunities.empty:
            for idx, row in opportunities.iterrows():
                # Render "Card" View
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin:0; color: #E2E8F0;">{row['Matchup']}</h3>
                        <span style="background: rgba(74, 222, 128, 0.1); color: #4ADE80; padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 14px;">
                            +{row['Edge']:.1f}% EDGE
                        </span>
                    </div>
                    <div style="display: flex; gap: 20px; margin-top: 15px;">
                        <div>
                            <div style="color: #94A3B8; font-size: 12px;">TARGET BET</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px; color: #fff;">Home ({row['Odds']})</div>
                        </div>
                        <div>
                            <div style="color: #94A3B8; font-size: 12px;">MODEL PROB</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px; color: #fff;">{row['Model Prob']*100:.1f}%</div>
                        </div>
                        <div>
                            <div style="color: #94A3B8; font-size: 12px;">KELLY STAKE</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px; color: #4ADE80;">${row['Stake']:.0f}</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #333; color: #CBD5E1; font-size: 14px;">
                        <i>"{row['Rationale']}"</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No opportunities found with >{min_edge}% Edge. Adjust filters in sidebar.")

    # --- TAB 2: PROP ENGINE ---
    with tab_props:
        st.dataframe(
            df[['Matchup', 'Prop', 'Rationale']],
            use_container_width=True,
            column_config={
                "Matchup": st.column_config.TextColumn("Game", width="medium"),
                "Prop": st.column_config.TextColumn("AI Recommendation", width="large"),
                "Rationale": st.column_config.TextColumn("Logic", width="large"),
            },
            hide_index=True
        )

    # --- TAB 3: PARLAY LAB ---
    with tab_parlay:
        # Find top 2 highest probability wins
        top_picks = df.sort_values(by='Model Prob', ascending=False).head(2)
        
        if len(top_picks) >= 2:
            g1 = top_picks.iloc[0]
            g2 = top_picks.iloc[1]
            
            comb_prob = g1['Model Prob'] * g2['Model Prob']
            fair_odds = int((1/comb_prob - 1) * 100)
            
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h2 style="color: #4ADE80; margin-bottom: 20px;">🚀 QUANT DOUBLE OF THE DAY</h2>
                <div style="font-size: 20px; font-weight: 700; margin-bottom: 10px;">
                    {g1['Matchup'].split('@')[1]} (ML)  +  {g2['Matchup'].split('@')[1]} (ML)
                </div>
                <div style="color: #94A3B8;">Combined Win Probability: <b style="color: #fff">{comb_prob*100:.1f}%</b></div>
                <div style="color: #94A3B8;">Target Odds: <b style="color: #fff">+{fair_odds}</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient data to generate a safe parlay.")

if __name__ == "__main__":
    main()


