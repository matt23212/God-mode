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
    page_title="GOD MODE // SAAS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS: PREMIUM SAAS THEME ---
st.markdown("""
<style>
    /* IMPORT INTER FONT */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* APP BACKGROUND */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }

    /* CARD STYLING */
    .css-1r6slb0, div[data-testid="stExpander"] {
        background-color: #1C1F26;
        border: 1px solid #2C303A;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }

    /* METRICS - SAAS STYLE */
    div[data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 28px;
        color: #FFFFFF;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: #9CA3AF;
        font-weight: 500;
    }

    /* HEADERS */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* ACCENT COLORS (Robinhood Green) */
    .highlight-green {
        color: #00C805;
        font-weight: 600;
    }
    
    /* DATAFRAMES */
    div[data-testid="stDataFrame"] {
        background-color: #1C1F26;
        border-radius: 8px;
        border: 1px solid #2C303A;
    }

    /* BUTTONS */
    .stButton>button {
        background-color: #00C805;
        color: #000000;
        border: none;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #00E006;
        box-shadow: 0 0 10px rgba(0, 200, 5, 0.4);
        color: #000000;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        color: #9CA3AF;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1C1F26;
        color: #00C805;
        border: 1px solid #2C303A;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. HARDCODED KEYRING (Your Keys) ---
ODDS_API_KEY = "34e5a58b5b50587ce21dbe0b33e344dc"
RAPID_API_KEY = "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c"
NEWS_API_KEY = "289796ecfb2c4d208506c26d37a4d9ba"
GEMINI_API_KEY = "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"

# --- 2. QUANT ENGINE (THE MATH) ---

class QuantEngine:
    @staticmethod
    def american_to_decimal(american_odds):
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def implied_prob(decimal_odds):
        return 1 / decimal_odds

    @staticmethod
    def kelly_criterion(decimal_odds, win_prob):
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        f_star = (b * p - q) / b
        return max(0, f_star) 

    @staticmethod
    def calculate_ev(decimal_odds, win_prob):
        return (win_prob * (decimal_odds - 1)) - (1 - win_prob)

# --- 3. DATA INGESTION LAYERS ---

@st.cache_data(ttl=900)
def fetch_market_data(sport_key='americanfootball_nfl'):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
    try:
        return requests.get(url, params=params).json()
    except:
        return []

@st.cache_data(ttl=3600)
def fetch_team_stats():
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
    headers = {"x-rapidapi-key": RAPID_API_KEY, "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"}
    try:
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def get_ai_prediction(matchup_str, stats_context):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    You are a Quantitative Sports Analyst for a hedge fund.
    Matchup: {matchup_str}
    Context: {str(stats_context)[:1000]}
    
    TASK: Return a strictly formatted JSON object with your win probability for the Home Team.
    
    JSON FORMAT:
    {{
        "home_win_prob": 0.65,
        "rationale": "One brief sentence explaining the edge.",
        "prop_idea": "Best single player prop."
    }}
    """
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return {"home_win_prob": 0.50, "rationale": "Model Unavailable", "prop_idea": "N/A"}

# --- 4. THE DASHBOARD ---

def main():
    # Sidebar
    st.sidebar.title("⚡ SETTINGS")
    st.sidebar.markdown("---")
    bankroll = st.sidebar.number_input("Portfolio Value ($)", value=1000, step=100)
    kelly_fraction = st.sidebar.slider("Kelly Aggression", 0.1, 1.0, 0.25)
    st.sidebar.info("v3.0.1 // SAAS BUILD")

    # Main Header
    st.title("God Mode Terminal")
    st.markdown("##### 🚀 AI-POWERED QUANTITATIVE BETTING ENGINE")
    st.markdown("---")
    
    # Top Stats Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Markets", "NFL", delta="Active", delta_color="normal")
    c2.metric("Model Confidence", "Gemini 2.0", delta="High")
    c3.metric("Bankroll Strategy", "Fractional Kelly")
    if c4.button("↻ Sync Markets"):
        st.cache_data.clear()

    st.write("") # Spacer

    tab1, tab2, tab3 = st.tabs(["🔥 Alpha Feed", "📊 Market Data", "🧩 Prop Intelligence"])

    # --- LOAD DATA ---
    with st.spinner("Analyzing spreads, DVOA, and injuries..."):
        odds_data = fetch_market_data()
        stats_data = fetch_team_stats()

    if not odds_data:
        st.error("⚠️ Market Data Unavailable. Check API Quotas.")
        return

    # --- PROCESSING ---
    market_rows = []
    
    # Limit to 8 games to keep it fast
    for game in odds_data[:8]: 
        home = game['home_team']
        away = game['away_team']
        
        # Best Odds Logic
        best_home = -9999
        if game['bookmakers']:
            for bm in game['bookmakers']:
                if bm['key'] in ['draftkings', 'fanduel', 'betmgm']:
                    for mkt in bm['markets']:
                        if mkt['key'] == 'h2h':
                            for outcome in mkt['outcomes']:
                                if outcome['name'] == home: best_home = outcome['price']
        
        if best_home == -9999: continue

        # AI & Quant Logic
        ai_data = get_ai_prediction(f"{away} @ {home}", stats_data)
        dec_odds = QuantEngine.american_to_decimal(best_home)
        true_prob = ai_data['home_win_prob']
        ev = QuantEngine.calculate_ev(dec_odds, true_prob)
        kelly_pct = QuantEngine.kelly_criterion(dec_odds, true_prob) * kelly_fraction
        
        market_rows.append({
            "Game": f"{away} @ {home}",
            "Odds": best_home,
            "True Prob": true_prob,
            "EV": ev,
            "Kelly": kelly_pct,
            "Rationale": ai_data['rationale'],
            "Prop Idea": ai_data['prop_idea']
        })

    df = pd.DataFrame(market_rows)

    # --- TAB 1: ALPHA FEED (The "Robinhood" View) ---
    with tab1:
        # Filter for +EV
        opportunities = df[df['EV'] > 0].copy()
        
        if not opportunities.empty:
            for idx, row in opportunities.iterrows():
                bet_amt = bankroll * row['Kelly']
                
                # HTML Card for "Robinhood" feel
                st.markdown(f"""
                <div style="background-color: #1C1F26; padding: 20px; border-radius: 12px; border: 1px solid #2C303A; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #fff;">{row['Game']}</h3>
                        <span style="background-color: rgba(0, 200, 5, 0.15); color: #00C805; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 14px;">
                            +{row['EV']*100:.1f}% EDGE
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 15px; color: #9CA3AF; font-size: 14px;">
                        <div>TARGET BET</div>
                        <div>PROBABILITY</div>
                        <div>KELLY STAKE</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: 600; color: #fff; font-family: monospace;">
                        <div>Home ({row['Odds']})</div>
                        <div>{row['True Prob']*100:.1f}%</div>
                        <div style="color: #00C805;">${bet_amt:.2f}</div>
                    </div>
                    <hr style="border-color: #333; margin: 15px 0;">
                    <p style="color: #E5E7EB; font-size: 14px; margin: 0;"><i>"{row['Rationale']}"</i></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No pure Alpha detected. Markets are efficient right now.")

    # --- TAB 2: MARKET DATA (The "Terminal" View) ---
    with tab2:
        st.dataframe(
            df[['Game', 'Odds', 'True Prob', 'EV', 'Kelly']],
            use_container_width=True,
            column_config={
                "True Prob": st.column_config.ProgressColumn("Win Probability", format="%.2f", min_value=0, max_value=1),
                "EV": st.column_config.NumberColumn("Expected Value", format="%.2f"),
                "Kelly": st.column_config.NumberColumn("Alloc %", format="%.3f")
            }
        )

    # --- TAB 3: PROP INTELLIGENCE ---
    with tab3:
        for idx, row in df.iterrows():
            with st.expander(f"🧩 {row['Game']} - Props"):
                st.markdown(f"**AI Recommendation:** `{row['Prop Idea']}`")
                st.caption(f"Based on stats context and defensive matchups.")

if __name__ == "__main__":
    main()


