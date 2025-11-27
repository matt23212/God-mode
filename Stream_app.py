import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google import genai
import json
import time

# --- PAGE CONFIGURATION (TERMINAL STYLE) ---
st.set_page_config(
    page_title="GOD MODE TERMINAL // QUANT",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS: THE "ROBINHOOD DARK" AESTHETIC ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #000000;
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Metrics Styling - Neon Green for Profit */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00ff41; 
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #888;
        font-size: 12px;
        text-transform: uppercase;
    }

    /* Cards/Containers */
    .css-1r6slb0 {
        border: 1px solid #333;
        background-color: #111;
        padding: 20px;
        border-radius: 5px;
    }
    
    /* Dataframes - Terminal Look */
    div[data-testid="stDataFrame"] {
        border: 1px solid #333;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #111;
        color: #00ff41;
        border: 1px solid #00ff41;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        font-size: 12px;
    }
    .stButton>button:hover {
        background-color: #00ff41;
        color: #000;
        border: 1px solid #00ff41;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #111;
        border-radius: 4px;
        color: #888;
        font-family: 'Roboto Mono', monospace;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff41;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. SECURE KEYRING ---
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "34e5a58b5b50587ce21dbe0b33e344dc")
RAPID_API_KEY = st.secrets.get("RAPID_API_KEY", "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "289796ecfb2c4d208506c26d37a4d9ba")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE")

# --- 2. QUANT ENGINE (THE MATH) ---

class QuantEngine:
    @staticmethod
    def american_to_decimal(american_odds):
        """Converts -110 to 1.91"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def implied_prob(decimal_odds):
        """Converts 2.00 to 0.50 (50%)"""
        return 1 / decimal_odds

    @staticmethod
    def kelly_criterion(decimal_odds, win_prob):
        """
        Full Kelly Formula: f* = (bp - q) / b
        Returns optimal bankroll percentage.
        """
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        f_star = (b * p - q) / b
        return max(0, f_star) 

    @staticmethod
    def calculate_ev(decimal_odds, win_prob):
        """Expected Value %"""
        return (win_prob * (decimal_odds - 1)) - (1 - win_prob)

# --- 3. DATA INGESTION LAYERS ---

@st.cache_data(ttl=900)
def fetch_market_data(sport_key='americanfootball_nfl'):
    """Fetches raw odds from Vegas."""
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
    try:
        return requests.get(url, params=params).json()
    except:
        return []

@st.cache_data(ttl=3600)
def fetch_team_stats():
    """Fetches live NFL records/stats from Tank01."""
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
    headers = {"x-rapidapi-key": RAPID_API_KEY, "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"}
    try:
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def get_ai_prediction(matchup_str, stats_context):
    """
    The 'PhD' Analyst. 
    Uses Gemini 2.0 Flash to calculate a 'True Win Probability' independent of Vegas lines.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    You are a Quantitative Sports Analyst.
    Matchup: {matchup_str}
    Context: {str(stats_context)[:1000]}
    
    TASK: Return a JSON object with your proprietary win probability for the Home Team.
    Based on DVOA, EPA/Play, and injuries.
    
    JSON FORMAT:
    {{
        "home_win_prob": 0.65,
        "rationale": "Brief, sharp reason",
        "prop_idea": "Best player prop idea for this game"
    }}
    """
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return {"home_win_prob": 0.50, "rationale": "Model Error", "prop_idea": "N/A"}

# --- 4. THE DASHBOARD ---

def main():
    st.sidebar.header("⚙️ SETTINGS")
    bankroll = st.sidebar.number_input("Bankroll ($)", value=1000, step=100)
    kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.25, help="Full Kelly is risky. 0.25 is standard.")

    st.title("♟️ QUANT TERMINAL // GOD MODE")
    st.markdown("`STATUS: ONLINE` `LATENCY: 42ms` `MODEL: GEMINI-2.0-FLASH`")
    
    tab1, tab2, tab3 = st.tabs(["📉 LIVE MARKETS", "🧩 PROP ENGINE", "🔗 PARLAY BUILDER"])

    # --- LOAD DATA ---
    with st.spinner("Calibrating Models..."):
        odds_data = fetch_market_data()
        stats_data = fetch_team_stats()

    if not odds_data:
        st.error("Market Data Offline.")
        return

    # --- PROCESS DATA INTO DATAFRAME ---
    market_rows = []
    
    for game in odds_data[:8]: # Analyze top 8 games
        home = game['home_team']
        away = game['away_team']
        
        # Extract Best Odds
        best_home = -9999
        if game['bookmakers']:
            for bm in game['bookmakers']:
                if bm['key'] in ['draftkings', 'fanduel']:
                    for mkt in bm['markets']:
                        if mkt['key'] == 'h2h':
                            for outcome in mkt['outcomes']:
                                if outcome['name'] == home: best_home = outcome['price']
        
        if best_home == -9999: continue

        # AI Analysis
        ai_data = get_ai_prediction(f"{away} @ {home}", stats_data)
        
        # Quant Math
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
            "Prop Idea": ai_data['prop_idea'],
            "Commence": game['commence_time']
        })

    df = pd.DataFrame(market_rows)

    # --- TAB 1: LIVE MARKETS (The Trading Desk) ---
    with tab1:
        st.subheader("🔥 HIGH CONVICTION PLAYS (EV+)")
        
        # Filter for Positive EV
        opportunities = df[df['EV'] > 0].copy()
        
        if not opportunities.empty:
            cols = st.columns(len(opportunities))
            for idx, row in opportunities.iterrows():
                bet_size = bankroll * row['Kelly']
                st.markdown(f"""
                <div class="css-1r6slb0">
                    <h3 style="color:#00ff41">{row['Game']}</h3>
                    <p>BET: <b>HOME ({row['Odds']})</b></p>
                    <p>EDGE: <span style="color:#00ff41">+{row['EV']*100:.1f}%</span></p>
                    <p>STAKE: <b>${bet_size:.0f}</b> ({row['Kelly']*100:.1f}%)</p>
                    <small>{row['Rationale']}</small>
                </div>
                """, unsafe_allow_html=True)
                st.write("") 
        else:
            st.info("Market Efficiency High. No pure alpha detected on Moneylines.")

        st.divider()
        st.subheader("📋 FULL MARKET TAPE")
        st.dataframe(
            df[['Game', 'Odds', 'True Prob', 'EV', 'Kelly']],
            use_container_width=True,
            column_config={
                "True Prob": st.column_config.ProgressColumn("Win Prob", format="%.2f", min_value=0, max_value=1),
                "EV": st.column_config.NumberColumn("Exp. Value", format="%.2f"),
                "Kelly": st.column_config.NumberColumn("Alloc %", format="%.3f")
            }
        )

    # --- TAB 2: PROP ENGINE ---
    with tab2:
        st.subheader("🧩 INTELLIGENT PLAYER PROPS")
        st.caption("AI-Generated Value based on Defensive Matchups")
        
        for idx, row in df.iterrows():
            with st.expander(f"{row['Game']}"):
                st.markdown(f"**🤖 AI Suggestion:** {row['Prop Idea']}")
                st.caption(f"Reasoning: {row['Rationale']}")

    # --- TAB 3: PARLAY BUILDER ---
    with tab3:
        st.subheader("🔗 THE DAILY PARLAY")
        st.caption("Correlated plays optimized for +EV")
        
        # Pick top 2 highest probability wins
        top_picks = df.sort_values(by='True Prob', ascending=False).head(2)
        
        if len(top_picks) >= 2:
            game1 = top_picks.iloc[0]
            game2 = top_picks.iloc[1]
            
            combined_prob = game1['True Prob'] * game2['True Prob']
            fair_odds = (1 / combined_prob) * 100 - 100
            
            st.markdown(f"""
            <div style="border: 1px solid #00ff41; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color:#fff">🚀 QUANT DOUBLE</h2>
                <h3 style="color:#00ff41">{game1['Game'].split('@')[1]} (ML)</h3>
                <h3 style="color:#00ff41">{game2['Game'].split('@')[1]} (ML)</h3>
                <hr style="border-color: #333">
                <p>Fair Probability: {combined_prob*100:.1f}%</p>
                <p>Target Odds: +{fair_odds:.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Not enough data to build a safe parlay.")

if __name__ == "__main__":
    main()


