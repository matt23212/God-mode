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
    page_title="GOD MODE TERMINAL // ALPHA",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS: THE "BLOOMBERG" AESTHETIC ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00ff41; 
        font-family: 'Courier New', monospace;
    }
    
    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #333;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1f1f1f;
        color: #00ff41;
        border: 1px solid #00ff41;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        background-color: #00ff41;
        color: #000;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e6e6e6;
        font-family: 'Roboto Mono', monospace;
        letter-spacing: -1px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. SECURE KEYRING ---
# (Falls back to hardcoded for your specific deploy if secrets fail)
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "34e5a58b5b50587ce21dbe0b33e344dc")
RAPID_API_KEY = st.secrets.get("RAPID_API_KEY", "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "289796ecfb2c4d208506c26d37a4d9ba")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE")

# --- 2. QUANTITATIVE ENGINE (THE MATH) ---

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
        Calculates optimal stake % using Full Kelly.
        f* = (bp - q) / b
        b = net odds (decimal - 1)
        p = win probability
        q = lose probability (1-p)
        """
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        
        f_star = (b * p - q) / b
        return max(0, f_star) # No negative bets

    @staticmethod
    def calculate_edge(implied_prob, true_prob):
        """Returns the Edge % (Expected Value)"""
        return true_prob - implied_prob

# --- 3. DATA INGESTION ---

@st.cache_data(ttl=900) # Cache for 15 mins
def fetch_market_data(sport_key='americanfootball_nfl'):
    """Fetches raw odds from The Odds API."""
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h', # Moneyline
        'oddsFormat': 'american',
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"MARKET DATA FAILURE: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_team_stats():
    """Fetches Tank01 NFL Stats (Records, etc)."""
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
    headers = {
        "x-rapidapi-key": RAPID_API_KEY,
        "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def get_ai_handicap(matchup_str, stats_context):
    """
    Uses Gemini 2.0 to generate a 'True Probability' based on deep analysis.
    Forces JSON output for mathematical parsing.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    ROLE: You are a PhD-level Sports Quantitative Analyst.
    TASK: Analyze this NFL matchup and output a strict JSON probability model.
    
    MATCHUP: {matchup_str}
    CONTEXT: {str(stats_context)[:1000]}
    
    REQUIREMENTS:
    1. Ignore the Vegas line. Calculate the 'True Win Probability' for the Home Team based on DVOA, injuries, and matchup history.
    2. Output ONLY valid JSON. No markdown. No conversational text.
    
    JSON FORMAT:
    {{
        "home_win_prob": 0.65,
        "reasoning": "Short summary of why",
        "risk_factor": "High/Med/Low"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        # Clean the response to ensure pure JSON
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"home_win_prob": 0.50, "reasoning": "AI Model Failure", "risk_factor": "High"}

# --- 4. THE UI RENDERER ---

def main():
    st.title("♟️ GOD MODE // QUANT TERMINAL")
    
    # Top Bar: Market Ticker
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label="MODEL VERSION", value="v2.4.0 (Kelly)")
    with col2: st.metric(label="MARKET STATUS", value="OPEN")
    with col3: st.metric(label="BANKROLL STRATEGY", value="FULL KELLY")
    with col4: 
        refresh = st.button("↻ REFRESH FEED")
        if refresh: st.cache_data.clear()

    # Fetch Data
    with st.spinner("Initializing Data Streams..."):
        odds_data = fetch_market_data()
        stats_data = fetch_team_stats()

    if not odds_data:
        st.warning("NO ACTIVE MARKETS DETECTED")
        return

    # Process Data into a DataFrame for the Terminal
    market_rows = []
    
    progress_bar = st.progress(0)
    total_games = len(odds_data[:10]) # Limit to 10 for API speed in this demo
    
    for i, game in enumerate(odds_data[:10]):
        progress_bar.progress((i + 1) / total_games)
        
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']
        
        # Get Best Odds (Finding the best line)
        best_home_odds = -9999
        best_away_odds = -9999
        
        # Simple parser to find DraftKings/FanDuel lines
        if game['bookmakers']:
            for bm in game['bookmakers']:
                if bm['key'] in ['draftkings', 'fanduel', 'betmgm']:
                    for mkt in bm['markets']:
                        if mkt['key'] == 'h2h':
                            for outcome in mkt['outcomes']:
                                if outcome['name'] == home_team:
                                    best_home_odds = outcome['price']
                                else:
                                    best_away_odds = outcome['price']
        
        if best_home_odds == -9999: continue # Skip if no odds found

        # 1. Run AI Model
        matchup_id = f"{away_team} @ {home_team}"
        ai_model = get_ai_handicap(matchup_id, stats_data)
        
        # 2. Run Quant Engine
        home_decimal = QuantEngine.american_to_decimal(best_home_odds)
        implied_prob = QuantEngine.implied_prob(home_decimal)
        true_prob = ai_model['home_win_prob']
        
        edge = QuantEngine.calculate_edge(implied_prob, true_prob)
        kelly_stake = QuantEngine.kelly_criterion(home_decimal, true_prob)
        
        # 3. Build Row
        market_rows.append({
            "Matchup": f"{away_team} @ {home_team}",
            "Time": commence_time,
            "Market Odds": best_home_odds,
            "Implied Prob": f"{implied_prob:.1%}",
            "AI Model Prob": f"{true_prob:.1%}",
            "Edge": edge, # Keep float for sorting
            "Kelly Stake": f"{kelly_stake:.1%}",
            "Risk": ai_model['risk_factor'],
            "Analysis": ai_model['reasoning']
        })

    progress_bar.empty()

    # Convert to DataFrame
    df = pd.DataFrame(market_rows)

    # --- TERMINAL DISPLAY ---
    
    st.markdown("### 📊 LIVE OPPORTUNITY BOARD")
    
    # Filter for Positive Edge
    opportunities = df[df['Edge'] > 0].copy()
    
    if not opportunities.empty:
        # Format the Edge column for display
        opportunities['Edge Display'] = opportunities['Edge'].apply(lambda x: f"+{x:.1%}")
        
        # Display as a high-density table
        st.dataframe(
            opportunities[['Matchup', 'Market Odds', 'Implied Prob', 'AI Model Prob', 'Edge Display', 'Kelly Stake', 'Risk', 'Analysis']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Edge Display": st.column_config.TextColumn(
                    "EDGE",
                    help="Positive Expected Value",
                    validate="^\\+[0-9.]+$" # Regex to ensure positive look
                ),
                "Kelly Stake": st.column_config.ProgressColumn(
                    "KELLY BET",
                    format="%s",
                    min_value=0,
                    max_value=0.2, # Cap visualization at 20%
                )
            }
        )
    else:
        st.info("NO POSITIVE EXPECTED VALUE (EV+) DETECTED IN CURRENT MARKETS. HOLD CASH.")

    # --- ALL GAMES (Raw Feed) ---
    with st.expander("Show All Market Data (Raw Feed)", expanded=False):
        st.dataframe(df)

if __name__ == "__main__":
    main()

