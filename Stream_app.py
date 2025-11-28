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

# --- CUSTOM CSS: PREMIUM FINTECH THEME ---
st.markdown("""
<style>
    /* GLOBAL THEME */
    .stApp {
        background-color: #000000;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #1F2937;
    }

    /* METRIC CARDS */
    .metric-container {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        border-color: #00C805;
    }

    /* TYPOGRAPHY */
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    
    /* ACCENTS */
    .text-green { color: #00C805; font-weight: 700; }
    .text-red { color: #FF453A; font-weight: 700; }
    .text-gray { color: #9CA3AF; font-size: 0.9em; }
    
    /* BADGES */
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-green { background: rgba(0, 200, 5, 0.15); color: #00C805; border: 1px solid #00C805; }
    .badge-gray { background: #374151; color: #D1D5DB; }

    /* LOGOS */
    .team-logo { width: 40px; height: 40px; object-fit: contain; margin-right: 12px; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: transparent;
        border: none;
        color: #6B7280;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: #00C805 !important; border-bottom: 2px solid #00C805; }
</style>
""", unsafe_allow_html=True)

# --- 1. KEYRING (Hardcoded for your deployment) ---
ODDS_API_KEY = "34e5a58b5b50587ce21dbe0b33e344dc"
RAPID_API_KEY = "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c"
NEWS_API_KEY = "289796ecfb2c4d208506c26d37a4d9ba"
GEMINI_API_KEY = "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"

# --- 2. ASSETS (Team Logos) ---
TEAM_LOGOS = {
    "Arizona Cardinals": "https://loodibee.com/wp-content/uploads/nfl-arizona-cardinals-team-logo-2-700x700.png",
    "Atlanta Falcons": "https://loodibee.com/wp-content/uploads/nfl-atlanta-falcons-team-logo-2-700x700.png",
    "Baltimore Ravens": "https://loodibee.com/wp-content/uploads/nfl-baltimore-ravens-team-logo-2-700x700.png",
    "Buffalo Bills": "https://loodibee.com/wp-content/uploads/nfl-buffalo-bills-team-logo-2-700x700.png",
    "Carolina Panthers": "https://loodibee.com/wp-content/uploads/nfl-carolina-panthers-team-logo-2-700x700.png",
    "Chicago Bears": "https://loodibee.com/wp-content/uploads/nfl-chicago-bears-team-logo-2-700x700.png",
    "Cincinnati Bengals": "https://loodibee.com/wp-content/uploads/nfl-cincinnati-bengals-team-logo-700x700.png",
    "Cleveland Browns": "https://loodibee.com/wp-content/uploads/nfl-cleveland-browns-team-logo-2-700x700.png",
    "Dallas Cowboys": "https://loodibee.com/wp-content/uploads/nfl-dallas-cowboys-team-logo-2-700x700.png",
    "Denver Broncos": "https://loodibee.com/wp-content/uploads/nfl-denver-broncos-team-logo-2-700x700.png",
    "Detroit Lions": "https://loodibee.com/wp-content/uploads/nfl-detroit-lions-team-logo-2-700x700.png",
    "Green Bay Packers": "https://loodibee.com/wp-content/uploads/nfl-green-bay-packers-team-logo-2-700x700.png",
    "Houston Texans": "https://loodibee.com/wp-content/uploads/nfl-houston-texans-team-logo-2-700x700.png",
    "Indianapolis Colts": "https://loodibee.com/wp-content/uploads/nfl-indianapolis-colts-team-logo-2-700x700.png",
    "Jacksonville Jaguars": "https://loodibee.com/wp-content/uploads/nfl-jacksonville-jaguars-team-logo-2-700x700.png",
    "Kansas City Chiefs": "https://loodibee.com/wp-content/uploads/nfl-kansas-city-chiefs-team-logo-2-700x700.png",
    "Las Vegas Raiders": "https://loodibee.com/wp-content/uploads/nfl-las-vegas-raiders-team-logo-2-700x700.png",
    "Los Angeles Chargers": "https://loodibee.com/wp-content/uploads/nfl-los-angeles-chargers-team-logo-2-700x700.png",
    "Los Angeles Rams": "https://loodibee.com/wp-content/uploads/nfl-los-angeles-rams-team-logo-2-700x700.png",
    "Miami Dolphins": "https://loodibee.com/wp-content/uploads/nfl-miami-dolphins-team-logo-2-700x700.png",
    "Minnesota Vikings": "https://loodibee.com/wp-content/uploads/nfl-minnesota-vikings-team-logo-2-700x700.png",
    "New England Patriots": "https://loodibee.com/wp-content/uploads/nfl-new-england-patriots-team-logo-2-700x700.png",
    "New Orleans Saints": "https://loodibee.com/wp-content/uploads/nfl-new-orleans-saints-team-logo-2-700x700.png",
    "New York Giants": "https://loodibee.com/wp-content/uploads/nfl-new-york-giants-team-logo-2-700x700.png",
    "New York Jets": "https://loodibee.com/wp-content/uploads/nfl-new-york-jets-team-logo-700x700.png",
    "Philadelphia Eagles": "https://loodibee.com/wp-content/uploads/nfl-philadelphia-eagles-team-logo-2-700x700.png",
    "Pittsburgh Steelers": "https://loodibee.com/wp-content/uploads/nfl-pittsburgh-steelers-team-logo-2-700x700.png",
    "San Francisco 49ers": "https://loodibee.com/wp-content/uploads/nfl-san-francisco-49ers-team-logo-2-700x700.png",
    "Seattle Seahawks": "https://loodibee.com/wp-content/uploads/nfl-seattle-seahawks-team-logo-2-700x700.png",
    "Tampa Bay Buccaneers": "https://loodibee.com/wp-content/uploads/nfl-tampa-bay-buccaneers-team-logo-2-700x700.png",
    "Tennessee Titans": "https://loodibee.com/wp-content/uploads/nfl-tennessee-titans-team-logo-2-700x700.png",
    "Washington Commanders": "https://loodibee.com/wp-content/uploads/washington-commanders-logo-700x700.png"
}

# --- 3. QUANT ENGINE (THE MATH) ---

class QuantEngine:
    @staticmethod
    def american_to_decimal(american_odds):
        return (american_odds / 100) + 1 if american_odds > 0 else (100 / abs(american_odds)) + 1

    @staticmethod
    def implied_prob(decimal_odds):
        return 1 / decimal_odds

    @staticmethod
    def bayesian_blend(market_prob, model_prob, confidence_score):
        """
        Blends Market Wisdom with AI Model based on Confidence.
        Confidence 0-10: 10 means trust AI 100%, 0 means trust Market 100%.
        """
        weight = confidence_score / 10.0
        return (model_prob * weight) + (market_prob * (1 - weight))

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

# --- 4. DATA INGESTION ---

@st.cache_data(ttl=900)
def fetch_market_data():
    """Fetches Odds. Handles empty states gracefully."""
    url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
    try:
        res = requests.get(url, params=params)
        return res.json() if res.status_code == 200 else []
    except:
        return []

@st.cache_data(ttl=3600)
def fetch_team_stats():
    """Fetches Stats. Returns empty dict on failure to prevent crash."""
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
    headers = {"x-rapidapi-key": RAPID_API_KEY, "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers)
        return res.json() if res.status_code == 200 else {}
    except:
        return {}

def get_ai_prediction(matchup, stats):
    """
    FAULT TOLERANT AI ANALYST.
    If stats are missing, it falls back to 'General Knowledge' mode.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    stats_context = str(stats)[:1000] if stats else "STATS_UNAVAILABLE_USE_GENERAL_KNOWLEDGE"
    
    prompt = f"""
    You are a PhD Sports Quantitative Analyst.
    MATCHUP: {matchup}
    STATS: {stats_context}
    
    TASK: Output a JSON probability model.
    1. If stats are missing, use your internal knowledge of NFL team strengths.
    2. 'home_win_prob': Your estimated win % for Home Team (0.0-1.0).
    3. 'confidence': 1-10 score of how strong this read is.
    
    OUTPUT JSON ONLY:
    {{
        "home_win_prob": 0.65,
        "confidence": 8,
        "rationale": "Chiefs passing DVOA ranks #1 vs weak secondary.",
        "prop_bet": "Player Name - Over/Under X Stat"
    }}
    """
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        # Fallback Default if AI fails
        return {"home_win_prob": 0.50, "confidence": 1, "rationale": "Model Unavailable", "prop_bet": "N/A"}

# --- 5. DASHBOARD UI ---

def main():
    # SIDEBAR
    with st.sidebar:
        st.header("⚡ PORTFOLIO")
        bankroll = st.number_input("Bankroll ($)", value=5000, step=100)
        kelly_fraction = st.slider("Kelly Fraction", 0.1, 0.5, 0.25)
        
        st.markdown("---")
        st.caption("MODEL SETTINGS")
        min_ev = st.slider("Min Edge %", 0.0, 10.0, 1.0)
        
        if st.button("↻ Refresh Market"):
            st.cache_data.clear()

    # HEADER
    st.title("GOD MODE // TERMINAL")
    st.markdown("##### 🚀 INSTITUTIONAL SPORTS ANALYTICS")
    st.markdown("---")

    # LOAD DATA
    with st.spinner("Initializing Quant Engine..."):
        odds_data = fetch_market_data()
        stats_data = fetch_team_stats()

    if not odds_data:
        st.error("Market Offline. No odds available.")
        return

    # PROCESSING LOOP
    opportunities = []
    
    for game in odds_data[:8]: 
        home = game['home_team']
        away = game['away_team']
        
        # Get Market Odds
        best_home = -9999
        if game['bookmakers']:
            for bm in game['bookmakers']:
                if bm['key'] in ['draftkings', 'fanduel', 'betmgm']:
                    for mkt in bm['markets']:
                        if mkt['key'] == 'h2h':
                            for outcome in mkt['outcomes']:
                                if outcome['name'] == home: best_home = outcome['price']
        
        if best_home == -9999: continue

        # AI & Quant Math
        ai_res = get_ai_prediction(f"{away} @ {home}", stats_data)
        
        dec_odds = QuantEngine.american_to_decimal(best_home)
        market_prob = QuantEngine.implied_prob(dec_odds)
        
        # BLENDED PROBABILITY (Bayesian)
        true_prob = QuantEngine.bayesian_blend(market_prob, ai_res['home_win_prob'], ai_res['confidence'])
        
        ev = QuantEngine.calculate_ev(dec_odds, true_prob)
        kelly = QuantEngine.kelly_criterion(dec_odds, true_prob) * kelly_fraction
        
        opportunities.append({
            "Game": f"{away} @ {home}",
            "Home": home,
            "Away": away,
            "Odds": best_home,
            "True Prob": true_prob,
            "EV": ev * 100,
            "Kelly": kelly,
            "Stake": bankroll * kelly,
            "Rationale": ai_res['rationale'],
            "Prop": ai_res['prop_bet'],
            "Confidence": ai_res['confidence']
        })

    df = pd.DataFrame(opportunities)

    # TABS
    tab1, tab2, tab3 = st.tabs(["🔥 ALPHA FEED", "📊 DATA GRID", "🧩 PROP LAB"])

    # --- TAB 1: ALPHA FEED (Visual) ---
    with tab1:
        valid_plays = df[df['EV'] >= min_ev].copy()
        
        if not valid_plays.empty:
            for _, row in valid_plays.iterrows():
                
                # Dynamic Badge Color
                conf_color = "#00C805" if row['Confidence'] >= 7 else "#F59E0B";
                
                # Logo Logic
                home_logo = TEAM_LOGOS.get(row['Home'], "https://cdn.freebiesupply.com/images/large/2x/nfl-logo-png-transparent.png")
                
                st.markdown(f"""
                <div class="metric-container">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div style="display: flex; align-items: center;">
                            <img src="{home_logo}" class="team-logo">
                            <div>
                                <h3 style="margin: 0; color: #fff;">{row['Home']}</h3>
                                <span class="text-gray">vs {row['Away']}</span>
                            </div>
                        </div>
                        <div class="badge badge-green">+{row['EV']:.1f}% EDGE</div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; background: #000; padding: 15px; border-radius: 8px;">
                        <div>
                            <div class="text-gray">SIGNAL</div>
                            <div class="mono" style="font-size: 1.2em; color: #fff;">HOME ({row['Odds']})</div>
                        </div>
                        <div>
                            <div class="text-gray">PROB</div>
                            <div class="mono" style="font-size: 1.2em; color: {conf_color};">{row['True Prob']*100:.1f}%</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="text-gray">KELLY BET</div>
                            <div class="mono text-green" style="font-size: 1.2em;">${row['Stake']:.0f}</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; font-size: 0.9em; color: #D1D5DB;">
                        <span style="color: {conf_color}; font-weight: 700;">AI ANALYST:</span> {row['Rationale']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No plays meet your Edge criteria. Market is efficient.")

    # --- TAB 2: DATA GRID (Terminal) ---
    with tab2:
        st.dataframe(
            df[['Home', 'Odds', 'True Prob', 'EV', 'Kelly', 'Confidence']],
            use_container_width=True,
            column_config={
                "True Prob": st.column_config.ProgressColumn("Win Prob", format="%.2f", min_value=0, max_value=1),
                "EV": st.column_config.NumberColumn("Edge %", format="%.1f"),
                "Kelly": st.column_config.NumberColumn("Alloc %", format="%.3f"),
                "Confidence": st.column_config.NumberColumn("Conf (1-10)", format="%d")
            }
        )

    # --- TAB 3: PROP LAB ---
    with tab3:
        for _, row in df.iterrows():
            with st.expander(f"🧩 {row['Home']} Props"):
                st.markdown(f"**AI Recommendation:** `{row['Prop']}`")
                st.caption(f"Context: {row['Rationale']}")

if __name__ == "__main__":
    main()


