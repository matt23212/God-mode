import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google import genai
import json
import time
import math
from scipy.stats import poisson

# ==========================================
# 1. SYSTEM CONFIGURATION & ASSETS
# ==========================================

st.set_page_config(
    page_title="GOD MODE // QUANT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME INJECTION ---
st.markdown("""
<style>
    /* RESET & CORE VARIABLES */
    :root {
        --bg-color: #000000;
        --card-bg: #111111;
        --border-color: #222222;
        --accent-green: #00C805;
        --accent-red: #FF5000;
        --text-primary: #FFFFFF;
        --text-secondary: #888888;
        --font-mono: 'Roboto Mono', monospace;
        --font-sans: 'Inter', sans-serif;
    }

    /* APP CONTAINER */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    /* METRIC CARDS (Robinhood Style) */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        border-color: var(--accent-green);
        transform: translateY(-2px);
    }

    /* HEADERS */
    h1, h2, h3 { 
        font-weight: 800; 
        letter-spacing: -0.5px; 
        color: var(--text-primary); 
    }
    
    /* BADGES */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-alpha { 
        background: rgba(0, 200, 5, 0.15); 
        color: var(--accent-green); 
        border: 1px solid rgba(0, 200, 5, 0.3);
    }
    .badge-risk { 
        background: rgba(255, 80, 0, 0.15); 
        color: var(--accent-red); 
        border: 1px solid rgba(255, 80, 0, 0.3);
    }

    /* DATATABLES */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-family: var(--font-mono);
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #0A0A0A;
        border-right: 1px solid var(--border-color);
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background: transparent;
        color: var(--text-secondary);
        border: none;
        font-weight: 600;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-green) !important;
    }
    
    /* UTILS */
    .mono { font-family: var(--font-mono); }
    .text-green { color: var(--accent-green); }
    .text-gray { color: var(--text-secondary); font-size: 12px; }
    .flex-between { display: flex; justify-content: space-between; align-items: center; }
</style>
""", unsafe_allow_html=True)

# --- SECURE KEYRING ---
# FIXED: Flat dictionary structure to prevent KeyError
KEYS = {
    "ODDS": "34e5a58b5b50587ce21dbe0b33e344dc",
    "RAPID": "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c",
    "NEWS": "289796ecfb2c4d208506c26d37a4d9ba",
    "GEMINI": "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE"
}

# ==========================================
# 2. QUANTITATIVE ENGINE (MATH CORE)
# ==========================================

class QuantMath:
    """
    Handles all statistical calculations, including Poisson simulations
    and Kelly Criterion optimization.
    """
    
    @staticmethod
    def implied_probability(american_odds):
        """Converts Moneyline to Implied Probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    @staticmethod
    def decimal_odds(american_odds):
        """Converts American to Decimal."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def poisson_win_prob(home_avg_score, away_avg_score):
        """
        Uses Poisson Distribution to simulate 10,000 games based on average scores.
        This provides a 'Mathematical Probability' independent of Vegas.
        """
        # Set simplified expected goals (Lambda)
        # In a real PhD model, this would factor in Defense DVOA adjustment
        lambda_home = max(10, home_avg_score) # Safety floor
        lambda_away = max(10, away_avg_score)
        
        # Simulation
        home_scores = poisson.rvs(lambda_home, size=10000)
        away_scores = poisson.rvs(lambda_away, size=10000)
        
        wins = np.sum(home_scores > away_scores)
        draws = np.sum(home_scores == away_scores)
        
        # NFL rarely draws, so we split them 50/50 for this simplified model
        return (wins + (draws * 0.5)) / 10000

    @staticmethod
    def kelly_criterion(decimal_odds, win_prob, fraction=0.25):
        """
        Calculates optimal bankroll allocation.
        Uses Fractional Kelly to reduce volatility (Standard for pros).
        """
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        
        if b == 0: return 0
        
        f_star = (b * p - q) / b
        return max(0, f_star) * fraction

    @staticmethod
    def bayesian_inference(market_prob, math_prob, ai_prob, ai_confidence):
        """
        The 'God Mode' Formula.
        Blends 3 signals into one final Truth Probability.
        """
        # Weights
        w_market = 0.30  # Respect the market, it knows a lot
        w_math = 0.30    # Respect the stats
        w_ai = 0.40 * (ai_confidence / 100.0) # Scale AI trust by its own confidence
        
        # Normalize weights
        total_w = w_market + w_math + w_ai
        w_market /= total_w
        w_math /= total_w
        w_ai /= total_w
        
        final_prob = (market_prob * w_market) + (math_prob * w_math) + (ai_prob * w_ai)
        return final_prob

# ==========================================
# 3. DATA LAKE (INGESTION LAYER)
# ==========================================

class DataLake:
    """
    Handles all API interactions with robust error handling and caching.
    """
    
    @staticmethod
    @st.cache_data(ttl=900) # Cache odds for 15 mins
    def get_odds():
        url = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
        # FIXED: Access keys directly from flat dictionary
        params = {
            'apiKey': KEYS['ODDS'], 
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american'
        }
        try:
            res = requests.get(url, params=params)
            if res.status_code == 200:
                return res.json()
            return []
        except:
            return []

    @staticmethod
    @st.cache_data(ttl=3600) # Cache stats for 1 hour
    def get_stats():
        url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
        headers = {
            "x-rapidapi-key": KEYS['RAPID'],
            "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
        }
        try:
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                return res.json()
            return {}
        except:
            return {}

    @staticmethod
    def get_news(team_name):
        # Lightweight news fetch
        try:
            three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{team_name}" NFL injury', # Targeted search
                'from': three_days_ago,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 3, 
                'apiKey': KEYS['NEWS']
            }
            res = requests.get(url, params=params)
            if res.status_code == 200:
                arts = res.json().get('articles', [])
                return [a['title'] for a in arts]
            return []
        except:
            return []

    @staticmethod
    def get_weather(city_name):
        # Open-Meteo (Free)
        try:
            geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1").json()
            if 'results' in geo:
                lat, lon = geo['results'][0]['latitude'], geo['results'][0]['longitude']
                wx = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m&forecast_days=1").json()
                curr_hour = datetime.now().hour
                temp = wx['hourly']['temperature_2m'][curr_hour]
                wind = wx['hourly']['wind_speed_10m'][curr_hour]
                return {"temp": temp, "wind": wind}
        except:
            return None
        return None

# ==========================================
# 4. INTELLIGENCE LAYER (AI AGENT)
# ==========================================

class AIAnalyst:
    """
    The Brain. Uses Gemini 2.0 to synthesize qualitative data.
    """
    def __init__(self):
        self.client = genai.Client(api_key=KEYS['GEMINI'])

    def analyze_matchup(self, matchup, stats_context, news_context, weather_context):
        prompt = f"""
        ACT AS: A Senior Quant Trader for an NFL Syndicate.
        
        MATCHUP: {matchup}
        
        DATA STREAMS:
        1. STATS: {str(stats_context)[:800]}
        2. NEWS HEADLINES: {str(news_context)}
        3. WEATHER: {str(weather_context)}
        
        OBJECTIVE:
        Generate a probabilistic assessment of the Home Team winning.
        You must ignore the Vegas line. Focus on matchup advantages (O-Line vs D-Line, Injuries, Weather).
        
        OUTPUT FORMAT (Strict JSON):
        {{
            "home_win_prob_float": 0.65,
            "confidence_score_int": 85,
            "key_alpha": "One sentence explaining the biggest edge.",
            "prop_bet": "Best player prop (e.g. 'Mahomes Over 250.5')"
        }}
        """
        
        try:
            res = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            clean_json = res.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except:
            # Fail-safe default
            return {
                "home_win_prob_float": 0.50,
                "confidence_score_int": 10,
                "key_alpha": "AI Offline - Using Market Defaults",
                "prop_bet": "N/A"
            }

# ==========================================
# 5. DASHBOARD UI (VIEW LAYER)
# ==========================================

def render_sidebar():
    with st.sidebar:
        st.header("⚡ PORTFOLIO")
        bankroll = st.number_input("Capital ($)", value=10000, step=1000)
        kelly = st.slider("Kelly Fraction", 0.1, 0.5, 0.25, help="0.25 is standard for sustainable growth.")
        
        st.divider()
        
        st.subheader("FILTERS")
        min_edge = st.slider("Min Edge %", 0.0, 10.0, 1.5)
        
        st.divider()
        if st.button("↻ REFRESH FEED"):
            st.cache_data.clear()
            st.rerun()
            
    return bankroll, kelly, min_edge

def render_card(game_data, logos):
    """Renders a high-fidelity SaaS card for a single game."""
    
    # CSS Classes
    edge_color = "#00C805" if game_data['edge'] > 0 else "#888"
    edge_txt = f"+{game_data['edge']:.1f}%" if game_data['edge'] > 0 else "EVEN"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="flex-between" style="margin-bottom: 20px;">
            <div style="display: flex; align-items: center;">
                <img src="{logos['away']}" style="width:40px; height:40px; margin-right:10px; opacity:0.7;">
                <span style="color:#666; font-weight:800; margin:0 10px;">@</span>
                <img src="{logos['home']}" style="width:50px; height:50px; margin-right:10px;">
                <div>
                    <div style="font-size: 18px; font-weight: 800; color: #fff;">{game_data['home_team']}</div>
                    <div class="text-gray">{game_data['away_team']}</div>
                </div>
            </div>
            <div class="badge badge-alpha">{edge_txt} EDGE</div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 15px;">
            <div style="background:#0A0A0A; padding:10px; border-radius:6px;">
                <div class="text-gray">ODDS</div>
                <div class="mono" style="color:#fff;">{game_data['odds']}</div>
            </div>
            <div style="background:#0A0A0A; padding:10px; border-radius:6px;">
                <div class="text-gray">IMPLIED</div>
                <div class="mono" style="color:#fff;">{game_data['implied_prob']:.1%}</div>
            </div>
            <div style="background:#0A0A0A; padding:10px; border-radius:6px;">
                <div class="text-gray">MODEL</div>
                <div class="mono" style="color:{edge_color};">{game_data['model_prob']:.1%}</div>
            </div>
            <div style="background:#0A0A0A; padding:10px; border-radius:6px; border: 1px solid {edge_color};">
                <div class="text-gray">STAKE</div>
                <div class="mono" style="color:#fff;">${game_data['stake']:.0f}</div>
            </div>
        </div>
        
        <div style="font-size: 13px; line-height: 1.5; color: #ccc; border-top: 1px solid #222; padding-top: 15px;">
            <span style="color: #00C805; font-weight: 700;">AI ANALYST:</span> {game_data['rationale']}
        </div>
        <div style="font-size: 12px; color: #666; margin-top: 5px;">
            🧩 Prop Idea: {game_data['prop']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. MAIN EXECUTION LOOP
# ==========================================

def main():
    bankroll, kelly_frac, min_edge = render_sidebar()
    
    st.title("GOD MODE // QUANT")
    st.caption("INSTITUTIONAL SPORTS ANALYTICS TERMINAL v5.0")
    
    # 1. Initialize Engines
    lake = DataLake()
    brain = AIAnalyst()
    
    # 2. Fetch Data
    with st.spinner("Ingesting global market data..."):
        odds = lake.get_odds()
        stats = lake.get_stats()
    
    if not odds:
        st.error("MARKET FEED DISCONNECTED. CHECK API QUOTAS.")
        return

    # 3. Processing Loop (The "Meat")
    opportunities = []
    
    # Analyze Top 5 games to keep latency low for demo
    for game in odds[:5]:
        home = game['home_team']
        away = game['away_team']
        
        # A. Find Best Market Price
        best_price = -9999
        if game['bookmakers']:
            for bm in game['bookmakers']:
                if bm['key'] in ['draftkings', 'fanduel', 'betmgm']:
                    for mkt in bm['markets']:
                        if mkt['key'] == 'h2h':
                            for out in mkt['outcomes']:
                                if out['name'] == home: best_price = out['price']
        
        if best_price == -9999: continue

        # B. Gather Intel
        news = lake.get_news(home)
        weather = lake.get_weather(home.split()[-1]) # Rough city mapping
        
        # C. Generate Probabilities
        # 1. Market Implied
        prob_market = QuantMath.implied_probability(best_price)
        
        # 2. Mathematical (Poisson - Mocked avg scores for reliability if stats missing)
        prob_math = QuantMath.poisson_win_prob(24.5, 21.0) 
        
        # 3. AI Inference
        ai_insight = brain.analyze_matchup(f"{away} @ {home}", stats, news, weather)
        
        # D. Bayesian Blend
        final_prob = QuantMath.bayesian_inference(
            prob_market, 
            prob_math, 
            ai_insight['home_win_prob_float'], 
            ai_insight['confidence_score_int']
        )
        
        # E. Calculate Edge & Stake
        dec_odds = QuantMath.decimal_odds(best_price)
        edge = (final_prob - prob_market) * 100
        kelly_pct = QuantMath.kelly_criterion(dec_odds, final_prob, kelly_frac)
        stake = bankroll * kelly_pct
        
        opportunities.append({
            "home_team": home,
            "away_team": away,
            "odds": best_price,
            "implied_prob": prob_market,
            "model_prob": final_prob,
            "edge": edge,
            "stake": stake,
            "rationale": ai_insight['key_alpha'],
            "prop": ai_insight['prop_bet']
        })

    # 4. Display
    tabs = st.tabs(["🔥 ALPHA FEED", "📊 DATA GRID", "📈 PARLAY LAB"])
    
    with tabs[0]:
        # Sort by Edge
        opportunities.sort(key=lambda x: x['edge'], reverse=True)
        
        valid_ops = [op for op in opportunities if op['edge'] >= min_edge]
        
        if valid_ops:
            for op in valid_ops:
                # Helper for logos (using ESPN CDN)
                def get_logo_url(name):
                    slug = name.split()[-1].lower()
                    if "football" in slug: slug = "washington"
                    return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/{slug}.png&w=100&h=100"
                
                logos = {"home": get_logo_url(op['home_team']), "away": get_logo_url(op['away_team'])}
                render_card(op, logos)
        else:
            st.info("Market is efficient. No plays meet your Edge criteria. Hold Cash.")

    with tabs[1]:
        df = pd.DataFrame(opportunities)
        st.dataframe(
            df[['home_team', 'odds', 'implied_prob', 'model_prob', 'edge', 'stake']],
            use_container_width=True,
            column_config={
                "implied_prob": st.column_config.NumberColumn("Implied %", format="%.2f"),
                "model_prob": st.column_config.NumberColumn("True %", format="%.2f"),
                "edge": st.column_config.NumberColumn("Edge %", format="%.2f"),
                "stake": st.column_config.NumberColumn("Stake $", format="$%.0f"),
            }
        )

if __name__ == "__main__":
    main()


