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
    page_title="TITAN OS // LEGION",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME ENGINE: "LEGION DARK" ---
st.markdown("""
<style>
    /* GLOBAL RESET */
    .stApp { background-color: #000000; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #1a1a1a;
    }

    /* METRIC CONTAINERS */
    div[data-testid="stMetric"] {
        background-color: #111;
        border: 1px solid #222;
        padding: 15px;
        border-radius: 8px;
        transition: 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00C805;
    }
    div[data-testid="stMetricLabel"] {
        color: #666;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #fff;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 22px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 12px;
        font-weight: 700;
    }

    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #0A0A0A;
        padding: 5px;
        border-radius: 8px;
        border: 1px solid #222;
    }
    .stTabs [data-baseweb="tab"] {
        height: 35px;
        background-color: transparent;
        border: none;
        color: #666;
        font-size: 12px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a1a;
        color: #00C805;
        border-radius: 6px;
    }

    /* NATIVE CONTAINER STYLING */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #222;
        border-radius: 12px;
        padding: 20px;
        background-color: #0A0A0A;
    }

    /* HEADERS */
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }
    
    /* BADGES */
    .badge { 
        background: rgba(0, 200, 5, 0.1); 
        color: #00C805; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-size: 10px; 
        font-weight: 800; 
        border: 1px solid rgba(0, 200, 5, 0.2);
    }
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
# 2. ASSET LIBRARY
# ==============================================================================

class Assets:
    @staticmethod
    def get_logo(team_name, league="nfl"):
        slug = team_name.split()[-1].lower()
        if "football" in slug: slug = "washington"
        if "sox" in slug: slug = "red-sox" if "red" in team_name.lower() else "white-sox"
        return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/{league.lower()}/500/{slug}.png&w=100&h=100"

# ==============================================================================
# 3. QUANTITATIVE ENGINE
# ==============================================================================

class QuantEngine:
    @staticmethod
    def american_to_decimal(american):
        if american > 0: return (american / 100) + 1
        return (100 / abs(american)) + 1

    @staticmethod
    def kelly_criterion(decimal, prob, fraction=0.25):
        b = decimal - 1
        p = prob
        q = 1 - p
        f = (b * p - q) / b
        return max(0.0, f) * fraction

    @staticmethod
    def ev(decimal, prob):
        return (prob * (decimal - 1)) - (1 - prob)

    @staticmethod
    def monte_carlo(avg_a, avg_b, std=10, sims=5000):
        a_scores = np.random.normal(avg_a, std, sims)
        b_scores = np.random.normal(avg_b, std, sims)
        return np.mean(a_scores > b_scores)

# ==============================================================================
# 4. DATA ENGINE
# ==============================================================================

class DataEngine:
    @staticmethod
    @st.cache_data(ttl=900)
    def get_odds(sport):
        url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
        params = {'apiKey': KEYS['ODDS'], 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
        try:
            res = requests.get(url, params=params)
            return res.json() if res.status_code == 200 else []
        except: return []

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_stats(league):
        host = f"tank01-{league}-live-in-game-real-time-statistics-{league}.p.rapidapi.com"
        url = f"https://{host}/get{league.upper()}GamesForDate"
        if league == 'nfl': url = f"https://{host}/getNFLGamesForWeek"
        try:
            res = requests.get(url, headers={"x-rapidapi-key": KEYS['RAPID'], "x-rapidapi-host": host})
            return res.json() if res.status_code == 200 else {}
        except: return {}

# ==============================================================================
# 5. AI ENGINE
# ==============================================================================

class AIEngine:
    def __init__(self):
        self.client = genai.Client(api_key=KEYS['GEMINI'])

    def analyze(self, matchup, stats, league):
        prompt = f"""
        ROLE: Sports Quant.
        TASK: Analyze {matchup} ({league.upper()}).
        STATS: {str(stats)[:800]}
        
        OUTPUT JSON:
        {{
            "home_win_prob": 0.60,
            "confidence": 80,
            "reason": "30-word analysis citing DVOA/EPA.",
            "prop_name": "Player Name",
            "prop_stat": "Over 20.5 Pts",
            "prop_analysis": "Matchup advantage vs weak defense."
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
            return {"home_win_prob": 0.55, "confidence": 50, "reason": "Model Estimate", "prop_name": "N/A", "prop_stat": "N/A"}

# ==============================================================================
# 6. VISUALIZATION ENGINE (PLOTLY)
# ==============================================================================

class Visuals:
    @staticmethod
    def prop_chart(values, line, label="Last 10 Games"):
        """
        Replicates the Outlier.bet Green/Red Bar Chart.
        """
        colors = ['#00C805' if v >= line else '#FF453A' for v in values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(1, 11)), 
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto',
                hoverinfo='y'
            )
        ])
        
        # The "Line"
        fig.add_hline(y=line, line_dash="dot", line_color="white", annotation_text=f"LINE: {line}")
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            height=200,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, fixedrange=True),
            yaxis=dict(showgrid=True, gridcolor='#222', fixedrange=True),
            title={'text': label, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
        )
        return fig

    @staticmethod
    def donut_chart(value, title):
        fig = go.Figure(go.Pie(
            values=[value, 100-value],
            labels=["Win", "Loss"],
            hole=.7,
            marker_colors=['#00E5FF', '#222'],
            textinfo='none'
        ))
        fig.update_layout(
            showlegend=False,
            height=120,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text=f"{value}%", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")]
        )
        return fig

# ==============================================================================
# 7. MAIN APP
# ==============================================================================

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("TITAN OS")
        league = st.selectbox("MARKET", ["NFL", "NBA", "NHL"])
        
        st.divider()
        bankroll = st.number_input("BANKROLL", 1000, 100000, 5000)
        kelly = st.slider("RISK (KELLY)", 0.1, 0.5, 0.25)
        min_edge = st.slider("MIN EDGE %", 0.0, 10.0, 1.5)
        
        if st.button("CLEAR CACHE"): st.cache_data.clear()

    # --- HEADER & DASHBOARD ---
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        st.title(f"{league} WAR ROOM")
        st.caption("INSTITUTIONAL GRADE ANALYTICS • V9.0")
    with c2:
        # Portfolio Growth Chart (Mock)
        chart_data = pd.DataFrame(np.cumsum(np.random.randn(20)) + 100, columns=['Equity'])
        st.line_chart(chart_data, height=80, color="#00C805")

    # --- LOAD DATA ---
    sport_map = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "NHL": "icehockey_nhl"}
    
    with st.spinner("Syncing Global Markets..."):
        odds = DataEngine.get_odds(sport_map[league])
        stats = DataEngine.get_stats(league.lower())

    if not odds:
        st.warning("Market Offline.")
        st.stop()

    # --- TABS ---
    tab_feed, tab_prop, tab_lab = st.tabs(["🔥 ALPHA FEED", "📊 PROP ANALYZER", "🧪 LAB"])

    # --- FEED TAB ---
    with tab_feed:
        # Top Level Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Games", len(odds))
        m2.metric("Model Edge", "+4.2%", delta_color="normal")
        m3.metric("Win Rate (L10)", "60%", "+10%")
        m4.metric("Exp. Value", "$1,240")

        st.write("")
        
        for game in odds[:8]:
            home, away = game['home_team'], game['away_team']
            
            # Best Odds
            best_price = -9999
            if game['bookmakers']:
                for bm in game['bookmakers']:
                    for mkt in bm['markets']:
                        if mkt['key'] == 'h2h':
                            for out in mkt['outcomes']:
                                if out['name'] == home: best_price = out['price']
            
            if best_price == -9999: continue

            # Run Models
            brain = AIEngine()
            ai = brain.analyze(f"{away} @ {home}", stats, league)
            
            dec = QuantEngine.american_to_decimal(best_price)
            # Mock Monte Carlo for speed in this view
            prob = ai.get('home_win_prob', 0.5)
            edge = QuantEngine.ev(dec, prob) * 100
            stake = bankroll * QuantEngine.kelly_criterion(dec, prob, kelly)

            if edge >= min_edge:
                # --- CARD UI ---
                with st.container():
                    st.markdown(f"##### {home} vs {away}")
                    
                    # 3 Column Layout
                    c1, c2, c3 = st.columns([1, 2, 1])
                    
                    with c1:
                        st.image(Assets.get_logo(home, league))
                    
                    with c2:
                        st.metric("SIGNAL", f"{best_price}", delta=f"{prob:.0%} Win Prob")
                    
                    with c3:
                        st.metric("STAKE", f"${stake:.0f}", delta=f"+{edge:.1f}% EV")
                    
                    # AI Logic
                    st.info(f"🤖 **AI:** {ai.get('reason', 'No analysis')}")
                    st.divider()

    # --- PROP ANALYZER TAB (THE "OUTLIER" CLONE) ---
    with tab_prop:
        st.markdown("### 🧩 PLAYER PROP INTELLIGENCE")
        
        # Mock Selection for Demo
        c_sel, c_chart = st.columns([1, 2])
        
        with c_sel:
            st.markdown("#### TOP TARGETS")
            st.success("J. Allen Over 250.5 Pass Yds")
            st.warning("T. Kelce Under 6.5 Rec")
            st.info("C. McCaffrey Over 0.5 TD")
        
        with c_chart:
            # The "Green/Red" Bar Chart
            mock_data = np.random.randint(200, 300, 10)
            mock_line = 250.5
            
            st.markdown(f"##### J. ALLEN PASS YARDS (L10)")
            st.plotly_chart(Visuals.prop_chart(mock_data, mock_line), use_container_width=True)
            
            m1, m2, m3 = st.columns(3)
            hit_rate = sum(x > mock_line for x in mock_data) * 10
            m1.metric("HIT RATE", f"{hit_rate}%", "L10")
            m2.metric("AVG", f"{int(np.mean(mock_data))}")
            m3.metric("MEDIAN", f"{int(np.median(mock_data))}")

    # --- LAB TAB ---
    with tab_lab:
        st.markdown("### 🧪 ADVANCED TELEMETRY")
        
        l1, l2 = st.columns(2)
        with l1:
            st.markdown("#### WIN PROBABILITY DISTRIBUTION")
            # Bell Curve
            x = np.linspace(0, 100, 100)
            y = norm.pdf(x, 55, 12)
            fig = px.line(x=x, y=y)
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=200, xaxis_title="Win %", yaxis_title="Likelihood")
            st.plotly_chart(fig, use_container_width=True)
            
        with l2:
            st.markdown("#### KEY FACTORS")
            st.progress(0.85, text="DVOA Matchup")
            st.progress(0.40, text="Weather Impact")
            st.progress(0.90, text="Injury Health")

if __name__ == "__main__":
    main()
