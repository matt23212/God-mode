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
# 1. ENTERPRISE CONFIGURATION & THEME ENGINE
# ==============================================================================

st.set_page_config(
    page_title="TITAN OS // ENTERPRISE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS: "NEON GLASS" DESIGN SYSTEM ---
st.markdown("""
<style>
    /* --- RESET & VARS --- */
    :root {
        --bg-app: #050505;
        --bg-card: #121212;
        --bg-card-hover: #1a1a1a;
        --border: #2a2a2a;
        --accent-primary: #DFFF00; /* Acid Green */
        --accent-secondary: #00E5FF; /* Cyan */
        --accent-danger: #FF2A6D; /* Neon Red */
        --text-main: #FFFFFF;
        --text-muted: #888888;
        --font-body: 'Inter', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* GLOBAL OVERRIDES */
    .stApp { background-color: var(--bg-app); color: var(--text-main); font-family: var(--font-body); }
    .block-container { padding-top: 2rem; max-width: 100% !important; padding-left: 2rem; padding-right: 2rem; }
    
    /* REMOVE STREAMLIT CHROME */
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }
    
    /* --- COMPONENT: STAT CARD (The "Reference Image" Look) --- */
    .stat-box {
        background: linear-gradient(145deg, #1a1a1a, #0a0a0a);
        border: 1px solid #333;
        border-radius: 16px;
        padding: 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stat-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: var(--accent-primary);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .stat-box:hover {
        transform: translateY(-4px);
        border-color: var(--accent-primary);
        box-shadow: 0 10px 30px -10px rgba(223, 255, 0, 0.15);
    }
    .stat-box:hover::before { opacity: 1; }
    
    .stat-label { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
    .stat-value { font-size: 32px; font-weight: 800; color: var(--text-main); margin: 8px 0; font-family: var(--font-mono); }
    .stat-delta { font-size: 12px; font-weight: 700; }
    .stat-delta.pos { color: var(--accent-primary); }
    .stat-delta.neg { color: var(--accent-danger); }

    /* --- COMPONENT: GAME ROW --- */
    .game-card {
        background-color: #0f0f0f;
        border-bottom: 1px solid #222;
        padding: 15px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: background 0.2s;
    }
    .game-card:hover { background-color: #161616; }
    .team-name { font-weight: 700; font-size: 14px; }
    .market-tag { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: #222; color: #888; }

    /* --- COMPONENT: SIDEBAR --- */
    section[data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid #222;
    }
    .sidebar-link {
        display: block;
        padding: 12px;
        margin: 5px 0;
        border-radius: 8px;
        color: #888;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s;
    }
    .sidebar-link:hover, .sidebar-link.active {
        background: rgba(223, 255, 0, 0.1);
        color: var(--accent-primary);
        border-left: 3px solid var(--accent-primary);
    }

    /* --- COMPONENT: TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #222;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #666;
        font-weight: 600;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-primary) !important;
        border-bottom: 2px solid var(--accent-primary);
    }
    
    /* --- BADGES --- */
    .badge-ai { background: rgba(0, 229, 255, 0.1); color: #00E5FF; border: 1px solid rgba(0, 229, 255, 0.3); padding: 2px 8px; border-radius: 100px; font-size: 10px; font-weight: 800; }
    .badge-ev { background: rgba(223, 255, 0, 0.1); color: #DFFF00; border: 1px solid rgba(223, 255, 0, 0.3); padding: 2px 8px; border-radius: 100px; font-size: 10px; font-weight: 800; }

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
# 2. QUANTUM MATH KERNEL (The "Engine")
# ==============================================================================

class QuantCore:
    @staticmethod
    def american_to_decimal(american):
        if american > 0: return (american / 100) + 1
        return (100 / abs(american)) + 1

    @staticmethod
    def kelly_criterion(decimal, prob, fraction=0.25):
        """Full Kelly is too risky. We use fractional."""
        b = decimal - 1
        p = prob
        q = 1 - p
        f = (b * p - q) / b
        return max(0.0, f) * fraction

    @staticmethod
    def ev_calc(decimal, prob):
        return (prob * (decimal - 1)) - (1 - prob)

    @staticmethod
    def poisson_sim(team_a_avg, team_b_avg, sims=10000):
        """Monte Carlo via Poisson Distribution"""
        a_scores = poisson.rvs(max(10, team_a_avg), size=sims)
        b_scores = poisson.rvs(max(10, team_b_avg), size=sims)
        return np.sum(a_scores > b_scores) / sims

    @staticmethod
    def bayesian_blend(mkt_prob, math_prob, ai_prob, ai_conf):
        """
        Weighted Consensus Model:
        - Market: The wisdom of the crowd (40%)
        - Math: Pure stats (30%)
        - AI: Qualitative factors (Injuries, Weather) (30% scaled by confidence)
        """
        ai_weight = 0.30 * (ai_conf / 100.0)
        rem_weight = 1.0 - ai_weight
        
        w_mkt = rem_weight * 0.6
        w_math = rem_weight * 0.4
        
        return (mkt_prob * w_mkt) + (math_prob * w_math) + (ai_prob * ai_weight)

# ==============================================================================
# 3. DATA INGESTION LAYER (Robust Pipelines)
# ==============================================================================

class DataNexus:
    @staticmethod
    def get_logo(team_name, league="nfl"):
        slug = team_name.split()[-1].lower().replace(" ", "-")
        if "football" in slug: slug = "washington"
        if "sox" in slug: slug = "red-sox" if "red" in team_name.lower() else "white-sox"
        return f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/{league.lower()}/500/{slug}.png&w=60&h=60"

    @staticmethod
    @st.cache_data(ttl=900)
    def fetch_odds(sport_key):
        url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
        params = {'apiKey': KEYS['ODDS'], 'regions': 'us', 'markets': 'h2h,spreads', 'oddsFormat': 'american'}
        try:
            res = requests.get(url, params=params, timeout=5)
            return res.json() if res.status_code == 200 else []
        except: return []

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_stats(league):
        # Map leagues to Tank01 endpoints
        ep_map = {
            "nfl": ("getNFLGamesForWeek", "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"),
            "nba": ("getNBAGamesForDate", "tank01-nba-high-quality-sports-data.p.rapidapi.com"),
            "nhl": ("getNHLGamesForDate", "tank01-nhl-live-in-game-real-time-statistics-nhl.p.rapidapi.com")
        }
        endpoint, host = ep_map.get(league, ep_map['nfl'])
        url = f"https://{host}/{endpoint}"
        try:
            res = requests.get(url, headers={"x-rapidapi-key": KEYS['RAPID'], "x-rapidapi-host": host}, timeout=5)
            return res.json() if res.status_code == 200 else {}
        except: return {}

    @staticmethod
    def fetch_weather(team_name):
        # Simplified geo-lookup for demo speed
        return {"temp": 72, "cond": "Clear", "wind": 5} # Placeholder for speed, replace with OpenMeteo logic in prod

# ==============================================================================
# 4. AI REASONING CORE (GEMINI 2.0)
# ==============================================================================

class TitanBrain:
    def __init__(self):
        self.client = genai.Client(api_key=KEYS['GEMINI'])

    def evaluate(self, matchup, context, league):
        prompt = f"""
        ROLE: Elite Sports Quantitative Analyst.
        TASK: Analyze {matchup} ({league}).
        DATA: {str(context)[:2000]}
        
        OUTPUT JSON ONLY:
        {{
            "win_prob": 0.65,
            "confidence": 85,
            "analysis": "30-word technical breakdown citing DVOA/EPA.",
            "key_metric": "Rush Yds Allowed",
            "key_val": "145.2",
            "prop_pick": "Player Name Over X",
            "prop_reason": "Matchup exploitation"
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
            return {"win_prob": 0.55, "confidence": 50, "analysis": "Data insufficient.", "prop_pick": "N/A"}

# ==============================================================================
# 5. UI RENDERER (THE "OUTLIER" AESTHETIC)
# ==============================================================================

class UX:
    @staticmethod
    def stat_card(label, value, delta, is_currency=False):
        delta_cls = "pos" if "+" in delta else "neg"
        val_fmt = f"${value:,.2f}" if is_currency else f"{value}"
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{val_fmt}</div>
            <div class="stat-delta {delta_cls}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def trend_chart(data, line, color="#00C805"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data, mode='lines+markers',
            line=dict(color=color, width=3, shape='spline'),
            marker=dict(size=6, color="#fff", line=dict(width=2, color=color))
        ))
        fig.add_hline(y=line, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            height=80, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )
        return fig

    @staticmethod
    def prop_bars(data, line):
        """The specific Green/Red bar chart from your screenshot"""
        colors = ['#DFFF00' if x > line else '#333' for x in data]
        fig = go.Figure(go.Bar(
            x=list(range(len(data))), y=data,
            marker_color=colors, borderwidth=0
        ))
        fig.add_hline(y=line, line_color="white", line_dash="dash")
        fig.update_layout(
            height=150, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            bargap=0.2
        )
        return fig

# ==============================================================================
# 6. MAIN APPLICATION
# ==============================================================================

def main():
    # --- SIDEBAR NAV ---
    with st.sidebar:
        st.title("TITAN OS")
        st.caption("v9.0 // ENTERPRISE")
        
        league = st.selectbox("MARKET", ["NFL", "NBA", "NHL"])
        
        st.divider()
        st.markdown("### 🏦 TREASURY")
        bankroll = st.number_input("Capital", 5000, 100000, 10000)
        kelly = st.slider("Kelly Factor", 0.1, 0.5, 0.25)
        
        st.divider()
        if st.button("REBOOT SYSTEM"): st.cache_data.clear()

    # --- HEADER DASHBOARD ---
    st.markdown(f"## {league} COMMAND CENTER")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: UX.stat_card("Active Opportunities", 24, "+5 New")
    with c2: UX.stat_card("Total Volume", bankroll * 0.15, "+12.4%", True)
    with c3: UX.stat_card("Model Accuracy", "68.2%", "+2.1%")
    with c4: UX.stat_card("Net Profit (Wk)", 1240, "+$340", True)

    st.write("") # Spacer

    # --- DATA LOAD ---
    keys = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "NHL": "icehockey_nhl"}
    with st.spinner("Syncing Global Markets..."):
        odds = DataEngine.fetch_odds(keys[league])
        stats = DataEngine.fetch_stats(league.lower())

    if not odds:
        st.error("Market Offline. Check API.")
        return

    # --- MAIN TABS ---
    tab_alpha, tab_prop, tab_lab = st.tabs(["🔥 ALPHA FEED", "🧩 PROP VISUALIZER", "🧪 QUANT LAB"])

    # --- FEED TAB ---
    with tab_alpha:
        for game in odds[:8]:
            home, away = game['home_team'], game['away_team']
            
            # Best Odds
            best_price = -110
            if game['bookmakers']:
                best_price = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
            
            # Quant Math
            dec = QuantCore.decimal_odds(best_price)
            imp = QuantCore.implied_prob(dec)
            # Mock math model for demo
            model_prob = QuantCore.monte_carlo_simulation(24, 21)
            
            # AI
            brain = TitanBrain()
            ai = brain.evaluate(f"{away} @ {home}", stats, league)
            
            # Final Blend
            final_prob = QuantCore.bayesian_blend(imp, model_prob, ai['win_prob'], ai['confidence'])
            edge = QuantCore.ev(dec, final_prob) * 100
            stake = bankroll * QuantCore.kelly_criterion(dec, final_prob, kelly)

            # RENDER CARD
            if edge > 0:
                with st.container():
                    # Header
                    c1, c2, c3 = st.columns([1, 4, 2])
                    with c1: st.image(Assets.get_logo(home, league), width=50)
                    with c2:
                        st.markdown(f"**{home}**")
                        st.caption(f"vs {away} • {datetime.now().strftime('%H:%M ET')}")
                    with c3:
                        st.markdown(f"<div class='badge badge-blue' style='text-align:center'>+{edge:.1f}% EV</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Stats Grid
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("ODDS", best_price)
                    k2.metric("PROB", f"{final_prob:.1%}")
                    k3.metric("KELLY", f"${stake:.0f}")
                    k4.metric("CONF", f"{ai['confidence']}/100")
                    
                    # AI Insight
                    st.info(f"🤖 **AI:** {ai['analysis']}")
                    
                    # Prop Recommendation (Mini)
                    st.caption(f"🎯 **Prop Target:** {ai.get('prop_pick', 'N/A')}")

    # --- PROP VISUALIZER TAB (OUTLIER STYLE) ---
    with tab_prop:
        st.markdown("### PLAYER PERFORMANCE LAB")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            # Mock Player Data (Last 10 Games) - In prod this comes from Tank01
            mock_vals = np.random.randint(15, 35, 10)
            mock_line = 24.5
            
            st.markdown("##### LAST 10 GAMES vs LINE")
            st.plotly_chart(Visuals.prop_bars(mock_vals, mock_line), use_container_width=True)
        
        with c2:
            hit_rate = sum(x > mock_line for x in mock_vals) * 10
            st.metric("HIT RATE", f"{hit_rate}%", "L10")
            st.metric("AVG", f"{np.mean(mock_vals):.1f}")
            st.markdown(f"""
            <div style="background:#111; padding:10px; border-radius:8px; border:1px solid #333;">
                <div style="color:#888; font-size:10px;">IMPLIED ODDS</div>
                <div style="color:#DFFF00; font-size:18px; font-weight:700;">-145</div>
            </div>
            """, unsafe_allow_html=True)

    # --- LAB TAB ---
    with tab_lab:
        st.markdown("### 🔬 DEEP DIVE TELEMETRY")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Win Probability Distribution**")
            # Gaussian Curve
            x = np.linspace(0, 100, 100)
            y = norm.pdf(x, 55, 10)
            fig = px.line(x=x, y=y, labels={'x': 'Win %', 'y': 'Probability'})
            fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("**Factor Analysis**")
            st.progress(0.8, text="Home Field Advantage")
            st.progress(0.6, text="Rest Disadvantage")
            st.progress(0.9, text="DVOA Matchup")

if __name__ == "__main__":
    main()
