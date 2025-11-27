import streamlit as st
import requests
from datetime import datetime, timedelta
from google import genai

# --- PAGE CONFIG (Makes it look like an app) ---
st.set_page_config(page_title="God Mode Terminal", page_icon="🏈", layout="centered")

# --- 1. THE KEYRING ---
# (Securely load keys from Streamlit Secrets or hardcode for personal prototype)
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "34e5a58b5b50587ce21dbe0b33e344dc")
RAPID_API_KEY = st.secrets.get("RAPID_API_KEY", "07d28ccf44mshdfc586c9867d85bp1e1c52jsn1c91d70acc9c")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "289796ecfb2c4d208506c26d37a4d9ba")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDuSrw5wSKaVk3nnaMhbfuufUuDXpMMDkE")

# --- 2. SETTINGS ---
SPORT = 'americanfootball_nfl'
REGION = 'us'
MARKETS = 'h2h'
TEAM_CITIES = {
    "Arizona Cardinals": "Glendale", "Atlanta Falcons": "Atlanta", 
    "Carolina Panthers": "Charlotte", "Dallas Cowboys": "Arlington", 
    "Detroit Lions": "Detroit", "Green Bay Packers": "Green Bay",
    "Las Vegas Raiders": "Paradise", "Los Angeles Chargers": "Inglewood", 
    "Los Angeles Rams": "Inglewood", "Miami Dolphins": "Miami Gardens",
    "Minnesota Vikings": "Minneapolis", "New England Patriots": "Foxborough",
    "New Orleans Saints": "New Orleans", "New York Giants": "East Rutherford",
    "New York Jets": "East Rutherford", "San Francisco 49ers": "Santa Clara",
    "Tampa Bay Buccaneers": "Tampa", "Tennessee Titans": "Nashville",
    "Washington Commanders": "Landover", "Kansas City Chiefs": "Kansas City",
    "Buffalo Bills": "Orchard Park", "Cleveland Browns": "Cleveland",
    "Chicago Bears": "Chicago", "Cincinnati Bengals": "Cincinnati",
    "Denver Broncos": "Denver", "Houston Texans": "Houston",
    "Indianapolis Colts": "Indianapolis", "Jacksonville Jaguars": "Jacksonville",
    "Philadelphia Eagles": "Philadelphia", "Pittsburgh Steelers": "Pittsburgh",
    "Seattle Seahawks": "Seattle", "Baltimore Ravens": "Baltimore"
}

# --- 3. FUNCTIONS ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to save API calls
def get_odds():
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'
    params = {'apiKey': ODDS_API_KEY, 'regions': REGION, 'markets': MARKETS, 'oddsFormat': 'american'}
    return requests.get(url, params=params).json()

@st.cache_data(ttl=3600)
def get_stats():
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
    headers = {"x-rapidapi-key": RAPID_API_KEY, "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"}
    res = requests.get(url, headers=headers)
    return res.json() if res.status_code == 200 else {}

def get_team_news(team_name):
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    url = "https://newsapi.org/v2/everything"
    params = {'q': f'"{team_name}" NFL', 'from': three_days_ago, 'sortBy': 'relevancy', 'language': 'en', 'pageSize': 2, 'apiKey': NEWS_API_KEY}
    res = requests.get(url, params=params)
    if res.status_code == 200:
        articles = res.json().get('articles', [])
        return "\n".join([f"- {a['title']}" for a in articles]) if articles else "No major news."
    return "News Error"

def get_weather(team_name):
    city = TEAM_CITIES.get(team_name, team_name.split(' ')[0])
    geo_res = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={'name': city, 'count': 1}).json()
    if 'results' not in geo_res: return "Location not found"
    lat, lon = geo_res['results'][0]['latitude'], geo_res['results'][0]['longitude']
    wx_res = requests.get("https://api.open-meteo.com/v1/forecast", params={'latitude': lat, 'longitude': lon, 'hourly': 'temperature_2m,precipitation_probability,wind_speed_10m', 'temperature_unit': 'fahrenheit', 'wind_speed_unit': 'mph', 'forecast_days': 1}).json()
    curr_hour = datetime.now().hour
    return f"{wx_res['hourly']['temperature_2m'][curr_hour]}°F | Wind: {wx_res['hourly']['wind_speed_10m'][curr_hour]}mph"

def run_god_mode(game, all_stats):
    client = genai.Client(api_key=GEMINI_API_KEY)
    home, away = game['home_team'], game['away_team']
    
    with st.spinner(f"🧠 Analyzing {away} @ {home}..."):
        odds_str = f"{game['bookmakers'][0]['title']}: {game['bookmakers'][0]['markets'][0]['outcomes']}" if game['bookmakers'] else "N/A"
        weather = get_weather(home)
        news_home, news_away = get_team_news(home), get_team_news(away)
        
        prompt = f"""
        Act as a 'God Mode' Sports Betting AI.
        Target: {away} @ {home}
        DATA: [ODDS: {odds_str}] [WEATHER: {weather}] 
        [NEWS {home}: {news_home}] [NEWS {away}: {news_away}]
        [STATS]: {str(all_stats)[:500]}
        
        OUTPUT:
        Produce a short report.
        1. VIBE: (One emoji + 1 sentence)
        2. WEATHER IMPACT: (High/Low)
        3. VERDICT: (Pick)
        """
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text

# --- 4. THE UI ---
st.title("👁️ God Mode Terminal")
st.caption("Powered by Gemini 2.0 • Odds • News • Stats")

if st.button("Run Analysis"):
    games = get_odds()
    stats = get_stats()
    
    if isinstance(games, list):
        for game in games[:3]: # Top 3 games
            with st.expander(f"🏈 {game['away_team']} @ {game['home_team']}", expanded=True):
                analysis = run_god_mode(game, stats)
                st.markdown(analysis)
    else:
        st.error("Could not load odds.")
