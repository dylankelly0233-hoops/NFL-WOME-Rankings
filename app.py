import streamlit as st
import nflreadpy as nfl
import pandas as pd

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NFL WOME Betting Model",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Wins Over Market Expectation (WOME)")
st.markdown("""
**The Strategy:** This dashboard identifies teams that have over-performed or under-performed 
market expectations (Closing Moneyline) up to the selected week. 
We look for **Regression to the Mean**: Betting on under-valued teams against over-valued teams.
""")

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_nfl_schedule(year):
    """
    Loads and caches the schedule data to avoid re-downloading on every click.
    """
    try:
        data = nfl.load_schedules(seasons=[year])
        # Convert polars to pandas if necessary (newer nflreadpy versions)
        if hasattr(data, "to_pandas"):
            data = data.to_pandas()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_implied_probability(moneyline):
    if pd.isna(moneyline): return 0.5
    if moneyline < 0:
        return abs(moneyline) / (abs(moneyline) + 100)
    else:
        return 100 / (moneyline + 100)

def get_win_result(row, team_type):
    score_diff = row['result']
    if pd.isna(score_diff): return 0 # Should not happen in filtered data
    if score_diff == 0: return 0.5
    
    if team_type == 'home':
        return 1.0 if score_diff > 0 else 0.0
    else:
        return 1.0 if score_diff < 0 else 0.0

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")
selected_season = st.sidebar.number_input("Season", min_value=2020, max_value=2026, value=2025)

# Load Data based on season
schedule = load_nfl_schedule(selected_season)

if schedule.empty:
    st.warning("No data found for this season.")
    st.stop()

# Determine available weeks
max_week = int(schedule['week'].max())
current_week_default = max(1, int(schedule[schedule['result'].isna()]['week'].min())) 
# Default to the first week without results, or week 1

selected_week = st.sidebar.slider("Target Week (Betting On)", 1, 22, current_week_default)
threshold = st.sidebar.slider("Bet Signal Threshold (Diff in WOME)", 0.5, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Threshold Tip:**\n\n"
    "Lower (0.5) = More Action\n\n"
    "Higher (1.5) = Sharper/Safer"
)

# -----------------------------------------------------------------------------
# 4. DATA PROCESSING
# -----------------------------------------------------------------------------
# Split data: Past (Completed games before target week) vs Upcoming (Target week)
past_games = schedule[(schedule['week'] < selected_week) & (schedule['result'].notna())].copy()
upcoming_games = schedule[schedule['week'] == selected_week].copy()

if past_games.empty:
    st.info(f"No completed games found before Week {selected_week}. Cannot calculate rankings yet.")
    st.stop()

# Calculate WOME
past_games['home_implied'] = past_games['home_moneyline'].apply(calculate_implied_probability)
past_games['away_implied'] = past_games['away_moneyline'].apply(calculate_implied_probability)
past_games['home_win'] = past_games.apply(lambda x: get_win_result(x, 'home'), axis=1)
past_games['away_win'] = past_games.apply(lambda x: get_win_result(x, 'away'), axis=1)

team_stats = {}

for _, row in past_games.iterrows():
    # Home
    h = row['home_team']
    if h not in team_stats: team_stats[h] = {'wins': 0, 'expected': 0}
    team_stats[h]['wins'] += row['home_win']
    team_stats[h]['expected'] += row['home_implied']
    
    # Away
    a = row['away_team']
    if a not in team_stats: team_stats[a] = {'wins': 0, 'expected': 0}
    team_stats[a]['wins'] += row['away_win']
    team_stats[a]['expected'] += row['away_implied']

# Create DataFrame
df_wome = pd.DataFrame.from_dict(team_stats, orient='index')
df_wome['WOME'] = df_wome['wins'] - df_wome['expected']
df_wome = df_wome.sort_values('WOME', ascending=False)
df_wome['Rank'] = range(1, len(df_wome) + 1)
df_wome['WOME'] = df_wome['WOME'].round(2)
df_wome['expected'] = df_wome['expected'].round(2)

# -----------------------------------------------------------------------------
# 5. DASHBOARD LAYOUT
# -----------------------------------------------------------------------------

tab1, tab2 = st.tabs(["🔥 Betting Signals", "📊 Team Rankings"])

# --- TAB 1: BETTING SIGNALS ---
with tab1:
    st.subheader(f"Week {selected_week} Analysis")
    st.write(f"Looking for WOME differentials > {threshold}")
    
    signals_found = False
    
    # Create columns for the game cards
    for _, game in upcoming_games.iterrows():
        home = game['home_team']
        away = game['away_team']
        
        if home in df_wome.index and away in df_wome.index:
            h_stat = df_wome.loc[home]
            a_stat = df_wome.loc[away]
            
            diff = h_stat['WOME'] - a_stat['WOME']
            
            # Formatting logic
            signal_color = "gray"
            recommendation = "No Value"
            
            # LOGIC
            if diff > threshold:
                # Home is overvalued, Away is undervalued -> BET AWAY
                signals_found = True
                st.success(f"**BET SIGNAL: {away}** (Fade {home})")
                with st.expander(f"See Details: {away} @ {home}"):
                    st.write(f"**{home}**: Rank #{h_stat['Rank']} (WOME: {h_stat['WOME']})")
                    st.write(f"**{away}**: Rank #{a_stat['Rank']} (WOME: {a_stat['WOME']})")
                    st.write(f"Differential: {diff:.2f}")

            elif diff < -threshold:
                # Away is overvalued, Home is undervalued -> BET HOME
                signals_found = True
                st.success(f"**BET SIGNAL: {home}** (Fade {away})")
                with st.expander(f"See Details: {away} @ {home}"):
                    st.write(f"**{home}**: Rank #{h_stat['Rank']} (WOME: {h_stat['WOME']})")
                    st.write(f"**{away}**: Rank #{a_stat['Rank']} (WOME: {a_stat['WOME']})")
                    st.write(f"Differential: {abs(diff):.2f}")

    if not signals_found:
        st.info("No games meet your threshold criteria this week.")

# --- TAB 2: RANKINGS ---
with tab2:
    st.subheader("Team Rankings by Performance vs Market")
    
    # Color formatter for the dataframe
    def color_wome(val):
        color = '#d4edda' if val > 0 else '#f8d7da'
        return f'background-color: {color}'

    # Display the dataframe with style
    st.dataframe(
        df_wome[['Rank', 'wins', 'expected', 'WOME']].style.applymap(color_wome, subset=['WOME']),
        use_container_width=True,
        height=800
    )
