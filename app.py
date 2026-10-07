import streamlit as st

from src.data import SEASONS
from src.loaders import MODEL_PATH, load_model, load_players

st.set_page_config(page_title="Forventede FPL-poeng", page_icon=":material/sports_soccer:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/players.py", title="Spillere", icon=":material/groups:"),
        st.Page("app_pages/best_team.py", title="Beste lag", icon=":material/trophy:", url_path="beste-lag"),
        st.Page("app_pages/my_team.py", title="Mitt lag", icon=":material/person:", url_path="mitt-lag"),
        st.Page("app_pages/models.py", title="Modeller", icon=":material/monitoring:", url_path="modeller"),
        st.Page("app_pages/live_test.py", title="Live-test", icon=":material/fact_check:", url_path="live-test"),
    ],
    position="top",
)

# Streamlit sletter verdien til et felt som ikke vises, så lag-ID-en må holdes i live når man bytter side.
if "entry_id" in st.session_state:
    st.session_state.entry_id = st.session_state.entry_id

st.title("Forventede FPL-poeng")

if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
    st.error("Fant ingen trent modell. Kjør `python -m src.train` først.", icon=":material/error:")
    st.stop()

bundle = load_model()
metrics = bundle["metrics"]

with st.sidebar:
    # Modellen med lavest feil på testsettet er forhåndsvalgt.
    model_names = list(bundle["models"])
    model_name = st.selectbox(
        "Modell",
        model_names,
        index=model_names.index(metrics["mae"].idxmin()),
        help="Modellen som brukes til prediksjonene på alle sidene. De sammenlignes på siden Modeller.",
    )
    season = st.selectbox("Sesong", SEASONS, index=len(SEASONS) - 1)
    played, upcoming = load_players(season)
    # Spillere uten kamp i neste runde har ingen motstander og vises ikke. Etter siste runde er det ingen igjen.
    upcoming = upcoming.dropna(subset=["opponent"])
    next_gameweek = None if upcoming.empty else int(upcoming["GW"].iat[0])
    # Første gameweek mangler i dataene, så vi må hente den fra de faktiske kampene.
    gameweeks = sorted((int(gw) for gw in played["GW"].unique()), reverse=True)
    gameweek = st.selectbox(
        "Runde",
        gameweeks if next_gameweek is None else [next_gameweek, *gameweeks],
        format_func=lambda gw: f"Runde {gw} (neste)" if gw == next_gameweek else f"Runde {gw}",
    )

is_played = gameweek != next_gameweek
if not is_played:
    players = upcoming.copy()
    caption = f"Prediksjoner for runde {gameweek} i {season}, basert på formen til og med runde {gameweek - 1}."
else:
    players = played[played["GW"] == gameweek].copy()
    caption = (
        f"Prediksjoner for runde {gameweek} i {season}, basert på formen før runden. "
        "Modellen er trent på disse kampene, så den treffer bedre her enn den vil gjøre på nye kamper."
    )

model = bundle["models"][model_name]
players["predicted_points"] = model.predict(players[bundle["features"]])
players["price"] = players["value"] / 10
players["fixture"] = players["opponent"] + players["was_home"].map({1: " (H)", 0: " (B)"})

# Delt med st.session_state for å kunne bruke dem i andre sider.
st.session_state.players = players
st.session_state.is_played = is_played
st.session_state.model = model

st.caption(f"{caption} {model_name} har en snittfeil (MAE) på {metrics.loc[model_name, 'mae']:.2f} poeng på testsettet.")

page.run()
