import streamlit as st

from src.charts import error_bars
from src.data import CURRENT_SEASON
from src.loaders import load_model, load_players
from src.models import BASELINE

MEASURES = {"Alle spillere": "mae", "Spillere som spilte": "mae_played"}
# Faste farger per modell, i en rekkefølge der naboene kan skilles fra hverandre også ved fargeblindhet.
COLORS = {"Gradient boosting": "#199e70", "Random forest": "#3987e5", "Lineær regresjon": "#d55181", BASELINE: "#c98500"}

bundle = load_model()
metrics, backtest = bundle["metrics"], bundle["backtest"]

st.caption(
    f"Bruker alltid {CURRENT_SEASON}, uansett hva som er valgt i sidepanelet. "
    "Feilen er gjennomsnittlig absolutt feil (MAE) i poeng per spiller, så lavere er bedre."
)

measure = st.segmented_control(
    "Mål feilen for",
    list(MEASURES),
    default="Alle spillere",
    help="De fleste spillerne i spillet spiller ikke og får 0 poeng, noe som trekker feilen for alle spillere ned.",
)
column = MEASURES[measure or "Alle spillere"]

best = metrics[column].idxmin()
improvement = 1 - metrics.loc[best, column] / metrics.loc[BASELINE, column]
with st.container(horizontal=True):
    st.metric("Beste modell", best, border=True)
    st.metric("Feil, beste modell", f"{metrics.loc[best, column]:.3f}", border=True)
    st.metric(f"Feil, {BASELINE.lower()}", f"{metrics.loc[BASELINE, column]:.3f}", border=True)
    st.metric("Forbedring fra baseline", f"{improvement:.1%}", border=True)

st.subheader("Feil på testsettet")
st.caption(
    f"Hver modell er trent på de eldste 80 % av kampene og testet på de nyeste 20 % "
    f"({bundle['n_test']} spillerkamper)."
)
chart, table = st.columns([3, 2])
with chart:
    st.altair_chart(error_bars(metrics[column]), alt="Feil på testsettet per modell")
with table:
    st.dataframe(
        metrics.sort_values(column),
        column_config={
            "mae": st.column_config.NumberColumn("Alle spillere", format="%.3f"),
            "mae_played": st.column_config.NumberColumn("Spillere som spilte", format="%.3f"),
        },
        alt="Feil på testsettet per modell som tabell",
    )

st.subheader("Feil per runde")
st.caption(
    "For hver runde denne sesongen er modellene trent bare på kampene før runden, og så brukt til å predikere den. "
    "Det viser hvor godt de ville truffet om de hadde vært i bruk da."
)
# Rundene vises som tekst slik at aksen ikke får desimaler.
per_gameweek = backtest.assign(GW="Runde " + backtest["GW"].astype(str).str.zfill(2))
st.line_chart(
    per_gameweek.pivot(index="GW", columns="model", values=column)[list(COLORS)],
    color=list(COLORS.values()),
    x_label="Runde",
    y_label="MAE (poeng)",
    alt="Feil per runde for hver modell",
)

_, upcoming = load_players(CURRENT_SEASON)
upcoming = upcoming.dropna(subset=["opponent"])
if not upcoming.empty:
    st.subheader(f"Prediksjoner for runde {int(upcoming['GW'].iat[0])}")
    st.caption("Hva hver modell forventer av hver spiller i neste runde. Klikk på en kolonne for å sortere.")
    predictions = upcoming[["name", "team"]].copy()
    predictions["fixture"] = upcoming["opponent"] + upcoming["was_home"].map({1: " (H)", 0: " (B)"})
    for name, model in bundle["models"].items():
        predictions[name] = model.predict(upcoming[bundle["features"]])
    # Sortert etter snittet av modellene, slik at ingen enkeltmodell bestemmer rekkefølgen.
    order = predictions[list(bundle["models"])].mean(axis=1).sort_values(ascending=False).index
    st.dataframe(
        predictions.loc[order],
        column_config={
            "name": "Spiller",
            "team": "Lag",
            "fixture": "Kamp",
            **{name: st.column_config.NumberColumn(name, format="%.2f") for name in bundle["models"]},
        },
        hide_index=True,
        height=420,
        alt="Prediksjoner for neste runde fra hver modell",
    )
