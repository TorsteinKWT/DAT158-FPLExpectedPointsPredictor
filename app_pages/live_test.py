import pandas as pd
import streamlit as st

from src.charts import error_bars
from src.data import CURRENT_SEASON, POSITION_NAMES
from src.loaders import load_players
from src.snapshot import load_selection, load_snapshots

st.caption(
    "Prediksjonene her ble lagret før rundene ble spilt, og sammenlignes med fasiten etterpå. "
    f"Bruker alltid {CURRENT_SEASON}, uansett hva som er valgt i sidepanelet."
)

snapshots, selection = load_snapshots(), load_selection()
if snapshots.empty:
    st.info("Ingen runder er lagret ennå. Kjør `python -m src.snapshot` før neste runde.", icon=":material/info:")
    st.stop()

# Modellene er kolonnene mellom kampen og tidspunktet filen ble lagret.
columns = list(snapshots.columns)
model_names = columns[columns.index("fixture") + 1 : columns.index("created_at")]

played, _ = load_players(CURRENT_SEASON)
actual = played.groupby(["GW", "element"], as_index=False)[["total_points", "minutes"]].sum()
results = snapshots.merge(actual, on=["GW", "element"], how="left")
played_gameweeks = sorted(results.loc[results["total_points"].notna(), "GW"].unique())

status = snapshots.groupby("GW", as_index=False).agg(saved=("created_at", "first"), players=("element", "count"))
status["saved"] = pd.to_datetime(status["saved"]).dt.tz_convert("Europe/Oslo").dt.strftime("%d.%m.%Y kl. %H:%M")
status["status"] = status["GW"].isin(played_gameweeks).map({True: "Spilt", False: "Venter på kampene"})
st.dataframe(
    status,
    column_config={"GW": "Runde", "saved": "Lagret", "players": "Spillere", "status": "Status"},
    hide_index=True,
    width="content",
    alt="Lagrede runder og om de er spilt",
)

scope = st.segmented_control(
    "Vis for",
    ["Alle spillere", "De utvalgte"],
    default="Alle spillere",
    help="De utvalgte er de fem med flest minutter per posisjon da den første runden ble lagret.",
)
if scope == "De utvalgte":
    results = results[results["element"].isin(selection["element"])]

st.subheader("Feil per modell")
finished = results.dropna(subset=["total_points"])
if finished.empty:
    st.info("Ingen av de lagrede rundene er ferdigspilt ennå, så det er ingen fasit å måle mot.", icon=":material/info:")
else:
    st.caption(
        f"Gjennomsnittlig absolutt feil (MAE) i poeng per spiller, så lavere er bedre. "
        f"Bygger på {len(finished)} spillerrunder."
    )
    errors = finished[model_names].sub(finished["total_points"], axis=0).abs()
    per_gameweek = errors.groupby(finished["GW"]).mean().T
    per_gameweek.columns = [f"Runde {gameweek}" for gameweek in per_gameweek.columns]
    per_gameweek["Samlet"] = errors.mean()
    chart, table = st.columns([3, 2])
    with chart:
        st.altair_chart(error_bars(per_gameweek["Samlet"]), alt="Samlet feil per modell på de lagrede rundene")
    with table:
        st.dataframe(
            per_gameweek.sort_values("Samlet"),
            column_config={column: st.column_config.NumberColumn(format="%.3f") for column in per_gameweek},
            alt="Feil per modell og runde",
        )

st.subheader("Prediksjoner og fasit")
gameweek = st.selectbox(
    "Runde", sorted(results["GW"].unique(), reverse=True), format_func=lambda gw: f"Runde {gw}", width=200
)
shown = results[results["GW"] == gameweek]
visible = ["name", "team", "position", "fixture", *model_names]
if gameweek in played_gameweeks:
    visible.append("total_points")
st.dataframe(
    shown.sort_values(model_names[-1], ascending=False)[visible].replace({"position": POSITION_NAMES}),
    column_config={
        "name": "Spiller",
        "team": "Lag",
        "position": "Posisjon",
        "fixture": "Kamp",
        **{name: st.column_config.NumberColumn(name, format="%.2f") for name in model_names},
        "total_points": st.column_config.NumberColumn("Faktiske poeng", format="%d"),
    },
    hide_index=True,
    height=420 if scope != "De utvalgte" else "content",
    alt="Lagrede prediksjoner per spiller, med faktiske poeng når runden er spilt",
)
