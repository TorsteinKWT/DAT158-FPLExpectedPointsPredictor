import streamlit as st

from src.data import POSITION_NAMES, POSITIONS

# Samme fargeskala som FPL bruker: grønn for lette kamper, rosa for vanskelige.
FDR_STYLES = {
    1: "background-color: #00FF87; color: #0B0E14",
    2: "background-color: #00FF87; color: #0B0E14",
    3: "background-color: #A3ADC2; color: #0B0E14",
    4: "background-color: #FF2882; color: #FFFFFF",
    5: "background-color: #FF2882; color: #FFFFFF",
}

players = st.session_state.players

with st.sidebar:
    positions = st.pills(
        "Posisjoner", POSITIONS, selection_mode="multi", default=POSITIONS, format_func=POSITION_NAMES.get
    )
    search = st.text_input("Søk etter spiller eller lag")

columns = ["name", "team", "position", "fixture", "fdr", "predicted_points"]
if st.session_state.is_played:
    columns.append("total_points")
columns += ["price", "total_points_roll", "minutes_roll"]
st.caption("Én rad per kamp, så spillere med dobbeltrunde kan stå to ganger.")

shown = players[players["position"].isin(positions)]
if search:
    matches = shown["name"].str.contains(search, case=False) | shown["team"].str.contains(search, case=False)
    shown = shown[matches]
shown = shown.sort_values("predicted_points", ascending=False)

st.dataframe(
    shown[columns].replace({"position": POSITION_NAMES}).style.map(FDR_STYLES.get, subset=["fdr"]),
    column_config={
        "name": "Spiller",
        "team": "Lag",
        "position": "Posisjon",
        "fixture": "Kamp",
        "fdr": st.column_config.NumberColumn(
            "FDR", format="%d", help="FPLs vanskelighetsgrad for kampen, fra 1 (lett) til 5 (vanskelig)."
        ),
        "price": st.column_config.NumberColumn("Pris", format="£%.1fm"),
        "total_points_roll": st.column_config.NumberColumn(
            "Poengsnitt", format="%.1f", help="Snitt over de siste fem kampene."
        ),
        "minutes_roll": st.column_config.NumberColumn(
            "Minutter", format="%.0f", help="Snitt over de siste fem kampene."
        ),
        "predicted_points": st.column_config.ProgressColumn(
            "Forventede poeng",
            format="%.2f",
            min_value=0,
            max_value=float(players["predicted_points"].max()),
        ),
        "total_points": "Faktiske poeng",
    },
    hide_index=True,
    height=600,
    alt="Spillere rangert etter forventede poeng",
)
