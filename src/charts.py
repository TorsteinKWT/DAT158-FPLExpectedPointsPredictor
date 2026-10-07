"""Diagrammer som brukes på flere sider i appen."""

import altair as alt
import pandas as pd


def error_bars(errors: pd.Series) -> alt.Chart:
    """Liggende søyler med feilen per modell, lavest øverst."""
    data = errors.rename("error").rename_axis("model").reset_index()
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            x=alt.X("error:Q", title="MAE (poeng)"),
            # Fast bredde på aksen, ellers kuttes de lengste modellnavnene når skrifttypen lastes inn sent.
            y=alt.Y("model:N", sort="x", title=None, axis=alt.Axis(labelLimit=220, minExtent=150)),
            tooltip=[alt.Tooltip("model:N", title="Modell"), alt.Tooltip("error:Q", title="MAE", format=".3f")],
        )
        .properties(height=180)
    )
