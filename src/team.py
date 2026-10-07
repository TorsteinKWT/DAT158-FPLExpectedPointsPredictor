"""Velg den beste startelleveren ut fra et sett med prediksjoner."""

import pandas as pd

TEAM_SIZE = 11
# Minimum og maksimum antall spillere for FPL formasjonen.
FORMATION_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def pick_best_team(players: pd.DataFrame, points: str = "predicted_points") -> pd.DataFrame:
    """Returner de elleve spillerne med høyest samlet poengsum i en gyldig formasjon.

    Bare formasjonsreglene gjelder: det er ikke noe budsjett og ingen grense per klubb.
    """
    ranked = players.sort_values(points, ascending=False).reset_index(drop=True)

    # Fyll inn minimum antall spillere for hver posisjon først, og fyll deretter opp til maks antall spillere.
    picked = []
    for position, (minimum, _) in FORMATION_LIMITS.items():
        picked += list(ranked.index[ranked["position"] == position][:minimum])
    counts = {position: minimum for position, (minimum, _) in FORMATION_LIMITS.items()}

    for index, position in ranked["position"].drop(picked).items():
        if len(picked) == TEAM_SIZE:
            break
        if counts[position] < FORMATION_LIMITS[position][1]:
            picked.append(index)
            counts[position] += 1

    return ranked.loc[picked].sort_values(points, ascending=False)
