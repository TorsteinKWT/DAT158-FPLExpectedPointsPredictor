"""Pick the best starting eleven from a set of predictions."""

import pandas as pd

TEAM_SIZE = 11
# Minimum and maximum number of players per position in a valid FPL formation.
FORMATION_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def pick_best_team(players: pd.DataFrame, points: str = "predicted_points") -> pd.DataFrame:
    """Return the eleven players with the highest total points in a valid formation.

    Only the formation rules apply: there is no budget and no limit per club.
    """
    ranked = players.sort_values(points, ascending=False).reset_index(drop=True)

    # Fill the minimum for each position first, then the open spots with the best players left.
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
