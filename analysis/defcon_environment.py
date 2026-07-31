"""Project Defensive Contribution (DefCon) environments by team for an upcoming season.

DefCon points are the dominant hidden driver of cheap-defender and cheap-midfielder
value since the rule arrived in 2025-26. This script measures each club's DefCon
environment from the completed season, separates *team strength* from *playing style*,
and projects the coming season after manager changes.

Two separate tables are produced because DEF DefCon (10+ CBIT) and MID DefCon
(12+ CBIRT) turn out to be only weakly correlated (r = +0.34) — they are different
phenomena and must be modelled separately.

Run:  python analysis/defcon_environment.py
"""

import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BASE_SEASON = "2025-26"
THRESH = {"DEF": 10, "MID": 12}

# 2026-27 Premier League. `manager` is who takes charge; `prev` is the club whose
# 2025-26 DefCon signature we attribute to them (None = no measurable PL signature).
# `note` records how confident the style prior is.
SQUADS_2627 = [
    # club,           manager,        new?,  prev club (2025-26 PL),  style prior
    ("Arsenal",        "Arteta",       False, "Arsenal",        None),
    ("Aston Villa",    "Emery",        False, "Aston Villa",    None),
    ("Bournemouth",    "Rose",         True,  None,             "high-press, Leipzig school"),
    ("Brentford",      "Andrews",      False, "Brentford",      None),
    ("Brighton",       "Hurzeler",     False, "Brighton",       None),
    ("Chelsea",        "Xabi Alonso",  True,  None,             "possession-dominant"),
    ("Coventry",       "Lampard",      True,  None,             "promoted"),
    ("Crystal Palace", "Sage",         True,  None,             "unknown in PL"),
    ("Everton",        "Moyes",        False, "Everton",        None),
    ("Fulham",         "Arbeloa",      True,  None,             "unknown in PL"),
    ("Hull",           "Jakirovic",    True,  None,             "promoted"),
    ("Ipswich",        "O'Neill",      True,  None,             "promoted, deep block"),
    ("Leeds",          "Farke",        False, "Leeds",          None),
    ("Liverpool",      "Iraola",       True,  "Bournemouth",    "high-press"),
    ("Man City",       "Maresca",      True,  "Chelsea",        "possession-dominant"),
    ("Man Utd",        "Amorim",       False, "Man Utd",        None),
    ("Newcastle",      "Howe",         False, "Newcastle",      None),
    ("Nott'm Forest",  "Glasner",      True,  "Crystal Palace", "deep block, 3-4-2-1"),
    ("Sunderland",     "Le Bris",      False, "Sunderland",     None),
    ("Spurs",          "De Zerbi",     True,  None,             "possession-leaning"),
]
PROMOTED = {"Coventry", "Hull", "Ipswich"}
# style priors for managers with no measurable PL DefCon signature, as a multiplier
# on the club's baseline. These are judgment, not measurement — kept explicit and mild.
STYLE_PRIOR = {
    "Xabi Alonso": 0.92,
    "De Zerbi": 0.95,
    "Rose": 1.06,
    "Sage": 1.00,
    "Arbeloa": 0.97,
    "Lampard": 1.00,
    "Jakirovic": 1.00,
    "O'Neill": 1.04,
}
# how much of a measured manager signature to carry to the new club
MANAGER_WEIGHT = 0.5


def team_table(season=BASE_SEASON):
    gw = pd.read_csv(os.path.join(DATA, season, "gws", "merged_gw.csv"))
    out = {}
    for pos, thr in THRESH.items():
        d = gw[(gw["position"] == pos) & (gw["minutes"] >= 60)].copy()
        d["hit"] = (d["defensive_contribution"] >= thr).astype(int)
        g = d.groupby("team").agg(
            **{f"{pos}_dc": ("defensive_contribution", "mean"),
               f"{pos}_hit": ("hit", "mean")})
        out[pos] = g
    T = pd.concat(out.values(), axis=1)

    T["xg"] = gw.groupby(["team", "fixture"])["expected_goals"].sum().groupby("team").mean()
    full = gw[gw["minutes"] >= 85]
    T["xga"] = (full.groupby(["team", "fixture"])["expected_goals_conceded"].max()
                .groupby("team").mean())
    T["xg_diff"] = T["xg"] - T["xga"]
    return T


def style_residuals(T):
    """Split each club's DefCon into the part team strength explains and the rest (style)."""
    for pos in THRESH:
        slope, icpt = np.polyfit(T["xg_diff"], T[f"{pos}_dc"], 1)
        T[f"{pos}_pred"] = slope * T["xg_diff"] + icpt
        T[f"{pos}_resid"] = T[f"{pos}_dc"] - T[f"{pos}_pred"]
    return T


def project(T):
    promo_base = {pos: T.loc[["Leeds", "Burnley", "Sunderland"], f"{pos}_dc"].mean()
                  for pos in THRESH}
    league = {pos: T[f"{pos}_dc"].mean() for pos in THRESH}
    rows = []
    for club, mgr, is_new, prev, note in SQUADS_2627:
        row = {"club": club, "manager": mgr, "new": "yes" if is_new else "-"}
        for pos in THRESH:
            if club in PROMOTED:
                base = promo_base[pos]
                conf = "low (promoted, no PL data)"
                if mgr in STYLE_PRIOR:
                    base *= STYLE_PRIOR[mgr]
            elif not is_new:
                base = T.loc[club, f"{pos}_dc"]
                conf = "high (squad + manager continuity)"
            else:
                # A club's style residual is part manager, part squad. Without multi-season
                # DefCon data the two cannot be separated, so split it evenly: the departing
                # manager takes half with them, the squad keeps half.
                pred = T.loc[club, f"{pos}_pred"]
                kept = (1 - MANAGER_WEIGHT) * T.loc[club, f"{pos}_resid"]
                if prev is not None:
                    base = pred + kept + MANAGER_WEIGHT * T.loc[prev, f"{pos}_resid"]
                    conf = "medium (measured manager signature)"
                else:
                    base = (pred + kept) * STYLE_PRIOR.get(mgr, 1.0)
                    conf = "low (style prior only)"
            row[f"{pos}_proj"] = base
            row[f"{pos}_2526"] = T.loc[club, f"{pos}_dc"] if club not in PROMOTED else np.nan
            row[f"{pos}_conf"] = conf
        rows.append(row)
    P = pd.DataFrame(rows)
    for pos in THRESH:
        P[f"{pos}_vs_avg"] = P[f"{pos}_proj"] / league[pos] - 1
    return P, league


def main():
    pd.set_option("display.width", 220)
    T = style_residuals(team_table())

    print("=== Drivers of DefCon, 2025-26 (n=20) ===")
    for pos in THRESH:
        r = np.corrcoef(T["xg_diff"], T[f"{pos}_dc"])[0, 1]
        print(f"  {pos}_dc vs team xG difference: r = {r:+.3f} (R2 {r**2:.2f})")
    r = np.corrcoef(T["DEF_dc"], T["MID_dc"])[0, 1]
    print(f"  DEF_dc vs MID_dc: r = {r:+.3f}  -> largely separate phenomena\n")

    print("=== Style residuals 2025-26 (DefCon above/below what team strength predicts) ===")
    print(T.sort_values("DEF_resid", ascending=False)[["DEF_dc", "DEF_resid", "MID_dc", "MID_resid"]]
          .to_string(float_format=lambda v: f"{v:+.2f}"))

    P, league = project(T)
    for pos, label in [("DEF", "DEFENDERS"), ("MID", "MIDFIELDERS")]:
        print(f"\n{'=' * 78}\nTOP TEAMS FOR {label} DefCon — 2026-27 PROJECTION\n{'=' * 78}")
        print(f"(league average 2025-26 = {league[pos]:.2f} DefCon actions/match)")
        cols = ["club", "manager", "new", f"{pos}_2526", f"{pos}_proj", f"{pos}_vs_avg", f"{pos}_conf"]
        s = P.sort_values(f"{pos}_proj", ascending=False)[cols].head(8).copy()
        s[f"{pos}_vs_avg"] = (s[f"{pos}_vs_avg"] * 100).map(lambda v: f"{v:+.0f}%")
        print(s.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n=== Biggest projected movers vs 2025-26 ===")
    P["DEF_delta"] = P["DEF_proj"] - P["DEF_2526"]
    P["MID_delta"] = P["MID_proj"] - P["MID_2526"]
    mv = P.dropna(subset=["DEF_delta"]).copy()
    mv["absmax"] = mv[["DEF_delta", "MID_delta"]].abs().max(axis=1)
    print(mv.nlargest(6, "absmax")[["club", "manager", "DEF_2526", "DEF_proj", "DEF_delta",
                                    "MID_2526", "MID_proj", "MID_delta"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


if __name__ == "__main__":
    main()
