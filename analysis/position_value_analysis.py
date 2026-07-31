"""Positional value analysis for a completed FPL season.

Answers two questions:
  1. Which positions make up the top 30 point scorers?
  2. Where does each £1m of squad budget buy the most points?

Run:  python analysis/position_value_analysis.py [--season 2025-26] [--compare 2024-25]
"""

import argparse
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

POS_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POS_ORDER = ["GK", "DEF", "MID", "FWD"]
# label drift across seasons: 2021-22 mixes GKP and GK for goalkeepers
POS_ALIASES = {"GKP": "GK"}
# "AM" (Assistant Manager) existed only in 2024-25 and is not an outfield position;
# kept as its own category so top-N accounting stays honest, excluded from squad maths
ALL_POS = POS_ORDER + ["AM"]
# squad rules: (min_in_XI, max_in_XI, squad_size)
XI_RULES = {"GK": (1, 1, 2), "DEF": (3, 5, 5), "MID": (2, 5, 5), "FWD": (1, 3, 3)}
# DefCon threshold on the position-aware `defensive_contribution` stat
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12, "GK": 999}


def load_season(season):
    """Return a per-player season frame with prices, points and derived stats."""
    gw = pd.read_csv(os.path.join(DATA, season, "gws", "merged_gw.csv"))
    raw = pd.read_csv(os.path.join(DATA, season, "players_raw.csv"))

    gw = gw.sort_values(["element", "GW"])
    gw["position"] = gw["position"].replace(POS_ALIASES)
    has_defcon = "defensive_contribution" in gw.columns
    if has_defcon:
        thresh = gw["position"].map(DEFCON_THRESHOLD).fillna(999)
        gw["defcon_hits"] = (gw["defensive_contribution"] >= thresh).astype(int)
    else:
        gw["defcon_hits"] = 0

    spec = {
        "name": ("name", "last"),
        "position": ("position", "last"),
        "team": ("team", "last"),
        "points": ("total_points", "sum"),
        "minutes": ("minutes", "sum"),
        "starts": ("starts", "sum"),
        "goals": ("goals_scored", "sum"),
        "assists": ("assists", "sum"),
        "clean_sheets": ("clean_sheets", "sum"),
        "bonus": ("bonus", "sum"),
        "defcon_hits": ("defcon_hits", "sum"),
        "start_price": ("value", "first"),
        "end_price": ("value", "last"),
        "mean_price": ("value", "mean"),
        "appearances": ("GW", "count"),
    }
    # older seasons don't carry every column (e.g. `starts` before 2021-22)
    spec = {k: v for k, v in spec.items() if v[0] in gw.columns}
    agg = gw.groupby("element").agg(**spec)
    agg["start_price"] /= 10.0
    agg["end_price"] /= 10.0
    agg["mean_price"] /= 10.0

    # ownership at season end, for "was this a template pick" context
    own = raw.set_index("id")["selected_by_percent"] if "id" in raw.columns else None
    if own is not None:
        agg["owned_pct"] = agg.index.map(own)

    agg["ppm"] = agg["points"] / agg["start_price"]
    agg["p90"] = np.where(agg["minutes"] > 0, agg["points"] / agg["minutes"] * 90, 0.0)
    agg["defcon_points"] = agg["defcon_hits"] * 2
    agg["defcon_share"] = np.where(agg["points"] > 0, agg["defcon_points"] / agg["points"], 0.0)
    agg["position"] = pd.Categorical(agg["position"], categories=ALL_POS, ordered=True)
    return agg.reset_index(), gw


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def top_n_breakdown(df, n=30):
    section(f"1. TOP {n} SCORERS — WHO THEY WERE")
    top = df.nlargest(n, "points").reset_index(drop=True)
    top.index += 1
    cols = ["name", "position", "points", "start_price", "end_price", "ppm", "minutes", "defcon_points"]
    show = top[cols].copy()
    show.columns = ["player", "pos", "pts", "£start", "£end", "pts/£m", "mins", "defcon"]
    print(show.to_string(float_format=lambda v: f"{v:.1f}"))

    section(f"2. POSITION BREAKDOWN OF THE TOP {n}")
    brk = top.groupby("position", observed=False).agg(
        n=("points", "size"),
        pts_mean=("points", "mean"),
        pts_total=("points", "sum"),
        price_start_mean=("start_price", "mean"),
        ppm_mean=("ppm", "mean"),
    )
    brk["share_of_slots_%"] = brk["n"] / n * 100
    print(brk.to_string(float_format=lambda v: f"{v:.1f}"))

    # how many of each position are even available to field
    section(f"3. TOP {n} SHARE vs SQUAD SLOTS AVAILABLE")
    rows = []
    for pos in POS_ORDER:
        lo, hi, squad = XI_RULES[pos]
        rows.append({
            "pos": pos,
            f"in_top{n}": int((top["position"] == pos).sum()),
            "squad_slots": squad,
            "XI_slots_min": lo,
            "XI_slots_max": hi,
            "pool_size": int((df["position"] == pos).sum()),
        })
    r = pd.DataFrame(rows)
    r["top_per_XI_slot"] = r[f"in_top{n}"] / r["XI_slots_max"]
    print(r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return top


def regulars(df, min_minutes=900):
    return df[df["minutes"] >= min_minutes].copy()


def value_by_position(df):
    section("4. POINTS-PER-MILLION BY POSITION (players with 900+ minutes)")
    reg = regulars(df)
    out = reg.groupby("position", observed=False).agg(
        n=("points", "size"),
        pts_median=("points", "median"),
        pts_p90=("points", lambda s: s.quantile(0.90)),
        pts_max=("points", "max"),
        price_median=("start_price", "median"),
        ppm_median=("ppm", "median"),
        ppm_p90=("ppm", lambda s: s.quantile(0.90)),
        ppm_max=("ppm", "max"),
    )
    print(out.to_string(float_format=lambda v: f"{v:.2f}"))

    section("5. PRICE-BAND EFFICIENCY (900+ minutes)")
    bands = [3.9, 4.5, 5.5, 6.5, 7.5, 9.0, 20.0]
    labels = ["<=4.5", "4.6-5.5", "5.6-6.5", "6.6-7.5", "7.6-9.0", ">9.0"]
    reg["band"] = pd.cut(reg["start_price"], bins=bands, labels=labels)
    tbl = reg.groupby(["position", "band"], observed=True).agg(
        n=("points", "size"),
        pts_mean=("points", "mean"),
        pts_max=("points", "max"),
        ppm_mean=("ppm", "mean"),
    ).dropna()
    print(tbl.to_string(float_format=lambda v: f"{v:.1f}"))


def marginal_return(df):
    """Slope of points vs price within each position = points bought per extra £1m."""
    section("6. MARGINAL RETURN ON SPEND — points gained per extra £1m (900+ mins)")
    reg = regulars(df)
    rows = []
    for pos in POS_ORDER:
        sub = reg[reg["position"] == pos]
        if len(sub) < 10:
            continue
        slope, intercept = np.polyfit(sub["start_price"], sub["points"], 1)
        corr = np.corrcoef(sub["start_price"], sub["points"])[0, 1]
        # residual spread tells you how reliable that slope is per pick
        resid = sub["points"] - (slope * sub["start_price"] + intercept)
        rows.append({
            "pos": pos,
            "n": len(sub),
            "pts_per_extra_£1m": slope,
            "baseline_pts_at_£4.0": slope * 4.0 + intercept,
            "corr(price,pts)": corr,
            "resid_sd": resid.std(),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nRead: a high slope means money spent on that position converts into points.")
    print("A high resid_sd means the slope is an average over very noisy individual picks.")


def replacement_value(df):
    """Value over replacement: points above what a cheap starter at that position gives."""
    section("7. VALUE OVER REPLACEMENT (VOR)")
    reg = regulars(df)
    repl, rows = {}, []
    for pos in POS_ORDER:
        cheap = reg[(reg["position"] == pos) & (reg["start_price"] <= 4.5)]
        wider = reg[(reg["position"] == pos) & (reg["start_price"] <= 5.0)]
        # fall back to the wider band when the <=£4.5 pool is too thin to be meaningful
        base = cheap if len(cheap) >= 5 else wider
        repl[pos] = base["points"].median() if len(base) else reg[reg["position"] == pos]["points"].quantile(0.25)
        rows.append({
            "pos": pos,
            "n_at<=£4.5": len(cheap),
            "median_pts<=£4.5": cheap["points"].median() if len(cheap) else np.nan,
            "n_at<=£5.0": len(wider),
            "median_pts<=£5.0": wider["points"].median() if len(wider) else np.nan,
            "baseline_used": repl[pos],
        })
    print("Replacement baseline = median points of a cheap regular starter (900+ mins).")
    print("A thin cheap pool is itself a finding: it means the position forces you to spend.")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.0f}"))

    reg["vor"] = reg["points"] - reg["position"].map(repl).astype(float)
    reg["premium_paid"] = reg["start_price"] - 4.5
    reg["vor_per_extra_m"] = np.where(reg["premium_paid"] > 0, reg["vor"] / reg["premium_paid"], np.nan)

    print("\nBest VOR per £1m spent above the £4.5 floor (min 900 mins, price > £5.5):")
    cand = reg[reg["start_price"] > 5.5].nlargest(20, "vor_per_extra_m")
    print(cand[["name", "position", "start_price", "points", "vor", "vor_per_extra_m"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\nPosition-level median VOR per extra £1m (price > £5.5):")
    print(reg[reg["start_price"] > 5.5].groupby("position", observed=False)["vor_per_extra_m"]
          .agg(["size", "median", "mean"]).to_string(float_format=lambda v: f"{v:.2f}"))
    return repl


def bust_risk(df):
    section("8. BUST RISK — did expensive picks actually deliver?")
    reg = regulars(df)
    prem = reg[reg["start_price"] >= 7.0]
    rows = []
    for pos in POS_ORDER:
        sub = prem[prem["position"] == pos]
        if not len(sub):
            continue
        rows.append({
            "pos": pos,
            "n_premium(>=£7.0)": len(sub),
            "median_pts": sub["points"].median(),
            "hit_rate_150+": (sub["points"] >= 150).mean() * 100,
            "bust_rate_<120": (sub["points"] < 120).mean() * 100,
            "ppm_median": sub["ppm"].median(),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    section("9. CEILING — is the position worth the captain armband?")
    rows = []
    for pos in POS_ORDER:
        sub = reg[reg["position"] == pos]
        rows.append({"pos": pos, "best": sub["points"].max(),
                     "2nd": sub["points"].nlargest(2).iloc[-1] if len(sub) > 1 else np.nan,
                     "5th": sub["points"].nlargest(5).iloc[-1] if len(sub) > 4 else np.nan,
                     "10th": sub["points"].nlargest(10).iloc[-1] if len(sub) > 9 else np.nan,
                     "20th": sub["points"].nlargest(20).iloc[-1] if len(sub) > 19 else np.nan})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    print("\nA position whose 1st and 10th best are close = no reason to pay up for the elite name.")


def defcon_effect(df):
    section("10. DEFENSIVE CONTRIBUTION POINTS — how much of each position's score is DefCon")
    reg = regulars(df)
    if reg["defcon_points"].sum() == 0:
        print("No defensive_contribution data for this season.")
        return
    out = reg.groupby("position", observed=False).agg(
        defcon_pts_mean=("defcon_points", "mean"),
        defcon_pts_max=("defcon_points", "max"),
        share_of_points=("defcon_share", "mean"),
    )
    out["share_of_points"] *= 100
    print(out.to_string(float_format=lambda v: f"{v:.1f}"))
    print("\nTop DefCon earners:")
    print(reg.nlargest(15, "defcon_points")[["name", "position", "start_price", "points", "defcon_points", "ppm"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))


def optimal_formation(df, xi_budget=84.0):
    """Hindsight-optimal XI shape: exact DP over 0.1m price units, FPL formation rules."""
    reg = df[df["minutes"] >= 500].copy()
    B = int(round(xi_budget * 10))
    NEG = -1e9

    def build(pos, max_n):
        sub = reg[reg["position"] == pos].sort_values("points", ascending=False).head(60)
        prices = (sub["start_price"] * 10).round().astype(int).to_numpy()
        pts = sub["points"].to_numpy(dtype=float)
        names = sub["name"].to_numpy()
        dp = np.full((max_n + 1, B + 1), NEG)
        dp[0, :] = 0.0
        parent = {}
        for i, (pr, pt, nm) in enumerate(zip(prices, pts, names)):
            for n in range(max_n, 0, -1):
                for c in range(B, pr - 1, -1):
                    cand = dp[n - 1, c - pr] + pt
                    if cand > dp[n, c] + 1e-9:
                        dp[n, c] = cand
                        parent[(n, c)] = (nm, pr, pt, n - 1, c - pr)
        return dp, parent

    tabs = {p: build(p, XI_RULES[p][1]) for p in POS_ORDER}

    def reconstruct(pos, n, c):
        dp, parent = tabs[pos]
        picks = []
        while n > 0 and (n, c) in parent:
            nm, pr, pt, n, c = parent[(n, c)]
            picks.append((nm, pr / 10.0, pt))
        return picks

    results = []
    for nd in range(3, 6):
        for nmid in range(2, 6):
            for nf in range(1, 4):
                if 1 + nd + nmid + nf != 11:
                    continue
                shape = [("GK", 1), ("DEF", nd), ("MID", nmid), ("FWD", nf)]
                # DP across positions, carrying the per-position spend that produced each total
                acc_pts = np.full(B + 1, NEG)
                acc_pts[0] = 0.0
                acc_split = [None] * (B + 1)
                acc_split[0] = []
                for pos, n in shape:
                    dp = tabs[pos][0][n]
                    new_pts = np.full(B + 1, NEG)
                    new_split = [None] * (B + 1)
                    for c in range(B + 1):
                        if acc_pts[c] < -1e8:
                            continue
                        for c2 in range(B - c + 1):
                            if dp[c2] < -1e8:
                                continue
                            t, v = c + c2, acc_pts[c] + dp[c2]
                            if v > new_pts[t]:
                                new_pts[t] = v
                                new_split[t] = acc_split[c] + [(pos, n, c2)]
                    acc_pts, acc_split = new_pts, new_split
                best_c = int(np.argmax(acc_pts))
                results.append({
                    "formation": f"1-{nd}-{nmid}-{nf}",
                    "pts": acc_pts[best_c],
                    "split": acc_split[best_c],
                })

    section("11. HINDSIGHT-OPTIMAL FORMATION & BUDGET SPLIT")
    print(f"XI budget: £{xi_budget:.1f}m (£100m squad minus ~£16m of bench fodder), min 500 mins")
    res = sorted(results, key=lambda r: -r["pts"])
    tbl = []
    for r in res:
        row = {"formation": r["formation"], "pts": r["pts"]}
        for pos, n, c in r["split"]:
            row[f"£{pos}"] = c / 10.0
        tbl.append(row)
    print(pd.DataFrame(tbl).head(8).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    best = res[0]
    print(f"\nThe optimal {best['formation']} XI ({best['pts']:.0f} pts):")
    for pos, n, c in best["split"]:
        picks = reconstruct(pos, n, c)
        for nm, pr, pt in sorted(picks, key=lambda x: -x[2]):
            print(f"  {pos:>3}  £{pr:>4.1f}m  {pt:>4.0f} pts  {nm}")
    print("\nBudget split of the optimal XI:")
    for pos, n, c in best["split"]:
        print(f"  {pos}: {n} players, £{c/10.0:.1f}m ({c/10.0/xi_budget*100:.0f}% of XI budget)")


def spend_curve(df):
    """How many points does the best-available player cost at each price point?"""
    section("12. BEST AVAILABLE PLAYER AT EACH PRICE POINT (900+ mins)")
    reg = regulars(df)
    bins = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
    rows = []
    for pos in POS_ORDER:
        sub = reg[reg["position"] == pos]
        row = {"pos": pos}
        for b in bins:
            cand = sub[sub["start_price"] <= b]
            row[f"<=£{b}"] = cand["points"].max() if len(cand) else np.nan
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    print("\nFlat rows = paying more bought you nothing at that position.")


def compare_seasons(seasons):
    section("13. SEASON-OVER-SEASON: POSITION SHARE OF THE TOP 30")
    rows = []
    for s in seasons:
        try:
            df, _ = load_season(s)
        except FileNotFoundError:
            continue
        top = df.nlargest(30, "points")
        counts = top["position"].value_counts()
        row = {"season": s}
        for pos in ALL_POS:
            row[pos] = int(counts.get(pos, 0))
        row["top30_cutoff_pts"] = top["points"].min()
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--history", nargs="*", default=["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    df, _ = load_season(args.season)
    n_am = int((df["position"] == "AM").sum())
    if n_am:
        print(f"Excluding {n_am} Assistant Manager elements from the squad-value analysis")
        df = df[df["position"] != "AM"].copy()
    df["position"] = pd.Categorical(df["position"].astype(str), categories=POS_ORDER, ordered=True)
    print(f"Season {args.season}: {len(df)} players, {df['points'].sum():.0f} total points recorded")

    top_n_breakdown(df, args.top)
    value_by_position(df)
    marginal_return(df)
    replacement_value(df)
    bust_risk(df)
    defcon_effect(df)
    optimal_formation(df)
    spend_curve(df)
    compare_seasons(args.history)


if __name__ == "__main__":
    main()
