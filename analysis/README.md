# Positional Value Analysis — 2025-26

`position_value_analysis.py` answers two questions from the season data in `data/`:

1. Which positions made up the top 30 point scorers?
2. Where does each £1m of squad budget buy the most points?

```bash
pip install pandas numpy
python analysis/position_value_analysis.py                 # defaults to 2025-26
python analysis/position_value_analysis.py --season 2024-25 --top 50
```

All prices are **start-of-season** prices (`value` in gameweek 1), so points-per-million
reflects what you would actually have paid at the initial squad deadline, not the
inflated end-of-season price. "Regulars" means 900+ minutes unless stated otherwise.

## Headline results (2025-26)

**Top 30 composition:** 14 MID, 10 DEF, 5 FWD, 1 GK.

**The defender share exploded.** Defenders in the top 30, by season:

| Season | GK | DEF | MID | FWD |
|---|---|---|---|---|
| 2021-22 | 4 | 9 | 15 | 2 |
| 2022-23 | 5 | 4 | 15 | 6 |
| 2023-24 | 1 | 3 | 18 | 8 |
| 2024-25 | 2 | 2 | 18 | 8 |
| **2025-26** | 1 | **10** | 14 | 5 |

This is the Defensive Contribution rule (introduced 2025-26) showing up in the data.
DefCon supplied 13.8% of the average defender's points and 7.5% of the average
midfielder's, and the biggest earners (Senesi, Elliot Anderson: 52 DefCon points each)
were cheap.

**Cheap defenders were the single best value in the game.** The four highest
points-per-million players in the top 30 were all defenders priced £4.0–4.5
(Guéhi 39.8, Senesi 38.9, Mukiele 37.8, Truffert 36.7 pts/£m) — against 17.1 for
Haaland at £14.0m.

**Money buys the least in midfield.** Regressing points on price among regulars:

| Pos | pts per extra £1m | corr(price, pts) | residual SD |
|---|---|---|---|
| GK | 33.4 | 0.38 | 35.1 |
| DEF | 19.1 | 0.26 | 38.1 |
| FWD | 16.7 | 0.62 | 37.3 |
| MID | 11.3 | 0.35 | 36.6 |

Midfield has by far the deepest cheap pool, so paying up adds the least. Forwards show
the tightest price/points relationship (r = 0.62) — the forward market was efficiently
priced, and there were only 2 sub-£4.5 forwards who played 900 minutes at all.

**Premium picks busted at roughly the rate they hit.** Among £7.0m+ regulars,
45.8% of midfielders and 50.0% of forwards finished under 120 points.

**Hindsight-optimal XI** (£84m, i.e. £100m squad less a £16m bench): 1-3-4-3 for
2141 points, spending 7% on GK, 18% on 3 defenders, 36% on 4 midfielders, 33% on
3 forwards. All eight legal formations landed within 27 points of each other, so
shape mattered far less than individual picks.

## What the script prints

| Section | Content |
|---|---|
| 1–3 | Top-N table, position breakdown, top-N share vs. available squad slots |
| 4–5 | Points-per-million by position and by price band |
| 6 | Marginal return on spend (points bought per extra £1m) |
| 7 | Value over replacement, with the replacement baseline shown explicitly |
| 8–9 | Bust rate of premium picks; ceiling by position (captaincy case) |
| 10 | Defensive Contribution points by position |
| 11 | Hindsight-optimal formation and budget split (exact DP knapsack) |
| 12 | Best available player at each price point |
| 13 | Season-over-season position share of the top 30 |

## Data caveats

- **Prices are FPL prices, not a market.** Start-of-season prices are set by FPL and
  reflect the *previous* season, so 2025-26 points-per-million partly measures which
  players FPL underpriced going in. It is descriptive of last season, not a forecast.
- **Single season, n=1.** The DefCon effect is one season of evidence. Section 13 exists
  to keep that in perspective.
- **Position labels drift between seasons.** 2021-22 mixes `GKP` and `GK`; the loader
  normalises them. 2024-25 contains `AM` (Assistant Manager) elements, which are excluded
  from squad maths but counted separately in the season comparison.
- **Survivorship in the price bands.** The 900-minute filter removes players who lost
  their place, which flatters every band — especially the cheap ones, where the
  rotation risk actually lives.
