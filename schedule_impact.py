"""
Schedule impact model (Part 3 / Question 9) -- corrected version.

What was wrong with the original model
---------------------------------------
1. Opponent strength leaked the team's own results.
   `opponent_win_pct` averaged each opponent's FULL-season win %, which includes
   the games that opponent played against the team being evaluated. A strong team
   beats its opponents, which lowers their records, so a strong team's schedule
   looks "easy" -- the model then credits schedule for wins that were really
   team quality. (The comment in the original cell said "excluding games vs
   select team", but the code did not exclude them.)
2. No control for team quality. With total wins regressed on opponent win %
   alone, anything correlated with being good (or bad) gets attributed to the
   schedule. That's why MIL came out at +36.6 wins and the back-to-back
   coefficient came out positive (more B2Bs -> more wins).
3. Team-season level with ~150 rows, scored in-sample only (R^2 = 0.30), so
   there was no check that the model generalises.

What this version does
----------------------
* Works game by game (~6,000 games instead of ~150 team-seasons).
* Opponent strength = opponent's win % EXCLUDING games against this team
  (leave-one-out), and team strength = team's win % excluding games against
  this opponent. Strengths are put on a log-odds scale.
* Team strength is in the model as a control, so schedule effects are
  estimated holding quality fixed.
* Schedule factors are entered for both sides: back-to-back, 4th game in 6
  nights, miles travelled into the game, plus home court.
* Logistic regression, validated leave-one-season-out (log loss and Brier
  score vs. a strength-only model and a coin flip).
* Schedule impact for each team-game = P(win | actual schedule) minus
  P(win | neutral schedule), where "neutral" means: an opponent of average
  strength among the OTHER 29 teams (a team can't play itself, so a good team's
  fair-schedule opponent is slightly below .500), and league-average
  rest/travel on both sides. Team strength and home/away are held at their real
  values. Summing over games gives wins gained or lost to the schedule.

Usage in the notebook (after the CSVs are loaded in the Setup cell):

    from schedule_impact import run_schedule_impact
    results = run_schedule_impact(schedule, game_data, locations, min_season=2019)
    results["validation"]          # out-of-sample table
    results["coefficients"]        # model coefficients
    results["team_impact"]         # one number per team (answer to Q9)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-3  # keeps logit finite for 0-win or undefeated records

SCHEDULE_FEATURES = [
    "b2b", "opp_b2b",
    "four_in_6", "opp_four_in_6",
    "travel_k", "opp_travel_k",   # thousands of miles travelled into the game
]
STRENGTH_FEATURES = ["team_str", "opp_str", "home"]
ALL_FEATURES = STRENGTH_FEATURES + SCHEDULE_FEATURES


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def build_schedule_features(schedule, locations):
    """Per team-game rest and travel features: b2b, 4th-in-6, miles travelled."""
    s = schedule[["season", "gamedate", "team", "opponent", "home"]].copy()
    s["gamedate"] = pd.to_datetime(s["gamedate"])
    s = s.sort_values(["team", "season", "gamedate"]).reset_index(drop=True)

    g = s.groupby(["team", "season"])["gamedate"]
    s["b2b"] = (s["gamedate"] - g.shift(1)).dt.days.eq(1).astype(int)
    # 4th game in 6 nights: the game 3 games ago was at most 5 days earlier
    s["four_in_6"] = (s["gamedate"] - g.shift(3)).dt.days.le(5).astype(int)

    loc = locations.set_index("team")[["latitude", "longitude"]]
    arena = np.where(s["home"] == 1, s["team"], s["opponent"])
    s["lat"] = loc.loc[arena, "latitude"].to_numpy()
    s["lon"] = loc.loc[arena, "longitude"].to_numpy()
    prev_lat = s.groupby(["team", "season"])["lat"].shift(1)
    prev_lon = s.groupby(["team", "season"])["lon"].shift(1)
    miles = _haversine_miles(prev_lat, prev_lon, s["lat"], s["lon"])
    s["travel_k"] = np.nan_to_num(miles, nan=0.0) / 1000.0
    return s[["season", "gamedate", "team", "opponent", "home",
              "b2b", "four_in_6", "travel_k"]]


def build_game_table(schedule, game_data, locations, min_season=2019):
    """One row per team-game with the win flag, strengths and schedule features."""
    gd = game_data.copy()
    gd["gamedate"] = pd.to_datetime(gd["gamedate"])
    if "gametype" in gd.columns:
        gd = gd[gd["gametype"] == 2]                      # regular season only
    gd = gd[gd["season"] >= min_season]
    results = (gd[["season", "gamedate", "off_team", "def_team", "off_win"]]
               .rename(columns={"off_team": "team", "def_team": "opponent",
                                "off_win": "win"})
               .drop_duplicates(["season", "gamedate", "team"]))

    feats = build_schedule_features(schedule[schedule["season"] >= min_season],
                                    locations)
    df = feats.merge(results, on=["season", "gamedate", "team", "opponent"],
                     how="inner")

    # Opponent's schedule features for the same game
    opp = feats.rename(columns={"team": "opponent", "opponent": "team",
                                "b2b": "opp_b2b", "four_in_6": "opp_four_in_6",
                                "travel_k": "opp_travel_k"}).drop(columns="home")
    df = df.merge(opp, on=["season", "gamedate", "team", "opponent"], how="left")
    df[["opp_b2b", "opp_four_in_6", "opp_travel_k"]] = (
        df[["opp_b2b", "opp_four_in_6", "opp_travel_k"]].fillna(0))

    # ---- Leave-one-out strengths --------------------------------------------
    tot = df.groupby(["season", "team"])["win"].agg(W="sum", G="count").reset_index()
    h2h = (df.groupby(["season", "team", "opponent"])["win"]
             .agg(w="sum", g="count").reset_index())
    h2h = h2h.merge(tot, on=["season", "team"])
    # team's win % excluding the games against this opponent
    h2h["team_loo_pct"] = (h2h["W"] - h2h["w"]) / (h2h["G"] - h2h["g"])
    pair = h2h[["season", "team", "opponent", "team_loo_pct"]]

    df = df.merge(pair, on=["season", "team", "opponent"], how="left")
    df = df.merge(pair.rename(columns={"team": "opponent", "opponent": "team",
                                       "team_loo_pct": "opp_loo_pct"}),
                  on=["season", "team", "opponent"], how="left")
    df["team_str"] = _logit(df["team_loo_pct"])
    df["opp_str"] = _logit(df["opp_loo_pct"])

    # Neutral opponent for each team-season: average LOO strength of the other
    # 29 teams (each counted once, regardless of how often they were played).
    opp_view = pair.rename(columns={"team": "opponent", "opponent": "team",
                                    "team_loo_pct": "opp_loo_pct"})
    opp_view["opp_str"] = _logit(opp_view["opp_loo_pct"])
    neutral = (opp_view.groupby(["season", "team"])["opp_str"].mean()
               .rename("neutral_opp_str").reset_index())
    df = df.merge(neutral, on=["season", "team"], how="left")
    return df


def _fit(X, y):
    # Large C = essentially unregularised; we want interpretable coefficients.
    return LogisticRegression(C=1e6, max_iter=2000).fit(X, y)


def validate(df):
    """Leave-one-season-out: does adding schedule factors improve predictions?"""
    rows = []
    for season in sorted(df["season"].unique()):
        train, test = df[df["season"] != season], df[df["season"] == season]
        y = test["win"]
        p_full = _fit(train[ALL_FEATURES], train["win"]).predict_proba(test[ALL_FEATURES])[:, 1]
        p_str = _fit(train[STRENGTH_FEATURES], train["win"]).predict_proba(test[STRENGTH_FEATURES])[:, 1]
        rows.append({
            "held_out_season": season, "games": len(test),
            "logloss_coinflip": log_loss(y, np.full(len(y), 0.5), labels=[0, 1]),
            "logloss_strength_only": log_loss(y, p_str, labels=[0, 1]),
            "logloss_full": log_loss(y, p_full, labels=[0, 1]),
            "brier_strength_only": brier_score_loss(y, p_str),
            "brier_full": brier_score_loss(y, p_full),
        })
    out = pd.DataFrame(rows)
    total = out.drop(columns="held_out_season").multiply(out["games"], axis=0).sum() / out["games"].sum()
    total["games"] = out["games"].sum()
    total["held_out_season"] = "ALL (weighted)"
    return pd.concat([out, total.to_frame().T], ignore_index=True)


def schedule_impact(df, model):
    """Wins gained (+) or lost (-) to the schedule, per team-game."""
    actual = model.predict_proba(df[ALL_FEATURES])[:, 1]
    neutral = df[ALL_FEATURES].copy()
    neutral["opp_str"] = df["neutral_opp_str"]
    for col in SCHEDULE_FEATURES:                    # league-average rest/travel
        neutral[col] = df[col].mean()
    baseline = model.predict_proba(neutral)[:, 1]
    out = df[["season", "gamedate", "team", "opponent", "win"]].copy()
    out["p_actual"], out["p_neutral"] = actual, baseline
    out["impact"] = actual - baseline
    # Split the effect: opponent strength vs. rest/travel
    only_opp = df[ALL_FEATURES].copy()
    only_opp["opp_str"] = df["neutral_opp_str"]
    out["impact_opponents"] = actual - model.predict_proba(only_opp)[:, 1]
    out["impact_rest_travel"] = out["impact"] - out["impact_opponents"]
    return out


def run_schedule_impact(schedule, game_data, locations, min_season=2019):
    df = build_game_table(schedule, game_data, locations, min_season)
    model = _fit(df[ALL_FEATURES], df["win"])
    coefs = pd.DataFrame({"feature": ["intercept"] + ALL_FEATURES,
                          "coef_log_odds": [model.intercept_[0], *model.coef_[0]]})
    coefs["odds_ratio"] = np.exp(coefs["coef_log_odds"])

    games = schedule_impact(df, model)
    team_impact = (games.groupby("team")[["impact", "impact_opponents", "impact_rest_travel"]]
                   .sum().round(2).sort_values("impact")
                   .rename(columns={"impact": "schedule_wins",
                                    "impact_opponents": "from_opponent_strength",
                                    "impact_rest_travel": "from_rest_travel"})
                   .reset_index())
    return {"games": df, "model": model, "coefficients": coefs,
            "validation": validate(df), "game_impact": games,
            "team_impact": team_impact}


if __name__ == "__main__":
    schedule = pd.read_csv("schedule.csv")
    game_data = pd.read_csv("team_game_data.csv")
    locations = pd.read_csv("locations.csv")
    r = run_schedule_impact(schedule, game_data, locations)
    pd.set_option("display.width", 140)
    print("Coefficients\n", r["coefficients"].round(3), "\n")
    print("Leave-one-season-out validation\n", r["validation"].round(4), "\n")
    print("Schedule impact by team (2019-20 to 2023-24)\n", r["team_impact"].to_string(index=False))
