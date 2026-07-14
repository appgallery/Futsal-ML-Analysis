# -*- coding: utf-8 -*-
"""Futsal_Best_Player_Pipeline.py

Authoritative Futsal "Best Player" Pipeline — ETL → Training → Inference
Directly matches the logic above the three hidden sections ("Don't Change it", "Trial and Checking", "lol")
of final_prod_player (1).py.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

pd.set_option("display.max_columns", 100)

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
PLAYERS_PATH = "players.csv"
MATCHES_PATH = "matches.csv"
OUT_PATH = "Futsal_match_ml.csv"

players_raw = pd.read_csv(PLAYERS_PATH)
matches_raw = pd.read_csv(MATCHES_PATH)

players = players_raw.copy()
matches = matches_raw.copy()

players["type"] = players["type"].astype(str).str.upper()

# ==========================================
# 2. ETL & FEATURE ENGINEERING (MATCHES THE AUTHORITATIVE LOGIC)
# ==========================================

# 2.1 Build one row per (match, player) with event counts from pivot table
player_match = pd.pivot_table(
    players,
    index=["match_id", "user_id"],
    columns="type",
    aggfunc="size",
    fill_value=0
).reset_index()

# Extract player descriptive info
player_info = (
    players[["match_id", "user_id", "team_id", "first_name", "last_name", "date_of_birth"]]
    .drop_duplicates(subset=["match_id", "user_id"])
)
player_match = player_match.merge(player_info, on=["match_id", "user_id"], how="left")

# Extract matches info
match_info = matches[[
    "id", "startDate", "homeTeamId", "awayTeamId",
    "winningTeam", "losingTeam", "winningTeamGoals", "losingTeamGoals",
    "homeTeamName", "awayTeamName"
]].copy()
match_info = match_info.rename(columns={"id": "match_id"})

player_match = player_match.merge(match_info, on="match_id", how="left")

# Calculate authoritative performance score
player_match["perf_score"] = (
    player_match["SCORE"] * 3
    - player_match["FOULS"] * 0.5
    - player_match["YELLOWCARD"] * 1
    - player_match["REDCARD"] * 3
)

# Identify best player in each match
player_match["best_match_score"] = (
    player_match.groupby("match_id")["perf_score"]
    .transform("max")
)
player_match["is_best_player"] = (
    player_match["perf_score"] == player_match["best_match_score"]
).astype(int)

# Re-sort for tie-breaking exactly as done in the original code
player_match = player_match.sort_values(
    ["match_id", "perf_score", "SCORE"],
    ascending=[True, False, False]
)
player_match["rank_in_match"] = (
    player_match.groupby("match_id").cumcount() + 1
)
player_match["is_best_player"] = (
    player_match["rank_in_match"] == 1
).astype(int)

# Identify is_winning_team and filter to winning team players only
player_match["is_winning_team"] = (
    player_match["team_id"] == player_match["winningTeam"]
).astype(int)

winning_players = player_match[player_match["is_winning_team"] == 1].copy()

# Recalculate rank inside team
winning_players["best_match_score"] = (
    winning_players.groupby("match_id")["perf_score"]
    .transform("max")
)
winning_players["is_best_player"] = (
    winning_players["perf_score"] == winning_players["best_match_score"]
).astype(int)

winning_players = winning_players.sort_values(
    ["match_id", "perf_score", "SCORE", "FOULS"],
    ascending=[True, False, False, True]
)
winning_players["rank_in_team"] = (
    winning_players.groupby("match_id").cumcount() + 1
)
winning_players["is_best_player"] = (
    winning_players["rank_in_team"] == 1
).astype(int)

# Format dates and sort by career timeline
winning_players["startDate"] = pd.to_datetime(winning_players["startDate"])
winning_players = winning_players.sort_values(["user_id", "startDate"]).reset_index(drop=True)

# Cumulative career features
winning_players["career_matches"] = (
    winning_players.groupby("user_id").cumcount()
)
winning_players["career_goals"] = (
    winning_players.groupby("user_id")["SCORE"].cumsum()
    - winning_players["SCORE"]
)
winning_players["career_avg_goals"] = (
    winning_players["career_goals"] / winning_players["career_matches"]
).fillna(0)

winning_players["career_fouls"] = (
    winning_players.groupby("user_id")["FOULS"].cumsum()
    - winning_players["FOULS"]
)
winning_players["career_yellow"] = (
    winning_players.groupby("user_id")["YELLOWCARD"].cumsum()
    - winning_players["YELLOWCARD"]
)
winning_players["career_red"] = (
    winning_players.groupby("user_id")["REDCARD"].cumsum()
    - winning_players["REDCARD"]
)

winning_players["career_perf"] = (
    winning_players.groupby("user_id")["perf_score"].cumsum()
    - winning_players["perf_score"]
)
winning_players["career_avg_perf"] = (
    winning_players["career_perf"] / winning_players["career_matches"]
).fillna(0)

# Rolling features (prior matches only)
winning_players["roll3_goals"] = (
    winning_players.groupby("user_id")["SCORE"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
).fillna(0)

winning_players["roll5_goals"] = (
    winning_players.groupby("user_id")["SCORE"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
).fillna(0)

winning_players["roll10_goals"] = (
    winning_players.groupby("user_id")["SCORE"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
).fillna(0)

winning_players["goal_trend"] = (
    winning_players["roll3_goals"] - winning_players["roll10_goals"]
)

winning_players["roll3_perf"] = (
    winning_players.groupby("user_id")["perf_score"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
).fillna(0)

winning_players["roll5_perf"] = (
    winning_players.groupby("user_id")["perf_score"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
).fillna(0)

winning_players["roll10_perf"] = (
    winning_players.groupby("user_id")["perf_score"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
).fillna(0)

winning_players["perf_trend"] = (
    winning_players["roll3_perf"] - winning_players["roll10_perf"]
)

# Experience feature
winning_players["log_experience"] = np.log1p(winning_players["career_matches"])

# Age features & cleaning
winning_players["date_of_birth"] = pd.to_datetime(winning_players["date_of_birth"], errors="coerce")
winning_players["age"] = (
    (winning_players["startDate"] - winning_players["date_of_birth"]).dt.days / 365.25
)
winning_players["age"] = winning_players["age"].fillna(winning_players["age"].median())

# Handle extreme age values
winning_players.loc[winning_players["age"] < 10, "age"] = np.nan

# Compute best player rate
winning_players = winning_players.sort_values(["user_id", "startDate"])
winning_players["career_best_player_rate"] = (
    winning_players.groupby("user_id")["is_best_player"]
    .transform(lambda x: x.shift(1).expanding().mean())
).fillna(0)

# Add missing rolling rate features
winning_players["prev_best_player"] = (
    winning_players.groupby("user_id")["is_best_player"]
    .shift(1)
    .fillna(0)
)
winning_players["roll5_best_player_rate"] = (
    winning_players.groupby("user_id")["prev_best_player"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)
winning_players["roll10_best_player_rate"] = (
    winning_players.groupby("user_id")["prev_best_player"]
    .transform(lambda x: x.rolling(10, min_periods=1).mean())
)

# Filter dataset to age >= 15 as in original line 564
df = winning_players[winning_players["age"] >= 15].copy()
df.loc[(df["age"] < 15) | (df["age"] > 50), "age"] = np.nan
df["age"] = df["age"].fillna(df["age"].median())

# Save engineered features
df.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH} — shape={df.shape}")

# ==========================================
# 3. TRAINING
# ==========================================
features = [
    "career_matches",
    "career_goals",
    "career_avg_goals",
    "career_avg_perf",
    "roll3_goals",
    "roll5_goals",
    "roll10_goals",
    "goal_trend",
    "roll3_perf",
    "roll5_perf",
    "roll10_perf",
    "perf_trend",
    "log_experience",
    "age",
    "career_best_player_rate",
    "roll5_best_player_rate",
    "roll10_best_player_rate"
]
target = "is_best_player"

df = df.sort_values("startDate").reset_index(drop=True)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Train: {X_train.shape} Test: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate: {y_test.mean():.4f}")

# Grid Search across models
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

models_to_search = {
    "xgb": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        ),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1]
        }
    ),
    "lightgbm": (
        LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            verbose=-1
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1]
        }
    ),
    "catboost": (
        CatBoostClassifier(
            verbose=0,
            random_state=42,
            auto_class_weights="Balanced"
        ),
        {
            "depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "iterations": [100, 200, 300]
        }
    ),
    "random_forest": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),
        {
            "n_estimators": [200, 400],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 3, 5]
        }
    )
}

best_model = None
best_score = -1
best_model_name = None
best_model_params = None

for name, (model, params) in models_to_search.items():
    print(f"Grid Searching model: {name}")
    grid = GridSearchCV(
        model,
        params,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    candidate = grid.best_estimator_
    preds = candidate.predict(X_test)
    score = f1_score(y_test, preds)
    print(f"Best params for {name}: {grid.best_params_} (F1 Score: {score:.4f})")
    
    if score > best_score:
        best_score = score
        best_model = candidate
        best_model_name = name
        best_model_params = grid.best_params_

print(f"Selected Best Model: {best_model_name} with F1 score: {best_score:.4f}")

# Calculate threshold and metrics
probs_test = best_model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.05, 0.50, 0.01)
best_threshold = 0.5
best_f1 = -1

for t in thresholds:
    preds = (probs_test >= t).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

print(f"Best Threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")

# Save artifacts
joblib.dump(best_model, "best_player_xgb.pkl")  # Kept name so inference function loading works
joblib.dump(features, "best_player_features.pkl")
joblib.dump(float(best_threshold), "best_threshold.pkl")
print("Saved artifacts: best_player_xgb.pkl, best_player_features.pkl, best_threshold.pkl")

# ==========================================
# 3.3 EXTRA EVALUATION METRICS (TOP-K ACCURACY, MRR, NDCG)
# ==========================================
test_df_eval = X_test.copy()
test_df_eval["match_id"] = df.loc[X_test.index, "match_id"]
test_df_eval["user_id"] = df.loc[X_test.index, "user_id"]
test_df_eval["is_best_player"] = y_test
test_df_eval["best_player_prob"] = probs_test

# Top-1, Top-3 Match Accuracy
top1 = (
    test_df_eval
    .sort_values(["match_id", "best_player_prob"], ascending=[True, False])
    .groupby("match_id")
    .head(1)
)
print("Top-1 Match Accuracy:", top1["is_best_player"].mean())

top3 = (
    test_df_eval
    .sort_values(["match_id", "best_player_prob"], ascending=[True, False])
    .groupby("match_id")
    .head(3)
)
top3_acc = (
    top3.groupby("match_id")["is_best_player"]
    .max()
    .mean()
)
print("Top-3 Match Accuracy:", top3_acc)

# MRR Scores
mrr_scores = []
for match_id, group in test_df_eval.groupby("match_id"):
    group = group.sort_values("best_player_prob", ascending=False).reset_index(drop=True)
    best_idx = group[group["is_best_player"] == 1].index
    if len(best_idx) > 0:
        rank = best_idx[0] + 1
        mrr_scores.append(1 / rank)
print("MRR =", np.mean(mrr_scores))

# Mean NDCG
from sklearn.metrics import ndcg_score
ndcg_scores = []
for match_id, group in test_df_eval.groupby("match_id"):
    if len(group) < 2:
        continue
    y_true = np.array([group["is_best_player"].values])
    y_score = np.array([group["best_player_prob"].values])
    ndcg_scores.append(ndcg_score(y_true, y_score))
print("Mean NDCG:", np.mean(ndcg_scores))

# Rank prediction metrics
rank_results = []
for match_id, group in test_df_eval.groupby("match_id"):
    group = group.sort_values("best_player_prob", ascending=False)
    actual_player = group.loc[group["is_best_player"] == 1, "user_id"]
    if len(actual_player) == 0:
        continue
    actual_player = actual_player.iloc[0]
    predicted_rank = (
        group.reset_index(drop=True)
             .query("user_id == @actual_player")
             .index[0] + 1
    )
    rank_results.append(predicted_rank)

rank_results = np.array(rank_results)
print("Matches:", len(rank_results))
print("Top-1:", np.mean(rank_results <= 1))
print("Top-2:", np.mean(rank_results <= 2))
print("Top-3:", np.mean(rank_results <= 3))
print("Top-5:", np.mean(rank_results <= 5))
print("Average Rank:", rank_results.mean())
print("Median Rank:", np.median(rank_results))

# ==========================================
# 4. INFERENCE FUNCTION
# ==========================================
def predict_best_player(match_prediction_json: dict, top_n: int = 5) -> dict:
    home_id = match_prediction_json["homeID"]
    away_id = match_prediction_json["awayID"]
    home_win_prob = match_prediction_json["homeWinProbability"]
    away_win_prob = match_prediction_json["awayWinProbability"]
    
    # Reload model and metadata
    best_model_loaded = joblib.load("best_player_xgb.pkl")
    features_loaded = joblib.load("best_player_features.pkl")
    
    # Group and pick most recent snapshot per user
    latest_player_features = df.sort_values("startDate").groupby("user_id").tail(1)
    
    # Identify team names
    home_name = latest_player_features.loc[latest_player_features["homeTeamId"] == home_id, "homeTeamName"].dropna().iloc[-1]
    away_name = latest_player_features.loc[latest_player_features["awayTeamId"] == away_id, "awayTeamName"].dropna().iloc[-1]
    
    # Determine the predicted winner
    if home_win_prob > away_win_prob:
        winning_team_id = home_id
        predicted_winner_name = home_name
    else:
        winning_team_id = away_id
        predicted_winner_name = away_name
        
    winning_players = latest_player_features[latest_player_features["team_id"] == winning_team_id].copy()
    
    if len(winning_players) == 0:
        return {"error": "No player data found"}
        
    X_players = winning_players[features_loaded]
    winning_players["best_player_probability"] = best_model_loaded.predict_proba(X_players)[:, 1]
    winning_players = winning_players.sort_values("best_player_probability", ascending=False)
    
    player_predictions = []
    for _, row in winning_players.iterrows():
        # Fallback to first_name + last_name if player_name isn't in columns
        player_name = row.get("player_name", f"{row.get('first_name', '')} {row.get('last_name', '')}").strip()
        
        prob = round(float(row["best_player_probability"]) * 100, 2)
        score = round(float(row["best_player_probability"]), 2)
        
        player_predictions.append({
            "playerName": player_name,
            "playerID": int(row["user_id"]),
            "playWellProbability": prob,
            "predictedPerfScore": score
        })
        
    return {
        "homeID": home_id,
        "awayID": away_id,
        "homeTeam": home_name,
        "awayTeam": away_name,
        "homeWinProbability": round(home_win_prob, 2),
        "awayWinProbability": round(away_win_prob, 2),
        "predictedWinner": predicted_winner_name,
        "predictedBestPlayer": player_predictions[0]["playerName"] if player_predictions else None,
        "winningTeamPlayers": player_predictions[:top_n],
    }