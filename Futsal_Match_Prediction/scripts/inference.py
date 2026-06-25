import joblib
from pydantic import BaseModel
from typing import List, Optional, Dict
from components.data_transformation import DataTransformation
from components.data_ingestion import DataIngestion
import pandas as pd
import numpy as np
import glob
import os

class MatchInput(BaseModel):
    matchID: Optional[int] = None
    homeID: int
    awayID: int

class MatchOutput(BaseModel):
    matchID: Optional[int] = None
    homeID: int
    awayID: int
    prediction: str
    confidence: float

class PlayerPrediction(BaseModel):
    playerName: str
    playWellProbability: float
    predictedPerfScore: float

class CommentaryOutput(BaseModel):
    matchSummary: str
    keyPlayerInsight: str
    overallAssessment: str

class MatchReportResponse(BaseModel):
    homeTeam: str
    awayTeam: str
    homeWinProbability: float
    awayWinProbability: float
    predictedWinner: str
    predictedBestPlayer: str
    winningTeamPlayers: List[PlayerPrediction]
    matchSummary: Optional[CommentaryOutput] = None
    narrativeContext: Optional[dict] = None

    
class Inference:
    def __init__(self):
        # Dynamically locate match classification model
        match_models = glob.glob("models/*_matches_model.pkl")
        if not match_models:
            raise FileNotFoundError("No match prediction model file (*_matches_model.pkl) found in models/")
        self.model_path = match_models[0]
        self.features_path = "features/match_trained_features.pkl"
        
        # Dynamically locate player regression model
        player_models = glob.glob("models/*_players_model.pkl")
        if not player_models:
            raise FileNotFoundError("No player performance model file (*_players_model.pkl) found in models/")
        self.player_model_path = player_models[0]
        self.player_features_path = "features/player_trained_features.pkl"
        
        print(f"Loading best match model: {self.model_path}")
        self.model = self.load_model(self.model_path)
        self.features = self.load_model(self.features_path)
        
        print(f"Loading best player model: {self.player_model_path}")
        self.player_model = self.load_model(self.player_model_path)
        self.player_features = self.load_model(self.player_features_path)
        
        print("Initializing Inference State... (This may take a few seconds)")
        self.dt = DataTransformation()
        raw_matches = DataIngestion().ingest_match_data()
        self.dt.transform(raw_matches, save_csv=False)
        self.team_memory = self.dt.team_memory
        self.h2h_memory = self.dt.h2h_memory
        print("Inference State Ready!")
    
    def load_model(self, model_path: str):
        return joblib.load(model_path)

    def test_match_infer(self, data: MatchInput) -> MatchOutput:
        # Resolve IDs
        match_id = data.matchID
        home_id = data.homeID
        away_id = data.awayID
        
        # Ensure memory exists
        self.dt.initialize_team_memory(home_id)
        self.dt.initialize_team_memory(away_id)
        
        hs = self.team_memory[home_id]
        aws = self.team_memory[away_id]
        
        # Compute Features
        home_win_rate = hs['wins'] / hs['matches'] if hs['matches'] > 0 else 0
        away_win_rate = aws['wins'] / aws['matches'] if aws['matches'] > 0 else 0
        
        home_recent = hs['results'][-5:]
        away_recent = aws['results'][-5:]
        
        home_wf = np.average(home_recent, weights=np.exp(np.linspace(0, 1, len(home_recent)))) if len(home_recent) > 0 else 0
        away_wf = np.average(away_recent, weights=np.exp(np.linspace(0, 1, len(away_recent)))) if len(away_recent) > 0 else 0
        diff_form = home_wf - away_wf
        
        home_attack = np.mean(hs['goals_scored'][-5:]) if len(hs['goals_scored']) > 0 else 0
        away_attack = np.mean(aws['goals_scored'][-5:]) if len(aws['goals_scored']) > 0 else 0
        diff_attack = home_attack - away_attack
        
        home_defense = np.mean(hs['goals_conceded'][-5:]) if len(hs['goals_conceded']) > 0 else 0
        away_defense = np.mean(aws['goals_conceded'][-5:]) if len(aws['goals_conceded']) > 0 else 0
        diff_defense = home_defense - away_defense
        
        home_goal_diff = np.mean(hs['goal_difference'][-5:]) if len(hs['goal_difference']) > 0 else 0
        away_goal_diff = np.mean(aws['goal_difference'][-5:]) if len(aws['goal_difference']) > 0 else 0
        diff_goal_diff = home_goal_diff - away_goal_diff
        
        diff_elo = hs['elo'] - aws['elo']
        
        pair = tuple(sorted([home_id, away_id]))
        self.dt.initialize_h2h_memory(pair)
        h2h = self.h2h_memory[pair]
        
        if h2h['matches'] > 0:
            if pair[0] == home_id:
                h2h_home_win_rate = h2h['team1_wins'] / h2h['matches']
            else:
                h2h_home_win_rate = h2h['team2_wins'] / h2h['matches']
        else:
            h2h_home_win_rate = 0.5
            
        feature_dict = {
            'home_win_rate': [home_win_rate],
            'away_win_rate': [away_win_rate],
            'diff_goal_diff': [diff_goal_diff],
            'diff_form': [diff_form],
            'diff_attack': [diff_attack],
            'diff_defense': [diff_defense],
            'diff_elo': [diff_elo],
            'h2h_home_win_rate': [h2h_home_win_rate]
        }
        
        # Use features loaded from .pkl
        feature_cols = self.features
        
        X_infer = pd.DataFrame(feature_dict)[feature_cols]
        
        # Predict
        prediction_num = self.model.predict(X_infer)[0]
        
        # Some models support predict_proba, some don't (like SVC without probability=True)
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(X_infer)[0]
                confidence = float(np.max(probs))
            except Exception:
                pass
            
        prediction_str = "Home Win" if prediction_num == 1 else "Away Win"
        return MatchOutput(
            matchID=match_id,
            homeID=home_id,
            awayID=away_id,
            prediction=prediction_str,
            confidence=confidence
        )

    def test_player_infer(self, match_pred: MatchOutput) -> MatchReportResponse:
        # Extract team IDs
        match_id = match_pred.matchID
        home_id = match_pred.homeID
        away_id = match_pred.awayID

        # Calculate win probabilities based on prediction outcome
        if match_pred.prediction == "Home Win":
            home_win_probability = round(match_pred.confidence * 100, 2)
            away_win_probability = round(100.0 - home_win_probability, 2)
        else:
            away_win_probability = round(match_pred.confidence * 100, 2)
            home_win_probability = round(100.0 - away_win_probability, 2)

        # Load player feature dataset
        historical_df = pd.read_csv("data/players_ml.csv")

        # Process dates
        historical_df['match_date'] = pd.to_datetime(
            historical_df['match_date'],
            errors='coerce'
        )
        historical_df = historical_df.sort_values('match_date')

        # Clean player rows
        historical_df = historical_df.dropna(subset=['user_id'])
        historical_df = historical_df.drop_duplicates()

        # Build dynamic clean team mapping
        home_mapping = historical_df[['homeTeamId', 'homeTeamName']].copy().rename(
            columns={'homeTeamId': 'team_id', 'homeTeamName': 'team_name'}
        )
        away_mapping = historical_df[['awayTeamId', 'awayTeamName']].copy().rename(
            columns={'awayTeamId': 'team_id', 'awayTeamName': 'team_name'}
        )
        team_mapping = pd.concat([home_mapping, away_mapping], axis=0).dropna()

        # Group by team_id and fetch most frequent team name
        team_mapping = (
            team_mapping
            .groupby('team_id')['team_name']
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
        )

        # Resolve Team Names automatically
        home_team_rows = team_mapping.loc[team_mapping['team_id'] == home_id, 'team_name']
        away_team_rows = team_mapping.loc[team_mapping['team_id'] == away_id, 'team_name']

        home_team_name = home_team_rows.iloc[0] if len(home_team_rows) > 0 else f"Home Team {home_id}"
        away_team_name = away_team_rows.iloc[0] if len(away_team_rows) > 0 else f"Away Team {away_id}"

        # Determine Winner dynamic details
        winner_id = home_id if match_pred.prediction == "Home Win" else away_id
        winner_team_name = home_team_name if match_pred.prediction == "Home Win" else away_team_name

        # Retrieve recent active players for the predicted WINNING team only
        latest_date = historical_df['match_date'].max()
        recent_cutoff = latest_date - pd.Timedelta(days=120)

        winning_players_df = historical_df[
            (historical_df['team_id'] == winner_id) &
            (historical_df['match_date'] >= recent_cutoff)
        ].copy()

        # Fallback if no players are active in past 120 days
        if winning_players_df.empty:
            winning_players_df = historical_df[historical_df['team_id'] == winner_id].copy()

        # Keep the most recent record of each player
        winning_players_df = winning_players_df.groupby('user_id').tail(1)

        # Keep top 8 players of the winning team by recent performance score
        winning_players_df = winning_players_df.sort_values('roll5_perf_score', ascending=False).head(8)

        if 'player_name' not in winning_players_df.columns:
            winning_players_df['player_name'] = (
                winning_players_df['first_name'].fillna('')
                + ' '
                + winning_players_df['last_name'].fillna('')
            ).str.strip()

        # Keep valid features only
        valid_features = [col for col in self.player_features if col in winning_players_df.columns]
        winning_players_df[valid_features] = winning_players_df[valid_features].fillna(0)

        # Predict play well probability
        if hasattr(self.player_model, 'predict_proba'):
            probs = self.player_model.predict_proba(winning_players_df[valid_features])[:, 1]
        else:
            probs = self.player_model.predict(winning_players_df[valid_features])
            
        winning_players_df['predicted_perf_score'] = probs  # Placeholder since we don't predict regression score anymore
        winning_players_df['play_well_probability'] = (probs * 100).round(2)

        # Sort player predictions
        winning_players_df = winning_players_df.sort_values('play_well_probability', ascending=False)


        di = DataIngestion()
        if di.engine is not None and match_id is not None:
            try:
                attendance_query = "SELECT CONCAT(u.first_name, ' ', u.last_name) AS player_name FROM futsaloz.player_attendance pa INNER JOIN users u ON pa.user_id = u.user_id WHERE pa.match_id = %s AND pa.is_present = 1"
                
                raw_conn = di.engine.raw_connection()
                try:
                    attendance_df = pd.read_sql(attendance_query, raw_conn, params=[match_id])
                finally:
                    raw_conn.close()
                
                # Names present in this match only
                match_players = set(
                    attendance_df["player_name"].str.strip().str.lower()
                )
            finally:
                di.close_connection()
        else:
            match_players = set()

        # Build Response Player List for predicted winning team only
        winning_players = []
        for _, row in winning_players_df.iterrows():
            player_name = row["player_name"].strip().lower()
            
            # If match_players is empty (no attendance data yet), assume all recent top players are available
            if not match_players or player_name in match_players:
                player_pred = PlayerPrediction(
                    playerName=row['player_name'],
                    playWellProbability=float(row['play_well_probability']),
                    predictedPerfScore=float(round(row['predicted_perf_score'], 2))
                )
                winning_players.append(player_pred)

        predicted_best_player = winning_players[0].playerName if len(winning_players) > 0 else "N/A"

        return MatchReportResponse(
            homeTeam=home_team_name,
            awayTeam=away_team_name,
            homeWinProbability=home_win_probability,
            awayWinProbability=away_win_probability,
            predictedWinner=winner_team_name,
            predictedBestPlayer=predicted_best_player,
            winningTeamPlayers=winning_players
        )