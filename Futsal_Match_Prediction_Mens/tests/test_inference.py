import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from scripts.inference import Inference, MatchInput, MatchOutput, MatchReportResponse, PlayerPrediction

@pytest.fixture
def mock_inference_dependencies(monkeypatch):
    # Mock glob to find fake models
    monkeypatch.setattr("glob.glob", lambda pattern: ["models/fake_model.pkl"])
    
    # Mock joblib.load
    def mock_load(path):
        if "features" in path:
            return ["feature1", "feature2", "feature3"]
        else:
            mock_model = MagicMock()
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = [[0.2, 0.8]]
            return mock_model
    monkeypatch.setattr("joblib.load", mock_load)
    
    # Mock DataTransformation
    mock_dt = MagicMock()
    mock_dt.team_memory = {10: {"matches": 10, "wins": 5, "goals_scored": [1,2,3,4,5], "goals_conceded": [1,1,1,1,1], "goal_difference": [0,1,2,3,4], "results": [1,0,1,0,1], "elo": 1500},
                           20: {"matches": 10, "wins": 2, "goals_scored": [0,1,0,1,0], "goals_conceded": [2,2,2,2,2], "goal_difference": [-2,-1,-2,-1,-2], "results": [0,0,0,0,0], "elo": 1400}}
    mock_dt.h2h_memory = {(10, 20): {"matches": 2, "team1_wins": 2, "team2_wins": 0}}
    monkeypatch.setattr("scripts.inference.DataTransformation", lambda: mock_dt)
    
    # Mock DataIngestion
    mock_di = MagicMock()
    mock_di_instance = MagicMock()
    mock_di_instance.ingest_match_data.return_value = pd.DataFrame()
    mock_di_instance.engine = None
    mock_di.return_value = mock_di_instance
    monkeypatch.setattr("scripts.inference.DataIngestion", mock_di)

def test_home_win_prediction(mock_inference_dependencies):
    inf = Inference()
    inf.model.predict.return_value = [1]
    
    result = inf.test_match_infer(MatchInput(homeID=10, awayID=20))
    
    assert result.prediction == "Home Win"
    assert result.confidence == 0.8

def test_away_win_prediction(mock_inference_dependencies):
    inf = Inference()
    inf.model.predict.return_value = [0]
    
    result = inf.test_match_infer(MatchInput(homeID=10, awayID=20))
    
    assert result.prediction == "Away Win"

def test_confidence_without_proba(mock_inference_dependencies, monkeypatch):
    inf = Inference()
    del inf.model.predict_proba
    
    result = inf.test_match_infer(MatchInput(homeID=10, awayID=20))
    
    assert result.confidence == 1.0

def test_unknown_team_id(mock_inference_dependencies):
    inf = Inference()
    # Mock initialize_team_memory to add unknown team to memory on the fly
    def side_effect(team_id):
        if team_id not in inf.dt.team_memory:
            inf.dt.team_memory[team_id] = {"matches": 0, "wins": 0, "goals_scored": [], "goals_conceded": [], "goal_difference": [], "results": [], "elo": 1500}
    inf.dt.initialize_team_memory.side_effect = side_effect
    
    def side_effect_h2h(pair):
        if pair not in inf.dt.h2h_memory:
            inf.dt.h2h_memory[pair] = {"matches": 0, "team1_wins": 0, "team2_wins": 0}
    inf.dt.initialize_h2h_memory.side_effect = side_effect_h2h
    
    result = inf.test_match_infer(MatchInput(homeID=99, awayID=100))
    
    assert result.prediction in ["Home Win", "Away Win"]
    assert inf.dt.initialize_team_memory.call_count >= 2

def test_feature_alignment(mock_inference_dependencies, monkeypatch):
    inf = Inference()
    # Verify that the dataframe created matches inf.features exactly
    called_with_df = None
    def mock_predict(df):
        nonlocal called_with_df
        called_with_df = df
        return [1]
    inf.model.predict = mock_predict
    
    # We need features to match the hardcoded feature_dict in Inference, or it'll fail when slicing
    inf.features = ['home_win_rate', 'away_win_rate', 'diff_goal_diff']
    inf.test_match_infer(MatchInput(homeID=10, awayID=20))
    
    assert called_with_df is not None
    assert list(called_with_df.columns) == inf.features

# --- Player Inference Tests ---
@pytest.fixture
def mock_player_infer_dependencies(monkeypatch, sample_players_df):
    def mock_read_csv(path):
        if "players_ml.csv" in path:
            df = sample_players_df.copy()
            # Needed columns for test_player_infer
            df['match_date'] = pd.to_datetime(df['created_on'])
            df['homeTeamId'] = df['homeTeamId']
            df['awayTeamId'] = df['awayTeamId']
            df['homeTeamName'] = df['homeTeamName']
            df['awayTeamName'] = df['awayTeamName']
            df['user_id'] = df['user_id']
            df['team_id'] = df['team_id']
            df['roll5_perf_score'] = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
            df['feature1'] = 0.5
            df['feature2'] = 0.5
            df['feature3'] = 0.5
            return df
        return pd.DataFrame()
    monkeypatch.setattr("pandas.read_csv", mock_read_csv)

def test_win_probability_home(mock_inference_dependencies, mock_player_infer_dependencies):
    inf = Inference()
    match_pred = MatchOutput(homeID=10, awayID=20, prediction="Home Win", confidence=0.75)
    
    report = inf.test_player_infer(match_pred)
    
    assert report.homeWinProbability == 75.0
    assert report.awayWinProbability == 25.0

def test_winning_team_players(mock_inference_dependencies, mock_player_infer_dependencies):
    inf = Inference()
    # Team 10 is home, Team 20 is away
    match_pred = MatchOutput(homeID=10, awayID=20, prediction="Home Win", confidence=0.8)
    
    report = inf.test_player_infer(match_pred)
    
    # John Doe and Mike Smith are on team 10 (home)
    assert len(report.winningTeamPlayers) > 0
    for p in report.winningTeamPlayers:
        assert p.playerName in ["John Doe", "Mike Smith"]

def test_best_player_is_first(mock_inference_dependencies, mock_player_infer_dependencies):
    inf = Inference()
    match_pred = MatchOutput(homeID=10, awayID=20, prediction="Home Win", confidence=0.8)
    
    # Force probabilities so we know who should be first
    inf.player_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8]]) # Assuming 2 players
    
    report = inf.test_player_infer(match_pred)
    
    # highest prob should be first
    assert report.predictedBestPlayer == report.winningTeamPlayers[0].playerName
    
    probs = [p.playWellProbability for p in report.winningTeamPlayers]
    assert probs == sorted(probs, reverse=True)

def test_no_attendance_data(mock_inference_dependencies, mock_player_infer_dependencies):
    inf = Inference()
    # By default mock_di.engine is None, meaning no DB connection for attendance
    match_pred = MatchOutput(homeID=10, awayID=20, prediction="Home Win", confidence=0.8)
    
    report = inf.test_player_infer(match_pred)
    
    # Should fall back to all players from the winning team
    assert len(report.winningTeamPlayers) > 0

def test_fallback_all_players(mock_inference_dependencies, monkeypatch, sample_players_df):
    inf = Inference()
    
    def mock_read_csv(path):
        df = sample_players_df.copy()
        # Set dates very old (> 120 days ago)
        df['match_date'] = pd.to_datetime('2020-01-01')
        df['roll5_perf_score'] = 1.0
        df['feature1'] = 0.5
        return df
    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    
    match_pred = MatchOutput(homeID=10, awayID=20, prediction="Home Win", confidence=0.8)
    report = inf.test_player_infer(match_pred)
    
    # It should still return players via the fallback
    assert len(report.winningTeamPlayers) > 0
