import pytest
import pandas as pd
from unittest.mock import MagicMock

@pytest.fixture
def sample_matches_df():
    """Minimal matches DataFrame with 5+ rows, all required columns."""
    data = {
        "id": [1, 2, 3, 4, 5, 6],
        "startDate": [
            "2023-01-01 10:00:00", "2023-01-08 10:00:00", "2023-01-15 10:00:00",
            "2023-01-22 10:00:00", "2023-01-29 10:00:00", "2023-02-05 10:00:00"
        ],
        "homeTeamId": [10, 10, 10, 20, 20, 10],
        "awayTeamId": [20, 20, 20, 10, 10, 30],
        "homeTeamName": ["Team A", "Team A", "Team A", "Team B", "Team B", "Team A"],
        "awayTeamName": ["Team B", "Team B", "Team B", "Team A", "Team A", "Team C"],
        "competitionName": ["League 1"] * 6,
        "seasonName": ["2023"] * 6,
        "winningTeam": [10, 20, 10, 10, 20, 10],
        "losingTeam": [20, 10, 20, 20, 10, 30],
        "winningTeamGoals": [3, 2, 4, 1, 3, 5],
        "losingTeamGoals": [1, 0, 2, 0, 1, 0],
        "winningTeamPoints": [3] * 6,
        "losingTeamPoints": [0] * 6,
        "match_id": [1, 2, 3, 4, 5, 6],
        "cmp_id": [130] * 6,
        "status": ["Completed"] * 6,
        "is_bye": [False] * 6,
        "is_forfeited": [False] * 6,
        "is_cancel": [False] * 6,
        "isDraw": [False] * 6,
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_players_df():
    """Minimal players DataFrame with SCORE/FOULS/YELLOWCARD/REDCARD events."""
    data = {
        "match_id": [1, 1, 1, 1, 2, 2],
        "user_id": [100, 101, 100, 102, 100, 101],
        "team_id": [10, 10, 10, 20, 10, 10],
        "homeTeamId": [10, 10, 10, 10, 10, 10],
        "awayTeamId": [20, 20, 20, 20, 20, 20],
        "homeTeamName": ["Team A"] * 6,
        "awayTeamName": ["Team B"] * 6,
        "type": ["SCORE", "FOULS", "YELLOWCARD", "REDCARD", "SCORE", "SCORE"],
        "assist_player_id": [101, None, None, None, None, 100],
        "first_name": ["John", "Mike", "John", "Chris", "John", "Mike"],
        "last_name": ["Doe", "Smith", "Doe", "Evans", "Doe", "Smith"],
        "date_of_birth": ["1990-01-01", "1992-05-05", "1990-01-01", "1988-08-08", "1990-01-01", "1992-05-05"],
        "created_on": ["2023-01-01 10:00:00"] * 6,
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_prediction_json():
    """Valid prediction JSON matching MatchReportResponse schema."""
    return {
        "homeTeam": "Team A",
        "awayTeam": "Team B",
        "homeWinProbability": 65.5,
        "awayWinProbability": 34.5,
        "predictedWinner": "Team A",
        "predictedBestPlayer": "John Doe",
        "winningTeamPlayers": [
            {
                "playerName": "John Doe",
                "playWellProbability": 85.0,
                "predictedPerfScore": 15.5
            },
            {
                "playerName": "Mike Smith",
                "playWellProbability": 70.0,
                "predictedPerfScore": 10.2
            }
        ],
        "matchSummary": None,
        "narrativeContext": None
    }

@pytest.fixture
def mock_inference_engine():
    """Mocked Inference class with canned predictions."""
    from scripts.inference import MatchOutput, MatchReportResponse, PlayerPrediction
    
    mock_engine = MagicMock()
    mock_engine.team_memory = {10: {}, 20: {}}
    
    mock_engine.test_match_infer.return_value = MatchOutput(
        matchID=1,
        homeID=10,
        awayID=20,
        prediction="Home Win",
        confidence=0.85
    )
    
    mock_engine.test_player_infer.return_value = MatchReportResponse(
        homeTeam="Team A",
        awayTeam="Team B",
        homeWinProbability=85.0,
        awayWinProbability=15.0,
        predictedWinner="Team A",
        predictedBestPlayer="John Doe",
        winningTeamPlayers=[
            PlayerPrediction(playerName="John Doe", playWellProbability=85.0, predictedPerfScore=15.5)
        ]
    )
    return mock_engine

@pytest.fixture
def sample_matches_csv(tmp_path, sample_matches_df):
    """Write sample matches to temp CSV, return path."""
    df = sample_matches_df.copy()
    df["outcome"] = df.apply(lambda row: 1 if row["winningTeam"] == row["homeTeamId"] else 0, axis=1)
    df["home_goals"] = df.apply(lambda row: row["winningTeamGoals"] if row["winningTeam"] == row["homeTeamId"] else row["losingTeamGoals"], axis=1)
    df["away_goals"] = df.apply(lambda row: row["winningTeamGoals"] if row["winningTeam"] == row["awayTeamId"] else row["losingTeamGoals"], axis=1)
    
    csv_path = tmp_path / "matches_ml.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def sample_players_csv(tmp_path, sample_players_df):
    """Write sample players to temp CSV, return path."""
    data = {
        "match_id": [1, 1],
        "user_id": [100, 101],
        "team_id": [10, 10],
        "homeTeamId": [10, 10],
        "awayTeamId": [20, 20],
        "homeTeamName": ["Team A", "Team A"],
        "awayTeamName": ["Team B", "Team B"],
        "match_date": ["2023-01-01", "2023-01-01"],
        "player_name": ["John Doe", "Mike Smith"],
        "goals": [1, 0],
        "assists": [0, 1],
        "perf_score": [5.0, 3.0],
        "career_matches": [10, 5],
        "career_goals": [5, 1],
        "career_assists": [2, 3],
        "career_gpg": [0.5, 0.2],
        "career_avg_perf": [4.0, 2.5]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "players_ml.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)
