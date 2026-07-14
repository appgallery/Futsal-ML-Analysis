import pytest
from pydantic import ValidationError
from scripts.inference import MatchInput, MatchOutput, PlayerPrediction, CommentaryOutput, MatchReportResponse

def test_match_input_valid():
    m = MatchInput(homeID=10, awayID=20)
    assert m.homeID == 10
    assert m.awayID == 20
    assert m.matchID is None

def test_match_input_optional_match_id():
    m = MatchInput(matchID=5, homeID=10, awayID=20)
    assert m.matchID == 5

def test_match_output_schema():
    m = MatchOutput(homeID=10, awayID=20, prediction="Home Win", confidence=0.85)
    assert m.prediction == "Home Win"
    assert m.confidence == 0.85

def test_player_prediction_schema():
    p = PlayerPrediction(playerName="John Doe", playWellProbability=80.5, predictedPerfScore=15.0)
    assert p.playerName == "John Doe"
    assert p.playWellProbability == 80.5

def test_commentary_output_schema():
    c = CommentaryOutput(matchSummary="Summary", keyPlayerInsight="Insight", overallAssessment="Assessment")
    assert c.matchSummary == "Summary"

def test_match_report_response_full():
    c = CommentaryOutput(matchSummary="Summary", keyPlayerInsight="Insight", overallAssessment="Assessment")
    p = PlayerPrediction(playerName="John Doe", playWellProbability=80.5, predictedPerfScore=15.0)
    
    m = MatchReportResponse(
        homeTeam="Team A",
        awayTeam="Team B",
        homeWinProbability=60.0,
        awayWinProbability=40.0,
        predictedWinner="Team A",
        predictedBestPlayer="John Doe",
        winningTeamPlayers=[p],
        matchSummary=c,
        narrativeContext={"some": "data"}
    )
    
    assert m.matchSummary.matchSummary == "Summary"
    assert len(m.winningTeamPlayers) == 1

def test_match_report_serialization():
    p = PlayerPrediction(playerName="John Doe", playWellProbability=80.5, predictedPerfScore=15.0)
    m = MatchReportResponse(
        homeTeam="Team A",
        awayTeam="Team B",
        homeWinProbability=60.0,
        awayWinProbability=40.0,
        predictedWinner="Team A",
        predictedBestPlayer="John Doe",
        winningTeamPlayers=[p]
    )
    
    # Serialize to dict
    d = m.model_dump()
    
    # Deserialize
    m2 = MatchReportResponse(**d)
    
    assert m2.homeTeam == "Team A"
    assert m2.winningTeamPlayers[0].playerName == "John Doe"
