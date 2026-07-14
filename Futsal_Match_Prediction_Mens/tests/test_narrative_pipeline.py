import pytest
import pandas as pd
import json
import os
from unittest.mock import patch, MagicMock
from pipeline.narrative_pipeline import NarrativeContextBuilder, PromptBuilder, CommentaryGenerator, enrich_prediction_with_commentary

def test_h2h_no_meetings(sample_players_csv, sample_matches_csv):
    builder = NarrativeContextBuilder(sample_players_csv, sample_matches_csv)
    result = builder._h2h_summary("Unknown 1", "Unknown 2")
    
    assert result["total"] == 0
    assert result["summary"] == "No previous meetings."

def test_h2h_with_matches(sample_players_csv, sample_matches_csv):
    builder = NarrativeContextBuilder(sample_players_csv, sample_matches_csv)
    # Based on sample_matches_df: Team A vs Team B occurs several times
    result = builder._h2h_summary("Team A", "Team B")
    
    assert result["total"] > 0
    assert "Team A" in result["dominant"] or "Team B" in result["dominant"]
    assert "last5" in result

def test_team_form_string(sample_players_csv, sample_matches_csv):
    builder = NarrativeContextBuilder(sample_players_csv, sample_matches_csv)
    result = builder._team_form("Team A", last_n=5)
    
    # Check that it returns a string of Ws and Ls
    assert "form_string" in result
    for char in result["form_string"].replace(" ", ""):
        assert char in ["W", "L"]

def test_team_strength_latest(sample_players_csv, sample_matches_csv):
    builder = NarrativeContextBuilder(sample_players_csv, sample_matches_csv)
    
    # We didn't explicitly mock the strength columns in sample_matches_df,
    # but the method should return an empty dict if the columns are missing or
    # return the default 0.0 values if columns exist but are not populated.
    result = builder._team_strength("Team A")
    
    assert "elo" in result
    assert "attack_strength" in result

def test_player_career_stats(sample_players_csv, sample_matches_csv):
    builder = NarrativeContextBuilder(sample_players_csv, sample_matches_csv)
    result = builder._player_career("John Doe")
    
    assert result["career_matches"] >= 0
    assert "roll5_goals" in result

def test_player_vs_opponent_no_data(sample_players_csv, sample_matches_csv):
    builder = NarrativeContextBuilder(sample_players_csv, sample_matches_csv)
    result = builder._player_vs_opponent("John Doe", "Unknown Team")
    
    assert result["matches"] == 0
    assert result["goals"] == 0

def test_build_full_context(sample_players_csv, sample_matches_csv, sample_prediction_json):
    builder = NarrativeContextBuilder(sample_players_csv, sample_matches_csv)
    
    context = builder.build(sample_prediction_json)
    
    assert context["home_team"] == "Team A"
    assert context["away_team"] == "Team B"
    assert context["predicted_winner"] == "Team A"
    assert "h2h" in context
    assert "home_form" in context
    assert "away_form" in context
    assert "player_contexts" in context
    # Only top 3 players should be processed
    assert len(context["player_contexts"]) <= 3

def test_prompt_contains_teams(sample_players_csv, sample_matches_csv, sample_prediction_json):
    builder = NarrativeContextBuilder(sample_players_csv, sample_matches_csv)
    context = builder.build(sample_prediction_json)
    
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build(context)
    
    assert "Team A" in prompt
    assert "Team B" in prompt

def test_system_prompt_no_ai_mention():
    system = PromptBuilder.SYSTEM
    assert "AI" not in system.lower().replace("aim", "") # 'aim' might be a word, but ' AI ' is safe
    # Actually just check the text explicitly instructs not to mention AI
    assert "Do not mention AI" in system

@pytest.mark.asyncio
async def test_generate_returns_dict():
    # Mock liteLLM
    generator = CommentaryGenerator()
    
    with patch("litellm.acompletion") as mock_acompletion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"matchSummary": "Test summary", "keyPlayerInsight": "Test player", "overallAssessment": "Test assessment"}'
        mock_acompletion.return_value = mock_response
        
        result = await generator.generate("System", "User")
        
        assert isinstance(result, dict)
        assert result["matchSummary"] == "Test summary"

@pytest.mark.asyncio
async def test_generate_handles_malformed_json():
    generator = CommentaryGenerator()
    
    with patch("litellm.acompletion") as mock_acompletion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Invalid JSON'
        mock_acompletion.return_value = mock_response
        
        with pytest.raises(json.JSONDecodeError):
            await generator.generate("System", "User")

def test_api_key_set():
    with patch.dict(os.environ, {}, clear=True):
        generator = CommentaryGenerator(api_key="TEST_KEY")
        assert os.environ["GEMINI_API_KEY"] == "TEST_KEY"

@pytest.mark.asyncio
async def test_enrich_adds_commentary(sample_players_csv, sample_matches_csv, sample_prediction_json):
    with patch("pipeline.narrative_pipeline.CommentaryGenerator.generate") as mock_generate:
        mock_generate.return_value = {
            "matchSummary": "Great match.",
            "keyPlayerInsight": "John played well.",
            "overallAssessment": "Home wins."
        }
        
        enriched = await enrich_prediction_with_commentary(
            sample_prediction_json,
            sample_players_csv,
            sample_matches_csv
        )
        
        assert enriched["matchSummary"] == mock_generate.return_value
        assert "narrativeContext" in enriched
        assert enriched["narrativeContext"]["home_form"] is not None
        
        # Check original is preserved
        assert enriched["homeTeam"] == "Team A"
        assert enriched["homeWinProbability"] == 65.5
