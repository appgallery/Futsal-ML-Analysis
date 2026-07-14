import pytest
import pandas as pd
import numpy as np
from components.data_transformation import DataMatchTransformation, DataPlayerTransformation

def test_drop_columns(sample_matches_df):
    dt = DataMatchTransformation()
    # Let's add some columns to drop
    df = sample_matches_df.copy()
    df['quaterFinal'] = 1
    df['note'] = 'test'
    
    result = dt.drop_columns(df)
    
    assert 'quaterFinal' not in result.columns
    assert 'note' not in result.columns
    assert 'homeTeamName' in result.columns # Should survive

def test_remove_invalid_matches(sample_matches_df):
    dt = DataMatchTransformation()
    df = sample_matches_df.copy()
    df.loc[0, 'is_cancel'] = True
    df.loc[1, 'is_forfeited'] = True
    
    result = dt.remove_invalid_matches(df)
    
    assert len(result) == len(df) - 2
    assert result['is_cancel'].sum() == 0
    assert result['is_forfeited'].sum() == 0

def test_remove_non_completed(sample_matches_df):
    dt = DataMatchTransformation()
    df = sample_matches_df.copy()
    df.loc[0, 'status'] = 'Scheduled'
    
    result = dt.remove_non_completed_matches(df)
    
    assert len(result) == len(df) - 1
    assert (result['status'] == 'Completed').all()

def test_clean_date_sorting(sample_matches_df):
    dt = DataMatchTransformation()
    df = sample_matches_df.copy()
    # Mess up the order
    df = pd.concat([df.iloc[2:], df.iloc[:2]])
    
    result = dt.clean_date(df)
    
    # Check if dates are monotonically increasing
    assert result['startDate'].is_monotonic_increasing

def test_create_outcome_home_win(sample_matches_df):
    dt = DataMatchTransformation()
    df = sample_matches_df.copy()
    # Set home win explicitly
    df.loc[0, 'homeTeamId'] = 10
    df.loc[0, 'awayTeamId'] = 20
    df.loc[0, 'winningTeam'] = 10
    
    result = dt.create_outcome(df)
    assert result.loc[0, 'outcome'] == 1

def test_create_outcome_away_win(sample_matches_df):
    dt = DataMatchTransformation()
    df = sample_matches_df.copy()
    # Set away win explicitly
    df.loc[1, 'homeTeamId'] = 10
    df.loc[1, 'awayTeamId'] = 20
    df.loc[1, 'winningTeam'] = 20
    
    result = dt.create_outcome(df)
    assert result.loc[1, 'outcome'] == 0

def test_create_home_away_goals(sample_matches_df):
    dt = DataMatchTransformation()
    df = sample_matches_df.copy()
    # Home win
    df.loc[0, 'winningTeam'] = df.loc[0, 'homeTeamId']
    df.loc[0, 'winningTeamGoals'] = 3
    df.loc[0, 'losingTeamGoals'] = 1
    
    # Away win
    df.loc[1, 'winningTeam'] = df.loc[1, 'awayTeamId']
    df.loc[1, 'winningTeamGoals'] = 4
    df.loc[1, 'losingTeamGoals'] = 2
    
    result = dt.create_home_away_goals(df)
    
    assert result.loc[0, 'home_goals'] == 3
    assert result.loc[0, 'away_goals'] == 1
    
    assert result.loc[1, 'home_goals'] == 2
    assert result.loc[1, 'away_goals'] == 4

def test_feature_columns_present(sample_matches_df):
    dt = DataMatchTransformation()
    df = dt.create_features(sample_matches_df.copy())
    for col in dt.feature_cols:
        assert col in df.columns

def test_elo_initial_values(sample_matches_df):
    dt = DataMatchTransformation()
    df = dt.clean_date(sample_matches_df)
    df = dt.create_outcome(df)
    df = dt.create_home_away_goals(df)
    df = dt.create_features(df)
    
    dt.feature_engineer(df.iloc[[0]])
    
    # Both start at 1500 before the match
    assert dt.team_memory[10]['elo'] != 1500 # Should be updated after the match
    
def test_elo_symmetric(sample_matches_df):
    dt = DataMatchTransformation()
    df = dt.clean_date(sample_matches_df)
    df = dt.create_outcome(df)
    df = dt.create_home_away_goals(df)
    df = dt.create_features(df)
    
    dt.feature_engineer(df.iloc[[0]])
    
    home_elo = dt.team_memory[10]['elo']
    away_elo = dt.team_memory[20]['elo']
    
    # Initial is 1500 each, total 3000
    assert abs((home_elo + away_elo) - 3000) < 1e-6

def test_win_rate_zero_matches(sample_matches_df):
    dt = DataMatchTransformation()
    df = dt.clean_date(sample_matches_df)
    df = dt.create_outcome(df)
    df = dt.create_home_away_goals(df)
    df = dt.create_features(df)
    
    result = dt.feature_engineer(df.iloc[[0]])
    # The first match should have win rate 0 because they had 0 matches before it
    assert result.loc[0, 'home_win_rate'] == 0
    assert result.loc[0, 'away_win_rate'] == 0

def test_weighted_form_empty(sample_matches_df):
    dt = DataMatchTransformation()
    df = dt.clean_date(sample_matches_df)
    df = dt.create_outcome(df)
    df = dt.create_home_away_goals(df)
    df = dt.create_features(df)
    
    result = dt.feature_engineer(df.iloc[[0]])
    assert result.loc[0, 'home_weighted_form'] == 0
    assert result.loc[0, 'away_weighted_form'] == 0

def test_h2h_first_meeting(sample_matches_df):
    dt = DataMatchTransformation()
    df = dt.clean_date(sample_matches_df)
    df = dt.create_outcome(df)
    df = dt.create_home_away_goals(df)
    df = dt.create_features(df)
    
    result = dt.feature_engineer(df.iloc[[0]])
    
    assert result.loc[0, 'h2h_matches'] == 0
    assert result.loc[0, 'h2h_home_win_rate'] == 0.5

def test_defense_strength_formula(sample_matches_df):
    dt = DataMatchTransformation()
    df = dt.clean_date(sample_matches_df)
    df = dt.create_outcome(df)
    df = dt.create_home_away_goals(df)
    df = dt.create_features(df)
    
    result = dt.feature_engineer(df.iloc[:2])
    # The second match should have defense strength based on first match
    # In match 1, home team 10 conceded 1 goal from team 20.
    assert result.loc[1, 'home_defense_strength'] == 1 + (1 + 1) # 3

def test_differential_features(sample_matches_df):
    dt = DataMatchTransformation()
    df = dt.clean_date(sample_matches_df)
    df = dt.create_outcome(df)
    df = dt.create_home_away_goals(df)
    df = dt.create_features(df)
    
    result = dt.feature_engineer(df.iloc[:2])
    
    assert result.loc[1, 'diff_form'] == result.loc[1, 'home_weighted_form'] - result.loc[1, 'away_weighted_form']
    assert result.loc[1, 'diff_attack'] == result.loc[1, 'home_attack_strength'] - result.loc[1, 'away_attack_strength']
    assert result.loc[1, 'diff_defense'] == result.loc[1, 'home_defense_strength'] - result.loc[1, 'away_defense_strength']


# --- DataPlayerTransformation Tests ---
def test_goals_aggregation(sample_players_df, sample_matches_df):
    pt = DataPlayerTransformation()
    result = pt.transform(sample_players_df, sample_matches_df)
    
    # user 100 has SCORE in match 1, and SCORE in match 2
    user_100_m1 = result[(result['user_id'] == 100) & (result['match_id'] == 1)]
    assert user_100_m1['goals'].iloc[0] == 1
    
def test_perf_score_formula(sample_players_df, sample_matches_df):
    pt = DataPlayerTransformation()
    result = pt.transform(sample_players_df, sample_matches_df)
    
    # user 100 in match 1: SCORE (1), YELLOWCARD (1)
    user_100_m1 = result[(result['user_id'] == 100) & (result['match_id'] == 1)].iloc[0]
    expected_score = (1 * 3) + (0 * 2) + (0 * 1.5) - (0 * 0.5) - 1 - (0 * 3)
    assert user_100_m1['perf_score'] == expected_score

def test_rolling_shift(sample_players_df, sample_matches_df):
    pt = DataPlayerTransformation()
    result = pt.transform(sample_players_df, sample_matches_df)
    
    user_100 = result[result['user_id'] == 100].sort_values('match_date')
    
    # First match rolling should be NaN or 0 (since it is shifted, rolling over previous)
    assert pd.isna(user_100['roll3_goals'].iloc[0]) or user_100['roll3_goals'].iloc[0] == 0
    # Second match should reflect first match's goal
    assert user_100['roll3_goals'].iloc[1] == 1

def test_is_best_player(sample_players_df, sample_matches_df):
    pt = DataPlayerTransformation()
    result = pt.transform(sample_players_df, sample_matches_df)
    
    # match 1 best player
    m1 = result[result['match_id'] == 1]
    best_players = m1[m1['is_best_player'] == 1]
    
    assert len(best_players) > 0
    assert best_players['perf_score'].iloc[0] == m1['perf_score'].max()

def test_filter_team_id_zero(sample_players_df, sample_matches_df):
    pt = DataPlayerTransformation()
    df = sample_players_df.copy()
    df.loc[0, 'team_id'] = 0
    
    result = pt.transform(df, sample_matches_df)
    assert 0 not in result['team_id'].values

def test_player_name_concat(sample_players_df, sample_matches_df):
    pt = DataPlayerTransformation()
    result = pt.transform(sample_players_df, sample_matches_df)
    assert "John Doe" in result['player_name'].values
