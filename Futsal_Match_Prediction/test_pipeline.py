import pandas as pd
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation

# Fix DataIngestion path for testing
class TestDataIngestion(DataIngestion):
    def __init__(self):
        super().__init__()
        self.match_path = "data/series-futsal-men-matches.csv"

# Monkey patch DataIngestion inside data_transformation module
import components.data_transformation as dt
dt.DataIngestion = TestDataIngestion

transformer = DataTransformation()

try:
    data = TestDataIngestion()
    matches = data.ingest_data()
    print("Ingested data shape:", matches.shape)
    
    # Drop columns
    matches = transformer.drop_columns(matches)
    
    # Remove Invalid Matches
    matches = transformer.remove_invalid_matches(matches)
    
    # Remove Non Completed Matches
    matches = transformer.remove_non_completed_matches(matches)
    
    # Convert Date & Sort
    matches = transformer.clean_date(matches)
    
    # Clean Draw Column
    matches = transformer.clean_draw(matches)
    
    print("Draws count:", matches['isDraw'].sum())
    
    # Create Outcome
    transformer.create_outcome(matches)
    
    # Create Home Away Goals
    transformer.create_home_away_goals(matches)
    
    # Print sample of winningTeam, homeTeamId, awayTeamId, outcome, isDraw, home_goals, away_goals
    print(matches[['winningTeam', 'homeTeamId', 'awayTeamId', 'outcome', 'isDraw', 'home_goals', 'away_goals']].head(20))
    
    # Check the create_features bug
    matches_after_create = transformer.create_features(matches)
    print("Type of matches after create_features:", type(matches_after_create))
    
except Exception as e:
    import traceback
    traceback.print_exc()
