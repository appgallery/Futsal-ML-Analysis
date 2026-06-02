import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine

class DataIngestion:

    def __init__(self):
        # --- OLD CSV LOGIC (Commented out for Production Reversion) ---
        # self.match_path = "data/series-futsal-men-matches.csv"
        # self.player_path = "data/players-futsal-men-scores.csv"

        # --- NEW DATABASE LOGIC ---
        # Get connection string from environment variables, fallback to None
        self.db_uri = os.environ.get("DATABASE_URI")
        if self.db_uri:
            self.engine = create_engine(self.db_uri)
        else:
            self.engine = None
            print("WARNING: DATABASE_URI environment variable not set. Please set it before running in production.")

    def read_match_data(self):
        # --- OLD CSV LOGIC ---
        # return pd.read_csv(self.match_path)

        # --- NEW DATABASE LOGIC ---
        if self.engine:
            # Assumes table name is 'futsal_men_matches'. Change if different.
            return pd.read_sql("SELECT * FROM futsal_men_matches", self.engine)
        else:
            raise ValueError("Cannot read match data from database: DATABASE_URI is not set.")
    
    def read_player_data(self):
        # --- OLD CSV LOGIC ---
        # return pd.read_csv(self.player_path)

        # --- NEW DATABASE LOGIC ---
        if self.engine:
            # Assumes table name is 'futsal_men_scores'. Change if different.
            return pd.read_sql("SELECT * FROM futsal_men_scores", self.engine)
        else:
             raise ValueError("Cannot read player data from database: DATABASE_URI is not set.")

    def ingest_match_data(self):
        matches_raw = self.read_match_data()
        matches = matches_raw.copy()
        return matches
    
    def ingest_player_data(self):
        player_raw = self.read_player_data()
        player = player_raw.copy()
        return player