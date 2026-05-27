import pandas as pd
import numpy as np

class DataIngestion:

    def __init__(self):
        self.match_path = "data/series-futsal-men-matches.csv"
        self.player_path = "data/players-futsal-men-scores.csv"

    def read_match_data(self):
        return pd.read_csv(self.match_path)
    
    def read_player_data(self):
        return pd.read_csv(self.player_path)

    def ingest_match_data(self):
        matches_raw = self.read_match_data()
        matches = matches_raw.copy()
        return matches
    
    def ingest_player_data(self):
        player_raw = self.read_player_data()
        player = player_raw.copy()
        return player

    
    
    